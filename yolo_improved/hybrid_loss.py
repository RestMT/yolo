from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.tal import bbox2dist

from .hybrid_config import HybridLossConfig


class HybridBboxLoss(BboxLoss):
    """Bounding-box loss with scale-adaptive CIoU, NWD, and shape terms."""

    def __init__(self, reg_max: int = 16, config: HybridLossConfig | None = None):
        """Initialize the hybrid localization loss while retaining the stock DFL/L1 term."""
        super().__init__(reg_max)
        self.config = config or HybridLossConfig()

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the hybrid box loss and the unchanged stock DFL/L1 loss."""
        weight = target_scores[fg_mask].sum(-1, keepdim=True)

        # Both inputs arrive in feature-grid coordinates. Convert them to the same image-normalized xyxy scale.
        batch_size, num_anchors = pred_bboxes.shape[:2]
        foreground_stride = stride.reshape(1, num_anchors, -1).expand(batch_size, -1, -1)[fg_mask].float()
        image_scale = imgsz[[1, 0, 1, 0]].float()
        pred_xyxy = pred_bboxes[fg_mask].float() * foreground_stride / image_scale
        target_xyxy = target_bboxes[fg_mask].float() * foreground_stride / image_scale

        ciou_loss = 1.0 - bbox_iou(pred_xyxy, target_xyxy, xywh=False, CIoU=True)
        pred_xywh = xyxy2xywh(pred_xyxy)
        target_xywh = xyxy2xywh(target_xyxy)
        pred_wh = pred_xywh[..., 2:].clamp_min(0)
        target_wh = target_xywh[..., 2:].clamp_min(0)

        eps = pred_xywh.new_tensor(self.config.eps)
        target_area = target_wh.prod(-1, keepdim=True)
        area_threshold = target_area.new_tensor(self.config.area_threshold)
        ciou_weight = torch.sigmoid(
            self.config.area_slope
            * (torch.log(target_area + eps) - torch.log(area_threshold + eps))
        )

        center_distance = (pred_xywh[..., :2] - target_xywh[..., :2]).pow(2).sum(-1, keepdim=True)
        size_distance = (pred_wh - target_wh).pow(2).sum(-1, keepdim=True) / 4
        wasserstein_distance = center_distance + size_distance
        nwd_denominator = pred_xywh.new_tensor(self.config.nwd_scale**2) + eps
        nwd_loss = -torch.expm1(-wasserstein_distance / nwd_denominator)

        pred_aspect = torch.log(pred_wh[..., :1] + eps) - torch.log(pred_wh[..., 1:] + eps)
        target_aspect = torch.log(target_wh[..., :1] + eps) - torch.log(target_wh[..., 1:] + eps)
        shape_loss = F.smooth_l1_loss(pred_aspect, target_aspect, reduction="none")

        hybrid_loss = (
            ciou_weight * ciou_loss
            + (1.0 - ciou_weight) * nwd_loss
            + self.config.shape_weight * shape_loss
        )
        loss_box = (hybrid_loss * weight.float()).sum() / target_scores_sum

        # Keep the stock DFL/L1 computation unchanged.
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_box, loss_dfl


class HybridDetectionLoss(v8DetectionLoss):
    """Detection loss that changes only the bounding-box loss implementation."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        config: HybridLossConfig | None = None,
    ):
        """Initialize the stock detection loss and replace only its bounding-box criterion."""
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        config = config or getattr(model, "hybrid_loss_config", None)
        self.bbox_loss = HybridBboxLoss(self.reg_max, config).to(self.device)
