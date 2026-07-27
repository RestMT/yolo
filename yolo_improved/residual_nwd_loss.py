from __future__ import annotations

import torch

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.ops import xyxy2xywh

from .residual_nwd_config import ResidualNWDLossConfig


class ResidualNWDBboxLoss(BboxLoss):
    """Stock CIoU plus an optional residual NWD term."""

    def __init__(self, reg_max: int = 16, config: ResidualNWDLossConfig | None = None):
        """Initialize the stock bounding-box loss and the local residual NWD configuration."""
        super().__init__(reg_max)
        self.config = config or ResidualNWDLossConfig()

    def calculate_alpha(self, target_area: torch.Tensor) -> torch.Tensor:
        """Return the per-box NWD coefficient for the configured mode."""
        if self.config.mode == "control":
            return torch.zeros_like(target_area)
        if self.config.mode == "constant-005":
            return torch.full_like(target_area, 0.05)
        if self.config.mode == "constant-010":
            return torch.full_like(target_area, 0.10)

        eps = target_area.new_tensor(self.config.eps)
        area_threshold = target_area.new_tensor(self.config.area_threshold)
        transition = torch.sigmoid(
            self.config.area_slope
            * (torch.log(target_area.clamp_min(0) + eps) - torch.log(area_threshold + eps))
        )
        return (self.config.alpha_max * (1.0 - transition)).clamp(0, self.config.alpha_max)

    def calculate_nwd_loss(self, pred_xywh: torch.Tensor, target_xywh: torch.Tensor) -> torch.Tensor:
        """Return NWD loss from normalized center-size bounding boxes."""
        eps = pred_xywh.new_tensor(self.config.eps)
        pred_wh = pred_xywh[..., 2:].clamp_min(eps)
        target_wh = target_xywh[..., 2:].clamp_min(eps)
        center_distance = (pred_xywh[..., :2] - target_xywh[..., :2]).pow(2).sum(-1, keepdim=True)
        size_distance = (pred_wh - target_wh).pow(2).sum(-1, keepdim=True) / 4
        wasserstein_distance = (center_distance + size_distance).clamp_min(0)
        normalized_distance = torch.sqrt(wasserstein_distance + eps) / pred_xywh.new_tensor(self.config.nwd_scale)
        return -torch.expm1(-normalized_distance)

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
        """Compute stock CIoU/DFL and add residual NWD outside control mode."""
        stock_box_loss, loss_dfl = super().forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride,
        )
        if self.config.mode == "control":
            return stock_box_loss, loss_dfl

        weight = target_scores[fg_mask].sum(-1, keepdim=True).float()
        batch_size, num_anchors = pred_bboxes.shape[:2]
        foreground_stride = stride.reshape(1, num_anchors, -1).expand(batch_size, -1, -1)[fg_mask].float()
        image_scale = imgsz[[1, 0, 1, 0]].float()
        pred_xywh = xyxy2xywh(pred_bboxes[fg_mask].float() * foreground_stride / image_scale)
        target_xywh = xyxy2xywh(target_bboxes[fg_mask].float() * foreground_stride / image_scale)

        eps = target_xywh.new_tensor(self.config.eps)
        target_area = target_xywh[..., 2:].clamp_min(eps).prod(-1, keepdim=True)
        residual_nwd = self.calculate_alpha(target_area) * self.calculate_nwd_loss(pred_xywh, target_xywh)
        residual_nwd = (residual_nwd * weight).sum() / target_scores_sum
        return stock_box_loss + residual_nwd, loss_dfl


class ResidualNWDDetectionLoss(v8DetectionLoss):
    """Detection loss that replaces only the bounding-box criterion with residual NWD."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        config: ResidualNWDLossConfig | None = None,
    ):
        """Initialize the stock detection loss and replace only its bounding-box criterion."""
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if config is None:
            config = getattr(model, "residual_nwd_loss_config", None)
        self.bbox_loss = ResidualNWDBboxLoss(self.reg_max, config).to(self.device)
