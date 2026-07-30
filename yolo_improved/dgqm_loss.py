# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Dual-geometry quality supervision layered on the unchanged E2.1b loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xyxy2xywh

from .contrast_ring_config import ContrastRingLossConfig
from .contrast_ring_loss import ContrastRingDetectionLoss
from .dgqm_config import DualGeometryQualityConfig, resolve_dual_geometry_quality_config


class DualGeometryQualityDetectionLoss(ContrastRingDetectionLoss):
    """E2.1b detection loss plus detached dual-geometry quality calibration."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        contrast_config: ContrastRingLossConfig | dict | None = None,
        quality_config: DualGeometryQualityConfig | dict | None = None,
    ):
        """Initialize unchanged E2.1b supervision and one local quality component."""
        self.quality_config = resolve_dual_geometry_quality_config(quality_config)
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2, config=contrast_config)
        self.trainable_quality = bool(getattr(model.model[-1], "trainable_quality", True))
        self.loss_names = (*self.loss_names, "quality_loss")

    def calculate_iou_quality(
        self,
        pred_xywh: torch.Tensor,
        target_xywh: torch.Tensor,
    ) -> torch.Tensor:
        """Return plain IoU quality in [0, 1] without CIoU penalties."""
        return bbox_iou(
            pred_xywh,
            target_xywh,
            xywh=True,
            eps=self.quality_config.eps,
        ).clamp(0.0, 1.0)

    def calculate_nwd_similarity(
        self,
        pred_xywh: torch.Tensor,
        target_xywh: torch.Tensor,
    ) -> torch.Tensor:
        """Return E1.1 normalized-Wasserstein similarity in [0, 1]."""
        return (1.0 - self.bbox_loss.calculate_nwd_loss(pred_xywh, target_xywh)).clamp(0.0, 1.0)

    def calculate_quality_target(
        self,
        pred_xywh: torch.Tensor,
        target_xywh: torch.Tensor,
    ) -> torch.Tensor:
        """Return the detached weighted IoU/NWD target in [0, 1]."""
        iou_quality = self.calculate_iou_quality(pred_xywh, target_xywh)
        nwd_quality = self.calculate_nwd_similarity(pred_xywh, target_xywh)
        quality_target = (
            self.quality_config.iou_weight * iou_quality
            + self.quality_config.nwd_weight * nwd_quality
        )
        return quality_target.clamp(0.0, 1.0).detach()

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict) -> tuple:
        """Reuse the E2.1b assignment once and append isolated quality supervision."""
        assignment_output: list[tuple] = []
        handle = self.assigner.register_forward_hook(lambda _, __, output: assignment_output.append(output))
        try:
            assigned, base_loss, base_items = super().get_assigned_targets_and_loss(preds, batch)
        finally:
            handle.remove()
        if len(assignment_output) != 1:
            raise RuntimeError(f"E10 expected one assignment pass, observed {len(assignment_output)}.")
        target_scores = assignment_output[0][2]
        fg_mask, _, target_bboxes, anchor_points, stride_tensor = assigned
        quality_fg_mask = fg_mask.bool()
        quality_logit = preds["quality"].permute(0, 2, 1).contiguous()
        signed_prediction = torch.tanh(quality_logit)
        differentiable_zero = signed_prediction.sum() * 0.0

        if not self.trainable_quality or self.quality_config.quality_gain == 0:
            quality_loss = differentiable_zero
        else:
            if quality_fg_mask.any():
                pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
                pred_bboxes = self.bbox_decode(anchor_points, pred_distri).detach()
                batch_size, num_anchors = pred_bboxes.shape[:2]
                foreground_stride = (
                    stride_tensor.reshape(1, num_anchors, -1)
                    .expand(batch_size, -1, -1)[quality_fg_mask]
                    .float()
                )
                imgsz = (
                    torch.tensor(
                        preds["feats"][0].shape[2:],
                        device=self.device,
                        dtype=pred_bboxes.dtype,
                    )
                    * self.stride[0]
                )
                image_scale = imgsz[[1, 0, 1, 0]].float()
                pred_xywh = xyxy2xywh(
                    pred_bboxes[quality_fg_mask].float() * foreground_stride / image_scale
                )
                target_xywh = xyxy2xywh(target_bboxes[quality_fg_mask].float() / image_scale)
                quality_target = self.calculate_quality_target(pred_xywh, target_xywh)
                signed_target = 2.0 * quality_target - 1.0
                positive_loss = F.smooth_l1_loss(
                    signed_prediction[quality_fg_mask].float(),
                    signed_target,
                    reduction="none",
                )
                positive_weight = target_scores[quality_fg_mask].sum(-1).float()
                positive_loss = (
                    positive_loss.squeeze(-1) * positive_weight
                ).sum() / positive_weight.sum().clamp_min(self.quality_config.eps)
            else:
                positive_loss = differentiable_zero

            if (~quality_fg_mask).any():
                negative_loss = signed_prediction[~quality_fg_mask].square().mean()
            else:
                negative_loss = differentiable_zero
            quality_loss = self.quality_config.quality_gain * (
                positive_loss + self.quality_config.negative_neutral_weight * negative_loss
            )

        loss = torch.cat((base_loss, quality_loss.reshape(1)))
        return assigned, loss, {**base_items, "quality_loss": quality_loss.detach()}
