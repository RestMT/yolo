# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Class-agnostic foregroundness supervision layered on unchanged E10 DGQM."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .contrast_ring_config import ContrastRingLossConfig
from .dgqm_config import DualGeometryQualityConfig
from .dgqm_loss import DualGeometryQualityDetectionLoss
from .foreground_config import (
    ForegroundnessFactorizationConfig,
    resolve_foregroundness_factorization_config,
)


class ForegroundnessFactorizedDetectionLoss(DualGeometryQualityDetectionLoss):
    """E10 detection loss plus detached anchor-level foregroundness supervision."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        contrast_config: ContrastRingLossConfig | dict | None = None,
        quality_config: DualGeometryQualityConfig | dict | None = None,
        foreground_config: ForegroundnessFactorizationConfig | dict | None = None,
    ):
        """Initialize unchanged E10 supervision and one local foreground component."""
        self.foreground_config = resolve_foregroundness_factorization_config(foreground_config)
        super().__init__(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            contrast_config=contrast_config,
            quality_config=quality_config,
        )
        self.trainable_foreground = bool(getattr(model.model[-1], "trainable_foreground", True))
        self.loss_names = (*self.loss_names, "foreground_loss")

    def calculate_foreground_loss(
        self,
        preds: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Return positive-anchor plus probability-weighted background foreground loss."""
        foreground_prediction = torch.tanh(
            preds["foreground"].permute(0, 2, 1).contiguous()
        )
        differentiable_zero = foreground_prediction.sum() * 0.0
        if not self.trainable_foreground or self.foreground_config.foreground_gain == 0:
            return differentiable_zero

        positive_mask = fg_mask.bool()
        negative_mask = ~positive_mask

        if positive_mask.any():
            positive_prediction = foreground_prediction[positive_mask].float()
            positive_target = torch.ones_like(positive_prediction)
            positive_loss = F.smooth_l1_loss(
                positive_prediction,
                positive_target,
                reduction="none",
            )
            positive_weight = target_scores.max(dim=-1).values[positive_mask].float()
            positive_loss = (
                positive_loss.squeeze(-1) * positive_weight
            ).sum() / positive_weight.sum().clamp_min(self.foreground_config.eps)
        else:
            positive_loss = differentiable_zero

        if negative_mask.any():
            anchor_probability = (
                preds["scores"]
                .permute(0, 2, 1)
                .detach()
                .sigmoid()
                .amax(dim=-1)
                .float()
            )
            hard_negative_weight = anchor_probability.pow(
                self.foreground_config.hard_negative_gamma
            )
            hard_negative_weight = hard_negative_weight * (
                anchor_probability >= self.foreground_config.hard_negative_min_probability
            )
            negative_prediction = foreground_prediction[negative_mask].float()
            negative_target = -torch.ones_like(negative_prediction)
            negative_loss = F.smooth_l1_loss(
                negative_prediction,
                negative_target,
                reduction="none",
            )
            negative_anchor_weight = hard_negative_weight[negative_mask]
            negative_loss = (
                negative_loss.squeeze(-1) * negative_anchor_weight
            ).sum() / negative_anchor_weight.sum().clamp_min(self.foreground_config.eps)
        else:
            negative_loss = differentiable_zero

        return self.foreground_config.foreground_gain * (
            positive_loss + self.foreground_config.negative_weight * negative_loss
        )

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict) -> tuple:
        """Reuse the single E10 assignment and append isolated foreground supervision."""
        assignment_output: list[tuple] = []
        handle = self.assigner.register_forward_hook(lambda _, __, output: assignment_output.append(output))
        try:
            assigned, base_loss, base_items = super().get_assigned_targets_and_loss(preds, batch)
        finally:
            handle.remove()
        if len(assignment_output) != 1:
            raise RuntimeError(f"E12 expected one assignment pass, observed {len(assignment_output)}.")

        fg_mask = assigned[0]
        target_scores = assignment_output[0][2]
        foreground_loss = self.calculate_foreground_loss(preds, fg_mask, target_scores)
        loss = torch.cat((base_loss, foreground_loss.reshape(1)))
        return assigned, loss, {**base_items, "foreground_loss": foreground_loss.detach()}
