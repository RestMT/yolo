# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Class-conditional suppression supervision layered on unchanged E10 DGQM."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .ccs_config import (
    ClassConditionalSuppressionConfig,
    resolve_class_conditional_suppression_config,
)
from .contrast_ring_config import ContrastRingLossConfig
from .dgqm_config import DualGeometryQualityConfig
from .dgqm_loss import DualGeometryQualityDetectionLoss


class ClassConditionalSuppressionDetectionLoss(DualGeometryQualityDetectionLoss):
    """E10 detection loss plus detached class-conditional hard-negative suppression."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        contrast_config: ContrastRingLossConfig | dict | None = None,
        quality_config: DualGeometryQualityConfig | dict | None = None,
        suppression_config: ClassConditionalSuppressionConfig | dict | None = None,
    ):
        """Initialize unchanged E10 supervision and one local suppression component."""
        self.suppression_config = resolve_class_conditional_suppression_config(suppression_config)
        super().__init__(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            contrast_config=contrast_config,
            quality_config=quality_config,
        )
        self.trainable_suppression = bool(getattr(model.model[-1], "trainable_suppression", True))
        self.loss_names = (*self.loss_names, "suppression_loss")

    def calculate_suppression_loss(
        self,
        preds: dict[str, torch.Tensor],
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Return neutral-positive plus probability-weighted hard-negative suppression loss."""
        suppression_prediction = torch.tanh(
            preds["class_suppression"].permute(0, 2, 1).contiguous()
        )
        differentiable_zero = suppression_prediction.sum() * 0.0
        if not self.trainable_suppression or self.suppression_config.suppression_gain == 0:
            return differentiable_zero

        positive_class_mask = target_scores > 0
        negative_class_mask = ~positive_class_mask

        if positive_class_mask.any():
            positive_prediction = suppression_prediction[positive_class_mask].float()
            positive_neutral_loss = F.smooth_l1_loss(
                positive_prediction,
                torch.zeros_like(positive_prediction),
                reduction="none",
            )
            positive_weight = target_scores[positive_class_mask].float()
            positive_neutral_loss = (
                positive_neutral_loss * positive_weight
            ).sum() / positive_weight.sum().clamp_min(self.suppression_config.eps)
        else:
            positive_neutral_loss = differentiable_zero

        if negative_class_mask.any():
            class_probability = (
                preds["scores"].permute(0, 2, 1).detach().sigmoid().float()
            )
            hard_negative_weight = class_probability.pow(self.suppression_config.hard_negative_gamma)
            hard_negative_weight = hard_negative_weight * (
                class_probability >= self.suppression_config.hard_negative_min_probability
            )
            negative_prediction = suppression_prediction[negative_class_mask].float()
            negative_target = -torch.ones_like(negative_prediction)
            negative_loss = F.smooth_l1_loss(
                negative_prediction,
                negative_target,
                reduction="none",
            )
            negative_weight = hard_negative_weight[negative_class_mask]
            negative_loss = (
                negative_loss * negative_weight
            ).sum() / negative_weight.sum().clamp_min(self.suppression_config.eps)
        else:
            negative_loss = differentiable_zero

        return self.suppression_config.suppression_gain * (
            self.suppression_config.positive_neutral_weight * positive_neutral_loss
            + negative_loss
        )

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict) -> tuple:
        """Reuse the single E10 assignment and append isolated suppression supervision."""
        assignment_output: list[tuple] = []
        handle = self.assigner.register_forward_hook(lambda _, __, output: assignment_output.append(output))
        try:
            assigned, base_loss, base_items = super().get_assigned_targets_and_loss(preds, batch)
        finally:
            handle.remove()
        if len(assignment_output) != 1:
            raise RuntimeError(f"E11 expected one assignment pass, observed {len(assignment_output)}.")

        target_scores = assignment_output[0][2]
        suppression_loss = self.calculate_suppression_loss(preds, target_scores)
        loss = torch.cat((base_loss, suppression_loss.reshape(1)))
        return assigned, loss, {**base_items, "suppression_loss": suppression_loss.detach()}
