from __future__ import annotations

from collections.abc import Sequence

import torch

from .class_balanced_config import (
    ClassBalancedPositiveConfig,
    resolve_class_balanced_positive_config,
    validate_class_balanced_positive_weights,
    validate_positive_class_weights,
)
from .contrast_ring_config import ContrastRingLossConfig, resolve_contrast_ring_config
from .contrast_ring_loss import ContrastRingBCEWithLogitsLoss, ContrastRingDetectionLoss
from .mutual_distillation_config import E2_1B_CONFIG


class ClassBalancedContrastRingBCEWithLogitsLoss(ContrastRingBCEWithLogitsLoss):
    """E2.1b BCE with an additional class multiplier only on positive elements."""

    def __init__(
        self,
        positive_class_weights: torch.Tensor | Sequence[float],
        config: ContrastRingLossConfig | dict | None = None,
    ):
        """Register validated mean-one positive class weights without changing E2.1b."""
        super().__init__(config=config)
        weights = validate_positive_class_weights(positive_class_weights)
        self.register_buffer("positive_class_weights", weights.view(1, 1, -1))

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """Apply class weights to soft positive targets while leaving every negative unchanged."""
        bce_loss = super().forward(pred_scores, target_scores)
        number_of_classes = pred_scores.shape[-1]
        if self.positive_class_weights.shape != (1, 1, number_of_classes):
            raise ValueError(
                f"positive class weights must have shape {(1, 1, number_of_classes)}, "
                f"got {tuple(self.positive_class_weights.shape)}."
            )

        positive_mask = target_scores > 0
        class_weights = self.positive_class_weights.to(device=bce_loss.device, dtype=bce_loss.dtype)
        positive_only_multiplier = torch.where(positive_mask, class_weights, class_weights.new_ones(()))
        return bce_loss * positive_only_multiplier


class ClassBalancedContrastRingDetectionLoss(ContrastRingDetectionLoss):
    """Fixed E2.1b supervision with class-balanced weights on positive classification elements only."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        positive_class_weights: torch.Tensor | Sequence[float] | None = None,
        class_balanced_config: ClassBalancedPositiveConfig | dict | None = None,
        config: ContrastRingLossConfig | dict | None = None,
    ):
        """Initialize E4 while explicitly disabling the stock global class-weight route."""
        if getattr(model, "class_weights", None) is not None:
            raise ValueError(
                "E4 does not allow model.class_weights because it would also weight negative classification elements."
            )
        if positive_class_weights is None:
            positive_class_weights = getattr(model, "positive_class_weights", None)
        if class_balanced_config is None:
            class_balanced_config = getattr(model, "class_balanced_positive_config", None)
        self.class_balanced_positive_config = resolve_class_balanced_positive_config(class_balanced_config)

        contrast_config = resolve_contrast_ring_config(config)
        if contrast_config != E2_1B_CONFIG:
            raise ValueError("E4 requires the fixed E2.1b contrast-ring configuration.")
        super().__init__(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            config=contrast_config,
        )

        weights = validate_class_balanced_positive_weights(
            positive_class_weights,
            config=self.class_balanced_positive_config,
            number_of_classes=self.nc,
        )

        if self.class_weights is not None:
            raise RuntimeError("stock global class weights must remain disabled in E4.")
        self.class_weights = None
        self.bce = ClassBalancedContrastRingBCEWithLogitsLoss(
            positive_class_weights=weights,
            config=contrast_config,
        ).to(self.device)
