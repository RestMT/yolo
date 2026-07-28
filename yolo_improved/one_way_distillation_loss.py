from __future__ import annotations

from functools import partial
from typing import Any

import torch

from ultralytics.utils.loss import E2ELoss

from .contrast_ring_loss import ContrastRingDetectionLoss
from .mutual_distillation_config import E2_1B_CONFIG
from .mutual_distillation_loss import MutualDistillationE2ELoss, _bernoulli_kl_loss
from .one_way_distillation_config import (
    OneWayDistillationConfig,
    resolve_one_way_distillation_config,
)


def _one2many_teacher_weights(
    one2many_logits: torch.Tensor,
    one2one_logits: torch.Tensor,
    one2many_fg_mask: torch.Tensor,
    confidence_temperature: float,
) -> torch.Tensor:
    """Return detached one-to-many teacher weights without computing the reverse direction."""
    if one2many_logits.shape != one2one_logits.shape:
        raise ValueError(
            f"classification logits must have matching shapes, got "
            f"{tuple(one2many_logits.shape)} and {tuple(one2one_logits.shape)}."
        )
    if one2many_fg_mask.shape != one2many_logits.shape[:2]:
        raise ValueError(
            f"one2many foreground mask must have shape {tuple(one2many_logits.shape[:2])}, "
            f"got {tuple(one2many_fg_mask.shape)}."
        )

    confidence_many = one2many_logits.detach().float().sigmoid().amax(dim=-1)
    confidence_one = one2one_logits.detach().float().sigmoid().amax(dim=-1)
    weights = (
        one2many_fg_mask.detach().float()
        * confidence_many
        * torch.sigmoid((confidence_many - confidence_one) / confidence_temperature)
    )
    return weights.detach()


class OneWayDistillationE2ELoss(MutualDistillationE2ELoss):
    """E2.1b supervision with one-to-many to one-to-one classification distillation."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: OneWayDistillationConfig | dict | None = None,
    ):
        """Initialize fixed E2.1b branches and the stock E2E weight schedule."""
        if config is None:
            config = getattr(model, "one_way_distillation_config", None)
        self.config = resolve_one_way_distillation_config(config)
        loss_fn = partial(ContrastRingDetectionLoss, config=E2_1B_CONFIG)
        E2ELoss.__init__(self, model, loss_fn=loss_fn)
        self.hyp = self.one2many.hyp
        zero = torch.zeros((), device=self.one2many.device)
        self.distill_cls_loss = zero
        self.distill_box_loss = zero
        self.distill_ramp = zero

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate E2.1b supervision and one-way classification distillation."""
        parsed_preds = self.one2many.parse_output(preds)
        if not isinstance(parsed_preds, dict) or "one2many" not in parsed_preds or "one2one" not in parsed_preds:
            raise TypeError("end-to-end predictions must contain one2many and one2one branches.")
        one2many_preds = parsed_preds["one2many"]
        one2one_preds = parsed_preds["one2one"]
        one2many_logits, one2one_logits, _, _ = self._prepare_branch_predictions(one2many_preds, one2one_preds)

        one2many_assignment, one2many_loss, _ = self.one2many.get_assigned_targets_and_loss(one2many_preds, batch)
        one2one_assignment, one2one_loss, one2one_items = self.one2one.get_assigned_targets_and_loss(
            one2one_preds, batch
        )
        batch_size, number_of_anchors = one2many_logits.shape[:2]
        self._validate_assignments(one2many_assignment, one2one_assignment, batch_size, number_of_anchors)

        supervised_loss = one2many_loss * batch_size * self.o2m + one2one_loss * batch_size * self.o2o
        ramp = self.calculate_ramp()
        zero = one2many_logits.new_zeros((), dtype=torch.float32)
        distill_cls_loss = zero
        scaled_cls_loss = zero

        distillation_enabled = ramp > 0 and self.config.classification_gain > 0
        if distillation_enabled:
            one2many_fg_mask = one2many_assignment[0]
            teacher_weights = _one2many_teacher_weights(
                one2many_logits.detach(),
                one2one_logits.detach(),
                one2many_fg_mask,
                self.config.confidence_temperature,
            )
            distill_cls_loss = _bernoulli_kl_loss(
                one2many_logits.detach(),
                one2one_logits,
                teacher_weights,
                self.config.temperature,
                self.config.eps,
            )
            scaled_cls_loss = ramp * self.config.classification_gain * self.hyp.cls * distill_cls_loss

        self.distill_cls_loss = distill_cls_loss.detach()
        self.distill_box_loss = zero
        self.distill_ramp = one2many_logits.new_tensor(ramp, dtype=torch.float32).detach()

        if not distillation_enabled:
            return supervised_loss, one2one_items

        distillation_vector = torch.stack((zero, scaled_cls_loss, zero))
        loss_items = dict(one2one_items)
        loss_items["cls_loss"] = loss_items["cls_loss"] + scaled_cls_loss.detach()
        return supervised_loss + distillation_vector * batch_size, loss_items
