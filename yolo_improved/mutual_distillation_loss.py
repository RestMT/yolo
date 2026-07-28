from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import E2ELoss

from .contrast_ring_loss import ContrastRingDetectionLoss
from .mutual_distillation_config import (
    E2_1B_CONFIG,
    MutualDistillationConfig,
    resolve_mutual_distillation_config,
)


def _directional_weights(
    one2many_logits: torch.Tensor,
    one2one_logits: torch.Tensor,
    one2many_fg_mask: torch.Tensor,
    one2one_fg_mask: torch.Tensor,
    confidence_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached adaptive teacher weights for both branch directions."""
    if one2many_logits.shape != one2one_logits.shape:
        raise ValueError(
            f"classification logits must have matching shapes, got "
            f"{tuple(one2many_logits.shape)} and {tuple(one2one_logits.shape)}."
        )
    expected_mask_shape = one2many_logits.shape[:2]
    if one2many_fg_mask.shape != expected_mask_shape or one2one_fg_mask.shape != expected_mask_shape:
        raise ValueError(
            f"foreground masks must have shape {tuple(expected_mask_shape)}, got "
            f"{tuple(one2many_fg_mask.shape)} and {tuple(one2one_fg_mask.shape)}."
        )

    confidence_many = one2many_logits.detach().float().sigmoid().amax(dim=-1)
    confidence_one = one2one_logits.detach().float().sigmoid().amax(dim=-1)
    many_to_one = (
        one2many_fg_mask.detach().float()
        * confidence_many
        * torch.sigmoid((confidence_many - confidence_one) / confidence_temperature)
    )
    one_to_many = (
        one2one_fg_mask.detach().float()
        * confidence_one
        * torch.sigmoid((confidence_one - confidence_many) / confidence_temperature)
    )
    return many_to_one.detach(), one_to_many.detach()


def _bernoulli_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    """Return weighted Bernoulli KL with a detached teacher and float32 arithmetic."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"teacher and student logits must have matching shapes, got "
            f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}."
        )
    if weights.shape != student_logits.shape[:2]:
        raise ValueError(
            f"weights must have shape {tuple(student_logits.shape[:2])}, got {tuple(weights.shape)}."
        )

    detached_weights = weights.detach().float()
    active_mask = detached_weights > 0
    teacher_probability = (
        (teacher_logits.detach().float()[active_mask] / temperature).sigmoid().clamp(eps, 1.0 - eps)
    )
    student_probability = (student_logits.float()[active_mask] / temperature).sigmoid().clamp(eps, 1.0 - eps)
    divergence = (
        teacher_probability * (teacher_probability.log() - student_probability.log())
        + (1.0 - teacher_probability)
        * (torch.log1p(-teacher_probability) - torch.log1p(-student_probability))
    )
    per_anchor_loss = divergence.mean(dim=-1)
    active_weights = detached_weights[active_mask]
    denominator = active_weights.sum().clamp_min(eps)
    return temperature**2 * (per_anchor_loss * active_weights).sum() / denominator


def _smooth_l1_loss(
    teacher_boxes: torch.Tensor,
    student_boxes: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return weighted Smooth L1 with a detached teacher and float32 arithmetic."""
    if teacher_boxes.shape != student_boxes.shape:
        raise ValueError(
            f"teacher and student boxes must have matching shapes, got "
            f"{tuple(teacher_boxes.shape)} and {tuple(student_boxes.shape)}."
        )
    if teacher_boxes.shape[-1] != 4:
        raise ValueError(f"decoded boxes must end in four coordinates, got {tuple(teacher_boxes.shape)}.")
    if weights.shape != student_boxes.shape[:2]:
        raise ValueError(f"weights must have shape {tuple(student_boxes.shape[:2])}, got {tuple(weights.shape)}.")

    detached_weights = weights.detach().float()
    active_mask = detached_weights > 0
    per_anchor_loss = F.smooth_l1_loss(
        student_boxes.float()[active_mask],
        teacher_boxes.detach().float()[active_mask],
        reduction="none",
    ).mean(dim=-1)
    active_weights = detached_weights[active_mask]
    denominator = active_weights.sum().clamp_min(eps)
    return (per_anchor_loss * active_weights).sum() / denominator


class MutualDistillationE2ELoss(E2ELoss):
    """E2.1b supervision with adaptive mutual distillation between YOLO26 branches."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: MutualDistillationConfig | dict | None = None,
    ):
        """Initialize fixed E2.1b branch criteria and the stock E2E weight schedule."""
        if config is None:
            config = getattr(model, "mutual_distillation_config", None)
        self.config = resolve_mutual_distillation_config(config)
        loss_fn = partial(ContrastRingDetectionLoss, config=E2_1B_CONFIG)
        super().__init__(model, loss_fn=loss_fn)
        self.hyp = self.one2many.hyp
        zero = torch.zeros((), device=self.one2many.device)
        self.distill_cls_loss = zero
        self.distill_box_loss = zero
        self.distill_ramp = zero

    def calculate_ramp(self) -> float:
        """Return the distillation ramp for the current zero-based epoch."""
        if self.updates < self.config.start_epoch:
            return 0.0
        return min((self.updates - self.config.start_epoch + 1) / self.config.warmup_epochs, 1.0)

    def _prepare_branch_predictions(
        self,
        one2many_preds: dict[str, torch.Tensor],
        one2one_preds: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate aligned branch layouts and return anchor-major logits and box distributions."""
        required_keys = {"boxes", "scores", "feats"}
        for name, branch in (("one2many", one2many_preds), ("one2one", one2one_preds)):
            if not isinstance(branch, dict):
                raise TypeError(f"{name} predictions must be a dict, got {type(branch).__name__}.")
            missing_keys = required_keys.difference(branch)
            if missing_keys:
                raise KeyError(f"{name} predictions are missing keys: {sorted(missing_keys)}.")

        features_many = one2many_preds["feats"]
        features_one = one2one_preds["feats"]
        if not isinstance(features_many, (list, tuple)) or not isinstance(features_one, (list, tuple)):
            raise TypeError("branch features must be ordered lists or tuples of detection levels.")
        if not features_many or len(features_many) != len(features_one):
            raise RuntimeError(
                f"branch detection-level counts must match and be nonzero, got "
                f"{len(features_many)} and {len(features_one)}."
            )
        if self.one2many.nc != self.one2one.nc or self.one2many.reg_max != self.one2one.reg_max:
            raise RuntimeError("branch class counts and reg_max values must match.")
        if len(features_many) != len(self.one2many.stride) or len(features_one) != len(self.one2one.stride):
            raise RuntimeError(
                f"branch detection-level counts must match their stride counts, got "
                f"{len(features_many)}/{len(self.one2many.stride)} and "
                f"{len(features_one)}/{len(self.one2one.stride)}."
            )

        for level, (feature_many, feature_one) in enumerate(zip(features_many, features_one)):
            if feature_many.ndim != 4 or feature_one.ndim != 4:
                raise RuntimeError(f"detection-level {level} features must be four-dimensional.")
            if feature_many.shape[0] != feature_one.shape[0] or feature_many.shape[-2:] != feature_one.shape[-2:]:
                raise RuntimeError(
                    f"detection-level {level} layouts do not match: "
                    f"{tuple(feature_many.shape)} versus {tuple(feature_one.shape)}."
                )

        boxes_many, boxes_one = one2many_preds["boxes"], one2one_preds["boxes"]
        scores_many, scores_one = one2many_preds["scores"], one2one_preds["scores"]
        for name, boxes, scores, features in (
            ("one2many", boxes_many, scores_many, features_many),
            ("one2one", boxes_one, scores_one, features_one),
        ):
            if boxes.ndim != 3 or scores.ndim != 3:
                raise RuntimeError(f"{name} boxes and scores must be three-dimensional.")
            if boxes.shape[0] != scores.shape[0] or boxes.shape[-1] != scores.shape[-1]:
                raise RuntimeError(
                    f"{name} box and score layouts do not match: {tuple(boxes.shape)} versus {tuple(scores.shape)}."
                )
            if any(feature.shape[0] != boxes.shape[0] for feature in features):
                raise RuntimeError(f"{name} feature and prediction batch sizes do not match.")
            if boxes.shape[1] != 4 * self.one2many.reg_max:
                raise RuntimeError(
                    f"{name} box channels must equal 4 * reg_max={4 * self.one2many.reg_max}, "
                    f"got {boxes.shape[1]}."
                )
            if scores.shape[1] != self.one2many.nc:
                raise RuntimeError(f"{name} score channels must equal nc={self.one2many.nc}, got {scores.shape[1]}.")
            expected_anchors = sum(feature.shape[-2] * feature.shape[-1] for feature in features)
            if boxes.shape[-1] != expected_anchors:
                raise RuntimeError(
                    f"{name} anchor count {boxes.shape[-1]} does not match its detection levels ({expected_anchors})."
                )

        if boxes_many.shape != boxes_one.shape or scores_many.shape != scores_one.shape:
            raise RuntimeError(
                f"branch prediction layouts must match, got boxes {tuple(boxes_many.shape)} and "
                f"{tuple(boxes_one.shape)}, scores {tuple(scores_many.shape)} and {tuple(scores_one.shape)}."
            )

        return (
            scores_many.permute(0, 2, 1).contiguous(),
            scores_one.permute(0, 2, 1).contiguous(),
            boxes_many.permute(0, 2, 1).contiguous(),
            boxes_one.permute(0, 2, 1).contiguous(),
        )

    @staticmethod
    def _validate_assignments(
        one2many_assignment: tuple,
        one2one_assignment: tuple,
        batch_size: int,
        number_of_anchors: int,
    ) -> None:
        """Validate assignment shapes and exact anchor/stride identity across branches."""
        for name, assignment in (("one2many", one2many_assignment), ("one2one", one2one_assignment)):
            if len(assignment) != 5:
                raise RuntimeError(f"{name} assignment must contain five tensors, got {len(assignment)}.")
            fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assignment
            if fg_mask.shape != (batch_size, number_of_anchors):
                raise RuntimeError(
                    f"{name} foreground mask must have shape {(batch_size, number_of_anchors)}, "
                    f"got {tuple(fg_mask.shape)}."
                )
            if target_gt_idx.shape != (batch_size, number_of_anchors):
                raise RuntimeError(
                    f"{name} target indices must have shape {(batch_size, number_of_anchors)}, "
                    f"got {tuple(target_gt_idx.shape)}."
                )
            if target_bboxes.shape != (batch_size, number_of_anchors, 4):
                raise RuntimeError(
                    f"{name} target boxes must have shape {(batch_size, number_of_anchors, 4)}, "
                    f"got {tuple(target_bboxes.shape)}."
                )
            if anchor_points.shape != (number_of_anchors, 2):
                raise RuntimeError(
                    f"{name} anchor points must have shape {(number_of_anchors, 2)}, "
                    f"got {tuple(anchor_points.shape)}."
                )
            if stride_tensor.shape != (number_of_anchors, 1):
                raise RuntimeError(
                    f"{name} stride tensor must have shape {(number_of_anchors, 1)}, "
                    f"got {tuple(stride_tensor.shape)}."
                )

        anchor_points_many, stride_many = one2many_assignment[3:]
        anchor_points_one, stride_one = one2one_assignment[3:]
        if not torch.equal(anchor_points_many, anchor_points_one):
            raise RuntimeError("one2many and one2one anchor points or their order do not match.")
        if not torch.equal(stride_many, stride_one):
            raise RuntimeError("one2many and one2one stride tensors or their anchor order do not match.")

    @staticmethod
    def _decode_normalized_boxes(
        criterion: ContrastRingDetectionLoss,
        distributions: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Decode stock DFL boxes and normalize xyxy coordinates to the image scale."""
        image_size = (
            torch.tensor(features[0].shape[-2:], device=distributions.device, dtype=torch.float32)
            * criterion.stride[0].detach().float()
        )
        image_scale = image_size[[1, 0, 1, 0]]
        decoded = criterion.bbox_decode(anchor_points.float(), distributions.float())
        return (decoded.float() * stride_tensor.float() / image_scale).clamp(0.0, 1.0)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate E2.1b supervision and warmed-up bidirectional distillation once per branch."""
        parsed_preds = self.one2many.parse_output(preds)
        if not isinstance(parsed_preds, dict) or "one2many" not in parsed_preds or "one2one" not in parsed_preds:
            raise TypeError("end-to-end predictions must contain one2many and one2one branches.")
        one2many_preds = parsed_preds["one2many"]
        one2one_preds = parsed_preds["one2one"]
        one2many_logits, one2one_logits, one2many_distributions, one2one_distributions = (
            self._prepare_branch_predictions(one2many_preds, one2one_preds)
        )

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
        distill_box_loss = zero
        scaled_cls_loss = zero
        scaled_box_loss = zero

        distillation_enabled = ramp > 0 and (self.config.classification_gain > 0 or self.config.box_gain > 0)
        if distillation_enabled:
            fg_many, _, _, anchor_points_many, stride_many = one2many_assignment
            fg_one, _, _, anchor_points_one, stride_one = one2one_assignment
            many_to_one_weights, one_to_many_weights = _directional_weights(
                one2many_logits,
                one2one_logits,
                fg_many,
                fg_one,
                self.config.confidence_temperature,
            )

            if self.config.classification_gain > 0:
                distill_cls_loss = (
                    _bernoulli_kl_loss(
                        one2many_logits,
                        one2one_logits,
                        many_to_one_weights,
                        self.config.temperature,
                        self.config.eps,
                    )
                    + _bernoulli_kl_loss(
                        one2one_logits,
                        one2many_logits,
                        one_to_many_weights,
                        self.config.temperature,
                        self.config.eps,
                    )
                )

            if self.config.box_gain > 0:
                normalized_many_boxes = self._decode_normalized_boxes(
                    self.one2many,
                    one2many_distributions,
                    anchor_points_many,
                    stride_many,
                    one2many_preds["feats"],
                )
                normalized_one_boxes = self._decode_normalized_boxes(
                    self.one2one,
                    one2one_distributions,
                    anchor_points_one,
                    stride_one,
                    one2one_preds["feats"],
                )
                distill_box_loss = (
                    _smooth_l1_loss(
                        normalized_many_boxes,
                        normalized_one_boxes,
                        many_to_one_weights,
                        self.config.eps,
                    )
                    + _smooth_l1_loss(
                        normalized_one_boxes,
                        normalized_many_boxes,
                        one_to_many_weights,
                        self.config.eps,
                    )
                )

            scaled_cls_loss = ramp * self.config.classification_gain * self.hyp.cls * distill_cls_loss
            scaled_box_loss = ramp * self.config.box_gain * self.hyp.box * distill_box_loss

        self.distill_cls_loss = distill_cls_loss.detach()
        self.distill_box_loss = distill_box_loss.detach()
        self.distill_ramp = one2many_logits.new_tensor(ramp, dtype=torch.float32).detach()

        if not distillation_enabled:
            return supervised_loss, one2one_items

        distillation_vector = torch.stack((scaled_box_loss, scaled_cls_loss, zero))
        loss_items = dict(one2one_items)
        loss_items["box_loss"] = loss_items["box_loss"] + scaled_box_loss.detach()
        loss_items["cls_loss"] = loss_items["cls_loss"] + scaled_cls_loss.detach()
        return supervised_loss + distillation_vector * batch_size, loss_items
