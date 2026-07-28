from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss
from yolo_improved import (
    ContrastRingDetectionLoss,
    ContrastRingDetectionModel,
    E2_1B_CONFIG,
    MutualDistillationConfig,
    MutualDistillationDetectionModel,
    MutualDistillationE2ELoss,
    MutualDistillationYOLO,
    ResidualNWDBboxLoss,
)
from yolo_improved.contrast_ring_loss import E2_LOCALIZATION_CONFIG
from yolo_improved.mutual_distillation_loss import (
    _bernoulli_kl_loss,
    _directional_weights,
    _smooth_l1_loss,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _synthetic_predictions(
    nc: int,
    reg_max: int,
    identical_branches: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Create small end-to-end head outputs without running model inference."""
    torch.manual_seed(31)
    feature_shapes = ((8, 8), (4, 4), (2, 2))
    number_of_anchors = sum(height * width for height, width in feature_shapes)
    shared_features = [torch.rand(1, 8, height, width) for height, width in feature_shapes]
    many_boxes = torch.randn(1, 4 * reg_max, number_of_anchors, requires_grad=True)
    many_scores = torch.randn(1, nc, number_of_anchors, requires_grad=True)
    if identical_branches:
        one_boxes = many_boxes.detach().clone().requires_grad_()
        one_scores = many_scores.detach().clone().requires_grad_()
    else:
        one_boxes = torch.randn(1, 4 * reg_max, number_of_anchors, requires_grad=True)
        one_scores = torch.randn(1, nc, number_of_anchors, requires_grad=True)
    return {
        "one2many": {
            "boxes": many_boxes,
            "scores": many_scores,
            "feats": shared_features,
        },
        "one2one": {
            "boxes": one_boxes,
            "scores": one_scores,
            "feats": [feature.detach() for feature in shared_features],
        },
    }


def _synthetic_batch(with_target: bool = True) -> dict[str, torch.Tensor]:
    """Create a 64-pixel batch with either one object or no targets."""
    if with_target:
        batch_idx = torch.tensor([0])
        classes = torch.tensor([[0.0]])
        boxes = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    else:
        batch_idx = torch.empty(0, dtype=torch.long)
        classes = torch.empty(0, 1)
        boxes = torch.empty(0, 4)
    return {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": batch_idx,
        "cls": classes,
        "bboxes": boxes,
    }


def _build_criteria(
    distillation_config: MutualDistillationConfig,
) -> tuple[E2ELoss, MutualDistillationE2ELoss]:
    """Build matching E2.1b and E3 criteria without loading weights."""
    e2_model = ContrastRingDetectionModel("yolo26n.yaml", nc=2, verbose=False, loss_config=E2_1B_CONFIG)
    e3_model = MutualDistillationDetectionModel(
        "yolo26n.yaml",
        nc=2,
        verbose=False,
        distillation_config=distillation_config,
    )
    e2_model.args = get_cfg()
    e3_model.args = get_cfg()
    return e2_model.init_criterion(), e3_model.init_criterion()


def test_mutual_distillation_config_defaults_and_invalid_values() -> None:
    """Validate fixed E3 defaults and reject every invalid coefficient class."""
    config = MutualDistillationConfig()
    assert config == MutualDistillationConfig(
        classification_gain=0.10,
        box_gain=0.05,
        temperature=2.0,
        confidence_temperature=0.10,
        start_epoch=3,
        warmup_epochs=5,
        eps=1e-6,
    )
    assert E2_1B_CONFIG.positive_gain == 0.25
    assert E2_1B_CONFIG.negative_gain == 0.25
    assert E2_1B_CONFIG.negative_gamma == 3.0
    assert E2_1B_CONFIG.contrast_tau == 0.25

    for kwargs in (
        {"classification_gain": -0.1},
        {"box_gain": float("inf")},
        {"temperature": 0},
        {"confidence_temperature": -0.1},
        {"start_epoch": -1},
        {"warmup_epochs": 0},
        {"eps": 0},
    ):
        with pytest.raises(ValueError):
            MutualDistillationConfig(**kwargs)


def test_directional_weights_are_bounded_detached_and_confidence_adaptive() -> None:
    """Check mask gating, valid bounds, detachment, and the stronger positive teacher."""
    many_logits = torch.tensor([[[4.0, -2.0], [0.0, -1.0]]], requires_grad=True)
    one_logits = torch.tensor([[[1.0, -2.0], [0.0, -1.0]]], requires_grad=True)
    fg_mask = torch.tensor([[True, True]])

    many_to_one, one_to_many = _directional_weights(
        many_logits,
        one_logits,
        fg_mask,
        fg_mask,
        confidence_temperature=0.10,
    )

    assert not many_to_one.requires_grad
    assert not one_to_many.requires_grad
    assert torch.isfinite(many_to_one).all() and torch.isfinite(one_to_many).all()
    assert torch.all((0 <= many_to_one) & (many_to_one <= 1))
    assert torch.all((0 <= one_to_many) & (one_to_many <= 1))
    assert many_to_one[0, 0] > one_to_many[0, 0]


def test_distillation_math_is_finite_zero_for_matches_and_detaches_teacher() -> None:
    """Check finite gradients, zero matching loss, empty masks, and teacher detachment."""
    teacher_logits = torch.tensor([[[2.0, -1.0], [0.5, -0.5]]], requires_grad=True)
    student_logits = teacher_logits.detach().clone().requires_grad_()
    weights = torch.tensor([[0.8, 0.3]])
    cls_loss = _bernoulli_kl_loss(teacher_logits, student_logits, weights, temperature=2.0, eps=1e-6)
    torch.testing.assert_close(cls_loss, torch.zeros_like(cls_loss), atol=1e-7, rtol=0)
    cls_loss.backward()
    assert teacher_logits.grad is None
    assert student_logits.grad is not None and torch.isfinite(student_logits.grad).all()

    student_logits.grad = None
    reverse_cls_loss = _bernoulli_kl_loss(
        student_logits,
        teacher_logits,
        weights,
        temperature=2.0,
        eps=1e-6,
    )
    reverse_cls_loss.backward()
    assert student_logits.grad is None
    assert teacher_logits.grad is not None and torch.isfinite(teacher_logits.grad).all()

    teacher_boxes = torch.tensor([[[0.1, 0.2, 0.5, 0.6], [0.2, 0.1, 0.7, 0.8]]], requires_grad=True)
    student_boxes = teacher_boxes.detach().clone().requires_grad_()
    box_loss = _smooth_l1_loss(teacher_boxes, student_boxes, weights, eps=1e-6)
    torch.testing.assert_close(box_loss, torch.zeros_like(box_loss), atol=0, rtol=0)
    box_loss.backward()
    assert teacher_boxes.grad is None
    assert student_boxes.grad is not None and torch.isfinite(student_boxes.grad).all()

    student_boxes.grad = None
    reverse_box_loss = _smooth_l1_loss(student_boxes, teacher_boxes, weights, eps=1e-6)
    reverse_box_loss.backward()
    assert student_boxes.grad is None
    assert teacher_boxes.grad is not None and torch.isfinite(teacher_boxes.grad).all()

    empty_weights = torch.zeros_like(weights)
    empty_teacher_logits = teacher_logits.detach().clone().requires_grad_()
    unmatched_student_logits = torch.randn_like(student_logits, requires_grad=True)
    empty_cls_loss = _bernoulli_kl_loss(
        empty_teacher_logits,
        unmatched_student_logits,
        empty_weights,
        temperature=2.0,
        eps=1e-6,
    )
    assert empty_cls_loss == 0
    empty_cls_loss.backward()
    assert empty_teacher_logits.grad is None
    assert unmatched_student_logits.grad is not None
    assert torch.count_nonzero(unmatched_student_logits.grad) == 0


def test_assignment_alignment_rejects_anchor_or_stride_mismatch() -> None:
    """Require identical anchor points, strides, and anchor ordering across branches."""
    batch_size, number_of_anchors = 1, 3
    assignment = (
        torch.tensor([[True, False, True]]),
        torch.zeros(batch_size, number_of_anchors, dtype=torch.long),
        torch.zeros(batch_size, number_of_anchors, 4),
        torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5]]),
        torch.tensor([[8.0], [8.0], [8.0]]),
    )
    MutualDistillationE2ELoss._validate_assignments(
        assignment,
        tuple(tensor.clone() for tensor in assignment),
        batch_size,
        number_of_anchors,
    )

    bad_anchors = list(copy.deepcopy(assignment))
    bad_anchors[3][[0, 1]] = bad_anchors[3][[1, 0]]
    with pytest.raises(RuntimeError, match="anchor points"):
        MutualDistillationE2ELoss._validate_assignments(
            assignment,
            tuple(bad_anchors),
            batch_size,
            number_of_anchors,
        )

    bad_stride = list(copy.deepcopy(assignment))
    bad_stride[4][1] = 16
    with pytest.raises(RuntimeError, match="stride tensors"):
        MutualDistillationE2ELoss._validate_assignments(
            assignment,
            tuple(bad_stride),
            batch_size,
            number_of_anchors,
        )


def test_ramp_uses_epoch_updates_and_reaches_one() -> None:
    """Check the requested zero-based start and five-epoch linear ramp."""
    criterion = object.__new__(MutualDistillationE2ELoss)
    criterion.config = MutualDistillationConfig()
    expected = {0: 0.0, 2: 0.0, 3: 0.2, 4: 0.4, 7: 1.0, 20: 1.0}
    for updates, ramp in expected.items():
        criterion.updates = updates
        assert criterion.calculate_ramp() == pytest.approx(ramp)


def test_zero_distillation_gains_reproduce_e2_1b() -> None:
    """Ensure disabled E3 coefficients reproduce the current E2.1b optimization vector."""
    e2_criterion, e3_criterion = _build_criteria(
        MutualDistillationConfig(classification_gain=0, box_gain=0)
    )
    predictions = _synthetic_predictions(nc=2, reg_max=e3_criterion.one2many.reg_max)
    batch = _synthetic_batch()

    e2_loss, e2_items = e2_criterion(predictions, batch)
    e3_loss, e3_items = e3_criterion(predictions, batch)

    torch.testing.assert_close(e3_loss, e2_loss, atol=1e-6, rtol=1e-6)
    assert e3_items.keys() == e2_items.keys()
    for name in e2_items:
        torch.testing.assert_close(e3_items[name], e2_items[name], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("with_target", (True, False))
def test_full_e3_loss_has_finite_gradients_and_zero_matching_distillation(with_target: bool) -> None:
    """Check finite losses/gradients and zero distillation for identical branch predictions."""
    _, criterion = _build_criteria(
        MutualDistillationConfig(start_epoch=0, warmup_epochs=1)
    )
    predictions = _synthetic_predictions(
        nc=2,
        reg_max=criterion.one2many.reg_max,
        identical_branches=True,
    )
    loss, _ = criterion(predictions, _synthetic_batch(with_target=with_target))
    loss.sum().backward()

    assert torch.isfinite(loss).all()
    assert criterion.distill_cls_loss.item() == pytest.approx(0, abs=1e-7)
    assert criterion.distill_box_loss.item() == pytest.approx(0, abs=1e-7)
    for branch in predictions.values():
        for name in ("boxes", "scores"):
            gradient = branch[name].grad
            assert gradient is not None and torch.isfinite(gradient).all()


def test_e3_reuses_e1_1_and_e2_1b_without_patching_stock_losses() -> None:
    """Check isolated inheritance and fixed branch criteria."""
    assert E2_LOCALIZATION_CONFIG.mode == "constant-010"
    assert E2_LOCALIZATION_CONFIG.nwd_scale == 0.10
    assert issubclass(MutualDistillationDetectionModel, ContrastRingDetectionModel)
    assert MutualDistillationYOLO is not YOLO
    assert BboxLoss.forward.__module__ == "ultralytics.utils.loss"
    assert v8DetectionLoss.get_assigned_targets_and_loss.__module__ == "ultralytics.utils.loss"

    model = MutualDistillationDetectionModel("yolo26n.yaml", nc=2, verbose=False)
    model.args = get_cfg()
    criterion = model.init_criterion()
    assert isinstance(criterion, MutualDistillationE2ELoss)
    assert (criterion.o2m, criterion.o2o, criterion.final_o2m) == pytest.approx((0.8, 0.2, 0.1))
    assert isinstance(criterion.one2many, ContrastRingDetectionLoss)
    assert isinstance(criterion.one2one, ContrastRingDetectionLoss)
    assert criterion.one2many.assigner.topk == 10
    assert criterion.one2one.assigner.topk == 7
    assert criterion.one2one.assigner.topk2 == 1
    assert isinstance(criterion.one2many.bbox_loss, ResidualNWDBboxLoss)
    assert criterion.one2many.bbox_loss.config == E2_LOCALIZATION_CONFIG
    assert criterion.one2one.bbox_loss.config == E2_LOCALIZATION_CONFIG
    assert criterion.one2many.contrast_ring_config == E2_1B_CONFIG
    assert criterion.one2one.contrast_ring_config == E2_1B_CONFIG


def test_e2_1b_and_e3_architectures_and_standard_state_dict_match() -> None:
    """Compare all parameter shapes and strictly load the standard pretrained state dict."""
    standard = YOLO(str(REPOSITORY_ROOT / "yolo26n.pt")).model
    channels = standard.yaml.get("channels") or 3
    model_yaml = copy.deepcopy(standard.yaml)
    number_of_classes = standard.model[-1].nc
    e2_model = ContrastRingDetectionModel(
        copy.deepcopy(model_yaml),
        ch=channels,
        nc=number_of_classes,
        verbose=False,
        loss_config=E2_1B_CONFIG,
    )
    e3_model = MutualDistillationDetectionModel(
        copy.deepcopy(model_yaml),
        ch=channels,
        nc=number_of_classes,
        verbose=False,
    )

    e2_parameters = [(name, tuple(parameter.shape)) for name, parameter in e2_model.named_parameters()]
    e3_parameters = [(name, tuple(parameter.shape)) for name, parameter in e3_model.named_parameters()]
    e2_state_shapes = {name: tuple(tensor.shape) for name, tensor in e2_model.state_dict().items()}
    e3_state_shapes = {name: tuple(tensor.shape) for name, tensor in e3_model.state_dict().items()}

    assert e3_parameters == e2_parameters
    assert e3_state_shapes == e2_state_shapes
    incompatible = e3_model.load_state_dict(standard.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
