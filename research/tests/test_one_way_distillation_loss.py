from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import E2ELoss
from yolo_improved import (
    ContrastRingDetectionModel,
    E2_1B_CONFIG,
    MutualDistillationConfig,
    MutualDistillationDetectionModel,
    MutualDistillationE2ELoss,
    ONE_WAY_DISTILLATION_VARIANTS,
    OneWayDistillationConfig,
    OneWayDistillationDetectionModel,
    OneWayDistillationE2ELoss,
    OneWayDistillationYOLO,
    get_one_way_distillation_config,
)
from yolo_improved.mutual_distillation_loss import _bernoulli_kl_loss
from yolo_improved.one_way_distillation_loss import _one2many_teacher_weights


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _synthetic_predictions(
    nc: int,
    reg_max: int,
    identical_branches: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Create small aligned branch predictions without model inference."""
    torch.manual_seed(37)
    feature_shapes = ((8, 8), (4, 4), (2, 2))
    number_of_anchors = sum(height * width for height, width in feature_shapes)
    features = [torch.rand(1, 8, height, width) for height, width in feature_shapes]
    many_boxes = torch.randn(1, 4 * reg_max, number_of_anchors, requires_grad=True)
    many_scores = torch.randn(1, nc, number_of_anchors, requires_grad=True)
    if identical_branches:
        one_boxes = many_boxes.detach().clone().requires_grad_()
        one_scores = many_scores.detach().clone().requires_grad_()
    else:
        one_boxes = torch.randn(1, 4 * reg_max, number_of_anchors, requires_grad=True)
        one_scores = torch.randn(1, nc, number_of_anchors, requires_grad=True)
    return {
        "one2many": {"boxes": many_boxes, "scores": many_scores, "feats": features},
        "one2one": {
            "boxes": one_boxes,
            "scores": one_scores,
            "feats": [feature.detach() for feature in features],
        },
    }


def _synthetic_batch(with_target: bool = True) -> dict[str, torch.Tensor]:
    """Create a small batch with one target or no positive targets."""
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
    config: OneWayDistillationConfig,
) -> tuple[E2ELoss, OneWayDistillationE2ELoss]:
    """Build E2.1b and E3.1 criteria with matching model metadata."""
    e2_model = ContrastRingDetectionModel("yolo26n.yaml", nc=2, verbose=False, loss_config=E2_1B_CONFIG)
    one_way_model = OneWayDistillationDetectionModel(
        "yolo26n.yaml",
        nc=2,
        verbose=False,
        distillation_config=config,
    )
    e2_model.args = get_cfg()
    one_way_model.args = get_cfg()
    return e2_model.init_criterion(), one_way_model.init_criterion()


def test_one_way_variants_and_validation() -> None:
    """Check all fixed variants and reject invalid E3.1 values."""
    assert ONE_WAY_DISTILLATION_VARIANTS == ("control", "cls-005", "cls-0025")
    expected_gains = {"control": 0.0, "cls-005": 0.05, "cls-0025": 0.025}
    for variant, gain in expected_gains.items():
        config = get_one_way_distillation_config(variant)
        assert config.classification_gain == gain
        assert config.temperature == 2.0
        assert config.confidence_temperature == 0.10
        assert config.start_epoch == 5
        assert config.warmup_epochs == 5
        assert config.eps == 1e-6

    for kwargs in (
        {"classification_gain": -0.1},
        {"temperature": 0},
        {"confidence_temperature": float("inf")},
        {"start_epoch": -1},
        {"warmup_epochs": 0},
        {"eps": 0},
    ):
        with pytest.raises(ValueError):
            OneWayDistillationConfig(**kwargs)
    with pytest.raises(ValueError):
        get_one_way_distillation_config("unknown")


def test_teacher_weights_use_only_positive_one2many_anchors_and_are_detached() -> None:
    """Check shape, masking, bounds, confidence adaptation, and detachment."""
    many_logits = torch.tensor([[[4.0, -2.0], [1.0, -1.0], [0.0, -1.0]]], requires_grad=True)
    one_logits = torch.tensor([[[1.0, -2.0], [4.0, -1.0], [0.0, -1.0]]], requires_grad=True)
    many_fg_mask = torch.tensor([[True, False, True]])
    weights = _one2many_teacher_weights(
        many_logits,
        one_logits,
        many_fg_mask,
        confidence_temperature=0.10,
    )

    assert weights.shape == many_fg_mask.shape
    assert not weights.requires_grad
    assert torch.isfinite(weights).all()
    assert torch.all((0 <= weights) & (weights <= 1))
    assert weights[0, 1] == 0
    assert weights[0, 0] > weights[0, 2]


def test_one_way_kl_detaches_teacher_and_updates_only_student() -> None:
    """Check identical/empty losses, finite AMP arithmetic, and gradient direction."""
    teacher = torch.tensor([[[3.0, -1.0], [0.5, -0.5]]], requires_grad=True)
    matching_student = teacher.detach().clone().requires_grad_()
    weights = torch.tensor([[0.8, 0.3]])
    matching_loss = _bernoulli_kl_loss(teacher.detach(), matching_student, weights, temperature=2.0, eps=1e-6)
    torch.testing.assert_close(matching_loss, torch.zeros_like(matching_loss), atol=1e-7, rtol=0)

    student = torch.tensor([[[0.0, 1.0], [-1.0, 0.5]]], requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = _bernoulli_kl_loss(teacher.detach(), student, weights, temperature=2.0, eps=1e-6)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert teacher.grad is None
    assert student.grad is not None and torch.isfinite(student.grad).all()

    empty_student = torch.randn_like(student, requires_grad=True)
    empty_loss = _bernoulli_kl_loss(
        teacher.detach(),
        empty_student,
        torch.zeros_like(weights),
        temperature=2.0,
        eps=1e-6,
    )
    assert empty_loss == 0
    empty_loss.backward()
    assert teacher.grad is None
    assert empty_student.grad is not None
    assert torch.count_nonzero(empty_student.grad) == 0


def test_one_way_source_has_no_reverse_or_box_distillation() -> None:
    """Require exactly one KL call and no reverse-weight, box-decode, or Smooth L1 call."""
    source = inspect.getsource(OneWayDistillationE2ELoss.__call__)
    assert source.count("_bernoulli_kl_loss(") == 1
    assert "_directional_weights" not in source
    assert "one_to_many_weights" not in source
    assert "_decode_normalized_boxes" not in source
    assert "_smooth_l1_loss" not in source
    assert "scaled_box_loss" not in source


def test_one_way_ramp_starts_at_epoch_five() -> None:
    """Check the requested zero-based five-epoch warmup."""
    criterion = object.__new__(OneWayDistillationE2ELoss)
    criterion.config = OneWayDistillationConfig()
    expected = {0: 0.0, 4: 0.0, 5: 0.2, 6: 0.4, 9: 1.0, 20: 1.0}
    for updates, ramp in expected.items():
        criterion.updates = updates
        assert criterion.calculate_ramp() == pytest.approx(ramp)


def test_control_reproduces_e2_1b_and_branch_logit_shapes_match() -> None:
    """Compare the control optimization vector and inspect aligned branch logits."""
    e2_criterion, control_criterion = _build_criteria(get_one_way_distillation_config("control"))
    predictions = _synthetic_predictions(nc=2, reg_max=control_criterion.one2many.reg_max)
    many_logits, one_logits, _, _ = control_criterion._prepare_branch_predictions(
        predictions["one2many"],
        predictions["one2one"],
    )
    assert many_logits.shape == one_logits.shape

    batch = _synthetic_batch()
    e2_loss, e2_items = e2_criterion(predictions, batch)
    control_loss, control_items = control_criterion(predictions, batch)
    torch.testing.assert_close(control_loss, e2_loss, atol=1e-6, rtol=1e-6)
    assert control_items.keys() == e2_items.keys()
    for name in e2_items:
        torch.testing.assert_close(control_items[name], e2_items[name], atol=1e-6, rtol=1e-6)


def test_active_distillation_changes_only_classification_component() -> None:
    """Ensure E3.1 leaves supervised box and DFL/L1 components unchanged."""
    _, control_criterion = _build_criteria(get_one_way_distillation_config("control"))
    _, active_criterion = _build_criteria(
        OneWayDistillationConfig(classification_gain=0.05, start_epoch=0, warmup_epochs=1)
    )
    predictions = _synthetic_predictions(nc=2, reg_max=active_criterion.one2many.reg_max)
    batch = _synthetic_batch()
    control_loss, _ = control_criterion(predictions, batch)
    active_loss, _ = active_criterion(predictions, batch)

    torch.testing.assert_close(active_loss[0], control_loss[0], atol=1e-7, rtol=0)
    torch.testing.assert_close(active_loss[2], control_loss[2], atol=1e-7, rtol=0)
    assert active_criterion.distill_box_loss == 0


def test_e3_remains_bidirectional_and_separate() -> None:
    """Check that E3 defaults and its bidirectional/box paths remain available."""
    assert MutualDistillationConfig() == MutualDistillationConfig(
        classification_gain=0.10,
        box_gain=0.05,
        temperature=2.0,
        confidence_temperature=0.10,
        start_epoch=3,
        warmup_epochs=5,
        eps=1e-6,
    )
    mutual_source = inspect.getsource(MutualDistillationE2ELoss.__call__)
    assert mutual_source.count("_bernoulli_kl_loss(") == 2
    assert mutual_source.count("_smooth_l1_loss(") == 2
    assert not issubclass(OneWayDistillationDetectionModel, MutualDistillationDetectionModel)
    assert OneWayDistillationYOLO is not YOLO


def test_e2_e3_and_e3_1_architectures_and_standard_state_dict_match() -> None:
    """Compare parameter shapes and strictly load the standard pretrained state dict."""
    standard = YOLO(str(REPOSITORY_ROOT / "yolo26n.pt")).model
    channels = standard.yaml.get("channels") or 3
    number_of_classes = standard.model[-1].nc
    model_yaml = copy.deepcopy(standard.yaml)
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
    one_way_model = OneWayDistillationDetectionModel(
        copy.deepcopy(model_yaml),
        ch=channels,
        nc=number_of_classes,
        verbose=False,
    )

    e2_parameters = [(name, tuple(parameter.shape)) for name, parameter in e2_model.named_parameters()]
    e3_parameters = [(name, tuple(parameter.shape)) for name, parameter in e3_model.named_parameters()]
    one_way_parameters = [(name, tuple(parameter.shape)) for name, parameter in one_way_model.named_parameters()]
    assert one_way_parameters == e2_parameters == e3_parameters

    e2_state = {name: tuple(tensor.shape) for name, tensor in e2_model.state_dict().items()}
    e3_state = {name: tuple(tensor.shape) for name, tensor in e3_model.state_dict().items()}
    one_way_state = {name: tuple(tensor.shape) for name, tensor in one_way_model.state_dict().items()}
    assert one_way_state == e2_state == e3_state

    incompatible = one_way_model.load_state_dict(standard.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
