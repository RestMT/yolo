# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Synthetic checks for E11 Class-Conditional Suppression-Calibrated DGQM."""

from __future__ import annotations

import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import (
    BoxClassAgreementQualityHead,
    ClassConditionalSuppressionDGQMDetect,
    ClassConditionalSuppressionHead,
    MorphologyAdaptiveAdapter,
)
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import get_flops, get_num_params
from yolo_improved.ccs_config import ClassConditionalSuppressionConfig
from yolo_improved.ccs_loss import ClassConditionalSuppressionDetectionLoss
from yolo_improved.ccs_model import (
    CCS_DGQM_VARIANTS,
    ClassConditionalSuppressionDetectionModel,
    build_ccs_dgqm_yolo,
    ccs_dgqm_yaml_path,
    collect_ccs_dgqm_diagnostics,
    remap_yolo26_ccs_dgqm_state_dict,
)
from yolo_improved.contrast_ring_loss import ContrastRingBCEWithLogitsLoss
from yolo_improved.dgqm_config import DualGeometryQualityConfig
from yolo_improved.dgqm_loss import DualGeometryQualityDetectionLoss
from yolo_improved.dgqm_model import (
    DualGeometryQualityDetectionModel,
    build_dgqm_yolo,
    dgqm_yaml_path,
)
from yolo_improved.madh_model import E2_1B_CONFIG
from yolo_improved.residual_nwd_loss import ResidualNWDBboxLoss


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STOCK_MODEL_DIRECTORY = REPOSITORY_ROOT / "ultralytics" / "cfg" / "models" / "26"
E11_MODEL_DIRECTORY = REPOSITORY_ROOT / "research" / "models"
STANDARD_N_WEIGHTS = REPOSITORY_ROOT / "yolo26n.pt"
SCALE_SETTINGS = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}


def _synthetic_batch(with_target: bool = True) -> dict[str, torch.Tensor]:
    """Return one small normalized detection batch."""
    if with_target:
        batch_idx = torch.tensor([0])
        cls = torch.tensor([[0.0]])
        bboxes = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    else:
        batch_idx = torch.empty(0, dtype=torch.long)
        cls = torch.empty(0, 1)
        bboxes = torch.empty(0, 4)
    return {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
    }


def _all_output_tensors(value):
    """Yield tensors from a nested output."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_output_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _all_output_tensors(item)


def _raw_e10_differences(
    e10: dict[str, dict[str, torch.Tensor]],
    e11: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Return requested E10 raw box, score, and quality differences."""
    return {
        f"{branch}_{component}": float(
            (e10[branch][component] - e11[branch][component]).abs().max().item()
        )
        for branch in ("one2many", "one2one")
        for component in ("boxes", "scores", "quality")
    }


def _assert_no_gradients(modules) -> None:
    """Assert that selected modules received no nonzero gradients."""
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for module in modules
        for parameter in module.parameters()
    )


def test_suppression_config_defaults_and_validation() -> None:
    """Validate exact E11 defaults and every numeric constraint."""
    config = ClassConditionalSuppressionConfig()
    assert config == ClassConditionalSuppressionConfig(
        suppression_gain=0.10,
        suppression_scale=1.0,
        hard_negative_gamma=2.0,
        hard_negative_min_probability=0.05,
        positive_neutral_weight=1.0,
        eps=1e-6,
    )
    for kwargs, message in (
        ({"suppression_gain": -0.1}, "suppression_gain"),
        ({"suppression_scale": 0.0}, "suppression_scale"),
        ({"hard_negative_gamma": -0.1}, "hard_negative_gamma"),
        ({"hard_negative_min_probability": -0.1}, "hard_negative_min_probability"),
        ({"hard_negative_min_probability": 1.0}, "hard_negative_min_probability"),
        ({"positive_neutral_weight": -0.1}, "positive_neutral_weight"),
        ({"eps": 0.0}, "eps"),
        ({"suppression_gain": float("nan")}, "suppression_gain"),
        ({"suppression_scale": float("inf")}, "suppression_scale"),
        ({"hard_negative_gamma": True}, "hard_negative_gamma"),
    ):
        with pytest.raises(ValueError, match=message):
            ClassConditionalSuppressionConfig(**kwargs)


def test_suppression_head_structure_zero_output_and_frozen_bypass() -> None:
    """Check agreement geometry, zero initialization, and control bypass."""
    trainable = ClassConditionalSuppressionHead(128, nc=5)
    assert trainable.hidden == 16
    assert trainable.reduce.conv.in_channels == 384
    assert trainable.reduce.conv.out_channels == 16
    assert trainable.context.conv.kernel_size == (3, 3)
    assert trainable.context.conv.groups == 16
    assert trainable.output.in_channels == 16
    assert trainable.output.out_channels == 5
    assert torch.count_nonzero(trainable.output.weight).item() == 0
    assert torch.count_nonzero(trainable.output.bias).item() == 0
    box = torch.randn(2, 128, 9, 11)
    cls = torch.randn(2, 128, 9, 11)
    output = trainable(box, cls)
    assert output.shape == (2, 5, 9, 11)
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)

    frozen = ClassConditionalSuppressionHead(128, nc=5, trainable_suppression=False)
    calls = []
    handles = [
        module.register_forward_hook(lambda *_args, name=name: calls.append(name))
        for name, module in (
            ("reduce", frozen.reduce),
            ("context", frozen.context),
            ("output", frozen.output),
        )
    ]
    try:
        frozen_output = frozen(box, cls)
    finally:
        for handle in handles:
            handle.remove()
    assert frozen_output.shape == (2, 5, 9, 11)
    assert torch.count_nonzero(frozen_output).item() == 0
    assert not calls
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


def test_detect_forward_detachment_shape_independence_and_suppression_only_inference() -> None:
    """Verify E10 reuse, detached suppression features, and nonpositive inference corrections."""
    head = ClassConditionalSuppressionDGQMDetect(
        nc=3,
        end2end=True,
        ch=(16, 24, 32),
    )
    head.stride[:] = torch.tensor([8.0, 16.0, 32.0])
    assert len(head.suppression_heads) == len(head.one2one_suppression_heads) == 3
    for first, second in zip(head.suppression_heads, head.one2one_suppression_heads):
        assert first is not second
        assert all(
            first_parameter.data_ptr() != second_parameter.data_ptr()
            for first_parameter, second_parameter in zip(first.parameters(), second.parameters())
        )

    head.train()
    features = [
        torch.randn(2, 16, 8, 8, requires_grad=True),
        torch.randn(2, 24, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    ]
    suppression_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    handles = [
        suppression_head.register_forward_pre_hook(
            lambda _, inputs: suppression_inputs.append((inputs[0], inputs[1]))
        )
        for suppression_head in (*head.suppression_heads, *head.one2one_suppression_heads)
    ]
    try:
        predictions = head(features)
    finally:
        for handle in handles:
            handle.remove()
    total_anchors = 8 * 8 + 4 * 4 + 2 * 2
    for branch in ("one2many", "one2one"):
        assert predictions[branch]["quality"].shape == (2, 1, total_anchors)
        assert predictions[branch]["class_suppression"].shape == (2, 3, total_anchors)
        assert torch.count_nonzero(predictions[branch]["quality"]).item() == 0
        assert torch.count_nonzero(predictions[branch]["class_suppression"]).item() == 0
    assert len(suppression_inputs) == 6
    assert all(
        not box_feature.requires_grad and not cls_feature.requires_grad
        for box_feature, cls_feature in suppression_inputs
    )

    head.eval()
    detached_features = [feature.detach() for feature in features]
    with torch.no_grad():
        _, raw_zero = head(detached_features)
        decoded_zero = head._inference(raw_zero["one2one"])
        for suppression_head in head.one2one_suppression_heads:
            suppression_head.output.bias[0] = 0.5
        _, raw_positive = head(detached_features)
        decoded_positive = head._inference(raw_positive["one2one"])
        for suppression_head in head.one2one_suppression_heads:
            suppression_head.output.bias[0] = -0.5
        _, raw_negative = head(detached_features)
        decoded_negative = head._inference(raw_negative["one2one"])

    torch.testing.assert_close(decoded_zero, decoded_positive, rtol=0, atol=0)
    correction = head.suppression_scale * torch.tanh(
        raw_negative["one2one"]["class_suppression"]
    ).clamp(max=0.0)
    assert correction.max().item() <= 0
    torch.testing.assert_close(decoded_zero[:, :4], decoded_negative[:, :4], rtol=0, atol=0)
    assert torch.all(decoded_negative[:, 4] < decoded_zero[:, 4])
    torch.testing.assert_close(decoded_negative[:, 5:], decoded_zero[:, 5:], rtol=0, atol=0)


def test_unified_yamls_and_all_virtual_scale_names() -> None:
    """Verify both unified definitions and all ten virtual E11 names."""
    e10_args = yaml_model_load(dgqm_yaml_path("n", "trainable"))["head"][-1][3]
    for variant in CCS_DGQM_VARIANTS:
        suffix = "ccs-dgqm-control" if variant == "control" else "ccs-dgqm"
        unified = E11_MODEL_DIRECTORY / f"yolo26-{suffix}.yaml"
        assert unified.is_file()
        assert not list(E11_MODEL_DIRECTORY.glob(f"yolo26?-{suffix}.yaml"))
        for size, scale_settings in SCALE_SETTINGS.items():
            virtual = ccs_dgqm_yaml_path(size, variant)
            assert not virtual.exists()
            target = yaml_model_load(virtual)
            e10 = yaml_model_load(dgqm_yaml_path(size, "trainable"))
            stock = yaml_model_load(STOCK_MODEL_DIRECTORY / f"yolo26{size}.yaml")
            assert target["scale"] == size
            assert tuple(target["scales"][size]) == scale_settings
            for key in ("nc", "end2end", "reg_max", "scales", "backbone", "scale"):
                assert target[key] == e10[key] == stock[key]
            assert target["head"][:-1] == e10["head"][:-1]
            assert target["head"][-1][:2] == e10["head"][-1][:2]
            assert target["head"][-1][2] == "ClassConditionalSuppressionDGQMDetect"
            args = target["head"][-1][3]
            assert args[: len(e10_args)] == e10_args
            assert args[len(e10_args) :] == [
                0.125,
                16,
                64,
                1.0,
                variant == "trainable",
            ]


@pytest.mark.parametrize("size", tuple(SCALE_SETTINGS))
def test_n_to_x_structure_transfer_parameters_and_flops(size: str) -> None:
    """Build both variants, retain complete E10, and measure suppression-only cost."""
    stock = DetectionModel(STOCK_MODEL_DIRECTORY / f"yolo26{size}.yaml", verbose=False)
    e10 = DualGeometryQualityDetectionModel(
        dgqm_yaml_path(size, "trainable"),
        verbose=False,
    )
    source_state = stock.state_dict()
    stock_parameters = get_num_params(stock)
    e10_parameters = get_num_params(e10)
    e10_flops = get_flops(e10, imgsz=640)

    for variant in CCS_DGQM_VARIANTS:
        target = ClassConditionalSuppressionDetectionModel(
            ccs_dgqm_yaml_path(size, variant),
            verbose=False,
        )
        head = target.model[-1]
        assert isinstance(head, ClassConditionalSuppressionDGQMDetect)
        assert head.f == [16, 19, 22]
        assert head.nl == 3
        assert head.stride.tolist() == [8.0, 16.0, 32.0]
        assert head.reg_max == 1
        assert isinstance(head.dfl, nn.Identity)
        for group in (
            head.box_adapters,
            head.cls_adapters,
            head.one2one_box_adapters,
            head.one2one_cls_adapters,
        ):
            assert len(group) == 3
            assert all(isinstance(adapter, MorphologyAdaptiveAdapter) for adapter in group)
            assert all(adapter.trainable_adapter for adapter in group)
        for group in (head.quality_heads, head.one2one_quality_heads):
            assert len(group) == 3
            assert all(isinstance(quality_head, BoxClassAgreementQualityHead) for quality_head in group)
            assert all(quality_head.trainable_quality for quality_head in group)
        for group in (head.suppression_heads, head.one2one_suppression_heads):
            assert len(group) == 3
            assert all(
                suppression_head.trainable_suppression == (variant == "trainable")
                for suppression_head in group
            )

        transferred, report = remap_yolo26_ccs_dgqm_state_dict(
            source_state,
            target.state_dict(),
            target_parameter_keys=dict(target.named_parameters()),
        )
        assert set(transferred) == set(source_state)
        assert not report.skipped_missing_keys
        assert not report.skipped_shape_keys
        assert report.new_adapter_keys
        assert report.new_quality_keys
        assert report.new_suppression_keys
        assert (
            set(report.new_adapter_keys)
            | set(report.new_quality_keys)
            | set(report.new_suppression_keys)
        ) == set(report.new_ccs_keys)
        for coverage_name in ("backbone", "neck", "cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            assert report.coverage(coverage_name).percentage == 100.0
        assert report.coverage("adapters").transferred_tensors == 0
        assert report.coverage("quality_heads").transferred_tensors == 0
        assert report.coverage("suppression_heads").transferred_tensors == 0

        target_parameters = get_num_params(target)
        suppression_parameters = sum(
            parameter.numel()
            for group in (head.suppression_heads, head.one2one_suppression_heads)
            for parameter in group.parameters()
        )
        assert target_parameters - e10_parameters == suppression_parameters
        parameter_increase = 100.0 * suppression_parameters / e10_parameters
        target_flops = get_flops(target, imgsz=640)
        gflops_increase = 100.0 * (target_flops - e10_flops) / e10_flops
        if variant == "control":
            assert target_flops == pytest.approx(e10_flops, rel=0, abs=1e-9)
        else:
            assert target_flops > e10_flops
        print(
            json.dumps(
                {
                    "scale": size,
                    "variant": variant,
                    "transferred_parameters": stock_parameters,
                    "pretrained_transfer_percent": 100.0 * stock_parameters / target_parameters,
                    "e10_parameters": e10_parameters,
                    "parameters": target_parameters,
                    "suppression_parameters": suppression_parameters,
                    "suppression_parameter_increase_percent": parameter_increase,
                    "e10_gflops": e10_flops,
                    "gflops": target_flops,
                    "suppression_gflops_increase_percent": gflops_increase,
                    "within_parameter_target": parameter_increase <= 2.0,
                    "within_gflops_target": gflops_increase <= 2.0,
                }
            )
        )
        del target
        gc.collect()

    del e10
    del stock
    gc.collect()


def test_positive_masks_negative_classes_and_hard_negative_weighting() -> None:
    """Check neutral positives, wrong/background negatives, thresholding, and p^gamma weights."""
    model = ClassConditionalSuppressionDetectionModel(
        ccs_dgqm_yaml_path("n", "trainable"),
        nc=3,
        verbose=False,
    )
    model.args = get_cfg()
    criterion = model.init_criterion().one2many
    probabilities = torch.tensor(
        [[[0.80, 0.40, 0.01], [0.60, 0.20, 0.10]]],
        dtype=torch.float32,
    )
    score_logits = torch.logit(probabilities).transpose(1, 2).contiguous()
    suppression_logits = torch.tensor(
        [[[0.20, -0.40], [0.30, 0.10], [-0.20, 0.50]]],
        requires_grad=True,
    )
    target_scores = torch.zeros(1, 2, 3)
    target_scores[0, 0, 1] = 0.75
    positive_class_mask = target_scores > 0
    negative_class_mask = ~positive_class_mask
    assert positive_class_mask.sum().item() == 1
    assert negative_class_mask[0, 0, 0] and negative_class_mask[0, 0, 2]
    assert negative_class_mask[0, 1].all()

    preds = {
        "scores": score_logits,
        "class_suppression": suppression_logits,
    }
    loss = criterion.calculate_suppression_loss(preds, target_scores)
    prediction = torch.tanh(suppression_logits.transpose(1, 2))
    positive_prediction = prediction[positive_class_mask]
    expected_positive = F.smooth_l1_loss(
        positive_prediction,
        torch.zeros_like(positive_prediction),
        reduction="none",
    )
    expected_positive = (
        expected_positive * target_scores[positive_class_mask]
    ).sum() / target_scores[positive_class_mask].sum()
    negative_probability = probabilities[negative_class_mask]
    negative_weight = negative_probability.square() * (negative_probability >= 0.05)
    expected_negative = F.smooth_l1_loss(
        prediction[negative_class_mask],
        -torch.ones_like(prediction[negative_class_mask]),
        reduction="none",
    )
    expected_negative = (expected_negative * negative_weight).sum() / negative_weight.sum()
    expected = 0.10 * (expected_positive + expected_negative)
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
    assert negative_weight[1].item() == 0
    loss.backward()
    assert suppression_logits.grad is not None
    assert torch.isfinite(suppression_logits.grad).all()
    assert torch.count_nonzero(suppression_logits.grad).item() > 0

    low_probability_logits = torch.logit(torch.full((1, 3, 2), 0.01))
    inactive_suppression_logits = torch.randn(1, 3, 2, requires_grad=True)
    differentiable_zero = criterion.calculate_suppression_loss(
        {
            "scores": low_probability_logits,
            "class_suppression": inactive_suppression_logits,
        },
        torch.zeros(1, 2, 3),
    )
    assert differentiable_zero.requires_grad
    torch.testing.assert_close(differentiable_zero, torch.zeros_like(differentiable_zero))
    differentiable_zero.backward()
    assert inactive_suppression_logits.grad is not None
    assert torch.count_nonzero(inactive_suppression_logits.grad).item() == 0


@pytest.mark.parametrize("branch_name", ("one2many", "one2one"))
def test_assignment_and_e10_losses_are_invariant_to_suppression(branch_name: str) -> None:
    """Ensure one assignment and unchanged E2.1b/E10 quality for arbitrary suppression logits."""
    model = ClassConditionalSuppressionDetectionModel(
        ccs_dgqm_yaml_path("n", "trainable"),
        verbose=False,
    )
    model.args = get_cfg()
    model.train()
    criterion = getattr(model.init_criterion(), branch_name)
    batch = _synthetic_batch()
    branch_predictions = model(batch["img"])[branch_name]

    assignment_calls = []
    handle = criterion.assigner.register_forward_hook(lambda *_args: assignment_calls.append(1))
    try:
        assigned_zero, losses_zero, items_zero = criterion.get_assigned_targets_and_loss(
            branch_predictions,
            batch,
        )
    finally:
        handle.remove()
    assert len(assignment_calls) == 1

    shifted_predictions = {
        **branch_predictions,
        "class_suppression": torch.full_like(branch_predictions["class_suppression"], 100.0),
    }
    assignment_calls = []
    handle = criterion.assigner.register_forward_hook(lambda *_args: assignment_calls.append(1))
    try:
        assigned_shifted, losses_shifted, items_shifted = criterion.get_assigned_targets_and_loss(
            shifted_predictions,
            batch,
        )
    finally:
        handle.remove()
    assert len(assignment_calls) == 1
    for zero_value, shifted_value in zip(assigned_zero, assigned_shifted):
        torch.testing.assert_close(zero_value, shifted_value, rtol=0, atol=0)
    torch.testing.assert_close(losses_zero[:4], losses_shifted[:4], rtol=0, atol=0)
    for name in ("box_loss", "cls_loss", "l1_loss", "quality_loss"):
        torch.testing.assert_close(items_zero[name], items_shifted[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("branch_name", "suppression_name"),
    (("one2many", "suppression_heads"), ("one2one", "one2one_suppression_heads")),
)
def test_suppression_gradient_isolation_and_head_gradients(
    branch_name: str,
    suppression_name: str,
) -> None:
    """Ensure suppression loss trains only its matching suppression heads."""
    model = ClassConditionalSuppressionDetectionModel(
        ccs_dgqm_yaml_path("n", "trainable"),
        verbose=False,
    )
    model.args = get_cfg()
    model.train()
    criterion = getattr(model.init_criterion(), branch_name)
    batch = _synthetic_batch()
    model.zero_grad(set_to_none=True)
    predictions = model(batch["img"])
    branch_predictions = {
        **predictions[branch_name],
        "scores": predictions[branch_name]["scores"] + 10.0,
    }
    suppression_loss = criterion.get_assigned_targets_and_loss(branch_predictions, batch)[1][4]
    assert suppression_loss > 0
    suppression_loss.backward()

    head = model.model[-1]
    suppression_heads = getattr(head, suppression_name)
    suppression_gradients = [
        parameter.grad
        for suppression_head in suppression_heads
        for parameter in suppression_head.parameters()
        if parameter.grad is not None
    ]
    assert suppression_gradients
    assert all(torch.isfinite(gradient).all() for gradient in suppression_gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in suppression_gradients)
    _assert_no_gradients(
        (
            *head.box_adapters,
            *head.cls_adapters,
            *head.one2one_box_adapters,
            *head.one2one_cls_adapters,
            *head.cv2,
            *head.cv3,
            *head.one2one_cv2,
            *head.one2one_cv3,
            *head.quality_heads,
            *head.one2one_quality_heads,
            *model.model[:-1],
        )
    )
    other_name = "one2one_suppression_heads" if suppression_name == "suppression_heads" else "suppression_heads"
    _assert_no_gradients(getattr(head, other_name))


def test_control_matches_e10_predictions_and_loss() -> None:
    """Verify disabled suppression is mathematically identical to trainable E10."""
    torch.manual_seed(111)
    e10 = DualGeometryQualityDetectionModel(
        dgqm_yaml_path("n", "trainable"),
        verbose=False,
    )
    control = ClassConditionalSuppressionDetectionModel(
        ccs_dgqm_yaml_path("n", "control"),
        verbose=False,
    )
    incompatible = control.load_state_dict(e10.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(
        key.startswith(("model.23.suppression_heads.", "model.23.one2one_suppression_heads."))
        for key in incompatible.missing_keys
    )
    e10.eval()
    control.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        e10_postprocessed, e10_raw = e10(image)
        control_postprocessed, control_raw = control(image)
        e10_decoded = e10.model[-1]._inference(e10_raw["one2one"])
        control_decoded = control.model[-1]._inference(control_raw["one2one"])
    assert max(_raw_e10_differences(e10_raw, control_raw).values()) <= 1e-6
    assert all(
        torch.count_nonzero(control_raw[branch]["class_suppression"]).item() == 0
        for branch in ("one2many", "one2one")
    )
    torch.testing.assert_close(e10_decoded, control_decoded, rtol=0, atol=1e-6)
    torch.testing.assert_close(e10_postprocessed, control_postprocessed, rtol=0, atol=1e-6)

    e10.args = get_cfg()
    control.args = get_cfg()
    e10.train()
    control.train()
    batch = _synthetic_batch()
    e10_loss = e10.init_criterion()(e10(batch["img"]), batch)[0]
    control_loss = control.init_criterion()(control(batch["img"]), batch)[0]
    assert control_loss.shape == (5,)
    torch.testing.assert_close(e10_loss, control_loss[:4], rtol=0, atol=1e-6)
    assert control_loss[4].item() == 0.0
    torch.testing.assert_close(e10_loss.sum(), control_loss.sum(), rtol=0, atol=1e-6)


@pytest.mark.skipif(not STANDARD_N_WEIGHTS.is_file(), reason="Local standard yolo26n.pt is required.")
def test_pretrained_equivalence_transfer_and_standard_checkpoint_loading(tmp_path: Path) -> None:
    """Check stock/E10 equivalence, exact transfer, zero suppression, and ordinary YOLO loading."""
    stock = YOLO(STANDARD_N_WEIGHTS, verbose=False).model
    e10 = build_dgqm_yolo("n", "trainable", verbose=False)
    control = build_ccs_dgqm_yolo("n", "control", verbose=False)
    trainable = build_ccs_dgqm_yolo("n", "trainable", verbose=False)
    stock.eval()
    e10.model.eval()
    control.model.eval()
    trainable.model.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        stock_postprocessed, stock_raw = stock(image)
        e10_postprocessed, e10_raw = e10.model(image)
        control_postprocessed, control_raw = control.model(image)
        trainable_postprocessed, trainable_raw = trainable.model(image)
        stock_decoded = stock.model[-1]._inference(stock_raw["one2one"])
        e10_decoded = e10.model.model[-1]._inference(e10_raw["one2one"])
        control_decoded = control.model.model[-1]._inference(control_raw["one2one"])
        trainable_decoded = trainable.model.model[-1]._inference(trainable_raw["one2one"])

    for raw in (control_raw, trainable_raw):
        for branch in ("one2many", "one2one"):
            for component in ("boxes", "scores"):
                torch.testing.assert_close(
                    stock_raw[branch][component],
                    raw[branch][component],
                    rtol=0,
                    atol=1e-6,
                )
            torch.testing.assert_close(
                e10_raw[branch]["quality"],
                raw[branch]["quality"],
                rtol=0,
                atol=1e-6,
            )
            assert torch.count_nonzero(raw[branch]["class_suppression"]).item() == 0
    for decoded in (e10_decoded, control_decoded, trainable_decoded):
        torch.testing.assert_close(stock_decoded, decoded, rtol=0, atol=1e-6)
    for postprocessed in (e10_postprocessed, control_postprocessed, trainable_postprocessed):
        torch.testing.assert_close(stock_postprocessed, postprocessed, rtol=0, atol=1e-6)

    for model in (control, trainable):
        report = model.ccs_dgqm_transfer_report
        assert set(report.exact_keys) == set(stock.state_dict())
        assert report.new_adapter_keys and report.new_quality_keys and report.new_suppression_keys
        assert all(
            report.coverage(name).percentage == 100.0
            for name in ("backbone", "neck", "cv2", "cv3", "one2one_cv2", "one2one_cv3")
        )
        head = model.model.model[-1]
        for suppression_head in (*head.suppression_heads, *head.one2one_suppression_heads):
            assert torch.count_nonzero(suppression_head.output.weight).item() == 0
            assert torch.count_nonzero(suppression_head.output.bias).item() == 0

    checkpoint = tmp_path / "synthetic-e11-ccs-dgqm.pt"
    trainable.save(checkpoint)
    loaded = YOLO(checkpoint, verbose=False)
    loaded_head = loaded.model.model[-1]
    assert isinstance(loaded_head, ClassConditionalSuppressionDGQMDetect)
    assert loaded_head.reg_max == 1
    assert loaded_head.stride.tolist() == [8.0, 16.0, 32.0]
    clean_load = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from ultralytics import YOLO; YOLO(sys.argv[1], verbose=False)",
            str(checkpoint),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert clean_load.returncode == 0, clean_load.stderr


def test_full_e10_suppression_loss_backward_amp_and_finite_outputs() -> None:
    """Verify five components, both assignments, suppression gradients, and CPU AMP."""
    model = ClassConditionalSuppressionDetectionModel(
        ccs_dgqm_yaml_path("n", "trainable"),
        verbose=False,
    )
    model.args = get_cfg()
    model.train()
    head = model.model[-1]
    with torch.no_grad():
        for tower in (*head.cv3, *head.one2one_cv3):
            tower[-1].bias.zero_()
    criterion = model.init_criterion()
    assert isinstance(criterion, E2ELoss)
    for branch in (criterion.one2many, criterion.one2one):
        assert isinstance(branch, ClassConditionalSuppressionDetectionLoss)
        assert isinstance(branch, DualGeometryQualityDetectionLoss)
        assert branch.reg_max == 1
        assert not branch.use_dfl
        assert branch.bbox_loss.dfl_loss is None
        assert isinstance(branch.bbox_loss, ResidualNWDBboxLoss)
        assert branch.bbox_loss.config.mode == "constant-010"
        assert branch.bbox_loss.config.nwd_scale == 0.10
        assert isinstance(branch.bce, ContrastRingBCEWithLogitsLoss)
        assert branch.contrast_ring_config == E2_1B_CONFIG
        assert branch.quality_config == DualGeometryQualityConfig()
        assert branch.suppression_config == ClassConditionalSuppressionConfig()
        assert branch.trainable_quality and branch.trainable_suppression
        assert branch.loss_names == (
            "box_loss",
            "cls_loss",
            "l1_loss",
            "quality_loss",
            "suppression_loss",
        )
    model.criterion = criterion

    batch = _synthetic_batch()
    loss, loss_items = model.loss(batch)
    assert loss.shape == (5,)
    assert set(loss_items) == {
        "box_loss",
        "cls_loss",
        "l1_loss",
        "quality_loss",
        "suppression_loss",
    }
    assert torch.isfinite(loss).all()
    assert all(torch.isfinite(value) for value in loss_items.values())
    assert loss[3] > 0 and loss[4] > 0
    loss.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    for group in (head.suppression_heads, head.one2one_suppression_heads):
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for suppression_head in group
            for parameter in suppression_head.parameters()
        )

    model.zero_grad(set_to_none=True)
    model.criterion = model.init_criterion()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_loss, amp_items = model.loss(_synthetic_batch())
    assert torch.isfinite(amp_loss).all()
    assert all(torch.isfinite(value) for value in amp_items.values())
    amp_loss.sum().backward()
    amp_gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert amp_gradients
    assert all(torch.isfinite(gradient).all() for gradient in amp_gradients)

    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(torch.rand(1, 3, 64, 64))
    assert all(torch.isfinite(tensor).all() for tensor in _all_output_tensors(output))


def test_validation_diagnostics_cover_all_branches_and_levels() -> None:
    """Measure and serialize all requested validation-batch diagnostics."""
    from research.scripts.train_e11 import _measure_correction_statistics

    model = ClassConditionalSuppressionDetectionModel(
        ccs_dgqm_yaml_path("n", "trainable"),
        verbose=False,
    )
    head = model.model[-1]
    with torch.no_grad():
        for quality_head in (*head.quality_heads, *head.one2one_quality_heads):
            quality_head.output.bias.fill_(0.2)
        for suppression_head in (*head.suppression_heads, *head.one2one_suppression_heads):
            midpoint = suppression_head.output.bias.numel() // 2
            suppression_head.output.bias[:midpoint] = -0.5
            suppression_head.output.bias[midpoint:] = 0.5

    statistics = _measure_correction_statistics(
        SimpleNamespace(model=model),
        torch.rand(1, 3, 64, 64),
    )
    assert set(statistics) == {
        (assignment, level)
        for assignment in ("one-to-many", "one-to-one")
        for level in ("P3", "P4", "P5")
    }
    diagnostics = collect_ccs_dgqm_diagnostics(model, statistics)
    assert len(diagnostics) == 6
    for item in diagnostics:
        assert item["shared_quality_output_norm"] > 0
        assert item["mean_shared_quality_correction"] > 0
        assert item["suppression_output_weight_norm"] == 0
        assert item["suppression_output_bias_norm"] > 0
        assert item["mean_suppression_correction"] < 0
        assert item["fraction_corrections_below_negative_0_1"] == pytest.approx(0.5)
        assert item["fraction_corrections_clamped_to_zero"] == pytest.approx(0.5)
        assert item["madh_box_alpha"] == 0
        assert item["madh_classification_alpha"] == 0
    json.dumps(diagnostics)


@pytest.mark.parametrize("size", tuple(SCALE_SETTINGS))
def test_n_to_x_latency_smoke(size: str) -> None:
    """Measure finite E10/E11 CPU latency without changing the prescribed architecture."""
    e10 = DualGeometryQualityDetectionModel(
        dgqm_yaml_path(size, "trainable"),
        verbose=False,
    ).eval()
    e11 = ClassConditionalSuppressionDetectionModel(
        ccs_dgqm_yaml_path(size, "trainable"),
        verbose=False,
    ).eval()
    image = torch.rand(1, 3, 640, 640)

    timings = {"e10": [], "e11": []}
    with torch.inference_mode():
        for _ in range(2):
            e10(image)
            e11(image)
        for repetition in range(9):
            ordered_models = (("e10", e10), ("e11", e11))
            if repetition % 2:
                ordered_models = tuple(reversed(ordered_models))
            for name, model in ordered_models:
                start = time.perf_counter()
                model(image)
                timings[name].append(1000.0 * (time.perf_counter() - start))

    e10_latency = statistics.median(timings["e10"])
    e11_latency = statistics.median(timings["e11"])
    assert e10_latency > 0 and e11_latency > 0
    assert torch.isfinite(torch.tensor((e10_latency, e11_latency))).all()
    print(
        json.dumps(
            {
                "scale": size,
                "device": "cpu",
                "imgsz": 640,
                "e10_latency_ms": e10_latency,
                "e11_latency_ms": e11_latency,
                "latency_increase_percent": 100.0 * (e11_latency - e10_latency) / e10_latency,
            }
        )
    )


def test_invalid_factory_inputs_and_run_names() -> None:
    """Reject unsupported E11 values and preserve exact run naming."""
    from research.scripts.train_e11 import _run_name

    for size in ("", "a", "N", "xx"):
        with pytest.raises(ValueError, match="Unsupported YOLO26 scale"):
            ccs_dgqm_yaml_path(size, "trainable")
    with pytest.raises(ValueError, match="Unsupported CCS-DGQM variant"):
        ccs_dgqm_yaml_path("n", "unknown")
    assert (
        _run_name("n", "trainable", 640, 30, -1, 42)
        == "yolo26n-ccs-dgqm_img640_e30_bauto_seed42"
    )
    assert (
        _run_name("n", "control", 640, 30, -8, 42)
        == "yolo26n-ccs-dgqm-control_img640_e30_b-8_seed42"
    )
