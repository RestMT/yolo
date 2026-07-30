# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Synthetic checks for E12 Foregroundness-Factorized DGQM."""

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
    ForegroundnessAgreementHead,
    ForegroundnessFactorizedDGQMDetect,
    MorphologyAdaptiveAdapter,
)
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import get_flops, get_num_params
from yolo_improved.contrast_ring_loss import ContrastRingBCEWithLogitsLoss
from yolo_improved.dgqm_config import DualGeometryQualityConfig
from yolo_improved.dgqm_loss import DualGeometryQualityDetectionLoss
from yolo_improved.dgqm_model import (
    DualGeometryQualityDetectionModel,
    build_dgqm_yolo,
    dgqm_yaml_path,
)
from yolo_improved.foreground_config import (
    ForegroundnessFactorizationConfig,
    resolve_foregroundness_factorization_config,
)
from yolo_improved.foreground_loss import ForegroundnessFactorizedDetectionLoss
from yolo_improved.foreground_model import (
    FF_DGQM_VARIANTS,
    ForegroundnessFactorizedDetectionModel,
    build_ff_dgqm_yolo,
    collect_ff_dgqm_diagnostics,
    ff_dgqm_yaml_path,
    remap_yolo26_ff_dgqm_state_dict,
)
from yolo_improved.madh_model import E2_1B_CONFIG
from yolo_improved.residual_nwd_loss import ResidualNWDBboxLoss


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STOCK_MODEL_DIRECTORY = REPOSITORY_ROOT / "ultralytics" / "cfg" / "models" / "26"
E12_MODEL_DIRECTORY = REPOSITORY_ROOT / "research" / "models"
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
    e12: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Return requested E10 raw box, score, and quality differences."""
    return {
        f"{branch}_{component}": float(
            (e10[branch][component] - e12[branch][component]).abs().max().item()
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


def test_foreground_config_defaults_resolver_and_validation() -> None:
    """Validate exact E12 defaults, resolver forms, and every numeric constraint."""
    config = ForegroundnessFactorizationConfig()
    assert config == ForegroundnessFactorizationConfig(
        foreground_gain=0.10,
        foreground_scale=0.5,
        hard_negative_gamma=2.0,
        hard_negative_min_probability=0.05,
        negative_weight=0.5,
        eps=1e-6,
    )
    assert resolve_foregroundness_factorization_config(None) == config
    assert resolve_foregroundness_factorization_config(config) is config
    assert resolve_foregroundness_factorization_config({"negative_weight": 0.25}).negative_weight == 0.25
    invalid_values = (
        ({"foreground_gain": -0.1}, "foreground_gain"),
        ({"foreground_scale": 0.0}, "foreground_scale"),
        ({"hard_negative_gamma": -0.1}, "hard_negative_gamma"),
        ({"hard_negative_min_probability": -0.1}, "hard_negative_min_probability"),
        ({"hard_negative_min_probability": 1.0}, "hard_negative_min_probability"),
        ({"negative_weight": -0.1}, "negative_weight"),
        ({"eps": 0.0}, "eps"),
        ({"foreground_gain": float("nan")}, "foreground_gain"),
        ({"foreground_scale": float("inf")}, "foreground_scale"),
        ({"hard_negative_gamma": True}, "hard_negative_gamma"),
    )
    for kwargs, message in invalid_values:
        with pytest.raises(ValueError, match=message):
            ForegroundnessFactorizationConfig(**kwargs)
    with pytest.raises(TypeError, match="foreground_config"):
        resolve_foregroundness_factorization_config("invalid")


def test_foreground_head_structure_zero_output_and_frozen_bypass() -> None:
    """Check E10 agreement reuse, zero initialization, and control bypass."""
    trainable = ForegroundnessAgreementHead(128)
    assert isinstance(trainable, BoxClassAgreementQualityHead)
    assert trainable.hidden == 16
    assert trainable.reduce.conv.in_channels == 384
    assert trainable.reduce.conv.out_channels == 16
    assert trainable.context.conv.kernel_size == (3, 3)
    assert trainable.context.conv.groups == 16
    assert trainable.output.in_channels == 16
    assert trainable.output.out_channels == 1
    assert torch.count_nonzero(trainable.output.weight).item() == 0
    assert torch.count_nonzero(trainable.output.bias).item() == 0
    box = torch.randn(2, 128, 9, 11)
    cls = torch.randn(2, 128, 9, 11)
    output = trainable(box, cls)
    assert output.shape == (2, 1, 9, 11)
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)

    frozen = ForegroundnessAgreementHead(128, trainable_foreground=False)
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
    assert frozen_output.shape == (2, 1, 9, 11)
    assert torch.count_nonzero(frozen_output).item() == 0
    assert not calls
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


def test_detect_forward_detachment_shape_independence_and_symmetric_inference() -> None:
    """Verify E10 reuse, detached foreground features, and symmetric bounded corrections."""
    head = ForegroundnessFactorizedDGQMDetect(
        nc=3,
        end2end=True,
        ch=(16, 24, 32),
    )
    head.stride[:] = torch.tensor([8.0, 16.0, 32.0])
    assert len(head.foreground_heads) == len(head.one2one_foreground_heads) == 3
    assert not hasattr(head, "suppression_heads")
    for first, second in zip(head.foreground_heads, head.one2one_foreground_heads):
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
    foreground_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    handles = [
        foreground_head.register_forward_pre_hook(
            lambda _, inputs: foreground_inputs.append((inputs[0], inputs[1]))
        )
        for foreground_head in (*head.foreground_heads, *head.one2one_foreground_heads)
    ]
    try:
        predictions = head(features)
    finally:
        for handle in handles:
            handle.remove()
    total_anchors = 8 * 8 + 4 * 4 + 2 * 2
    for branch in ("one2many", "one2one"):
        assert set(predictions[branch]) == {"boxes", "scores", "quality", "foreground", "feats"}
        assert predictions[branch]["quality"].shape == (2, 1, total_anchors)
        assert predictions[branch]["foreground"].shape == (2, 1, total_anchors)
        assert torch.count_nonzero(predictions[branch]["quality"]).item() == 0
        assert torch.count_nonzero(predictions[branch]["foreground"]).item() == 0
    assert len(foreground_inputs) == 6
    assert all(
        not box_feature.requires_grad and not cls_feature.requires_grad
        for box_feature, cls_feature in foreground_inputs
    )

    head.eval()
    detached_features = [feature.detach() for feature in features]
    with torch.no_grad():
        _, raw_zero = head(detached_features)
        decoded_zero = head._inference(raw_zero["one2one"])
        for foreground_head in head.one2one_foreground_heads:
            foreground_head.output.bias.fill_(100.0)
        _, raw_positive = head(detached_features)
        decoded_positive = head._inference(raw_positive["one2one"])
        for foreground_head in head.one2one_foreground_heads:
            foreground_head.output.bias.fill_(-100.0)
        _, raw_negative = head(detached_features)
        decoded_negative = head._inference(raw_negative["one2one"])

    positive_correction = head.foreground_scale * torch.tanh(raw_positive["one2one"]["foreground"])
    negative_correction = head.foreground_scale * torch.tanh(raw_negative["one2one"]["foreground"])
    assert positive_correction.max().item() <= 0.5
    assert positive_correction.min().item() >= 0
    assert negative_correction.min().item() >= -0.5
    assert negative_correction.max().item() <= 0
    torch.testing.assert_close(decoded_zero[:, :4], decoded_positive[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(decoded_zero[:, :4], decoded_negative[:, :4], rtol=0, atol=0)
    assert torch.all(decoded_positive[:, 4:] > decoded_zero[:, 4:])
    assert torch.all(decoded_negative[:, 4:] < decoded_zero[:, 4:])


def test_unified_yamls_and_all_virtual_scale_names() -> None:
    """Verify both unified definitions and all ten virtual E12 names."""
    e10_args = yaml_model_load(dgqm_yaml_path("n", "trainable"))["head"][-1][3]
    for variant in FF_DGQM_VARIANTS:
        suffix = "ff-dgqm-control" if variant == "control" else "ff-dgqm"
        unified = E12_MODEL_DIRECTORY / f"yolo26-{suffix}.yaml"
        assert unified.is_file()
        assert not list(E12_MODEL_DIRECTORY.glob(f"yolo26?-{suffix}.yaml"))
        for size, scale_settings in SCALE_SETTINGS.items():
            virtual = ff_dgqm_yaml_path(size, variant)
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
            assert target["head"][-1][2] == "ForegroundnessFactorizedDGQMDetect"
            args = target["head"][-1][3]
            assert args[: len(e10_args)] == e10_args
            assert args[len(e10_args) :] == [
                0.125,
                16,
                64,
                0.5,
                variant == "trainable",
            ]


@pytest.mark.parametrize("size", tuple(SCALE_SETTINGS))
def test_n_to_x_structure_transfer_parameters_and_flops(size: str) -> None:
    """Build both variants, retain complete E10, and measure foreground-only cost."""
    stock = DetectionModel(STOCK_MODEL_DIRECTORY / f"yolo26{size}.yaml", verbose=False)
    e10 = DualGeometryQualityDetectionModel(
        dgqm_yaml_path(size, "trainable"),
        verbose=False,
    )
    source_state = stock.state_dict()
    stock_parameters = get_num_params(stock)
    e10_parameters = get_num_params(e10)
    e10_flops = get_flops(e10, imgsz=640)

    for variant in FF_DGQM_VARIANTS:
        target = ForegroundnessFactorizedDetectionModel(
            ff_dgqm_yaml_path(size, variant),
            verbose=False,
        )
        head = target.model[-1]
        assert isinstance(head, ForegroundnessFactorizedDGQMDetect)
        assert head.f == [16, 19, 22]
        assert head.nl == 3
        assert head.stride.tolist() == [8.0, 16.0, 32.0]
        assert head.reg_max == 1
        assert isinstance(head.dfl, nn.Identity)
        assert not hasattr(head, "suppression_heads")
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
        for group in (head.foreground_heads, head.one2one_foreground_heads):
            assert len(group) == 3
            assert all(
                foreground_head.trainable_foreground == (variant == "trainable")
                for foreground_head in group
            )

        transferred, report = remap_yolo26_ff_dgqm_state_dict(
            source_state,
            target.state_dict(),
            target_parameter_keys=dict(target.named_parameters()),
        )
        assert set(transferred) == set(source_state)
        assert not report.skipped_missing_keys
        assert not report.skipped_shape_keys
        assert report.new_adapter_keys
        assert report.new_quality_keys
        assert report.new_foreground_keys
        assert (
            set(report.new_adapter_keys)
            | set(report.new_quality_keys)
            | set(report.new_foreground_keys)
        ) == set(report.new_ff_keys)
        for coverage_name in ("backbone", "neck", "cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            assert report.coverage(coverage_name).percentage == 100.0
        assert report.coverage("adapters").transferred_tensors == 0
        assert report.coverage("quality_heads").transferred_tensors == 0
        assert report.coverage("foreground_heads").transferred_tensors == 0

        target_parameters = get_num_params(target)
        foreground_parameters = sum(
            parameter.numel()
            for group in (head.foreground_heads, head.one2one_foreground_heads)
            for parameter in group.parameters()
        )
        assert target_parameters - e10_parameters == foreground_parameters
        parameter_increase = 100.0 * foreground_parameters / e10_parameters
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
                    "foreground_parameters": foreground_parameters,
                    "foreground_parameter_increase_percent": parameter_increase,
                    "e10_gflops": e10_flops,
                    "gflops": target_flops,
                    "foreground_gflops_increase_percent": gflops_increase,
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


def test_anchor_targets_hard_negative_max_probability_and_no_classwise_negatives() -> None:
    """Check one ±1 target per anchor and max-class hard-negative weighting."""
    model = ForegroundnessFactorizedDetectionModel(
        ff_dgqm_yaml_path("n", "trainable"),
        nc=3,
        verbose=False,
    )
    model.args = get_cfg()
    criterion = model.init_criterion().one2many
    probabilities = torch.tensor(
        [[[0.70, 0.10, 0.20], [0.20, 0.80, 0.30], [0.01, 0.02, 0.03]]],
        dtype=torch.float32,
    )
    score_logits = torch.logit(probabilities).transpose(1, 2).contiguous()
    foreground_logits = torch.tensor([[[0.20, -0.40, 0.50]]], requires_grad=True)
    fg_mask = torch.tensor([[True, False, False]])
    target_scores = torch.zeros(1, 3, 3)
    target_scores[0, 0, 0] = 0.75
    preds = {
        "scores": score_logits,
        "foreground": foreground_logits,
    }
    loss = criterion.calculate_foreground_loss(preds, fg_mask, target_scores)

    prediction = torch.tanh(foreground_logits.transpose(1, 2))
    assert prediction.numel() == fg_mask.numel()
    positive_prediction = prediction[fg_mask]
    positive_target = torch.ones_like(positive_prediction)
    expected_positive = F.smooth_l1_loss(
        positive_prediction,
        positive_target,
        reduction="none",
    ).squeeze(-1)
    positive_weight = target_scores.max(dim=-1).values[fg_mask]
    expected_positive = (expected_positive * positive_weight).sum() / positive_weight.sum()
    anchor_probability = probabilities.amax(dim=-1)
    negative_anchor_probability = anchor_probability[~fg_mask]
    negative_weight = negative_anchor_probability.square() * (negative_anchor_probability >= 0.05)
    negative_prediction = prediction[~fg_mask]
    negative_target = -torch.ones_like(negative_prediction)
    expected_negative = F.smooth_l1_loss(
        negative_prediction,
        negative_target,
        reduction="none",
    ).squeeze(-1)
    expected_negative = (expected_negative * negative_weight).sum() / negative_weight.sum()
    expected = 0.10 * (expected_positive + 0.5 * expected_negative)
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
    assert negative_anchor_probability.tolist() == pytest.approx([0.8, 0.03])
    assert negative_weight[1].item() == 0

    changed_positive_classes = score_logits.clone()
    changed_positive_classes[:, 1:, 0] = 100.0
    unchanged_loss = criterion.calculate_foreground_loss(
        {"scores": changed_positive_classes, "foreground": foreground_logits},
        fg_mask,
        target_scores,
    )
    torch.testing.assert_close(loss, unchanged_loss, rtol=0, atol=0)
    loss.backward()
    assert foreground_logits.grad is not None
    assert torch.isfinite(foreground_logits.grad).all()
    assert torch.count_nonzero(foreground_logits.grad).item() > 0

    inactive_logits = torch.randn(1, 1, 2, requires_grad=True)
    differentiable_zero = criterion.calculate_foreground_loss(
        {
            "scores": torch.logit(torch.full((1, 3, 2), 0.01)),
            "foreground": inactive_logits,
        },
        torch.zeros(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, 3),
    )
    assert differentiable_zero.requires_grad
    torch.testing.assert_close(differentiable_zero, torch.zeros_like(differentiable_zero))
    differentiable_zero.backward()
    assert inactive_logits.grad is not None
    assert torch.count_nonzero(inactive_logits.grad).item() == 0


@pytest.mark.parametrize("branch_name", ("one2many", "one2one"))
def test_assignment_and_e10_losses_are_invariant_to_foreground(branch_name: str) -> None:
    """Ensure one assignment and unchanged E2.1b/E10 quality for arbitrary foreground logits."""
    model = ForegroundnessFactorizedDetectionModel(
        ff_dgqm_yaml_path("n", "trainable"),
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
        "foreground": torch.full_like(branch_predictions["foreground"], 100.0),
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
    ("branch_name", "foreground_name"),
    (("one2many", "foreground_heads"), ("one2one", "one2one_foreground_heads")),
)
def test_foreground_gradient_isolation_and_head_gradients(
    branch_name: str,
    foreground_name: str,
) -> None:
    """Ensure foreground loss trains only its matching foreground heads."""
    model = ForegroundnessFactorizedDetectionModel(
        ff_dgqm_yaml_path("n", "trainable"),
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
    foreground_loss = criterion.get_assigned_targets_and_loss(branch_predictions, batch)[1][4]
    assert foreground_loss > 0
    foreground_loss.backward()

    head = model.model[-1]
    foreground_heads = getattr(head, foreground_name)
    foreground_gradients = [
        parameter.grad
        for foreground_head in foreground_heads
        for parameter in foreground_head.parameters()
        if parameter.grad is not None
    ]
    assert foreground_gradients
    assert all(torch.isfinite(gradient).all() for gradient in foreground_gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in foreground_gradients)
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
    other_name = "one2one_foreground_heads" if foreground_name == "foreground_heads" else "foreground_heads"
    _assert_no_gradients(getattr(head, other_name))


def test_control_matches_e10_predictions_and_loss() -> None:
    """Verify disabled foregroundness is mathematically identical to trainable E10."""
    torch.manual_seed(112)
    e10 = DualGeometryQualityDetectionModel(
        dgqm_yaml_path("n", "trainable"),
        verbose=False,
    )
    control = ForegroundnessFactorizedDetectionModel(
        ff_dgqm_yaml_path("n", "control"),
        verbose=False,
    )
    incompatible = control.load_state_dict(e10.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(
        key.startswith(("model.23.foreground_heads.", "model.23.one2one_foreground_heads."))
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
        torch.count_nonzero(control_raw[branch]["foreground"]).item() == 0
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
    """Check stock/E10 equivalence, exact transfer, zero foreground, and ordinary YOLO loading."""
    stock = YOLO(STANDARD_N_WEIGHTS, verbose=False).model
    e10 = build_dgqm_yolo("n", "trainable", verbose=False)
    control = build_ff_dgqm_yolo("n", "control", verbose=False)
    trainable = build_ff_dgqm_yolo("n", "trainable", verbose=False)
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
            assert torch.count_nonzero(raw[branch]["foreground"]).item() == 0
            assert "class_suppression" not in raw[branch]
    for decoded in (e10_decoded, control_decoded, trainable_decoded):
        torch.testing.assert_close(stock_decoded, decoded, rtol=0, atol=1e-6)
    for postprocessed in (e10_postprocessed, control_postprocessed, trainable_postprocessed):
        torch.testing.assert_close(stock_postprocessed, postprocessed, rtol=0, atol=1e-6)

    for model in (control, trainable):
        report = model.ff_dgqm_transfer_report
        assert set(report.exact_keys) == set(stock.state_dict())
        assert report.new_adapter_keys and report.new_quality_keys and report.new_foreground_keys
        assert all(
            report.coverage(name).percentage == 100.0
            for name in ("backbone", "neck", "cv2", "cv3", "one2one_cv2", "one2one_cv3")
        )
        head = model.model.model[-1]
        for foreground_head in (*head.foreground_heads, *head.one2one_foreground_heads):
            assert torch.count_nonzero(foreground_head.output.weight).item() == 0
            assert torch.count_nonzero(foreground_head.output.bias).item() == 0

    checkpoint = tmp_path / "synthetic-e12-ff-dgqm.pt"
    trainable.save(checkpoint)
    loaded = YOLO(checkpoint, verbose=False)
    loaded_head = loaded.model.model[-1]
    assert isinstance(loaded_head, ForegroundnessFactorizedDGQMDetect)
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


def test_full_e10_foreground_loss_backward_amp_and_finite_outputs() -> None:
    """Verify five components, both assignments, foreground gradients, and CPU AMP."""
    model = ForegroundnessFactorizedDetectionModel(
        ff_dgqm_yaml_path("n", "trainable"),
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
        assert isinstance(branch, ForegroundnessFactorizedDetectionLoss)
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
        assert branch.foreground_config == ForegroundnessFactorizationConfig()
        assert branch.trainable_quality and branch.trainable_foreground
        assert branch.loss_names == (
            "box_loss",
            "cls_loss",
            "l1_loss",
            "quality_loss",
            "foreground_loss",
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
        "foreground_loss",
    }
    assert torch.isfinite(loss).all()
    assert all(torch.isfinite(value) for value in loss_items.values())
    assert loss[3] > 0 and loss[4] > 0
    loss.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    for group in (head.foreground_heads, head.one2one_foreground_heads):
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for foreground_head in group
            for parameter in foreground_head.parameters()
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


def test_post_training_diagnostics_reset_exhausted_validation_loader() -> None:
    """Reset the persistent validation iterator before post-training diagnostics."""
    from research.scripts.train_e12 import _next_validation_batch

    expected_batch = {"img": torch.ones(1, 3, 8, 8)}

    class ExhaustedValidationLoader:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def __iter__(self):
            return iter((expected_batch,)) if self.reset_calls else iter(())

    validation_loader = ExhaustedValidationLoader()
    model = SimpleNamespace(
        trainer=SimpleNamespace(
            validator=SimpleNamespace(dataloader=validation_loader),
        )
    )

    assert _next_validation_batch(model) is expected_batch
    assert validation_loader.reset_calls == 1


def test_validation_assignment_diagnostics_cover_all_branches_and_levels() -> None:
    """Measure and serialize all requested validation-assignment diagnostics."""
    from research.scripts.train_e12 import _measure_assignment_statistics

    model = ForegroundnessFactorizedDetectionModel(
        ff_dgqm_yaml_path("n", "trainable"),
        verbose=False,
    )
    model.args = get_cfg()
    head = model.model[-1]
    with torch.no_grad():
        for quality_head in (*head.quality_heads, *head.one2one_quality_heads):
            quality_head.output.bias.fill_(0.2)
        for group in (head.foreground_heads, head.one2one_foreground_heads):
            for foreground_head, bias in zip(group, (0.4, -0.4, 0.0)):
                foreground_head.output.bias.fill_(bias)

    statistics = _measure_assignment_statistics(
        SimpleNamespace(model=model),
        _synthetic_batch(),
    )
    assert set(statistics) == {
        (assignment, level)
        for assignment in ("one-to-many", "one-to-one")
        for level in ("P3", "P4", "P5")
    }
    assert any(
        item["mean_positive_foreground_correction"] is not None for item in statistics.values()
    )
    assert all(
        item["mean_background_foreground_correction"] is not None for item in statistics.values()
    )
    diagnostics = collect_ff_dgqm_diagnostics(model, statistics)
    assert len(diagnostics) == 6
    for item in diagnostics:
        assert item["quality_output_weight_norm"] == 0
        assert item["mean_quality_correction"] > 0
        assert item["foreground_output_weight_norm"] == 0
        assert -0.5 <= item["mean_foreground_correction"] <= 0.5
        assert item["madh_box_alpha"] == 0
        assert item["madh_classification_alpha"] == 0
        if item["level"] == "P3":
            assert item["foreground_output_bias"] == pytest.approx(0.4)
            assert item["fraction_corrections_above_positive_0_1"] == 1
            assert item["fraction_corrections_below_negative_0_1"] == 0
        elif item["level"] == "P4":
            assert item["foreground_output_bias"] == pytest.approx(-0.4)
            assert item["fraction_corrections_above_positive_0_1"] == 0
            assert item["fraction_corrections_below_negative_0_1"] == 1
        else:
            assert item["foreground_output_bias"] == 0
            assert item["fraction_corrections_above_positive_0_1"] == 0
            assert item["fraction_corrections_below_negative_0_1"] == 0
    json.dumps(diagnostics)


@pytest.mark.parametrize("size", tuple(SCALE_SETTINGS))
def test_n_to_x_latency_smoke(size: str) -> None:
    """Measure finite E10/E12 CPU latency without changing the prescribed architecture."""
    e10 = DualGeometryQualityDetectionModel(
        dgqm_yaml_path(size, "trainable"),
        verbose=False,
    ).eval()
    e12 = ForegroundnessFactorizedDetectionModel(
        ff_dgqm_yaml_path(size, "trainable"),
        verbose=False,
    ).eval()
    image = torch.rand(1, 3, 640, 640)

    timings = {"e10": [], "e12": []}
    with torch.inference_mode():
        for _ in range(2):
            e10(image)
            e12(image)
        for repetition in range(9):
            ordered_models = (("e10", e10), ("e12", e12))
            if repetition % 2:
                ordered_models = tuple(reversed(ordered_models))
            for name, model in ordered_models:
                start = time.perf_counter()
                model(image)
                timings[name].append(1000.0 * (time.perf_counter() - start))

    e10_latency = statistics.median(timings["e10"])
    e12_latency = statistics.median(timings["e12"])
    assert e10_latency > 0 and e12_latency > 0
    assert torch.isfinite(torch.tensor((e10_latency, e12_latency))).all()
    print(
        json.dumps(
            {
                "scale": size,
                "device": "cpu",
                "imgsz": 640,
                "e10_latency_ms": e10_latency,
                "e12_latency_ms": e12_latency,
                "latency_increase_percent": 100.0 * (e12_latency - e10_latency) / e10_latency,
            }
        )
    )


def test_invalid_factory_inputs_and_run_names() -> None:
    """Reject unsupported E12 values and preserve exact run naming."""
    from research.scripts.train_e12 import _run_name

    for size in ("", "a", "N", "xx"):
        with pytest.raises(ValueError, match="Unsupported YOLO26 scale"):
            ff_dgqm_yaml_path(size, "trainable")
    for variant in ("", "foreground", "Control", "train"):
        with pytest.raises(ValueError, match="Unsupported FF-DGQM variant"):
            ff_dgqm_yaml_path("n", variant)
    with pytest.raises(ValueError, match="default E10"):
        build_ff_dgqm_yolo(
            "n",
            "trainable",
            quality_config=DualGeometryQualityConfig(quality_gain=0.0),
        )
    assert _run_name("n", "control", 640, 100, -1, 42) == (
        "yolo26n-ff-dgqm-control_img640_e100_bauto_seed42"
    )
    assert _run_name("x", "trainable", 512, 50, 8, 7) == (
        "yolo26x-ff-dgqm_img512_e50_b8_seed7"
    )
