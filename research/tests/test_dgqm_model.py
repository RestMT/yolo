# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Synthetic checks for E10 Dual-Geometry Quality-Calibrated MADH."""

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
from torch import nn

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import (
    BoxClassAgreementQualityHead,
    DualGeometryQualityMorphologyDetect,
)
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import get_flops, get_num_params
from yolo_improved.contrast_ring_loss import ContrastRingBCEWithLogitsLoss
from yolo_improved.contrast_ring_model import ContrastRingDetectionModel
from yolo_improved.dgqm_config import DualGeometryQualityConfig
from yolo_improved.dgqm_loss import DualGeometryQualityDetectionLoss
from yolo_improved.dgqm_model import (
    DGQM_VARIANTS,
    DualGeometryQualityDetectionModel,
    build_dgqm_yolo,
    collect_dgqm_diagnostics,
    dgqm_yaml_path,
    remap_yolo26_dgqm_state_dict,
)
from yolo_improved.madh_model import E2_1B_CONFIG, madh_yaml_path, remap_yolo26_madh_state_dict
from yolo_improved.residual_nwd_loss import ResidualNWDBboxLoss


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STOCK_MODEL_DIRECTORY = REPOSITORY_ROOT / "ultralytics" / "cfg" / "models" / "26"
E10_MODEL_DIRECTORY = REPOSITORY_ROOT / "research" / "models"
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


def _raw_differences(
    first: dict[str, dict[str, torch.Tensor]],
    second: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Return requested raw box/score maximum absolute differences."""
    return {
        f"{branch}_{component}": float(
            (first[branch][component] - second[branch][component]).abs().max().item()
        )
        for branch in ("one2many", "one2one")
        for component in ("boxes", "scores")
    }


def _assert_no_gradients(modules) -> None:
    """Assert that selected modules received no nonzero gradients."""
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for module in modules
        for parameter in module.parameters()
    )


def test_quality_config_defaults_and_validation() -> None:
    """Validate the exact default target/loss coefficients and invalid inputs."""
    config = DualGeometryQualityConfig()
    assert config == DualGeometryQualityConfig(
        iou_weight=0.75,
        nwd_weight=0.25,
        quality_gain=0.25,
        negative_neutral_weight=0.05,
        quality_scale=1.0,
        eps=1e-6,
    )
    for kwargs, message in (
        ({"iou_weight": -0.1, "nwd_weight": 1.1}, "iou_weight"),
        ({"nwd_weight": -0.1, "iou_weight": 1.1}, "nwd_weight"),
        ({"iou_weight": 0.4, "nwd_weight": 0.5}, "must equal 1"),
        ({"quality_gain": -0.1}, "quality_gain"),
        ({"negative_neutral_weight": -0.1}, "negative_neutral_weight"),
        ({"quality_scale": 0.0}, "quality_scale"),
        ({"eps": 0.0}, "eps"),
        ({"iou_weight": float("nan")}, "iou_weight"),
        ({"quality_scale": True}, "quality_scale"),
    ):
        with pytest.raises(ValueError, match=message):
            DualGeometryQualityConfig(**kwargs)


def test_quality_head_structure_zero_output_and_frozen_bypass() -> None:
    """Check agreement geometry, exact zero initialization, and control bypass."""
    trainable = BoxClassAgreementQualityHead(128)
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

    frozen = BoxClassAgreementQualityHead(128, trainable_quality=False)
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


def test_detect_forward_quality_shape_detachment_independence_and_calibration() -> None:
    """Verify full E8 paths, detached quality features, and inference calibration."""
    head = DualGeometryQualityMorphologyDetect(
        nc=3,
        end2end=True,
        ch=(16, 24, 32),
    )
    head.stride[:] = torch.tensor([8.0, 16.0, 32.0])
    assert len(head.quality_heads) == len(head.one2one_quality_heads) == 3
    for first, second in zip(head.quality_heads, head.one2one_quality_heads):
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
    quality_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    handles = [
        quality_head.register_forward_pre_hook(
            lambda _, inputs: quality_inputs.append((inputs[0], inputs[1]))
        )
        for quality_head in (*head.quality_heads, *head.one2one_quality_heads)
    ]
    try:
        predictions = head(features)
    finally:
        for handle in handles:
            handle.remove()
    total_anchors = 8 * 8 + 4 * 4 + 2 * 2
    for branch in ("one2many", "one2one"):
        assert predictions[branch]["quality"].shape == (2, 1, total_anchors)
        assert torch.count_nonzero(predictions[branch]["quality"]).item() == 0
    assert len(quality_inputs) == 6
    assert all(
        not box_feature.requires_grad and not cls_feature.requires_grad
        for box_feature, cls_feature in quality_inputs
    )

    head.eval()
    with torch.no_grad():
        _, raw_zero = head([feature.detach() for feature in features])
        decoded_zero = head._inference(raw_zero["one2one"])
        for quality_head in head.one2one_quality_heads:
            quality_head.output.bias.fill_(0.5)
        _, raw_shifted = head([feature.detach() for feature in features])
        decoded_shifted = head._inference(raw_shifted["one2one"])
    torch.testing.assert_close(decoded_zero[:, :4], decoded_shifted[:, :4], rtol=0, atol=0)
    expected_scores = (
        raw_shifted["one2one"]["scores"]
        + head.quality_scale * torch.tanh(raw_shifted["one2one"]["quality"])
    ).sigmoid()
    torch.testing.assert_close(decoded_shifted[:, 4:], expected_scores, rtol=0, atol=0)
    assert torch.all(decoded_shifted[:, 4:] > decoded_zero[:, 4:])


def test_validation_diagnostic_corrections_use_quality_heads() -> None:
    """Measure all six correction means on synthetic validation images."""
    from research.scripts.train_e10 import _measure_quality_corrections

    model = DualGeometryQualityDetectionModel(
        dgqm_yaml_path("n", "trainable"),
        verbose=False,
    )
    corrections = _measure_quality_corrections(
        SimpleNamespace(model=model),
        torch.rand(1, 3, 64, 64),
    )
    assert set(corrections) == {
        (assignment, level)
        for assignment in ("one-to-many", "one-to-one")
        for level in ("P3", "P4", "P5")
    }
    assert all(correction == 0.0 for correction in corrections.values())


def test_unified_yamls_and_all_virtual_scale_names() -> None:
    """Verify both unified E10 definitions and all ten virtual names."""
    for variant in DGQM_VARIANTS:
        suffix = "dgqm-control" if variant == "control" else "dgqm"
        unified = E10_MODEL_DIRECTORY / f"yolo26-{suffix}.yaml"
        assert unified.is_file()
        assert not list(E10_MODEL_DIRECTORY.glob(f"yolo26?-{suffix}.yaml"))
        for size, scale_settings in SCALE_SETTINGS.items():
            virtual = dgqm_yaml_path(size, variant)
            assert not virtual.exists()
            target = yaml_model_load(virtual)
            stock = yaml_model_load(STOCK_MODEL_DIRECTORY / f"yolo26{size}.yaml")
            assert target["scale"] == size
            assert tuple(target["scales"][size]) == scale_settings
            for key in ("nc", "end2end", "reg_max", "scales", "backbone", "scale"):
                assert target[key] == stock[key]
            assert target["head"][:-1] == stock["head"][:-1]
            assert target["head"][-1][:2] == stock["head"][-1][:2]
            assert target["head"][-1][2] == "DualGeometryQualityMorphologyDetect"
            args = target["head"][-1][3]
            assert args == [
                "nc",
                0.125,
                16,
                96,
                7,
                2,
                0.5,
                0.125,
                16,
                64,
                1.0,
                variant == "trainable",
                True,
            ]


@pytest.mark.parametrize("size", tuple(SCALE_SETTINGS))
def test_n_to_x_structure_transfer_parameters_and_flops(size: str) -> None:
    """Build both variants, preserve E8, and measure quality-only model cost."""
    stock = DetectionModel(STOCK_MODEL_DIRECTORY / f"yolo26{size}.yaml", verbose=False)
    madh = ContrastRingDetectionModel(
        madh_yaml_path(size, "trainable"),
        verbose=False,
        loss_config=E2_1B_CONFIG,
    )
    source_state = stock.state_dict()
    stock_parameters = get_num_params(stock)
    madh_parameters = get_num_params(madh)
    madh_flops = get_flops(madh, imgsz=640)

    for variant in DGQM_VARIANTS:
        target = DualGeometryQualityDetectionModel(
            dgqm_yaml_path(size, variant),
            verbose=False,
        )
        head = target.model[-1]
        assert isinstance(head, DualGeometryQualityMorphologyDetect)
        assert head.f == [16, 19, 22]
        assert head.nl == 3
        assert head.stride.tolist() == [8.0, 16.0, 32.0]
        assert head.reg_max == 1
        assert isinstance(head.dfl, nn.Identity)
        assert len(head.box_adapters) == len(head.cls_adapters) == 3
        assert len(head.one2one_box_adapters) == len(head.one2one_cls_adapters) == 3
        assert len(head.quality_heads) == len(head.one2one_quality_heads) == 3
        assert all(
            adapter.trainable_adapter
            for group in (
                head.box_adapters,
                head.cls_adapters,
                head.one2one_box_adapters,
                head.one2one_cls_adapters,
            )
            for adapter in group
        )
        assert all(
            quality_head.trainable_quality == (variant == "trainable")
            for group in (head.quality_heads, head.one2one_quality_heads)
            for quality_head in group
        )

        transferred, report = remap_yolo26_dgqm_state_dict(
            source_state,
            target.state_dict(),
            target_parameter_keys=dict(target.named_parameters()),
        )
        assert set(transferred) == set(source_state)
        assert not report.skipped_missing_keys
        assert not report.skipped_shape_keys
        assert report.new_adapter_keys
        assert report.new_quality_keys
        assert set(report.new_adapter_keys).isdisjoint(report.new_quality_keys)
        for coverage_name in ("backbone", "neck", "cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            assert report.coverage(coverage_name).percentage == 100.0
        assert report.coverage("adapters").transferred_tensors == 0
        assert report.coverage("quality_heads").transferred_tensors == 0

        target_parameters = get_num_params(target)
        quality_parameters = sum(
            parameter.numel()
            for group in (head.quality_heads, head.one2one_quality_heads)
            for parameter in group.parameters()
        )
        assert target_parameters - madh_parameters == quality_parameters
        parameter_increase = 100.0 * quality_parameters / madh_parameters
        target_flops = get_flops(target, imgsz=640)
        gflops_increase = 100.0 * (target_flops - madh_flops) / madh_flops
        if variant == "control":
            assert target_flops == pytest.approx(madh_flops, rel=0, abs=1e-9)
        else:
            assert target_flops > madh_flops
        print(
            json.dumps(
                {
                    "scale": size,
                    "variant": variant,
                    "transferred_parameters": stock_parameters,
                    "pretrained_transfer_percent": 100.0 * stock_parameters / target_parameters,
                    "e8_parameters": madh_parameters,
                    "parameters": target_parameters,
                    "quality_parameters": quality_parameters,
                    "quality_parameter_increase_percent": parameter_increase,
                    "e8_gflops": madh_flops,
                    "gflops": target_flops,
                    "quality_gflops_increase_percent": gflops_increase,
                    "within_parameter_target": parameter_increase <= 2.0,
                    "within_gflops_target": gflops_increase <= 2.0,
                }
            )
        )
        del target
        gc.collect()

    del madh
    del stock
    gc.collect()


def test_dual_geometry_targets_and_positive_negative_losses() -> None:
    """Check plain IoU, NWD similarity, signed bounds, and both quality terms."""
    model = DualGeometryQualityDetectionModel(
        dgqm_yaml_path("n", "trainable"),
        verbose=False,
    )
    model.args = get_cfg()
    criterion = model.init_criterion().one2many
    assert isinstance(criterion, DualGeometryQualityDetectionLoss)
    pred_xywh = torch.tensor(
        (
            (0.50, 0.50, 0.20, 0.20),
            (0.10, 0.10, 0.05, 0.05),
            (0.90, 0.90, 0.10, 0.10),
        ),
        dtype=torch.float32,
    )
    target_xywh = torch.tensor(
        (
            (0.50, 0.50, 0.20, 0.20),
            (0.12, 0.12, 0.05, 0.05),
            (0.10, 0.10, 0.10, 0.10),
        ),
        dtype=torch.float32,
    )
    iou_quality = criterion.calculate_iou_quality(pred_xywh, target_xywh)
    nwd_quality = criterion.calculate_nwd_similarity(pred_xywh, target_xywh)
    quality_target = criterion.calculate_quality_target(pred_xywh, target_xywh)
    signed_target = 2.0 * quality_target - 1.0
    assert torch.all((0 <= iou_quality) & (iou_quality <= 1))
    assert torch.all((0 <= nwd_quality) & (nwd_quality <= 1))
    assert torch.all((0 <= quality_target) & (quality_target <= 1))
    assert torch.all((-1 <= signed_target) & (signed_target <= 1))
    assert not quality_target.requires_grad
    torch.testing.assert_close(
        quality_target,
        (0.75 * iou_quality + 0.25 * nwd_quality).clamp(0, 1),
        rtol=0,
        atol=0,
    )
    assert iou_quality[0].item() == pytest.approx(1.0, abs=3e-5)

    model.train()
    predictions = model(_synthetic_batch()["img"])
    assignment_calls = []
    handle = criterion.assigner.register_forward_hook(lambda *_args: assignment_calls.append(1))
    try:
        _, positive_losses, positive_items = criterion.get_assigned_targets_and_loss(
            predictions["one2many"],
            _synthetic_batch(),
        )
    finally:
        handle.remove()
    assert len(assignment_calls) == 1
    assert positive_losses.shape == (4,)
    assert positive_losses[3] > 0
    assert positive_items["quality_loss"] > 0

    empty_predictions = model(_synthetic_batch(with_target=False)["img"])["one2many"]
    quality_logit = torch.full_like(empty_predictions["quality"], 0.5, requires_grad=True)
    empty_predictions = {**empty_predictions, "quality": quality_logit}
    _, negative_losses, negative_items = criterion.get_assigned_targets_and_loss(
        empty_predictions,
        _synthetic_batch(with_target=False),
    )
    expected_negative = (
        criterion.quality_config.quality_gain
        * criterion.quality_config.negative_neutral_weight
        * torch.tanh(torch.tensor(0.5)).square()
    )
    torch.testing.assert_close(negative_losses[3], expected_negative, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(negative_items["quality_loss"], expected_negative, rtol=1e-6, atol=1e-7)
    negative_losses[3].backward()
    assert quality_logit.grad is not None
    assert torch.isfinite(quality_logit.grad).all()
    assert torch.count_nonzero(quality_logit.grad).item() > 0


@pytest.mark.parametrize(
    ("branch_name", "quality_name"),
    (("one2many", "quality_heads"), ("one2one", "one2one_quality_heads")),
)
def test_quality_loss_gradient_isolation_and_assignment_invariance(
    branch_name: str,
    quality_name: str,
) -> None:
    """Ensure each quality loss trains only its own quality heads and never changes assignment."""
    model = DualGeometryQualityDetectionModel(
        dgqm_yaml_path("n", "trainable"),
        verbose=False,
    )
    model.args = get_cfg()
    model.train()
    criterion_wrapper = model.init_criterion()
    branch_criterion = getattr(criterion_wrapper, branch_name)
    batch = _synthetic_batch()
    predictions = model(batch["img"])
    branch_predictions = predictions[branch_name]

    assigned_zero = branch_criterion.get_assigned_targets_and_loss(branch_predictions, batch)[0]
    shifted_predictions = {
        **branch_predictions,
        "quality": torch.full_like(branch_predictions["quality"], 100.0),
    }
    assigned_shifted = branch_criterion.get_assigned_targets_and_loss(shifted_predictions, batch)[0]
    for zero_value, shifted_value in zip(assigned_zero, assigned_shifted):
        torch.testing.assert_close(zero_value, shifted_value, rtol=0, atol=0)

    model.zero_grad(set_to_none=True)
    predictions = model(batch["img"])
    quality_loss = branch_criterion.get_assigned_targets_and_loss(predictions[branch_name], batch)[1][3]
    assert quality_loss > 0
    quality_loss.backward()
    head = model.model[-1]
    quality_heads = getattr(head, quality_name)
    quality_gradients = [
        parameter.grad
        for quality_head in quality_heads
        for parameter in quality_head.parameters()
        if parameter.grad is not None
    ]
    assert quality_gradients
    assert all(torch.isfinite(gradient).all() for gradient in quality_gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in quality_gradients)
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
            *model.model[:-1],
        )
    )
    other_quality_name = "one2one_quality_heads" if quality_name == "quality_heads" else "quality_heads"
    _assert_no_gradients(getattr(head, other_quality_name))


def test_control_matches_e8_predictions_and_loss() -> None:
    """Verify disabled quality is mathematically identical to E8-MADH."""
    torch.manual_seed(101)
    e8 = ContrastRingDetectionModel(
        madh_yaml_path("n", "trainable"),
        verbose=False,
        loss_config=E2_1B_CONFIG,
    )
    control = DualGeometryQualityDetectionModel(
        dgqm_yaml_path("n", "control"),
        verbose=False,
    )
    incompatible = control.load_state_dict(e8.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(
        key.startswith(("model.23.quality_heads.", "model.23.one2one_quality_heads."))
        for key in incompatible.missing_keys
    )
    e8.eval()
    control.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        e8_output, e8_raw = e8(image)
        control_output, control_raw = control(image)
    assert max(_raw_differences(e8_raw, control_raw).values()) <= 1e-6
    torch.testing.assert_close(e8_output, control_output, rtol=0, atol=1e-6)

    e8.args = get_cfg()
    control.args = get_cfg()
    e8.train()
    control.train()
    batch = _synthetic_batch()
    e8_predictions = e8(batch["img"])
    control_predictions = control(batch["img"])
    e8_loss = e8.init_criterion()(e8_predictions, batch)[0]
    control_loss = control.init_criterion()(control_predictions, batch)[0]
    assert control_loss.shape == (4,)
    torch.testing.assert_close(e8_loss, control_loss[:3], rtol=0, atol=1e-6)
    assert control_loss[3].item() == 0.0
    torch.testing.assert_close(e8_loss.sum(), control_loss.sum(), rtol=0, atol=1e-6)


@pytest.mark.skipif(not STANDARD_N_WEIGHTS.is_file(), reason="Local standard yolo26n.pt is required.")
def test_pretrained_fp32_equivalence_postprocess_transfer_and_standard_load(tmp_path: Path) -> None:
    """Check all initial equivalence surfaces and ordinary YOLO checkpoint loading."""
    stock = YOLO(STANDARD_N_WEIGHTS, verbose=False).model
    control = build_dgqm_yolo("n", "control", verbose=False)
    trainable = build_dgqm_yolo("n", "trainable", verbose=False)
    madh = ContrastRingDetectionModel(
        madh_yaml_path("n", "trainable"),
        verbose=False,
        loss_config=E2_1B_CONFIG,
    )
    transferred, _ = remap_yolo26_madh_state_dict(
        stock.state_dict(),
        madh.state_dict(),
        target_parameter_keys=dict(madh.named_parameters()),
    )
    madh.load_state_dict(transferred, strict=False)

    stock.eval()
    madh.eval()
    control.model.eval()
    trainable.model.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        stock_postprocessed, stock_raw = stock(image)
        madh_postprocessed, madh_raw = madh(image)
        control_postprocessed, control_raw = control.model(image)
        trainable_postprocessed, trainable_raw = trainable.model(image)
        stock_decoded = stock.model[-1]._inference(stock_raw["one2one"])
        madh_decoded = madh.model[-1]._inference(madh_raw["one2one"])
        control_decoded = control.model.model[-1]._inference(control_raw["one2one"])
        trainable_decoded = trainable.model.model[-1]._inference(trainable_raw["one2one"])

    for raw in (control_raw, trainable_raw):
        differences = _raw_differences(stock_raw, raw)
        assert max(differences.values()) <= 1e-6
    for decoded in (control_decoded, trainable_decoded):
        torch.testing.assert_close(stock_decoded, decoded, rtol=0, atol=1e-6)
    for postprocessed in (control_postprocessed, trainable_postprocessed):
        torch.testing.assert_close(stock_postprocessed, postprocessed, rtol=0, atol=1e-6)
    assert max(_raw_differences(madh_raw, control_raw).values()) <= 1e-6
    torch.testing.assert_close(madh_decoded, control_decoded, rtol=0, atol=1e-6)
    torch.testing.assert_close(madh_postprocessed, control_postprocessed, rtol=0, atol=1e-6)

    for model in (control, trainable):
        report = model.dgqm_transfer_report
        assert set(report.exact_keys) == set(stock.state_dict())
        assert report.new_adapter_keys
        assert report.new_quality_keys
        assert all(
            report.coverage(name).percentage == 100.0
            for name in ("backbone", "neck", "cv2", "cv3", "one2one_cv2", "one2one_cv3")
        )
        diagnostics = collect_dgqm_diagnostics(model)
        assert len(diagnostics) == 6
        assert all(item["quality_output_weight_norm"] == 0 for item in diagnostics)
        assert all(item["quality_output_bias"] == 0 for item in diagnostics)
        json.dumps(diagnostics)

    checkpoint = tmp_path / "synthetic-e10-dgqm.pt"
    trainable.save(checkpoint)
    loaded = YOLO(checkpoint, verbose=False)
    loaded_head = loaded.model.model[-1]
    assert isinstance(loaded_head, DualGeometryQualityMorphologyDetect)
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


def test_full_e2_1b_quality_loss_backward_amp_and_finite_outputs() -> None:
    """Verify all four components, both branches, quality gradients, and CPU AMP."""
    model = DualGeometryQualityDetectionModel(
        dgqm_yaml_path("n", "trainable"),
        verbose=False,
    )
    model.args = get_cfg()
    model.train()
    criterion = model.init_criterion()
    assert isinstance(criterion, E2ELoss)
    for branch in (criterion.one2many, criterion.one2one):
        assert isinstance(branch, DualGeometryQualityDetectionLoss)
        assert branch.reg_max == 1
        assert not branch.use_dfl
        assert branch.bbox_loss.dfl_loss is None
        assert isinstance(branch.bbox_loss, ResidualNWDBboxLoss)
        assert branch.bbox_loss.config.mode == "constant-010"
        assert branch.bbox_loss.config.nwd_scale == 0.10
        assert isinstance(branch.bce, ContrastRingBCEWithLogitsLoss)
        assert branch.contrast_ring_config == E2_1B_CONFIG
        assert branch.loss_names == ("box_loss", "cls_loss", "l1_loss", "quality_loss")
    model.criterion = criterion

    batch = _synthetic_batch()
    loss, loss_items = model.loss(batch)
    assert loss.shape == (4,)
    assert set(loss_items) == {"box_loss", "cls_loss", "l1_loss", "quality_loss"}
    assert torch.isfinite(loss).all()
    assert all(torch.isfinite(value) for value in loss_items.values())
    assert loss[3] > 0
    loss.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    head = model.model[-1]
    for group in (head.quality_heads, head.one2one_quality_heads):
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for quality_head in group
            for parameter in quality_head.parameters()
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


@pytest.mark.parametrize("size", tuple(SCALE_SETTINGS))
def test_n_to_x_latency_smoke(size: str) -> None:
    """Measure finite E8/E10 CPU latency without enforcing an architecture change."""
    e8 = ContrastRingDetectionModel(
        madh_yaml_path(size, "trainable"),
        verbose=False,
        loss_config=E2_1B_CONFIG,
    ).eval()
    e10 = DualGeometryQualityDetectionModel(
        dgqm_yaml_path(size, "trainable"),
        verbose=False,
    ).eval()
    image = torch.rand(1, 3, 640, 640)

    def measure(model: nn.Module) -> float:
        with torch.inference_mode():
            for _ in range(2):
                model(image)
            timings = []
            for _ in range(5):
                start = time.perf_counter()
                model(image)
                timings.append(1000.0 * (time.perf_counter() - start))
        return statistics.median(timings)

    e8_latency = measure(e8)
    e10_latency = measure(e10)
    assert e8_latency > 0 and e10_latency > 0
    assert torch.isfinite(torch.tensor((e8_latency, e10_latency))).all()
    print(
        json.dumps(
            {
                "scale": size,
                "device": "cpu",
                "imgsz": 640,
                "e8_latency_ms": e8_latency,
                "e10_latency_ms": e10_latency,
                "latency_increase_percent": 100.0 * (e10_latency - e8_latency) / e8_latency,
            }
        )
    )


def test_invalid_factory_inputs_and_run_names() -> None:
    """Reject unsupported E10 values and preserve exact run naming."""
    from research.scripts.train_e10 import _run_name

    for size in ("", "a", "N", "xx"):
        with pytest.raises(ValueError, match="Unsupported YOLO26 scale"):
            dgqm_yaml_path(size, "trainable")
    with pytest.raises(ValueError, match="Unsupported DGQM variant"):
        dgqm_yaml_path("n", "unknown")
    assert _run_name("n", "trainable", 640, 30, -1, 42) == "yolo26n-dgqm_img640_e30_bauto_seed42"
    assert _run_name("n", "trainable", 640, 30, -8, 42) == "yolo26n-dgqm_img640_e30_b-8_seed42"
    assert _run_name("n", "control", 640, 30, -8, 42) == "yolo26n-dgqm-control_img640_e30_b-8_seed42"
