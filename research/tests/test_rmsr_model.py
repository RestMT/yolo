# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Synthetic CPU checks for E6 zero-initialized Residual Multi-Scale Refinement."""

from __future__ import annotations

import copy
import gc
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import C3k2, C3k2RMSR, Concat, Detect
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.torch_utils import get_flops, get_num_params
from yolo_improved import (
    E2_1B_CONFIG,
    ContrastRingDetectionLoss,
    ContrastRingDetectionModel,
    ContrastRingYOLO,
    ResidualNWDBboxLoss,
    build_rmsr_yolo,
    rmsr_yaml_path,
)
from yolo_improved.contrast_ring_loss import ContrastRingBCEWithLogitsLoss, E2_LOCALIZATION_CONFIG


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STOCK_MODEL_DIRECTORY = REPOSITORY_ROOT / "ultralytics" / "cfg" / "models" / "26"
E6_MODEL_DIRECTORY = REPOSITORY_ROOT / "research" / "models"
STANDARD_N_WEIGHTS = REPOSITORY_ROOT / "yolo26n.pt"
SCALE_SETTINGS = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}


def _state_shapes(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    """Return state shapes relative to one module."""
    return {name: tuple(value.shape) for name, value in module.state_dict().items()}


def _output_signature(value):
    """Return a nested type-and-shape signature without retaining tensors."""
    if isinstance(value, torch.Tensor):
        return "tensor", tuple(value.shape)
    if isinstance(value, dict):
        return {key: _output_signature(item) for key, item in value.items()}
    if isinstance(value, list):
        return "list", [_output_signature(item) for item in value]
    if isinstance(value, tuple):
        return "tuple", tuple(_output_signature(item) for item in value)
    return type(value).__name__


def _output_tensors(value):
    """Yield all tensors from a nested model output."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _output_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _output_tensors(item)


def _has_nonzero_finite_gradient(module: torch.nn.Module) -> bool:
    """Return whether a module has at least one finite, nonzero parameter gradient."""
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


def test_control_exactly_matches_base_forward_and_gradients() -> None:
    """Ensure control bypasses refinement and exactly reproduces base outputs and gradients."""
    torch.manual_seed(61)
    control = C3k2RMSR(16, 24, n=2, c3k=True, trainable_refinement=False)
    reference = copy.deepcopy(control.base)
    branch_calls: list[str] = []
    handles = (
        control.local_branch.register_forward_hook(lambda *_: branch_calls.append("local")),
        control.context_branch.register_forward_hook(lambda *_: branch_calls.append("context")),
        control.fusion.register_forward_hook(lambda *_: branch_calls.append("fusion")),
    )
    x_control = torch.randn(2, 16, 12, 12, requires_grad=True)
    x_reference = x_control.detach().clone().requires_grad_()
    weights = torch.randn(2, 24, 12, 12)
    try:
        control_output = control(x_control)
        reference_output = reference(x_reference)
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(control_output, reference_output, rtol=0, atol=0)
    assert control_output.shape == (2, 24, 12, 12)
    assert not branch_calls
    assert not control.gate_raw.requires_grad
    assert "gate_raw" in dict(control.named_buffers())
    assert all(
        not parameter.requires_grad
        for branch in (control.local_branch, control.context_branch, control.fusion)
        for parameter in branch.parameters()
    )

    (control_output * weights).sum().backward()
    (reference_output * weights).sum().backward()
    torch.testing.assert_close(x_control.grad, x_reference.grad, rtol=0, atol=0)
    for (control_name, control_parameter), (reference_name, reference_parameter) in zip(
        control.base.named_parameters(), reference.named_parameters()
    ):
        assert control_name == reference_name
        torch.testing.assert_close(control_parameter.grad, reference_parameter.grad, rtol=0, atol=0)
    assert all(
        parameter.grad is None
        for branch in (control.local_branch, control.context_branch, control.fusion)
        for parameter in branch.parameters()
    )


def test_trainable_zero_gate_gradients_dilation_and_amp() -> None:
    """Check zero-start equivalence, gate/residual gradients, dilation, finite AMP, and validation."""
    for gate_max in (0, -1, float("inf"), float("nan"), True, "invalid"):
        with pytest.raises(ValueError, match="gate_max"):
            C3k2RMSR(8, 8, gate_max=gate_max)
    with pytest.raises(ValueError, match="trainable_refinement"):
        C3k2RMSR(8, 8, trainable_refinement=1)

    torch.manual_seed(67)
    rmsr = C3k2RMSR(16, 24, n=2, c3k=True, gate_max=1.0, trainable_refinement=True)
    reference = copy.deepcopy(rmsr.base)
    assert isinstance(rmsr.gate_raw, torch.nn.Parameter)
    assert rmsr.gate_raw.item() == 0
    assert rmsr.local_branch.conv.dilation == (1, 1)
    assert rmsr.context_branch.conv.dilation == (2, 2)
    assert rmsr.local_branch.conv.groups == 24
    assert rmsr.context_branch.conv.groups == 24

    x = torch.randn(2, 16, 12, 12)
    x_rmsr = x.clone().requires_grad_()
    x_reference = x.clone().requires_grad_()
    output = rmsr(x_rmsr)
    reference_output = reference(x_reference)
    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    assert torch.isfinite(output).all()

    weights = torch.randn_like(output)
    (output * weights).sum().backward()
    assert rmsr.gate_raw.grad is not None
    assert torch.isfinite(rmsr.gate_raw.grad)
    assert rmsr.gate_raw.grad.abs().item() > 0

    rmsr.zero_grad(set_to_none=True)
    rmsr.gate_raw.data.fill_(0.5)
    nonzero_output = rmsr(torch.randn(2, 16, 12, 12))
    nonzero_output.square().mean().backward()
    for branch in (rmsr.local_branch, rmsr.context_branch, rmsr.fusion):
        assert _has_nonzero_finite_gradient(branch)

    rmsr.zero_grad(set_to_none=True)
    rmsr.gate_raw.data.fill_(0.25)
    amp_input = torch.randn(2, 16, 12, 12, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_output = rmsr(amp_input)
        amp_loss = amp_output.float().square().mean()
    amp_loss.backward()
    assert torch.isfinite(amp_loss)
    assert torch.isfinite(amp_output).all()
    assert amp_input.grad is not None and torch.isfinite(amp_input.grad).all()


@pytest.mark.parametrize("variant", ("control", "trainable"))
@pytest.mark.parametrize("size", tuple(SCALE_SETTINGS))
def test_unified_yaml_scaling_detect_and_model_cost(size: str, variant: str) -> None:
    """Build n-x variants and verify scaling, stock topology, Detect outputs, parameters, and FLOPs."""
    depth, width, max_channels = SCALE_SETTINGS[size]
    virtual_e6_yaml = rmsr_yaml_path(size, variant)
    unified_name = "yolo26-rmsr-control.yaml" if variant == "control" else "yolo26-rmsr.yaml"
    unified_e6_yaml = E6_MODEL_DIRECTORY / unified_name
    stock_virtual_yaml = STOCK_MODEL_DIRECTORY / f"yolo26{size}.yaml"

    assert unified_e6_yaml.is_file()
    assert not virtual_e6_yaml.exists()
    assert not list(E6_MODEL_DIRECTORY.glob(f"yolo26?-{'rmsr-control' if variant == 'control' else 'rmsr'}.yaml"))
    e6_definition = yaml_model_load(virtual_e6_yaml)
    stock_definition = yaml_model_load(stock_virtual_yaml)
    assert e6_definition["scale"] == size
    assert tuple(e6_definition["scales"][size]) == SCALE_SETTINGS[size]
    assert e6_definition["backbone"] == stock_definition["backbone"]
    for index, (e6_layer, stock_layer) in enumerate(zip(e6_definition["head"], stock_definition["head"])):
        if index != 5:
            assert e6_layer == stock_layer

    stock_model = DetectionModel(stock_virtual_yaml, verbose=False)
    stock_layer16_shapes = _state_shapes(stock_model.model[16])
    stock_parameters = get_num_params(stock_model)
    stock_flops = get_flops(stock_model, imgsz=640)
    del stock_model
    gc.collect()

    e6_model = ContrastRingDetectionModel(virtual_e6_yaml, verbose=False, loss_config=E2_1B_CONFIG)
    rmsr = e6_model.model[16]
    assert isinstance(rmsr, C3k2RMSR)
    assert _state_shapes(rmsr.base) == stock_layer16_shapes
    assert len(rmsr.base.m) == max(round(2 * depth), 1)
    assert rmsr.base.cv2.conv.out_channels == make_divisible(min(256, max_channels) * width, 8)
    assert rmsr.trainable_refinement == (variant == "trainable")
    assert rmsr.gate_max == 1.0
    assert rmsr.gate_raw.item() == 0
    assert isinstance(e6_model.model[15], Concat)
    assert isinstance(e6_model.model[23], Detect)
    assert e6_model.model[23].f == [16, 19, 22]
    assert e6_model.model[23].nl == 3
    assert e6_model.stride.tolist() == [8.0, 16.0, 32.0]
    assert len(e6_model.model) == 24

    e6_parameters = get_num_params(e6_model)
    e6_flops = get_flops(e6_model, imgsz=640)
    assert e6_parameters > stock_parameters
    if variant == "control":
        assert e6_flops == pytest.approx(stock_flops, rel=0, abs=1e-9)
        assert all(
            not parameter.requires_grad
            for branch in (rmsr.local_branch, rmsr.context_branch, rmsr.fusion)
            for parameter in branch.parameters()
        )
    else:
        assert e6_flops > stock_flops
        assert all(
            parameter.requires_grad
            for branch in (rmsr.local_branch, rmsr.context_branch, rmsr.fusion)
            for parameter in branch.parameters()
        )
    assert e6_parameters / stock_parameters < 1.30
    assert e6_flops / stock_flops < 1.30
    del e6_model
    gc.collect()


def test_full_model_output_amp_and_fixed_e2_1b() -> None:
    """Verify finite full-model AMP output, stock output format, and unchanged E2.1b criterion."""
    e6 = ContrastRingDetectionModel(rmsr_yaml_path("n", "trainable"), verbose=False, loss_config=E2_1B_CONFIG)
    stock = DetectionModel(STOCK_MODEL_DIRECTORY / "yolo26n.yaml", verbose=False)
    e6.eval()
    stock.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        e6_output = e6(image)
    with torch.no_grad():
        stock_output = stock(image)

    assert _output_signature(e6_output) == _output_signature(stock_output)
    assert all(torch.isfinite(tensor).all() for tensor in _output_tensors(e6_output))
    assert e6.contrast_ring_loss_config == E2_1B_CONFIG
    assert E2_LOCALIZATION_CONFIG.mode == "constant-010"
    assert E2_LOCALIZATION_CONFIG.nwd_scale == 0.10
    e6.args = get_cfg()
    criterion = e6.init_criterion()
    assert isinstance(criterion, E2ELoss)
    for branch in (criterion.one2many, criterion.one2one):
        assert isinstance(branch, ContrastRingDetectionLoss)
        assert isinstance(branch.bbox_loss, ResidualNWDBboxLoss)
        assert branch.bbox_loss.config == E2_LOCALIZATION_CONFIG
        assert isinstance(branch.bce, ContrastRingBCEWithLogitsLoss)
        assert branch.contrast_ring_config == E2_1B_CONFIG

    del criterion
    del stock
    del e6
    gc.collect()


@pytest.mark.skipif(not STANDARD_N_WEIGHTS.is_file(), reason="Local standard yolo26n.pt is required.")
@pytest.mark.parametrize("variant", ("control", "trainable"))
def test_pretrained_transfer_and_standard_checkpoint_loading(variant: str, tmp_path: Path) -> None:
    """Check complete layer-16 remap, untouched RMSR state, zero gate, and ordinary YOLO loading."""
    e6 = build_rmsr_yolo("n", variant, verbose=False)
    source = YOLO(STANDARD_N_WEIGHTS, verbose=False).model
    source_state = source.state_dict()
    target_state = e6.model.state_dict()
    report = e6.rmsr_transfer_report

    assert not report.skipped_shape_keys
    source_layer16 = {key for key in source_state if key.startswith("model.16.")}
    expected_layer16 = {key.replace("model.16.", "model.16.base.", 1) for key in source_layer16}
    assert expected_layer16 == set(report.remapped_layer16_keys)
    for source_key, source_tensor in source_state.items():
        target_key = (
            source_key.replace("model.16.", "model.16.base.", 1)
            if source_key.startswith("model.16.")
            else source_key
        )
        torch.testing.assert_close(target_state[target_key], source_tensor, rtol=0, atol=0)

    new_prefixes = (
        "model.16.local_branch.",
        "model.16.context_branch.",
        "model.16.fusion.",
        "model.16.gate_raw",
    )
    transferred = set(report.exact_keys) | set(report.remapped_layer16_keys)
    assert report.missing_target_keys
    assert all(key.startswith(new_prefixes) for key in report.missing_target_keys)
    assert not any(key.startswith(new_prefixes) for key in transferred)
    assert torch.count_nonzero(target_state["model.16.gate_raw"]).item() == 0
    assert torch.count_nonzero(target_state["model.16.local_branch.conv.weight"]).item() > 0
    assert report.transferred_parameter_elements == sum(parameter.numel() for parameter in source.parameters())
    assert report.transfer_percentage > 99
    assert e6.ckpt["model"] is e6.model

    checkpoint = tmp_path / f"synthetic-e6-{variant}.pt"
    e6.save(checkpoint)
    loaded = YOLO(checkpoint, verbose=False)
    loaded_rmsr = loaded.model.model[16]
    assert isinstance(loaded_rmsr, C3k2RMSR)
    assert loaded_rmsr.trainable_refinement == (variant == "trainable")
    assert loaded_rmsr.gate_raw.item() == 0
    assert loaded.model.model[23].f == [16, 19, 22]
    assert loaded.model.stride.tolist() == [8.0, 16.0, 32.0]

    if variant == "trainable":
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

    del loaded
    del source
    del e6
    gc.collect()


def test_invalid_factory_inputs_and_batch_tag() -> None:
    """Reject unsupported factory values and preserve the requested negative batch tag."""
    from research.scripts.train_e6 import _run_name

    for size in ("", "a", "N", "xx"):
        with pytest.raises(ValueError, match="Unsupported YOLO26 scale"):
            rmsr_yaml_path(size, "trainable")
    with pytest.raises(ValueError, match="Unsupported RMSR variant"):
        rmsr_yaml_path("n", "unknown")
    assert _run_name("n", "trainable", 640, 100, -1, 42).endswith("_bauto_seed42")
    assert _run_name("n", "trainable", 640, 100, -8, 42).endswith("_b-8_seed42")
    assert _run_name("n", "control", 640, 100, -8, 42).startswith("yolo26n-rmsr-control_")
