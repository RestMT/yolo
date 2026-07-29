# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Synthetic CPU checks for E8 Morphology-Adaptive Decoupled Head."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import (
    Detect,
    MorphologyAdaptiveAdapter,
    MorphologyAdaptiveDetect,
)
from ultralytics.nn.modules.conv import autopad
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import get_flops, get_num_params
from yolo_improved.contrast_ring_loss import ContrastRingBCEWithLogitsLoss, ContrastRingDetectionLoss
from yolo_improved.contrast_ring_model import ContrastRingDetectionModel
from yolo_improved.madh_model import (
    E2_1B_CONFIG,
    MADH_VARIANTS,
    build_madh_yolo,
    collect_madh_gate_values,
    madh_yaml_path,
    remap_yolo26_madh_state_dict,
)
from yolo_improved.residual_nwd_loss import ResidualNWDBboxLoss


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STOCK_MODEL_DIRECTORY = REPOSITORY_ROOT / "ultralytics" / "cfg" / "models" / "26"
E8_MODEL_DIRECTORY = REPOSITORY_ROOT / "research" / "models"
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


def _assert_raw_predictions_equal(
    stock: dict[str, dict[str, torch.Tensor]],
    target: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Compare all requested raw end-to-end prediction tensors."""
    for branch in ("one2many", "one2one"):
        for component in ("boxes", "scores"):
            difference = (stock[branch][component] - target[branch][component]).abs().max().item()
            assert difference <= 1e-6
            torch.testing.assert_close(stock[branch][component], target[branch][component], rtol=0, atol=1e-6)


def _synthetic_batch() -> dict[str, torch.Tensor]:
    """Return one small normalized detection batch."""
    return {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }


def test_adapter_router_identity_gradients_and_tuple_kernels() -> None:
    """Check spatial routing, asymmetric kernels, zero identity, and staged gradients."""
    assert autopad((1, 7)) == [0, 3]
    assert autopad((7, 1)) == [3, 0]
    torch.manual_seed(81)
    adapter = MorphologyAdaptiveAdapter(32)
    assert adapter.hidden == 16
    assert adapter.horizontal_branch.conv.kernel_size == (1, 7)
    assert adapter.vertical_branch.conv.kernel_size == (7, 1)
    assert adapter.horizontal_branch.conv.padding == (0, 3)
    assert adapter.vertical_branch.conv.padding == (3, 0)
    assert adapter.context_branch.conv.dilation == (2, 2)
    for branch in (
        adapter.local_branch,
        adapter.horizontal_branch,
        adapter.vertical_branch,
        adapter.context_branch,
    ):
        assert branch.conv.groups == adapter.hidden

    adapter.eval()
    x = torch.randn(2, 32, 11, 13)
    reduced = adapter.reduce(x)
    branches = (
        adapter.local_branch(reduced),
        adapter.horizontal_branch(reduced),
        adapter.vertical_branch(reduced),
        adapter.context_branch(reduced),
    )
    assert all(branch.shape == (2, 16, 11, 13) for branch in branches)
    routing_weights = adapter.router(reduced).softmax(dim=1)
    torch.testing.assert_close(routing_weights, torch.full_like(routing_weights, 0.25), rtol=0, atol=0)
    torch.testing.assert_close(routing_weights.sum(dim=1), torch.ones_like(routing_weights[:, 0]), rtol=0, atol=0)
    assert torch.count_nonzero(adapter.project.conv.weight).item() > 0
    torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)

    adapter.train()
    adapter.zero_grad(set_to_none=True)
    x_zero = torch.randn(2, 32, 11, 13, requires_grad=True)
    output_zero = adapter(x_zero)
    weights = torch.randn_like(output_zero)
    (output_zero * weights).sum().backward()
    assert adapter.gate_raw.grad is not None
    assert torch.isfinite(adapter.gate_raw.grad)
    assert adapter.gate_raw.grad.abs().item() > 0
    for name, parameter in adapter.named_parameters():
        if name != "gate_raw" and parameter.grad is not None:
            assert torch.count_nonzero(parameter.grad).item() == 0

    adapter.zero_grad(set_to_none=True)
    adapter.gate_raw.data.fill_(0.25)
    output_nonzero = adapter(torch.randn(2, 32, 11, 13))
    (output_nonzero * torch.randn_like(output_nonzero)).sum().backward()
    for module in (
        adapter.reduce,
        adapter.local_branch,
        adapter.horizontal_branch,
        adapter.vertical_branch,
        adapter.context_branch,
        adapter.router,
        adapter.project,
    ):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)


def test_control_bypasses_every_adapter_operation_and_preserves_input_gradient() -> None:
    """Ensure control returns the original tensor without executing morphology modules."""
    control = MorphologyAdaptiveAdapter(32, trainable_adapter=False)
    calls: list[str] = []
    modules = {
        "reduce": control.reduce,
        "local": control.local_branch,
        "horizontal": control.horizontal_branch,
        "vertical": control.vertical_branch,
        "context": control.context_branch,
        "router": control.router,
        "project": control.project,
    }
    handles = [
        module.register_forward_hook(lambda _, __, ___, name=name: calls.append(name))
        for name, module in modules.items()
    ]
    x = torch.randn(2, 32, 9, 11, requires_grad=True)
    weights = torch.randn_like(x)
    try:
        output = control(x)
    finally:
        for handle in handles:
            handle.remove()
    assert output is x
    assert not calls
    assert all(not parameter.requires_grad for parameter in control.parameters())
    (output * weights).sum().backward()
    torch.testing.assert_close(x.grad, weights, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in control.parameters())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"channels": 0}, "channels"),
        ({"channels": True}, "channels"),
        ({"channels": 16, "hidden_ratio": 0}, "hidden_ratio"),
        ({"channels": 16, "hidden_ratio": float("nan")}, "hidden_ratio"),
        ({"channels": 16, "min_hidden": 0}, "min_hidden"),
        ({"channels": 16, "min_hidden": 24, "max_hidden": 16}, "max_hidden"),
        ({"channels": 16, "strip_kernel": 2}, "strip_kernel"),
        ({"channels": 16, "strip_kernel": 4}, "strip_kernel"),
        ({"channels": 16, "context_dilation": 0}, "context_dilation"),
        ({"channels": 16, "gate_max": float("inf")}, "gate_max"),
        ({"channels": 16, "trainable_adapter": 1}, "trainable_adapter"),
    ),
)
def test_adapter_validation(kwargs: dict, message: str) -> None:
    """Reject invalid adapter geometry and nonfinite coefficients."""
    with pytest.raises(ValueError, match=message):
        MorphologyAdaptiveAdapter(**kwargs)


def test_unified_yamls_differ_from_stock_only_by_detect() -> None:
    """Verify both unified E8 definitions and all ten virtual scale names."""
    for variant in MADH_VARIANTS:
        suffix = "madh-control" if variant == "control" else "madh"
        unified = E8_MODEL_DIRECTORY / f"yolo26-{suffix}.yaml"
        assert unified.is_file()
        assert not list(E8_MODEL_DIRECTORY.glob(f"yolo26?-{suffix}.yaml"))
        for size, scale_settings in SCALE_SETTINGS.items():
            virtual = madh_yaml_path(size, variant)
            assert not virtual.exists()
            target = yaml_model_load(virtual)
            stock = yaml_model_load(STOCK_MODEL_DIRECTORY / f"yolo26{size}.yaml")
            assert target["scale"] == size
            assert tuple(target["scales"][size]) == scale_settings
            for key in ("nc", "end2end", "reg_max", "scales", "backbone", "scale"):
                assert target[key] == stock[key]
            assert target["head"][:-1] == stock["head"][:-1]
            assert target["head"][-1][:2] == stock["head"][-1][:2]
            assert target["head"][-1][2] == "MorphologyAdaptiveDetect"
            assert target["head"][-1][3][-1] == (variant == "trainable")


@pytest.mark.parametrize("size", tuple(SCALE_SETTINGS))
def test_n_to_x_structure_transfer_and_model_cost(size: str) -> None:
    """Build stock/control/trainable and verify topology, transfer, parameters, and FLOPs."""
    stock = DetectionModel(STOCK_MODEL_DIRECTORY / f"yolo26{size}.yaml", verbose=False)
    stock_head = stock.model[-1]
    stock_state = stock.state_dict()
    stock_parameters = get_num_params(stock)
    stock_flops = get_flops(stock, imgsz=640)
    stock_tower_shapes = {
        name: _state_shapes(getattr(stock_head, name))
        for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3")
    }

    for variant in MADH_VARIANTS:
        target = ContrastRingDetectionModel(
            madh_yaml_path(size, variant),
            verbose=False,
            loss_config=E2_1B_CONFIG,
        )
        head = target.model[-1]
        assert isinstance(head, MorphologyAdaptiveDetect)
        assert head.f == [16, 19, 22]
        assert head.nl == 3
        assert head.stride.tolist() == [8.0, 16.0, 32.0]
        assert head.reg_max == 1
        assert isinstance(head.dfl, nn.Identity)
        assert target.yaml["scale"] == size
        assert len(target.model) == 24
        for name, shapes in stock_tower_shapes.items():
            assert _state_shapes(getattr(head, name)) == shapes

        adapter_groups = (
            head.box_adapters,
            head.cls_adapters,
            head.one2one_box_adapters,
            head.one2one_cls_adapters,
        )
        assert all(len(group) == 3 for group in adapter_groups)
        assert all(
            adapter.trainable_adapter == (variant == "trainable")
            for group in adapter_groups
            for adapter in group
        )
        for first_group, second_group in (
            (head.box_adapters, head.cls_adapters),
            (head.box_adapters, head.one2one_box_adapters),
            (head.cls_adapters, head.one2one_cls_adapters),
        ):
            for first, second in zip(first_group, second_group):
                assert first is not second
                assert all(
                    first_parameter.data_ptr() != second_parameter.data_ptr()
                    for first_parameter, second_parameter in zip(first.parameters(), second.parameters())
                )

        transferred, report = remap_yolo26_madh_state_dict(
            stock_state,
            target.state_dict(),
            target_parameter_keys=dict(target.named_parameters()),
        )
        assert not report.skipped_missing_keys
        assert not report.skipped_shape_keys
        assert report.new_adapter_keys
        assert all(
            key.startswith(
                (
                    "model.23.box_adapters.",
                    "model.23.cls_adapters.",
                    "model.23.one2one_box_adapters.",
                    "model.23.one2one_cls_adapters.",
                )
            )
            for key in report.new_adapter_keys
        )
        assert set(transferred) == set(stock_state)
        for coverage_name in ("backbone", "neck", "cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            assert report.coverage(coverage_name).percentage == 100
        assert report.coverage("adapters").transferred_tensors == 0

        target_parameters = get_num_params(target)
        target_flops = get_flops(target, imgsz=640)
        assert target_parameters > stock_parameters
        if variant == "control":
            assert target_flops == pytest.approx(stock_flops, rel=0, abs=1e-9)
        else:
            assert target_flops > stock_flops
        if size == "n":
            assert target_parameters / stock_parameters <= 1.10
            assert target_flops / stock_flops <= 1.10
        del target
        gc.collect()

    del stock
    gc.collect()


@pytest.mark.skipif(not STANDARD_N_WEIGHTS.is_file(), reason="Local standard yolo26n.pt is required.")
def test_pretrained_fp32_equivalence_and_standard_checkpoint_loading(tmp_path: Path) -> None:
    """Check exact transfer, zero-gate equivalence, and ordinary YOLO checkpoint loading."""
    stock = YOLO(STANDARD_N_WEIGHTS, verbose=False).model
    control = build_madh_yolo("n", "control", verbose=False)
    trainable = build_madh_yolo("n", "trainable", verbose=False)
    stock.eval()
    control.model.eval()
    trainable.model.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        stock_decoded, stock_raw = stock(image)
        control_decoded, control_raw = control.model(image)
        trainable_decoded, trainable_raw = trainable.model(image)

    _assert_raw_predictions_equal(stock_raw, control_raw)
    _assert_raw_predictions_equal(stock_raw, trainable_raw)
    for decoded in (control_decoded, trainable_decoded):
        difference = (stock_decoded - decoded).abs().max().item()
        assert difference <= 1e-6
        torch.testing.assert_close(stock_decoded, decoded, rtol=0, atol=1e-6)

    source_state = stock.state_dict()
    for model in (control, trainable):
        report = model.madh_transfer_report
        target_state = model.model.state_dict()
        assert set(report.exact_keys) == set(source_state)
        assert not report.skipped_shape_keys
        assert report.new_adapter_keys
        for key, source_tensor in source_state.items():
            torch.testing.assert_close(target_state[key], source_tensor, rtol=0, atol=0)
        gates = collect_madh_gate_values(model)
        assert len(gates) == 12
        assert all(gate["gate_raw"] == 0.0 and gate["alpha"] == 0.0 for gate in gates)
        json.dumps(gates)

    checkpoint = tmp_path / "synthetic-e8-madh.pt"
    trainable.save(checkpoint)
    loaded = YOLO(checkpoint, verbose=False)
    loaded_head = loaded.model.model[-1]
    assert isinstance(loaded_head, MorphologyAdaptiveDetect)
    assert loaded_head.reg_max == 1
    assert loaded_head.stride.tolist() == [8.0, 16.0, 32.0]
    assert len(collect_madh_gate_values(loaded.model)) == 12
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
    del trainable
    del control
    del stock
    gc.collect()


def test_e2_1b_synthetic_backward_amp_and_finite_outputs() -> None:
    """Verify unchanged E2.1b, finite full-model backward, and CPU AMP."""
    model = ContrastRingDetectionModel(
        madh_yaml_path("n", "trainable"),
        verbose=False,
        loss_config=E2_1B_CONFIG,
    )
    model.args = get_cfg()
    model.train()
    criterion = model.init_criterion()
    assert isinstance(criterion, E2ELoss)
    for branch in (criterion.one2many, criterion.one2one):
        assert isinstance(branch, ContrastRingDetectionLoss)
        assert branch.reg_max == 1
        assert not branch.use_dfl
        assert branch.bbox_loss.dfl_loss is None
        assert isinstance(branch.bbox_loss, ResidualNWDBboxLoss)
        assert branch.bbox_loss.config.mode == "constant-010"
        assert branch.bbox_loss.config.nwd_scale == 0.10
        assert isinstance(branch.bce, ContrastRingBCEWithLogitsLoss)
        assert branch.contrast_ring_config == E2_1B_CONFIG
    model.criterion = criterion

    batch = _synthetic_batch()
    loss, loss_items = model.loss(batch)
    assert loss.shape == (3,)
    assert set(loss_items) == {"box_loss", "cls_loss", "l1_loss"}
    assert torch.isfinite(loss).all()
    assert all(torch.isfinite(value) for value in loss_items.values())
    loss.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    head = model.model[-1]
    gate_gradients = [
        adapter.gate_raw.grad
        for attribute in (
            "box_adapters",
            "cls_adapters",
            "one2one_box_adapters",
            "one2one_cls_adapters",
        )
        for adapter in getattr(head, attribute)
    ]
    assert all(gradient is not None and torch.isfinite(gradient) for gradient in gate_gradients)

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
    del model
    gc.collect()


def test_invalid_factory_inputs_and_run_names() -> None:
    """Reject unsupported E8 values and preserve requested negative batch tags."""
    from research.scripts.train_e8 import _run_name

    for size in ("", "a", "N", "xx"):
        with pytest.raises(ValueError, match="Unsupported YOLO26 scale"):
            madh_yaml_path(size, "trainable")
    with pytest.raises(ValueError, match="Unsupported MADH variant"):
        madh_yaml_path("n", "unknown")
    assert _run_name("n", "trainable", 640, 30, -1, 42) == "yolo26n-madh_img640_e30_bauto_seed42"
    assert _run_name("n", "trainable", 640, 30, -8, 42) == "yolo26n-madh_img640_e30_b-8_seed42"
    assert _run_name("n", "control", 640, 30, -8, 42) == "yolo26n-madh-control_img640_e30_b-8_seed42"
