# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Synthetic CPU checks for E9 Geometry-Preserving Spatial Morphology."""

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
    GeometryPreservingSpatialMorphologyDetect,
    SpatialMorphologyClassificationAdapter,
)
from ultralytics.nn.modules.conv import autopad
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import get_flops, get_num_params
from yolo_improved.contrast_ring_loss import ContrastRingBCEWithLogitsLoss, ContrastRingDetectionLoss
from yolo_improved.contrast_ring_model import ContrastRingDetectionModel
from yolo_improved.gpsm_model import (
    E2_1B_CONFIG,
    GPSM_VARIANTS,
    build_gpsm_yolo,
    collect_gpsm_parameters,
    gpsm_yaml_path,
    remap_yolo26_gpsm_state_dict,
)
from yolo_improved.madh_model import madh_yaml_path
from yolo_improved.residual_nwd_loss import ResidualNWDBboxLoss


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STOCK_MODEL_DIRECTORY = REPOSITORY_ROOT / "ultralytics" / "cfg" / "models" / "26"
E9_MODEL_DIRECTORY = REPOSITORY_ROOT / "research" / "models"
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


def _nonzero_finite_gradients(module: nn.Module) -> list[torch.Tensor]:
    """Return finite nonzero gradients owned by a module."""
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
    ]
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    return gradients


def test_adapter_router_spatial_gate_gradients_and_tuple_kernels() -> None:
    """Check experts, spatial routing, zero identity, and staged adapter gradients."""
    assert autopad((1, 7)) == [0, 3]
    assert autopad((7, 1)) == [3, 0]
    torch.manual_seed(91)
    adapter = SpatialMorphologyClassificationAdapter(32)
    assert adapter.hidden == 16
    assert not hasattr(adapter, "gate_raw")
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
    stacked = torch.stack(branches, dim=1)
    assert stacked.shape == (2, 4, 16, 11, 13)
    routing_weights = adapter.router(reduced).softmax(dim=1).unsqueeze(2)
    assert routing_weights.shape == (2, 4, 1, 11, 13)
    torch.testing.assert_close(routing_weights, torch.full_like(routing_weights, 0.25), rtol=0, atol=0)
    torch.testing.assert_close(
        routing_weights.sum(dim=1),
        torch.ones_like(routing_weights[:, 0]),
        rtol=0,
        atol=0,
    )
    gate = adapter.gate_max * torch.tanh(adapter.spatial_gate(reduced))
    assert gate.shape == (2, 1, 11, 13)
    assert torch.count_nonzero(gate).item() == 0
    assert torch.count_nonzero(adapter.project.conv.weight).item() > 0
    torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)

    adapter.train()
    adapter.zero_grad(set_to_none=True)
    x_zero = torch.randn(2, 32, 11, 13, requires_grad=True)
    output_zero = adapter(x_zero)
    (output_zero * torch.randn_like(output_zero)).sum().backward()
    assert _nonzero_finite_gradients(adapter.spatial_gate)
    for name, parameter in adapter.named_parameters():
        if not name.startswith("spatial_gate.") and parameter.grad is not None:
            assert torch.count_nonzero(parameter.grad).item() == 0

    adapter.zero_grad(set_to_none=True)
    adapter.spatial_gate.bias.data.fill_(0.25)
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
        adapter.spatial_gate,
    ):
        assert _nonzero_finite_gradients(module)


def test_control_bypasses_every_adapter_operation_and_preserves_input_gradient() -> None:
    """Ensure control returns the original tensor without executing morphology modules."""
    control = SpatialMorphologyClassificationAdapter(32, trainable_adapter=False)
    calls: list[str] = []
    modules = {
        "reduce": control.reduce,
        "local": control.local_branch,
        "horizontal": control.horizontal_branch,
        "vertical": control.vertical_branch,
        "context": control.context_branch,
        "router": control.router,
        "project": control.project,
        "spatial_gate": control.spatial_gate,
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
        SpatialMorphologyClassificationAdapter(**kwargs)


def test_unified_yamls_differ_from_stock_only_by_detect() -> None:
    """Verify both unified E9 definitions and all ten virtual scale names."""
    for variant in GPSM_VARIANTS:
        suffix = "gpsm-control" if variant == "control" else "gpsm"
        unified = E9_MODEL_DIRECTORY / f"yolo26-{suffix}.yaml"
        assert unified.is_file()
        assert not list(E9_MODEL_DIRECTORY.glob(f"yolo26?-{suffix}.yaml"))
        for size, scale_settings in SCALE_SETTINGS.items():
            virtual = gpsm_yaml_path(size, variant)
            assert not virtual.exists()
            target = yaml_model_load(virtual)
            stock = yaml_model_load(STOCK_MODEL_DIRECTORY / f"yolo26{size}.yaml")
            assert target["scale"] == size
            assert tuple(target["scales"][size]) == scale_settings
            for key in ("nc", "end2end", "reg_max", "scales", "backbone", "scale"):
                assert target[key] == stock[key]
            assert target["head"][:-1] == stock["head"][:-1]
            assert target["head"][-1][:2] == stock["head"][-1][:2]
            assert target["head"][-1][2] == "GeometryPreservingSpatialMorphologyDetect"
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
    madh = ContrastRingDetectionModel(
        madh_yaml_path(size, "trainable"),
        verbose=False,
        loss_config=E2_1B_CONFIG,
    )
    madh_parameters = get_num_params(madh)
    madh_flops = get_flops(madh, imgsz=640)

    for variant in GPSM_VARIANTS:
        target = ContrastRingDetectionModel(
            gpsm_yaml_path(size, variant),
            verbose=False,
            loss_config=E2_1B_CONFIG,
        )
        head = target.model[-1]
        assert isinstance(head, GeometryPreservingSpatialMorphologyDetect)
        assert head.f == [16, 19, 22]
        assert head.nl == 3
        assert head.stride.tolist() == [8.0, 16.0, 32.0]
        assert head.reg_max == 1
        assert isinstance(head.dfl, nn.Identity)
        assert target.yaml["scale"] == size
        assert len(target.model) == 24
        assert len(head.cls_adapters) == 3
        assert not hasattr(head, "box_adapters")
        assert not hasattr(head, "one2one_box_adapters")
        assert not hasattr(head, "one2one_cls_adapters")
        for name, shapes in stock_tower_shapes.items():
            assert _state_shapes(getattr(head, name)) == shapes

        transferred, report = remap_yolo26_gpsm_state_dict(
            stock_state,
            target.state_dict(),
            target_parameter_keys=dict(target.named_parameters()),
        )
        assert not report.skipped_missing_keys
        assert not report.skipped_shape_keys
        assert report.new_gpsm_keys
        assert all(key.startswith("model.23.cls_adapters.") for key in report.new_gpsm_keys)
        assert set(transferred) == set(stock_state)
        for coverage_name in ("backbone", "neck", "cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            assert report.coverage(coverage_name).percentage == 100
        assert report.coverage("cls_adapters").transferred_tensors == 0

        target_parameters = get_num_params(target)
        target_flops = get_flops(target, imgsz=640)
        assert stock_parameters < target_parameters < madh_parameters
        if variant == "control":
            assert target_flops == pytest.approx(stock_flops, rel=0, abs=1e-9)
        else:
            assert stock_flops < target_flops < madh_flops
        if size == "n":
            assert target_parameters / stock_parameters <= 1.05
            assert target_flops / stock_flops <= 1.05
        del target
        gc.collect()

    del madh
    del stock
    gc.collect()


def test_shared_adapter_features_and_detached_one_to_one_path() -> None:
    """Verify adapters run once, box inputs stay stock, and one-to-one sees detached adapted semantics."""
    head = GeometryPreservingSpatialMorphologyDetect(nc=3, end2end=True, ch=(16, 24, 32))
    for adapter in head.cls_adapters:
        adapter.spatial_gate.bias.data.fill_(0.25)
    head.train()
    inputs = [
        torch.randn(2, 16, 8, 8, requires_grad=True),
        torch.randn(2, 24, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    ]
    captured: dict[str, list[torch.Tensor]] = {
        "adapter": [],
        "box": [],
        "cls": [],
        "one2one_box": [],
        "one2one_cls": [],
    }
    handles = []
    for index in range(3):
        handles.extend(
            (
                head.cls_adapters[index].register_forward_hook(
                    lambda _, __, output: captured["adapter"].append(output)
                ),
                head.cv2[index].register_forward_pre_hook(
                    lambda _, args: captured["box"].append(args[0])
                ),
                head.cv3[index].register_forward_pre_hook(
                    lambda _, args: captured["cls"].append(args[0])
                ),
                head.one2one_cv2[index].register_forward_pre_hook(
                    lambda _, args: captured["one2one_box"].append(args[0])
                ),
                head.one2one_cv3[index].register_forward_pre_hook(
                    lambda _, args: captured["one2one_cls"].append(args[0])
                ),
            )
        )
    try:
        predictions = head(inputs)
    finally:
        for handle in handles:
            handle.remove()

    assert len(captured["adapter"]) == 3
    for index, source in enumerate(inputs):
        adapted = captured["adapter"][index]
        assert captured["box"][index] is source
        assert captured["box"][index].requires_grad
        assert captured["cls"][index] is adapted
        assert captured["cls"][index].requires_grad
        assert captured["one2one_box"][index].data_ptr() == source.data_ptr()
        assert not captured["one2one_box"][index].requires_grad
        assert captured["one2one_cls"][index].data_ptr() == adapted.data_ptr()
        assert not captured["one2one_cls"][index].requires_grad
        assert not torch.equal(adapted, source)
        assert predictions["one2many"]["feats"][index] is source
        assert predictions["one2one"]["feats"][index].data_ptr() == source.data_ptr()
        assert not predictions["one2one"]["feats"][index].requires_grad


def test_one_to_one_classification_does_not_train_adapter_or_backbone() -> None:
    """Verify detached one-to-one classification gradients stop before shared adapters."""
    head = GeometryPreservingSpatialMorphologyDetect(nc=3, end2end=True, ch=(16, 24, 32))
    head.train()
    inputs = [
        torch.randn(2, 16, 8, 8, requires_grad=True),
        torch.randn(2, 24, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    ]
    scores = head(inputs)["one2one"]["scores"]
    (scores * torch.randn_like(scores)).sum().backward()
    assert all(feature.grad is None for feature in inputs)
    assert all(parameter.grad is None for adapter in head.cls_adapters for parameter in adapter.parameters())
    assert _nonzero_finite_gradients(head.one2one_cv3)


def test_one_to_many_classification_trains_spatial_gate_and_backbone() -> None:
    """Verify dense classification supervision reaches the shared adapter and input features."""
    head = GeometryPreservingSpatialMorphologyDetect(nc=3, end2end=True, ch=(16, 24, 32))
    head.train()
    inputs = [
        torch.randn(2, 16, 8, 8, requires_grad=True),
        torch.randn(2, 24, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    ]
    scores = head(inputs)["one2many"]["scores"]
    (scores * torch.randn_like(scores)).sum().backward()
    assert all(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in inputs)
    assert all(_nonzero_finite_gradients(adapter.spatial_gate) for adapter in head.cls_adapters)
    assert _nonzero_finite_gradients(head.cv3)


def test_box_forward_and_gradients_match_stock_detect() -> None:
    """Verify both box paths use the unchanged towers and original or detached features."""
    torch.manual_seed(92)
    stock = Detect(nc=3, reg_max=1, end2end=True, ch=(16, 24, 32))
    target = GeometryPreservingSpatialMorphologyDetect(nc=3, end2end=True, ch=(16, 24, 32))
    incompatible = target.load_state_dict(stock.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith("cls_adapters.") for key in incompatible.missing_keys)
    stock.train()
    target.train()
    stock_inputs = [
        torch.randn(2, 16, 8, 8, requires_grad=True),
        torch.randn(2, 24, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    ]
    target_inputs = [feature.detach().clone().requires_grad_(True) for feature in stock_inputs]
    stock_predictions = stock(stock_inputs)
    target_predictions = target(target_inputs)
    for branch in ("one2many", "one2one"):
        torch.testing.assert_close(
            stock_predictions[branch]["boxes"],
            target_predictions[branch]["boxes"],
            rtol=0,
            atol=0,
        )

    weights = torch.randn_like(stock_predictions["one2many"]["boxes"])
    (stock_predictions["one2many"]["boxes"] * weights).sum().backward()
    (target_predictions["one2many"]["boxes"] * weights).sum().backward()
    for stock_feature, target_feature in zip(stock_inputs, target_inputs):
        torch.testing.assert_close(stock_feature.grad, target_feature.grad, rtol=0, atol=0)
    for stock_parameter, target_parameter in zip(stock.cv2.parameters(), target.cv2.parameters()):
        torch.testing.assert_close(stock_parameter.grad, target_parameter.grad, rtol=0, atol=0)
    assert all(parameter.grad is None for adapter in target.cls_adapters for parameter in adapter.parameters())


@pytest.mark.skipif(not STANDARD_N_WEIGHTS.is_file(), reason="Local standard yolo26n.pt is required.")
def test_pretrained_fp32_equivalence_and_standard_checkpoint_loading(tmp_path: Path) -> None:
    """Check exact transfer, zero-gate equivalence, and ordinary YOLO checkpoint loading."""
    stock = YOLO(STANDARD_N_WEIGHTS, verbose=False).model
    control = build_gpsm_yolo("n", "control", verbose=False)
    trainable = build_gpsm_yolo("n", "trainable", verbose=False)
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
        report = model.gpsm_transfer_report
        target_state = model.model.state_dict()
        assert set(report.exact_keys) == set(source_state)
        assert not report.skipped_shape_keys
        assert report.new_gpsm_keys
        for key, source_tensor in source_state.items():
            torch.testing.assert_close(target_state[key], source_tensor, rtol=0, atol=0)
        diagnostics = collect_gpsm_parameters(model)
        assert len(diagnostics) == 3
        assert all(item["spatial_gate_weight_norm"] == 0.0 for item in diagnostics)
        assert all(item["spatial_gate_bias"] == 0.0 for item in diagnostics)
        assert all(item["router_weight_norm"] == 0.0 for item in diagnostics)
        assert all(item["router_bias"] == [0.0, 0.0, 0.0, 0.0] for item in diagnostics)
        assert all(item["project_weight_norm"] > 0.0 for item in diagnostics)
        json.dumps(diagnostics)

    checkpoint = tmp_path / "synthetic-e9-gpsm.pt"
    trainable.save(checkpoint)
    loaded = YOLO(checkpoint, verbose=False)
    loaded_head = loaded.model.model[-1]
    assert isinstance(loaded_head, GeometryPreservingSpatialMorphologyDetect)
    assert loaded_head.reg_max == 1
    assert loaded_head.stride.tolist() == [8.0, 16.0, 32.0]
    assert len(collect_gpsm_parameters(loaded.model)) == 3
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
        gpsm_yaml_path("n", "trainable"),
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
    assert all(_nonzero_finite_gradients(adapter.spatial_gate) for adapter in head.cls_adapters)

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
    """Reject unsupported E9 values and preserve requested negative batch tags."""
    from research.scripts.train_e9 import _run_name

    for size in ("", "a", "N", "xx"):
        with pytest.raises(ValueError, match="Unsupported YOLO26 scale"):
            gpsm_yaml_path(size, "trainable")
    with pytest.raises(ValueError, match="Unsupported GPSM variant"):
        gpsm_yaml_path("n", "unknown")
    assert _run_name("n", "trainable", 640, 30, -1, 42) == "yolo26n-gpsm_img640_e30_bauto_seed42"
    assert _run_name("n", "trainable", 640, 30, -8, 42) == "yolo26n-gpsm_img640_e30_b-8_seed42"
    assert _run_name("n", "control", 640, 30, -8, 42) == "yolo26n-gpsm-control_img640_e30_b-8_seed42"
