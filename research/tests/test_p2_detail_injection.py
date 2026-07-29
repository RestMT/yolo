# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Synthetic CPU checks for the isolated E5 P2 Detail Injection experiment."""

from __future__ import annotations

import gc
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Concat, Conv, Detect
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
    build_p2_detail_injection_yolo,
    p2_detail_injection_yaml_path,
    remap_yolo26_p2di_state_dict,
)
from yolo_improved.contrast_ring_loss import ContrastRingBCEWithLogitsLoss, E2_LOCALIZATION_CONFIG


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STOCK_MODEL_DIRECTORY = REPOSITORY_ROOT / "ultralytics" / "cfg" / "models" / "26"
E5_MODEL_DIRECTORY = REPOSITORY_ROOT / "research" / "models"
STANDARD_N_WEIGHTS = REPOSITORY_ROOT / "yolo26n.pt"
SCALE_SETTINGS = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}


def _state_shapes(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    """Return state shapes relative to one model layer."""
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


@pytest.mark.parametrize("size", tuple(SCALE_SETTINGS))
def test_unified_yaml_scaling_backbone_and_model_cost(size: str) -> None:
    """Build every scale and verify unified YAML scaling, backbone identity, Detect, parameters, and FLOPs."""
    depth, width, max_channels = SCALE_SETTINGS[size]
    virtual_e5_yaml = p2_detail_injection_yaml_path(size)
    unified_e5_yaml = E5_MODEL_DIRECTORY / "yolo26-p2di.yaml"
    stock_virtual_yaml = STOCK_MODEL_DIRECTORY / f"yolo26{size}.yaml"

    assert unified_e5_yaml.is_file()
    assert not virtual_e5_yaml.exists()
    assert not list(E5_MODEL_DIRECTORY.glob("yolo26?-p2di.yaml"))

    e5_definition = yaml_model_load(virtual_e5_yaml)
    stock_definition = yaml_model_load(stock_virtual_yaml)
    assert e5_definition["scale"] == size
    assert tuple(e5_definition["scales"][size]) == SCALE_SETTINGS[size]
    assert e5_definition["backbone"] == stock_definition["backbone"]

    stock_model = DetectionModel(stock_virtual_yaml, verbose=False)
    stock_backbone_shapes = [_state_shapes(layer) for layer in stock_model.model[:11]]
    stock_parameters = get_num_params(stock_model)
    stock_flops = get_flops(stock_model, imgsz=640)
    del stock_model
    gc.collect()

    e5_model = ContrastRingDetectionModel(
        virtual_e5_yaml,
        verbose=False,
        loss_config=E2_1B_CONFIG,
    )
    assert e5_model.yaml["scale"] == size
    assert [_state_shapes(layer) for layer in e5_model.model[:11]] == stock_backbone_shapes
    assert len(e5_model.model[2].m) == max(round(2 * depth), 1)
    assert e5_model.model[0].conv.out_channels == make_divisible(min(64, max_channels) * width, 8)
    assert e5_model.model[7].conv.out_channels == make_divisible(min(1024, max_channels) * width, 8)
    assert isinstance(e5_model.model[15], Conv)
    assert isinstance(e5_model.model[16], Concat)
    assert isinstance(e5_model.model[24], Detect)
    assert e5_model.model[24].f == [17, 20, 23]
    assert e5_model.model[24].nl == 3
    assert e5_model.stride.tolist() == [8.0, 16.0, 32.0]

    e5_parameters = get_num_params(e5_model)
    e5_flops = get_flops(e5_model, imgsz=640)
    assert e5_parameters > stock_parameters
    assert stock_flops > 0 and e5_flops > stock_flops
    assert e5_parameters / stock_parameters < 1.25
    assert e5_flops / stock_flops < 1.25
    del e5_model
    gc.collect()


def test_p2_route_forward_output_format_and_fixed_e2_1b() -> None:
    """Check P2/4 -> P3/8 geometry, three-level output compatibility, and fixed E2.1b supervision."""
    e5_model = ContrastRingDetectionModel(
        p2_detail_injection_yaml_path("n"),
        verbose=False,
        loss_config=E2_1B_CONFIG,
    )
    stock_model = DetectionModel(STOCK_MODEL_DIRECTORY / "yolo26n.yaml", verbose=False)
    captured: dict[str, object] = {}

    def capture_spatial(name: str):
        def hook(_module, _inputs, output):
            captured[name] = tuple(output.shape[-2:])

        return hook

    def capture_concat_inputs(_module, inputs):
        captured["concat_inputs"] = tuple(tuple(tensor.shape[-2:]) for tensor in inputs[0])

    handles = (
        e5_model.model[2].register_forward_hook(capture_spatial("p2")),
        e5_model.model[15].register_forward_hook(capture_spatial("p2_projection")),
        e5_model.model[16].register_forward_pre_hook(capture_concat_inputs),
    )
    e5_model.eval()
    stock_model.eval()
    image = torch.rand(1, 3, 64, 64)
    try:
        with torch.no_grad():
            e5_output = e5_model(image)
            stock_output = stock_model(image)
    finally:
        for handle in handles:
            handle.remove()

    assert captured["p2"] == (16, 16)
    assert captured["p2_projection"] == (8, 8)
    assert captured["concat_inputs"] == ((8, 8), (8, 8), (8, 8))
    assert _output_signature(e5_output) == _output_signature(stock_output)
    assert e5_model.model[24].nl == 3
    assert e5_model.model[24].stride.tolist() == [8.0, 16.0, 32.0]
    assert all(
        layer.__class__.__module__.startswith(("ultralytics.nn.modules", "torch.nn.modules"))
        for layer in e5_model.model
    )

    assert e5_model.contrast_ring_loss_config == E2_1B_CONFIG
    assert E2_1B_CONFIG.inner_kernel == 3
    assert E2_1B_CONFIG.outer_kernel == 7
    assert E2_1B_CONFIG.contrast_tau == 0.25
    assert E2_1B_CONFIG.positive_gain == 0.25
    assert E2_1B_CONFIG.negative_gain == 0.25
    assert E2_1B_CONFIG.negative_gamma == 3.0
    assert E2_1B_CONFIG.eps == 1e-6
    assert E2_LOCALIZATION_CONFIG.mode == "constant-010"
    assert E2_LOCALIZATION_CONFIG.nwd_scale == 0.10

    e5_model.args = get_cfg()
    criterion = e5_model.init_criterion()
    assert isinstance(criterion, E2ELoss)
    for branch in (criterion.one2many, criterion.one2one):
        assert isinstance(branch, ContrastRingDetectionLoss)
        assert isinstance(branch.bbox_loss, ResidualNWDBboxLoss)
        assert branch.bbox_loss.config == E2_LOCALIZATION_CONFIG
        assert isinstance(branch.bce, ContrastRingBCEWithLogitsLoss)
        assert branch.contrast_ring_config == E2_1B_CONFIG

    del criterion
    del stock_model
    del e5_model
    gc.collect()


@pytest.mark.skipif(not STANDARD_N_WEIGHTS.is_file(), reason="Local standard yolo26n.pt is required.")
def test_exact_pretrained_transfer_and_standard_checkpoint_loading(tmp_path: Path) -> None:
    """Verify the exact layer map, protected P2 initialization, Detect transfer, and ordinary YOLO loading."""
    e5 = build_p2_detail_injection_yolo("n", STANDARD_N_WEIGHTS, verbose=False)
    source = YOLO(STANDARD_N_WEIGHTS, verbose=False).model
    target = e5.model
    source_state = source.state_dict()
    target_state = target.state_dict()
    report = e5.p2di_transfer_report

    assert not report.skipped_absent_keys
    assert len(report.skipped_shape_keys) == 1
    assert report.skipped_shape_keys[0].startswith("model.17.")
    assert all(key in report.transferred_keys for key in target_state if 0 <= int(key.split(".")[1]) <= 10)
    for target_key in report.transferred_keys:
        target_index = int(target_key.split(".")[1])
        if target_index <= 14:
            source_key = target_key
        elif target_index >= 17:
            source_index = 16 if target_index == 17 else target_index - 1
            source_key = target_key.replace(f"model.{target_index}.", f"model.{source_index}.", 1)
        else:
            continue
        torch.testing.assert_close(target_state[target_key], source_state[source_key], rtol=0, atol=0)

    p2_projection_keys = {key for key in target_state if key.startswith("model.15.")}
    assert p2_projection_keys
    assert p2_projection_keys <= set(report.newly_initialized_keys)
    assert not p2_projection_keys & set(report.transferred_keys)
    assert torch.count_nonzero(target_state["model.15.conv.weight"]) > 0

    detect_box_keys = {
        key
        for key in target_state
        if key.startswith(("model.24.cv2.", "model.24.one2one_cv2."))
    }
    assert detect_box_keys and detect_box_keys <= set(report.transferred_keys)
    assert e5.ckpt["model"] is target

    with pytest.raises(ValueError, match="Invalid YOLO26 -> E5 layer mapping"):
        remap_yolo26_p2di_state_dict(source_state, target_state, layer_index_map={})

    checkpoint = tmp_path / "synthetic-e5.pt"
    e5.save(checkpoint)
    loaded = YOLO(checkpoint, verbose=False)
    assert isinstance(loaded.model.model[-1], Detect)
    assert loaded.model.model[-1].i == 24
    assert loaded.model.model[-1].nl == 3
    assert loaded.model.stride.tolist() == [8.0, 16.0, 32.0]
    assert all(
        layer.__class__.__module__.startswith(("ultralytics.nn.modules", "torch.nn.modules"))
        for layer in loaded.model.model
    )
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
    del e5
    gc.collect()


@pytest.mark.parametrize("size", ("", "a", "N", "xx"))
def test_invalid_scale_is_rejected(size: str) -> None:
    """Reject unsupported E5 scale names before model construction."""
    with pytest.raises(ValueError, match="Unsupported YOLO26 scale"):
        p2_detail_injection_yaml_path(size)
