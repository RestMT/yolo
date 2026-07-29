# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Synthetic CPU checks for E7 distributional YOLO26 box regression."""

from __future__ import annotations

import gc
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils.loss import DFLoss, E2ELoss
from ultralytics.utils.torch_utils import get_flops, get_num_params
from yolo_improved import build_regmax_yolo
from yolo_improved.contrast_ring_loss import ContrastRingBCEWithLogitsLoss, ContrastRingDetectionLoss
from yolo_improved.contrast_ring_model import ContrastRingDetectionModel
from yolo_improved.regmax_model import (
    E2_1B_CONFIG,
    REG_MAX_VALUES,
    regmax_yaml_path,
    remap_yolo26_regmax_state_dict,
)
from yolo_improved.residual_nwd_loss import ResidualNWDBboxLoss


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STOCK_MODEL_DIRECTORY = REPOSITORY_ROOT / "ultralytics" / "cfg" / "models" / "26"
E7_MODEL_DIRECTORY = REPOSITORY_ROOT / "research" / "models"
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


def _final_box_keys(detect: Detect) -> set[str]:
    """Return final regression-convolution parameters for both end-to-end heads."""
    return {
        f"model.23.{head_name}.{level_index}.{len(level) - 1}.{parameter_name}"
        for head_name in ("cv2", "one2one_cv2")
        for level_index, level in enumerate(getattr(detect, head_name))
        for parameter_name in ("weight", "bias")
    }


def _assert_detect(detect: Detect, reg_max: int) -> None:
    """Check the complete E7 Detect distributional structure."""
    assert detect.f == [16, 19, 22]
    assert detect.nl == 3
    assert detect.stride.tolist() == [8.0, 16.0, 32.0]
    assert detect.reg_max == reg_max
    assert detect.no == detect.nc + 4 * reg_max
    assert not isinstance(detect.dfl, nn.Identity)
    assert detect.dfl.c1 == reg_max
    for head_name in ("cv2", "one2one_cv2"):
        head = getattr(detect, head_name)
        assert len(head) == 3
        assert [level[-1].out_channels for level in head] == [4 * reg_max] * 3


def _synthetic_batch() -> dict[str, torch.Tensor]:
    """Return one small normalized detection batch."""
    return {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }


def test_unified_yamls_differ_from_stock_only_by_reg_max() -> None:
    """Verify both unified E7 definitions and all ten virtual scale names."""
    for reg_max in REG_MAX_VALUES:
        unified = E7_MODEL_DIRECTORY / f"yolo26-regmax{reg_max}.yaml"
        assert unified.is_file()
        assert not list(E7_MODEL_DIRECTORY.glob(f"yolo26?-regmax{reg_max}.yaml"))
        for size, scale_settings in SCALE_SETTINGS.items():
            virtual = regmax_yaml_path(size, reg_max)
            assert not virtual.exists()
            target = yaml_model_load(virtual)
            stock = yaml_model_load(STOCK_MODEL_DIRECTORY / f"yolo26{size}.yaml")
            assert target["scale"] == size
            assert tuple(target["scales"][size]) == scale_settings
            for key in ("nc", "end2end", "scales", "backbone", "head", "scale"):
                assert target[key] == stock[key]
            assert stock["reg_max"] == 1
            assert target["reg_max"] == reg_max
            assert target["head"][-1] == [[16, 19, 22], 1, "Detect", ["nc"]]


@pytest.mark.parametrize("size", tuple(SCALE_SETTINGS))
def test_n_to_x_detect_scaling_transfer_and_cost(size: str) -> None:
    """Build stock/regmax4/regmax8 and check scaling, transfer isolation, parameters, and FLOPs."""
    stock = DetectionModel(STOCK_MODEL_DIRECTORY / f"yolo26{size}.yaml", verbose=False)
    stock_state = stock.state_dict()
    stock_non_detect_shapes = {
        key: tuple(tensor.shape) for key, tensor in stock_state.items() if not key.startswith("model.23.")
    }
    stock_classification_shapes = {
        key: tuple(tensor.shape)
        for key, tensor in stock_state.items()
        if key.startswith(("model.23.cv3.", "model.23.one2one_cv3."))
    }
    stock_parameters = get_num_params(stock)
    stock_flops = get_flops(stock, imgsz=640)
    previous_parameters = stock_parameters
    previous_flops = stock_flops

    for reg_max in REG_MAX_VALUES:
        target = ContrastRingDetectionModel(
            regmax_yaml_path(size, reg_max),
            verbose=False,
            loss_config=E2_1B_CONFIG,
        )
        detect = target.model[-1]
        _assert_detect(detect, reg_max)
        assert target.yaml["scale"] == size
        assert len(target.model) == 24
        target_state = target.state_dict()
        assert {
            key: tuple(tensor.shape) for key, tensor in target_state.items() if not key.startswith("model.23.")
        } == stock_non_detect_shapes
        assert {
            key: tuple(tensor.shape)
            for key, tensor in target_state.items()
            if key.startswith(("model.23.cv3.", "model.23.one2one_cv3."))
        } == stock_classification_shapes

        transferred, report = remap_yolo26_regmax_state_dict(
            stock_state,
            target_state,
            target_parameter_keys=dict(target.named_parameters()),
            target_reg_max=reg_max,
        )
        assert not report.skipped_missing_keys
        assert report.coverage("backbone").percentage == 100
        assert report.coverage("neck").percentage == 100
        assert report.coverage("classification heads").percentage == 100
        assert _final_box_keys(detect) <= set(report.skipped_shape_keys)
        assert set(report.skipped_shape_keys).isdisjoint(transferred)
        assert all(
            key.startswith(("model.23.cv2.", "model.23.one2one_cv2."))
            for key in report.skipped_shape_keys
        )
        assert "model.23.dfl.conv.weight" in report.newly_initialized_keys
        assert report.coverage("one-to-many box heads").transferred_parameter_elements == report.coverage(
            "one-to-one box heads"
        ).transferred_parameter_elements

        parameters = get_num_params(target)
        flops = get_flops(target, imgsz=640)
        assert parameters > previous_parameters
        assert flops > previous_flops
        previous_parameters = parameters
        previous_flops = flops
        del target
        gc.collect()

    del stock
    gc.collect()


@pytest.mark.parametrize("reg_max", REG_MAX_VALUES)
def test_e2_1b_dfl_forward_backward_and_amp(reg_max: int) -> None:
    """Check both DFL branches, finite synthetic loss/backward, and CPU AMP."""
    model = ContrastRingDetectionModel(
        regmax_yaml_path("n", reg_max),
        verbose=False,
        loss_config=E2_1B_CONFIG,
    )
    model.args = get_cfg()
    model.train()
    criterion = model.init_criterion()
    assert isinstance(criterion, E2ELoss)
    for branch in (criterion.one2many, criterion.one2one):
        assert isinstance(branch, ContrastRingDetectionLoss)
        assert branch.use_dfl
        assert branch.reg_max == reg_max
        assert branch.loss_names[2] == "dfl_loss"
        assert isinstance(branch.bbox_loss, ResidualNWDBboxLoss)
        assert isinstance(branch.bbox_loss.dfl_loss, DFLoss)
        assert branch.bbox_loss.dfl_loss.reg_max == reg_max
        assert branch.bbox_loss.config.mode == "constant-010"
        assert branch.bbox_loss.config.nwd_scale == 0.10
        assert isinstance(branch.bce, ContrastRingBCEWithLogitsLoss)
        assert branch.contrast_ring_config == E2_1B_CONFIG
    model.criterion = criterion

    batch = _synthetic_batch()
    predictions = model(batch["img"])
    for branch_name in ("one2many", "one2one"):
        branch_predictions = predictions[branch_name]
        assert branch_predictions["boxes"].shape[1] == 4 * reg_max
        assert branch_predictions["scores"].shape[1] == 80
        assert len(branch_predictions["feats"]) == 3
    loss, loss_items = model.loss(batch, predictions)
    assert loss.shape == (3,)
    assert set(loss_items) == {"box_loss", "cls_loss", "dfl_loss"}
    assert torch.isfinite(loss).all()
    assert all(torch.isfinite(value) for value in loss_items.values())
    loss.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)

    model.zero_grad(set_to_none=True)
    model.criterion = model.init_criterion()
    amp_batch = _synthetic_batch()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_loss, amp_items = model.loss(amp_batch)
    assert torch.isfinite(amp_loss).all()
    assert all(torch.isfinite(value) for value in amp_items.values())
    amp_loss.sum().backward()
    amp_gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert amp_gradients
    assert all(torch.isfinite(gradient).all() for gradient in amp_gradients)

    del criterion
    del model
    gc.collect()


@pytest.mark.skipif(not STANDARD_N_WEIGHTS.is_file(), reason="Local standard yolo26n.pt is required.")
@pytest.mark.parametrize("reg_max", REG_MAX_VALUES)
def test_pretrained_transfer_and_standard_checkpoint_loading(reg_max: int, tmp_path: Path) -> None:
    """Verify real n-weight transfer and ordinary YOLO loading of a saved E7 checkpoint."""
    model = build_regmax_yolo("n", reg_max, verbose=False)
    source = YOLO(STANDARD_N_WEIGHTS, verbose=False).model
    source_state = source.state_dict()
    target_state = model.model.state_dict()
    report = model.regmax_transfer_report
    detect = model.model.model[-1]

    assert report.source_checkpoint == "yolo26n.pt"
    assert report.target_reg_max == reg_max
    assert report.coverage("backbone").percentage == 100
    assert report.coverage("neck").percentage == 100
    assert report.coverage("classification heads").percentage == 100
    assert _final_box_keys(detect) <= set(report.skipped_shape_keys)
    for key in report.exact_keys:
        torch.testing.assert_close(target_state[key], source_state[key], rtol=0, atol=0)
    assert all(
        key.startswith(("model.23.cv2.", "model.23.one2one_cv2."))
        for key in report.skipped_shape_keys
    )
    dfl_weight = target_state["model.23.dfl.conv.weight"].flatten()
    torch.testing.assert_close(dfl_weight, torch.arange(reg_max, dtype=dfl_weight.dtype), rtol=0, atol=0)
    for head_name in ("cv2", "one2one_cv2"):
        for level in getattr(detect, head_name):
            assert torch.all(level[-1].bias == 2.0)
    assert model.ckpt["model"] is model.model

    checkpoint = tmp_path / f"synthetic-e7-regmax{reg_max}.pt"
    model.save(checkpoint)
    loaded = YOLO(checkpoint, verbose=False)
    loaded_detect = loaded.model.model[-1]
    _assert_detect(loaded_detect, reg_max)
    assert loaded.model.yaml["reg_max"] == reg_max
    assert hasattr(loaded_detect, "one2one_cv2")
    assert hasattr(loaded_detect, "one2one_cv3")

    if reg_max == 8:
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
    del model
    gc.collect()


def test_invalid_factory_inputs_and_run_names() -> None:
    """Reject unsupported E7 values and preserve requested negative batch tags."""
    from research.scripts.train_e7 import _run_name

    for size in ("", "a", "N", "xx"):
        with pytest.raises(ValueError, match="Unsupported YOLO26 scale"):
            regmax_yaml_path(size, 8)
    for reg_max in (1, 2, 16, 4.0, True, "8"):
        with pytest.raises(ValueError, match="Unsupported E7 reg_max"):
            regmax_yaml_path("n", reg_max)
    assert _run_name("n", 8, 640, 30, -1, 42) == "yolo26n-regmax8_img640_e30_bauto_seed42"
    assert _run_name("n", 8, 640, 30, -8, 42) == "yolo26n-regmax8_img640_e30_b-8_seed42"
