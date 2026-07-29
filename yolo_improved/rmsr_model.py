# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Construction and pretrained-weight transfer for E6 Residual Multi-Scale Refinement."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from ultralytics.nn.modules import C3k2, C3k2RMSR, Detect
from ultralytics.nn.tasks import load_checkpoint

from .contrast_ring_config import ContrastRingLossConfig
from .contrast_ring_model import ContrastRingYOLO


RMSR_VARIANTS = ("control", "trainable")
E2_1B_CONFIG = ContrastRingLossConfig(
    inner_kernel=3,
    outer_kernel=7,
    contrast_tau=0.25,
    positive_gain=0.25,
    negative_gain=0.25,
    negative_gamma=3.0,
    eps=1e-6,
)


@dataclass(frozen=True)
class RMSRTransferReport:
    """Exact and remapped pretrained-transfer results for one E6 model."""

    exact_keys: tuple[str, ...]
    remapped_layer16_keys: tuple[str, ...]
    skipped_shape_keys: tuple[str, ...]
    missing_target_keys: tuple[str, ...]
    new_rmsr_parameter_keys: tuple[str, ...]
    transferred_parameter_elements: int
    target_parameter_elements: int
    new_rmsr_parameter_elements: int
    gate_value: float

    @property
    def transfer_percentage(self) -> float:
        """Return target parameter-element coverage as a percentage."""
        return 100.0 * self.transferred_parameter_elements / self.target_parameter_elements


def _underlying_model(model: ContrastRingYOLO | nn.Module) -> nn.Module:
    """Return the PyTorch model from a ContrastRingYOLO facade or a direct module."""
    return model.model if isinstance(model, ContrastRingYOLO) else model


def remap_yolo26_rmsr_state_dict(
    source_state_dict: Mapping[str, Tensor],
    target_state_dict: Mapping[str, Tensor],
    *,
    target_parameter_keys: Collection[str] | None = None,
) -> tuple[dict[str, Tensor], RMSRTransferReport]:
    """Map a standard YOLO26 state dictionary into an E6 target without tensor reshaping.

    Args:
        source_state_dict (Mapping[str, Tensor]): Standard YOLO26 source state.
        target_state_dict (Mapping[str, Tensor]): E6 RMSR target state.
        target_parameter_keys (Collection[str] | None): Target parameter names for element accounting.

    Returns:
        tuple[dict[str, Tensor], RMSRTransferReport]: Compatible tensors and an immutable transfer report.
    """
    source_layer16 = {key for key in source_state_dict if key.startswith("model.16.")}
    target_base_layer16 = {key for key in target_state_dict if key.startswith("model.16.base.")}
    if not source_layer16:
        raise ValueError("Source state dict does not contain the standard YOLO26 layer 16.")
    if not target_base_layer16:
        raise ValueError("Target state dict does not contain C3k2RMSR base parameters at layer 16.")

    parameter_keys = set(target_state_dict) if target_parameter_keys is None else set(target_parameter_keys)
    remapped: dict[str, Tensor] = {}
    exact_keys: list[str] = []
    remapped_layer16_keys: list[str] = []
    skipped_shape_keys: list[str] = []
    source_keys_without_target: list[str] = []
    for source_key, source_tensor in source_state_dict.items():
        target_key = source_key
        is_layer16_remap = False
        if target_key not in target_state_dict and source_key.startswith("model.16."):
            target_key = source_key.replace("model.16.", "model.16.base.", 1)
            is_layer16_remap = True

        target_tensor = target_state_dict.get(target_key)
        if target_tensor is None:
            source_keys_without_target.append(source_key)
        elif source_tensor.shape != target_tensor.shape:
            skipped_shape_keys.append(target_key)
        else:
            remapped[target_key] = source_tensor
            (remapped_layer16_keys if is_layer16_remap else exact_keys).append(target_key)

    if source_keys_without_target:
        raise ValueError(f"Source state keys have no E6 target: {sorted(source_keys_without_target)}.")

    transferred_keys = set(remapped)
    missing_target_keys = tuple(sorted(set(target_state_dict) - transferred_keys))
    new_rmsr_prefixes = (
        "model.16.local_branch.",
        "model.16.context_branch.",
        "model.16.fusion.",
        "model.16.gate_raw",
    )
    new_rmsr_parameter_keys = tuple(
        sorted(key for key in parameter_keys if key.startswith(new_rmsr_prefixes))
    )
    transferred_parameter_elements = sum(
        target_state_dict[key].numel() for key in transferred_keys & parameter_keys
    )
    target_parameter_elements = sum(target_state_dict[key].numel() for key in parameter_keys)
    new_rmsr_parameter_elements = sum(target_state_dict[key].numel() for key in new_rmsr_parameter_keys)
    gate = target_state_dict.get("model.16.gate_raw")
    if gate is None or gate.numel() != 1:
        raise ValueError("E6 target state dict must contain the scalar model.16.gate_raw.")

    return remapped, RMSRTransferReport(
        exact_keys=tuple(sorted(exact_keys)),
        remapped_layer16_keys=tuple(sorted(remapped_layer16_keys)),
        skipped_shape_keys=tuple(sorted(skipped_shape_keys)),
        missing_target_keys=missing_target_keys,
        new_rmsr_parameter_keys=new_rmsr_parameter_keys,
        transferred_parameter_elements=transferred_parameter_elements,
        target_parameter_elements=target_parameter_elements,
        new_rmsr_parameter_elements=new_rmsr_parameter_elements,
        gate_value=float(gate.detach().cpu().item()),
    )


def print_rmsr_transfer_report(report: RMSRTransferReport) -> None:
    """Print the required E6 pretrained-transfer summary."""
    print(f"Transferred exact keys: {len(report.exact_keys)}")
    print(f"Transferred remapped layer-16 keys: {len(report.remapped_layer16_keys)}")
    print(f"Skipped shape mismatches: {len(report.skipped_shape_keys)}")
    print(f"Missing target keys: {len(report.missing_target_keys)}")
    print(
        f"New RMSR parameters: {len(report.new_rmsr_parameter_keys)} tensors, "
        f"{report.new_rmsr_parameter_elements} elements"
    )
    print(f"Gate value after loading: {report.gate_value:.10f}")
    print(f"Transfer percentage by parameter elements: {report.transfer_percentage:.6f}%")


def load_yolo26_rmsr_pretrained(
    target: ContrastRingYOLO | nn.Module,
    pretrained: str | Path,
    *,
    verbose: bool = True,
) -> RMSRTransferReport:
    """Load all compatible standard YOLO26 tensors into the E6 base architecture."""
    target_model = _underlying_model(target)
    source_model, _ = load_checkpoint(pretrained, device="cpu")
    if source_model.yaml.get("scale") != target_model.yaml.get("scale"):
        raise ValueError(
            f"Checkpoint scale {source_model.yaml.get('scale')!r} does not match "
            f"E6 target scale {target_model.yaml.get('scale')!r}."
        )
    if not isinstance(source_model.model[16], C3k2):
        raise ValueError("Pretrained checkpoint must contain the standard C3k2 at layer 16.")
    if not isinstance(target_model.model[16], C3k2RMSR):
        raise ValueError("Target model must contain C3k2RMSR at layer 16.")
    if (
        not isinstance(source_model.model[23], Detect)
        or not isinstance(target_model.model[23], Detect)
        or source_model.model[23].f != [16, 19, 22]
        or target_model.model[23].f != [16, 19, 22]
    ):
        raise ValueError("Source and target must retain the standard Detect layer 23 over P3, P4, and P5.")

    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    remapped, report = remap_yolo26_rmsr_state_dict(
        source_state_dict,
        target_state_dict,
        target_parameter_keys=dict(target_model.named_parameters()),
    )
    incompatible = target_model.load_state_dict(remapped, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected remapped target keys: {incompatible.unexpected_keys}.")
    reported_missing = set(report.missing_target_keys)
    actual_missing = set(incompatible.missing_keys)
    if actual_missing - reported_missing:
        raise RuntimeError("load_state_dict reported missing keys outside the new RMSR state.")
    silently_initialized = reported_missing - actual_missing
    if any(not key.endswith(".num_batches_tracked") for key in silently_initialized):
        raise RuntimeError("Reported RMSR target keys do not match load_state_dict missing keys.")

    transferred_keys = set(report.exact_keys) | set(report.remapped_layer16_keys)
    expected_base_layer16 = {
        key.replace("model.16.", "model.16.base.", 1)
        for key in source_state_dict
        if key.startswith("model.16.")
    }
    if not expected_base_layer16 or not expected_base_layer16 <= set(report.remapped_layer16_keys):
        raise RuntimeError("The complete standard layer 16 was not transferred into C3k2RMSR.base.")
    for layer_range, description in (
        (range(0, 11), "backbone"),
        ((13, 17, 18, 19, 20, 21, 22), "P4/P5 neck"),
    ):
        required = {
            key
            for key in target_state_dict
            if int(key.split(".")[1]) in layer_range
        }
        if not required <= transferred_keys:
            raise RuntimeError(f"Incomplete pretrained transfer for the {description}.")

    compatible_detect = {
        key
        for key, tensor in source_state_dict.items()
        if key.startswith("model.23.") and key in target_state_dict and tensor.shape == target_state_dict[key].shape
    }
    detect_box = {
        key
        for key in target_state_dict
        if key.startswith(("model.23.cv2.", "model.23.one2one_cv2."))
    }
    if not compatible_detect <= transferred_keys or not detect_box <= transferred_keys:
        raise RuntimeError("Compatible Detect classification or box tensors were not fully transferred.")

    forbidden_prefixes = (
        "model.16.local_branch.",
        "model.16.context_branch.",
        "model.16.fusion.",
        "model.16.gate_raw",
    )
    if any(key.startswith(forbidden_prefixes) for key in transferred_keys):
        raise RuntimeError("Standard YOLO26 tensors must not be copied into new RMSR parameters.")
    gate = target_model.model[16].gate_raw.detach()
    if torch.count_nonzero(gate).item() != 0:
        raise RuntimeError("The RMSR gate must remain exactly zero after pretrained loading.")

    if verbose:
        print_rmsr_transfer_report(report)
    return report


def rmsr_yaml_path(size: str, variant: str) -> Path:
    """Return the virtual scale-specific E6 YAML path."""
    if size not in {"n", "s", "m", "l", "x"}:
        raise ValueError(f"Unsupported YOLO26 scale {size!r}; expected one of n, s, m, l, x.")
    if variant not in RMSR_VARIANTS:
        raise ValueError(f"Unsupported RMSR variant {variant!r}; expected control or trainable.")
    model_directory = Path(__file__).resolve().parents[1] / "research" / "models"
    suffix = "rmsr-control" if variant == "control" else "rmsr"
    unified_yaml = model_directory / f"yolo26-{suffix}.yaml"
    if not unified_yaml.is_file():
        raise FileNotFoundError(f"E6 architecture YAML not found: {unified_yaml}")
    return model_directory / f"yolo26{size}-{suffix}.yaml"


def build_rmsr_yolo(size: str, variant: str, verbose: bool = False) -> ContrastRingYOLO:
    """Build E6 with fixed E2.1b supervision and standard pretrained YOLO26 weights."""
    architecture_yaml = rmsr_yaml_path(size, variant)
    model = ContrastRingYOLO(architecture_yaml, loss_config=E2_1B_CONFIG, verbose=verbose)
    rmsr_layer = model.model.model[16]
    if model.model.yaml.get("scale") != size:
        raise RuntimeError(f"Expected E6 scale {size!r}, received {model.model.yaml.get('scale')!r}.")
    if model.model.contrast_ring_loss_config != E2_1B_CONFIG:
        raise RuntimeError("E6 must use the fixed E2.1b loss configuration.")
    if not isinstance(rmsr_layer, C3k2RMSR) or rmsr_layer.trainable_refinement != (variant == "trainable"):
        raise RuntimeError(f"E6 YAML does not implement the requested {variant!r} variant.")
    if rmsr_layer.gate_max != 1.0 or torch.count_nonzero(rmsr_layer.gate_raw.detach()).item() != 0:
        raise RuntimeError("E6 must start with gate_max=1.0 and an exactly zero raw gate.")

    pretrained = Path(f"yolo26{size}.pt")
    report = load_yolo26_rmsr_pretrained(model, pretrained, verbose=verbose)
    model.rmsr_variant = variant
    model.rmsr_transfer_report = report
    model.rmsr_pretrained = str(pretrained)
    model.ckpt_path = str(pretrained)
    model.ckpt = {"model": model.model}
    return model
