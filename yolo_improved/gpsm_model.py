# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Construction and exact pretrained transfer for E9 Geometry-Preserving Spatial Morphology."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn

from ultralytics.nn.modules import (
    Detect,
    GeometryPreservingSpatialMorphologyDetect,
    SpatialMorphologyClassificationAdapter,
)
from ultralytics.nn.tasks import load_checkpoint, yaml_model_load
from ultralytics.utils.loss import E2ELoss

from .contrast_ring_loss import ContrastRingDetectionLoss, E2_LOCALIZATION_CONFIG
from .contrast_ring_model import ContrastRingDetectionModel, ContrastRingYOLO
from .madh_model import E2_1B_CONFIG, remap_yolo26_madh_state_dict
from .residual_nwd_loss import ResidualNWDBboxLoss


GPSM_VARIANTS = ("control", "trainable")
MODEL_SIZES = ("n", "s", "m", "l", "x")
_GPSM_PREFIX = "model.23.cls_adapters."


@dataclass(frozen=True)
class GPSMTransferCoverage:
    """Pretrained parameter coverage for one E9 architecture region."""

    name: str
    transferred_tensors: int
    target_tensors: int
    transferred_parameter_elements: int
    target_parameter_elements: int

    @property
    def percentage(self) -> float:
        """Return target parameter-element coverage for this region."""
        if self.target_parameter_elements == 0:
            return 100.0
        return 100.0 * self.transferred_parameter_elements / self.target_parameter_elements


@dataclass(frozen=True)
class GPSMTransferReport:
    """Exact-key pretrained-transfer results for one E9 model."""

    source_checkpoint: str
    target_architecture: str
    exact_keys: tuple[str, ...]
    skipped_missing_keys: tuple[str, ...]
    skipped_shape_keys: tuple[str, ...]
    new_gpsm_keys: tuple[str, ...]
    transferred_parameter_elements: int
    target_parameter_elements: int
    new_gpsm_parameter_elements: int
    coverages: tuple[GPSMTransferCoverage, ...]

    @property
    def transfer_percentage(self) -> float:
        """Return target parameter-element coverage as a percentage."""
        return 100.0 * self.transferred_parameter_elements / self.target_parameter_elements

    def coverage(self, name: str) -> GPSMTransferCoverage:
        """Return one named architecture-region coverage record."""
        try:
            return next(item for item in self.coverages if item.name == name)
        except StopIteration as error:
            raise KeyError(name) from error


def _underlying_model(model: ContrastRingYOLO | nn.Module) -> nn.Module:
    """Return the PyTorch detection model from a facade or direct module."""
    return model.model if isinstance(model, ContrastRingYOLO) else model


def remap_yolo26_gpsm_state_dict(
    source_state_dict: Mapping[str, Tensor],
    target_state_dict: Mapping[str, Tensor],
    *,
    target_parameter_keys: Collection[str] | None = None,
) -> tuple[dict[str, Tensor], GPSMTransferReport]:
    """Reuse E8 exact-key selection and expose E9 classification-adapter coverage."""
    transferred, madh_report = remap_yolo26_madh_state_dict(
        source_state_dict,
        target_state_dict,
        target_parameter_keys=target_parameter_keys,
    )
    coverages = tuple(
        GPSMTransferCoverage(
            name="cls_adapters" if coverage.name == "adapters" else coverage.name,
            transferred_tensors=coverage.transferred_tensors,
            target_tensors=coverage.target_tensors,
            transferred_parameter_elements=coverage.transferred_parameter_elements,
            target_parameter_elements=coverage.target_parameter_elements,
        )
        for coverage in madh_report.coverages
    )
    return transferred, GPSMTransferReport(
        source_checkpoint=madh_report.source_checkpoint,
        target_architecture=madh_report.target_architecture,
        exact_keys=madh_report.exact_keys,
        skipped_missing_keys=madh_report.skipped_missing_keys,
        skipped_shape_keys=madh_report.skipped_shape_keys,
        new_gpsm_keys=madh_report.new_adapter_keys,
        transferred_parameter_elements=madh_report.transferred_parameter_elements,
        target_parameter_elements=madh_report.target_parameter_elements,
        new_gpsm_parameter_elements=madh_report.new_adapter_parameter_elements,
        coverages=coverages,
    )


def print_gpsm_transfer_report(report: GPSMTransferReport) -> None:
    """Print the required E9 pretrained-transfer summary and regional coverage."""
    print(f"Transferred tensors: {len(report.exact_keys)}")
    print(f"Transferred parameter elements: {report.transferred_parameter_elements}")
    print(f"Skipped shape mismatches: {len(report.skipped_shape_keys)}")
    print(
        f"New GPSM tensors: {len(report.new_gpsm_keys)} tensors, "
        f"{report.new_gpsm_parameter_elements} parameter elements"
    )
    print(f"Transfer percentage: {report.transfer_percentage:.6f}%")
    print("Transfer coverage:")
    for coverage in report.coverages:
        print(
            f"  {coverage.name}: {coverage.transferred_tensors}/{coverage.target_tensors} tensors, "
            f"{coverage.percentage:.6f}% parameter elements"
        )


def collect_gpsm_parameters(model: ContrastRingYOLO | nn.Module) -> tuple[dict[str, object], ...]:
    """Return JSON-ready spatial-gate, router, and projection diagnostics for P3/P4/P5."""
    underlying = _underlying_model(model)
    head = underlying.model[-1]
    if not isinstance(head, GeometryPreservingSpatialMorphologyDetect):
        raise ValueError("Model does not contain GeometryPreservingSpatialMorphologyDetect.")
    levels = ("P3/8", "P4/16", "P5/32")
    return tuple(
        {
            "detection_level": level,
            "spatial_gate_weight_norm": float(adapter.spatial_gate.weight.detach().float().norm().cpu().item()),
            "spatial_gate_bias": float(adapter.spatial_gate.bias.detach().float().cpu().item()),
            "router_weight_norm": float(adapter.router.weight.detach().float().norm().cpu().item()),
            "router_bias": [float(value) for value in adapter.router.bias.detach().float().cpu().tolist()],
            "project_weight_norm": float(adapter.project.conv.weight.detach().float().norm().cpu().item()),
        }
        for level, adapter in zip(levels, head.cls_adapters)
    )


def _validate_gpsm_head(
    model: nn.Module,
    variant: str,
) -> GeometryPreservingSpatialMorphologyDetect:
    """Validate E9 topology, stock box path, and classification-only adapter state."""
    head = model.model[-1]
    if not isinstance(head, GeometryPreservingSpatialMorphologyDetect):
        raise RuntimeError("E9 must use GeometryPreservingSpatialMorphologyDetect.")
    if head.f != [16, 19, 22] or head.nl != 3:
        raise RuntimeError("E9 must retain exactly the P3/8, P4/16, and P5/32 levels.")
    if head.stride.tolist() != [8.0, 16.0, 32.0]:
        raise RuntimeError(f"E9 Detect strides must be [8, 16, 32], got {head.stride.tolist()}.")
    if head.reg_max != 1 or head.no != head.nc + 4 or not isinstance(head.dfl, nn.Identity):
        raise RuntimeError("E9 must retain stock YOLO26 reg_max=1 without DFL.")
    for forbidden in ("box_adapters", "one2one_box_adapters", "one2one_cls_adapters"):
        if hasattr(head, forbidden):
            raise RuntimeError(f"E9 must not create {forbidden}.")
    if len(head.cls_adapters) != 3:
        raise RuntimeError("E9 must contain exactly one shared classification adapter per detection level.")

    trainable = variant == "trainable"
    for adapter in head.cls_adapters:
        if not isinstance(adapter, SpatialMorphologyClassificationAdapter):
            raise RuntimeError("E9 classification adapter has an unexpected type.")
        if adapter.trainable_adapter != trainable or adapter.gate_max != 0.5:
            raise RuntimeError(f"E9 adapter does not implement the requested {variant!r} variant.")
        if hasattr(adapter, "gate_raw"):
            raise RuntimeError("E9 must use a spatial gate instead of the E8 scalar gate.")
        if torch.count_nonzero(adapter.spatial_gate.weight.detach()).item() != 0:
            raise RuntimeError("Every E9 spatial-gate weight must start at zero.")
        if torch.count_nonzero(adapter.spatial_gate.bias.detach()).item() != 0:
            raise RuntimeError("Every E9 spatial-gate bias must start at zero.")
        if torch.count_nonzero(adapter.router.weight.detach()).item() != 0:
            raise RuntimeError("Every E9 router weight must start at zero.")
        if torch.count_nonzero(adapter.router.bias.detach()).item() != 0:
            raise RuntimeError("Every E9 router bias must start at zero.")
        if any(parameter.requires_grad != trainable for parameter in adapter.parameters()):
            raise RuntimeError(f"E9 adapter gradients do not match the requested {variant!r} variant.")
    return head


def _validate_criterion(model: ContrastRingDetectionModel) -> None:
    """Verify unchanged E2.1b supervision and the stock reg_max=1 L1 branch."""
    criterion = model.init_criterion()
    if not isinstance(criterion, E2ELoss):
        raise RuntimeError("E9 must retain the stock end-to-end loss wrapper.")
    for name, branch in (("one-to-many", criterion.one2many), ("one-to-one", criterion.one2one)):
        if not isinstance(branch, ContrastRingDetectionLoss):
            raise RuntimeError(f"E9 {name} must reuse ContrastRingDetectionLoss.")
        if branch.reg_max != 1 or branch.use_dfl or branch.bbox_loss.dfl_loss is not None:
            raise RuntimeError(f"E9 {name} must retain stock YOLO26 normalized L1 regression.")
        if not isinstance(branch.bbox_loss, ResidualNWDBboxLoss):
            raise RuntimeError(f"E9 {name} must retain the E1.1 residual NWD box criterion.")
        if branch.bbox_loss.config != E2_LOCALIZATION_CONFIG:
            raise RuntimeError(f"E9 {name} changed the E1.1 localization configuration.")
        if branch.contrast_ring_config != E2_1B_CONFIG:
            raise RuntimeError(f"E9 {name} changed the E2.1b classification configuration.")


def load_yolo26_gpsm_pretrained(
    target: ContrastRingYOLO | nn.Module,
    pretrained: str | Path,
    *,
    verbose: bool = True,
) -> GPSMTransferReport:
    """Load every standard YOLO26 tensor while preserving all new E9 adapter state."""
    target_model = _underlying_model(target)
    target_head = target_model.model[-1]
    if not isinstance(target_head, GeometryPreservingSpatialMorphologyDetect):
        raise ValueError("Target must contain GeometryPreservingSpatialMorphologyDetect.")

    source_model, _ = load_checkpoint(pretrained, device="cpu")
    source_head = source_model.model[-1]
    if type(source_head) is not Detect or source_head.reg_max != 1:
        raise ValueError("Pretrained source must be a standard YOLO26 Detect checkpoint with reg_max=1.")
    if source_head.f != [16, 19, 22] or source_head.nl != 3:
        raise ValueError("Pretrained source does not have the stock YOLO26 P3/P4/P5 topology.")
    if source_model.yaml.get("scale") != target_model.yaml.get("scale"):
        raise ValueError(
            f"Checkpoint scale {source_model.yaml.get('scale')!r} does not match "
            f"E9 target scale {target_model.yaml.get('scale')!r}."
        )

    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    transferred, report = remap_yolo26_gpsm_state_dict(
        source_state_dict,
        target_state_dict,
        target_parameter_keys=dict(target_model.named_parameters()),
    )
    variant = "trainable" if target_head.cls_adapters[0].trainable_adapter else "control"
    suffix = "gpsm" if variant == "trainable" else "gpsm-control"
    target_architecture = Path(target_model.yaml["yaml_file"]).with_name(f"yolo26-{suffix}.yaml")
    report = replace(
        report,
        source_checkpoint=str(pretrained),
        target_architecture=str(target_architecture),
    )
    initial_adapter_state = {
        key: target_state_dict[key].detach().clone() for key in report.new_gpsm_keys
    }

    incompatible = target_model.load_state_dict(transferred, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected E9 transfer keys: {incompatible.unexpected_keys}.")
    reported_missing = set(report.new_gpsm_keys)
    actual_missing = set(incompatible.missing_keys)
    if actual_missing - reported_missing:
        raise RuntimeError(f"load_state_dict reported untracked E9 keys: {sorted(actual_missing - reported_missing)}.")
    silently_initialized = reported_missing - actual_missing
    if any(not key.endswith(".num_batches_tracked") for key in silently_initialized):
        raise RuntimeError("E9 transfer report does not match load_state_dict missing keys.")

    loaded_state_dict = target_model.state_dict()
    for key, initial_tensor in initial_adapter_state.items():
        if not torch.equal(loaded_state_dict[key], initial_tensor):
            raise RuntimeError(f"New E9 adapter tensor {key!r} was modified during pretrained transfer.")
    if report.skipped_missing_keys or report.skipped_shape_keys:
        raise RuntimeError("Stock YOLO26 and E9 standard tensors must match by exact key and shape.")
    if not report.new_gpsm_keys or any(not key.startswith(_GPSM_PREFIX) for key in report.new_gpsm_keys):
        raise RuntimeError("Only E9 classification-adapter tensors may remain newly initialized.")
    for coverage_name in ("backbone", "neck", "cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        coverage = report.coverage(coverage_name)
        if coverage.transferred_tensors != coverage.target_tensors or coverage.percentage != 100.0:
            raise RuntimeError(f"Pretrained transfer is incomplete for E9 {coverage_name}.")
    adapter_coverage = report.coverage("cls_adapters")
    if adapter_coverage.transferred_tensors != 0 or adapter_coverage.transferred_parameter_elements != 0:
        raise RuntimeError("No standard YOLO26 tensor may be copied into E9 classification adapters.")
    _validate_gpsm_head(target_model, variant)

    if verbose:
        print_gpsm_transfer_report(report)
    return report


def gpsm_yaml_path(size: str, variant: str) -> Path:
    """Return the virtual scale-specific E9 YAML path."""
    if size not in MODEL_SIZES:
        raise ValueError(f"Unsupported YOLO26 scale {size!r}; expected one of n, s, m, l, x.")
    if variant not in GPSM_VARIANTS:
        raise ValueError(f"Unsupported GPSM variant {variant!r}; expected control or trainable.")
    model_directory = Path(__file__).resolve().parents[1] / "research" / "models"
    suffix = "gpsm-control" if variant == "control" else "gpsm"
    unified_yaml = model_directory / f"yolo26-{suffix}.yaml"
    if not unified_yaml.is_file():
        raise FileNotFoundError(f"E9 architecture YAML not found: {unified_yaml}")
    return model_directory / f"yolo26{size}-{suffix}.yaml"


def _validate_stock_architecture(target_model: nn.Module, size: str) -> None:
    """Ensure E9 changes only the final Detect module and its adapter arguments."""
    repository_root = Path(__file__).resolve().parents[1]
    stock = yaml_model_load(repository_root / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{size}.yaml")
    target = target_model.yaml
    for key in ("nc", "end2end", "reg_max", "scales", "backbone", "scale"):
        if target.get(key) != stock.get(key):
            raise RuntimeError(f"E9 architecture unexpectedly changes stock YOLO26 key {key!r}.")
    if target["head"][:-1] != stock["head"][:-1]:
        raise RuntimeError("E9 must retain every stock YOLO26 head layer before Detect.")
    if target["head"][-1][0:2] != stock["head"][-1][0:2]:
        raise RuntimeError("E9 must retain stock YOLO26 Detect inputs and repeat count.")
    if target["head"][-1][2] != "GeometryPreservingSpatialMorphologyDetect":
        raise RuntimeError("E9 YAML must replace only Detect with GeometryPreservingSpatialMorphologyDetect.")


def build_gpsm_yolo(size: str, variant: str, verbose: bool = False) -> ContrastRingYOLO:
    """Build E9 with fixed E2.1b supervision and standard pretrained YOLO26 weights."""
    architecture_yaml = gpsm_yaml_path(size, variant)
    model = ContrastRingYOLO(architecture_yaml, loss_config=E2_1B_CONFIG, verbose=verbose)
    if model.model.yaml.get("scale") != size:
        raise RuntimeError(f"Expected E9 scale {size!r}, received {model.model.yaml.get('scale')!r}.")
    if model.model.contrast_ring_loss_config != E2_1B_CONFIG:
        raise RuntimeError("E9 must use the fixed E2.1b loss configuration.")
    _validate_stock_architecture(model.model, size)
    _validate_gpsm_head(model.model, variant)
    _validate_criterion(model.model)

    pretrained = Path(f"yolo26{size}.pt")
    report = load_yolo26_gpsm_pretrained(model, pretrained, verbose=verbose)
    model.gpsm_variant = variant
    model.gpsm_transfer_report = report
    model.gpsm_pretrained = str(pretrained)
    model.ckpt_path = str(pretrained)
    model.ckpt = {"model": model.model}
    return model
