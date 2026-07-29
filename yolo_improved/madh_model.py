# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Construction and exact pretrained transfer for E8 Morphology-Adaptive Decoupled Head."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn

from ultralytics.nn.modules import Detect, MorphologyAdaptiveAdapter, MorphologyAdaptiveDetect
from ultralytics.nn.tasks import load_checkpoint, yaml_model_load
from ultralytics.utils.loss import E2ELoss

from .contrast_ring_config import ContrastRingLossConfig
from .contrast_ring_loss import ContrastRingDetectionLoss, E2_LOCALIZATION_CONFIG
from .contrast_ring_model import ContrastRingDetectionModel, ContrastRingYOLO
from .residual_nwd_loss import ResidualNWDBboxLoss


MADH_VARIANTS = ("control", "trainable")
MODEL_SIZES = ("n", "s", "m", "l", "x")
E2_1B_CONFIG = ContrastRingLossConfig(
    inner_kernel=3,
    outer_kernel=7,
    contrast_tau=0.25,
    positive_gain=0.25,
    negative_gain=0.25,
    negative_gamma=3.0,
    eps=1e-6,
)
_ADAPTER_PREFIXES = (
    "model.23.box_adapters.",
    "model.23.cls_adapters.",
    "model.23.one2one_box_adapters.",
    "model.23.one2one_cls_adapters.",
)


@dataclass(frozen=True)
class MADHTransferCoverage:
    """Pretrained parameter coverage for one E8 architecture region."""

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
class MADHTransferReport:
    """Exact-key pretrained-transfer results for one E8 model."""

    source_checkpoint: str
    target_architecture: str
    exact_keys: tuple[str, ...]
    skipped_missing_keys: tuple[str, ...]
    skipped_shape_keys: tuple[str, ...]
    new_adapter_keys: tuple[str, ...]
    transferred_parameter_elements: int
    target_parameter_elements: int
    new_adapter_parameter_elements: int
    coverages: tuple[MADHTransferCoverage, ...]

    @property
    def transfer_percentage(self) -> float:
        """Return target parameter-element coverage as a percentage."""
        return 100.0 * self.transferred_parameter_elements / self.target_parameter_elements

    def coverage(self, name: str) -> MADHTransferCoverage:
        """Return one named architecture-region coverage record."""
        try:
            return next(item for item in self.coverages if item.name == name)
        except StopIteration as error:
            raise KeyError(name) from error


def _underlying_model(model: ContrastRingYOLO | nn.Module) -> nn.Module:
    """Return the PyTorch detection model from a facade or direct module."""
    return model.model if isinstance(model, ContrastRingYOLO) else model


def _is_layer_range(key: str, start: int, stop: int) -> bool:
    """Return whether a state key belongs to a model layer in [start, stop)."""
    parts = key.split(".", 2)
    return len(parts) > 2 and parts[0] == "model" and parts[1].isdigit() and start <= int(parts[1]) < stop


def _coverage(
    name: str,
    target_state_dict: Mapping[str, Tensor],
    transferred_keys: set[str],
    parameter_keys: set[str],
    predicate,
) -> MADHTransferCoverage:
    """Calculate tensor and parameter-element coverage for selected target keys."""
    target_keys = {key for key in target_state_dict if predicate(key)}
    transferred = target_keys & transferred_keys
    target_parameters = target_keys & parameter_keys
    transferred_parameters = transferred & parameter_keys
    return MADHTransferCoverage(
        name=name,
        transferred_tensors=len(transferred),
        target_tensors=len(target_keys),
        transferred_parameter_elements=sum(target_state_dict[key].numel() for key in transferred_parameters),
        target_parameter_elements=sum(target_state_dict[key].numel() for key in target_parameters),
    )


def remap_yolo26_madh_state_dict(
    source_state_dict: Mapping[str, Tensor],
    target_state_dict: Mapping[str, Tensor],
    *,
    target_parameter_keys: Collection[str] | None = None,
) -> tuple[dict[str, Tensor], MADHTransferReport]:
    """Select exact-name, exact-shape standard YOLO26 tensors for an E8 target."""
    parameter_keys = set(target_state_dict) if target_parameter_keys is None else set(target_parameter_keys)
    transferred: dict[str, Tensor] = {}
    skipped_missing_keys: list[str] = []
    skipped_shape_keys: list[str] = []
    for key, source_tensor in source_state_dict.items():
        target_tensor = target_state_dict.get(key)
        if target_tensor is None:
            skipped_missing_keys.append(key)
        elif source_tensor.shape != target_tensor.shape:
            skipped_shape_keys.append(key)
        else:
            transferred[key] = source_tensor

    transferred_keys = set(transferred)
    new_adapter_keys = tuple(sorted(set(target_state_dict) - transferred_keys))
    transferred_parameter_elements = sum(
        target_state_dict[key].numel() for key in transferred_keys & parameter_keys
    )
    target_parameter_elements = sum(target_state_dict[key].numel() for key in parameter_keys)
    adapter_parameter_keys = {key for key in parameter_keys if key.startswith(_ADAPTER_PREFIXES)}
    coverages = (
        _coverage(
            "backbone",
            target_state_dict,
            transferred_keys,
            parameter_keys,
            lambda key: _is_layer_range(key, 0, 11),
        ),
        _coverage(
            "neck",
            target_state_dict,
            transferred_keys,
            parameter_keys,
            lambda key: _is_layer_range(key, 11, 23),
        ),
        _coverage(
            "cv2",
            target_state_dict,
            transferred_keys,
            parameter_keys,
            lambda key: key.startswith("model.23.cv2."),
        ),
        _coverage(
            "cv3",
            target_state_dict,
            transferred_keys,
            parameter_keys,
            lambda key: key.startswith("model.23.cv3."),
        ),
        _coverage(
            "one2one_cv2",
            target_state_dict,
            transferred_keys,
            parameter_keys,
            lambda key: key.startswith("model.23.one2one_cv2."),
        ),
        _coverage(
            "one2one_cv3",
            target_state_dict,
            transferred_keys,
            parameter_keys,
            lambda key: key.startswith("model.23.one2one_cv3."),
        ),
        _coverage(
            "adapters",
            target_state_dict,
            transferred_keys,
            parameter_keys,
            lambda key: key.startswith(_ADAPTER_PREFIXES),
        ),
    )
    return transferred, MADHTransferReport(
        source_checkpoint="<state_dict>",
        target_architecture="<state_dict>",
        exact_keys=tuple(sorted(transferred)),
        skipped_missing_keys=tuple(sorted(skipped_missing_keys)),
        skipped_shape_keys=tuple(sorted(skipped_shape_keys)),
        new_adapter_keys=new_adapter_keys,
        transferred_parameter_elements=transferred_parameter_elements,
        target_parameter_elements=target_parameter_elements,
        new_adapter_parameter_elements=sum(target_state_dict[key].numel() for key in adapter_parameter_keys),
        coverages=coverages,
    )


def print_madh_transfer_report(report: MADHTransferReport) -> None:
    """Print the required E8 pretrained-transfer summary and regional coverage."""
    print(f"Exact transferred tensors: {len(report.exact_keys)}")
    print(f"Transferred parameter elements: {report.transferred_parameter_elements}")
    print(f"Skipped shape mismatches: {len(report.skipped_shape_keys)}")
    print(
        f"New adapter tensors: {len(report.new_adapter_keys)} tensors, "
        f"{report.new_adapter_parameter_elements} parameter elements"
    )
    print(f"Transfer percentage: {report.transfer_percentage:.6f}%")
    print("Transfer coverage:")
    for coverage in report.coverages:
        print(
            f"  {coverage.name}: {coverage.transferred_tensors}/{coverage.target_tensors} tensors, "
            f"{coverage.percentage:.6f}% parameter elements"
        )


def _all_adapters(head: MorphologyAdaptiveDetect):
    """Yield branch, level, and adapter for all twelve independent E8 adapters."""
    levels = ("P3/8", "P4/16", "P5/32")
    for assignment, box_name, cls_name in (
        ("one-to-many", "box_adapters", "cls_adapters"),
        ("one-to-one", "one2one_box_adapters", "one2one_cls_adapters"),
    ):
        for task, attribute in (("box", box_name), ("cls", cls_name)):
            adapters = getattr(head, attribute)
            for level, adapter in zip(levels, adapters):
                yield assignment, task, level, adapter


def collect_madh_gate_values(model: ContrastRingYOLO | nn.Module) -> tuple[dict[str, str | float], ...]:
    """Return JSON-ready gate_raw and alpha values for every E8 adapter."""
    underlying = _underlying_model(model)
    head = underlying.model[-1]
    if not isinstance(head, MorphologyAdaptiveDetect):
        raise ValueError("Model does not contain MorphologyAdaptiveDetect.")
    return tuple(
        {
            "assignment": assignment,
            "task": task,
            "detection_level": level,
            "gate_raw": float(adapter.gate_raw.detach().float().cpu().item()),
            "alpha": float(
                (adapter.gate_max * torch.tanh(adapter.gate_raw.detach().float())).cpu().item()
            ),
        }
        for assignment, task, level, adapter in _all_adapters(head)
    )


def _validate_madh_head(model: nn.Module, variant: str) -> MorphologyAdaptiveDetect:
    """Validate E8 topology, fixed reg_max, and independent adapter state."""
    head = model.model[-1]
    if not isinstance(head, MorphologyAdaptiveDetect):
        raise RuntimeError("E8 must use MorphologyAdaptiveDetect.")
    if head.f != [16, 19, 22] or head.nl != 3:
        raise RuntimeError("E8 must retain exactly the P3/8, P4/16, and P5/32 levels.")
    if head.stride.tolist() != [8.0, 16.0, 32.0]:
        raise RuntimeError(f"E8 Detect strides must be [8, 16, 32], got {head.stride.tolist()}.")
    if head.reg_max != 1 or head.no != head.nc + 4 or not isinstance(head.dfl, nn.Identity):
        raise RuntimeError("E8 must retain stock YOLO26 reg_max=1 without DFL.")

    trainable = variant == "trainable"
    adapter_groups = (
        head.box_adapters,
        head.cls_adapters,
        head.one2one_box_adapters,
        head.one2one_cls_adapters,
    )
    if any(len(group) != 3 for group in adapter_groups):
        raise RuntimeError("E8 must contain independent adapters for all four branches and three levels.")
    for group in adapter_groups:
        for adapter in group:
            if not isinstance(adapter, MorphologyAdaptiveAdapter) or adapter.trainable_adapter != trainable:
                raise RuntimeError(f"E8 adapter does not implement the requested {variant!r} variant.")
            if adapter.gate_max != 0.5 or torch.count_nonzero(adapter.gate_raw.detach()).item() != 0:
                raise RuntimeError("Every E8 adapter must start with gate_max=0.5 and a zero gate.")
            if torch.count_nonzero(adapter.router.weight.detach()).item() != 0:
                raise RuntimeError("Every E8 router weight must start at zero.")
            if torch.count_nonzero(adapter.router.bias.detach()).item() != 0:
                raise RuntimeError("Every E8 router bias must start at zero.")
            if any(parameter.requires_grad != trainable for parameter in adapter.parameters()):
                raise RuntimeError(f"E8 adapter gradients do not match the requested {variant!r} variant.")

    for first, second in (
        (head.box_adapters, head.cls_adapters),
        (head.box_adapters, head.one2one_box_adapters),
        (head.cls_adapters, head.one2one_cls_adapters),
    ):
        for first_adapter, second_adapter in zip(first, second):
            if any(
                first_parameter.data_ptr() == second_parameter.data_ptr()
                for first_parameter, second_parameter in zip(
                    first_adapter.parameters(), second_adapter.parameters()
                )
            ):
                raise RuntimeError("E8 box/class and one-to-many/one-to-one adapters must be independent.")
    return head


def _validate_criterion(model: ContrastRingDetectionModel) -> None:
    """Verify unchanged E2.1b supervision and the reg_max=1 L1 branch."""
    criterion = model.init_criterion()
    if not isinstance(criterion, E2ELoss):
        raise RuntimeError("E8 must retain the stock end-to-end loss wrapper.")
    for name, branch in (("one-to-many", criterion.one2many), ("one-to-one", criterion.one2one)):
        if not isinstance(branch, ContrastRingDetectionLoss):
            raise RuntimeError(f"E8 {name} must reuse ContrastRingDetectionLoss.")
        if branch.reg_max != 1 or branch.use_dfl or branch.bbox_loss.dfl_loss is not None:
            raise RuntimeError(f"E8 {name} must retain stock YOLO26 normalized L1 regression.")
        if not isinstance(branch.bbox_loss, ResidualNWDBboxLoss):
            raise RuntimeError(f"E8 {name} must retain the E1.1 residual NWD box criterion.")
        if branch.bbox_loss.config != E2_LOCALIZATION_CONFIG:
            raise RuntimeError(f"E8 {name} changed the E1.1 localization configuration.")
        if branch.contrast_ring_config != E2_1B_CONFIG:
            raise RuntimeError(f"E8 {name} changed the E2.1b classification configuration.")


def load_yolo26_madh_pretrained(
    target: ContrastRingYOLO | nn.Module,
    pretrained: str | Path,
    *,
    verbose: bool = True,
) -> MADHTransferReport:
    """Load every standard YOLO26 tensor while preserving all new E8 adapter state."""
    target_model = _underlying_model(target)
    target_head = target_model.model[-1]
    if not isinstance(target_head, MorphologyAdaptiveDetect):
        raise ValueError("Target must contain MorphologyAdaptiveDetect.")

    source_model, _ = load_checkpoint(pretrained, device="cpu")
    source_head = source_model.model[-1]
    if type(source_head) is not Detect or source_head.reg_max != 1:
        raise ValueError("Pretrained source must be a standard YOLO26 Detect checkpoint with reg_max=1.")
    if source_head.f != [16, 19, 22] or source_head.nl != 3:
        raise ValueError("Pretrained source does not have the stock YOLO26 P3/P4/P5 topology.")
    if source_model.yaml.get("scale") != target_model.yaml.get("scale"):
        raise ValueError(
            f"Checkpoint scale {source_model.yaml.get('scale')!r} does not match "
            f"E8 target scale {target_model.yaml.get('scale')!r}."
        )

    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    transferred, report = remap_yolo26_madh_state_dict(
        source_state_dict,
        target_state_dict,
        target_parameter_keys=dict(target_model.named_parameters()),
    )
    variant = "trainable" if target_head.box_adapters[0].trainable_adapter else "control"
    suffix = "madh" if variant == "trainable" else "madh-control"
    target_architecture = Path(target_model.yaml["yaml_file"]).with_name(f"yolo26-{suffix}.yaml")
    report = replace(
        report,
        source_checkpoint=str(pretrained),
        target_architecture=str(target_architecture),
    )
    initial_adapter_state = {
        key: target_state_dict[key].detach().clone() for key in report.new_adapter_keys
    }

    incompatible = target_model.load_state_dict(transferred, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected E8 transfer keys: {incompatible.unexpected_keys}.")
    reported_missing = set(report.new_adapter_keys)
    actual_missing = set(incompatible.missing_keys)
    if actual_missing - reported_missing:
        raise RuntimeError(f"load_state_dict reported untracked E8 keys: {sorted(actual_missing - reported_missing)}.")
    silently_initialized = reported_missing - actual_missing
    if any(not key.endswith(".num_batches_tracked") for key in silently_initialized):
        raise RuntimeError("E8 transfer report does not match load_state_dict missing keys.")

    loaded_state_dict = target_model.state_dict()
    for key, initial_tensor in initial_adapter_state.items():
        if not torch.equal(loaded_state_dict[key], initial_tensor):
            raise RuntimeError(f"New E8 adapter tensor {key!r} was modified during pretrained transfer.")
    if report.skipped_missing_keys or report.skipped_shape_keys:
        raise RuntimeError("Stock YOLO26 and E8 standard tensors must match by exact key and shape.")
    if not report.new_adapter_keys or any(
        not key.startswith(_ADAPTER_PREFIXES) for key in report.new_adapter_keys
    ):
        raise RuntimeError("Only E8 adapter tensors may remain newly initialized.")
    for coverage_name in ("backbone", "neck", "cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        coverage = report.coverage(coverage_name)
        if coverage.transferred_tensors != coverage.target_tensors or coverage.percentage != 100.0:
            raise RuntimeError(f"Pretrained transfer is incomplete for E8 {coverage_name}.")
    adapter_coverage = report.coverage("adapters")
    if adapter_coverage.transferred_tensors != 0 or adapter_coverage.transferred_parameter_elements != 0:
        raise RuntimeError("No standard YOLO26 tensor may be copied into E8 adapters.")
    _validate_madh_head(target_model, variant)

    if verbose:
        print_madh_transfer_report(report)
    return report


def madh_yaml_path(size: str, variant: str) -> Path:
    """Return the virtual scale-specific E8 YAML path."""
    if size not in MODEL_SIZES:
        raise ValueError(f"Unsupported YOLO26 scale {size!r}; expected one of n, s, m, l, x.")
    if variant not in MADH_VARIANTS:
        raise ValueError(f"Unsupported MADH variant {variant!r}; expected control or trainable.")
    model_directory = Path(__file__).resolve().parents[1] / "research" / "models"
    suffix = "madh-control" if variant == "control" else "madh"
    unified_yaml = model_directory / f"yolo26-{suffix}.yaml"
    if not unified_yaml.is_file():
        raise FileNotFoundError(f"E8 architecture YAML not found: {unified_yaml}")
    return model_directory / f"yolo26{size}-{suffix}.yaml"


def _validate_stock_architecture(target_model: nn.Module, size: str) -> None:
    """Ensure E8 changes only the final Detect module and its adapter arguments."""
    repository_root = Path(__file__).resolve().parents[1]
    stock = yaml_model_load(repository_root / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{size}.yaml")
    target = target_model.yaml
    for key in ("nc", "end2end", "reg_max", "scales", "backbone", "scale"):
        if target.get(key) != stock.get(key):
            raise RuntimeError(f"E8 architecture unexpectedly changes stock YOLO26 key {key!r}.")
    if target["head"][:-1] != stock["head"][:-1]:
        raise RuntimeError("E8 must retain every stock YOLO26 head layer before Detect.")
    if target["head"][-1][0:2] != stock["head"][-1][0:2]:
        raise RuntimeError("E8 must retain stock YOLO26 Detect inputs and repeat count.")
    if target["head"][-1][2] != "MorphologyAdaptiveDetect":
        raise RuntimeError("E8 YAML must replace only Detect with MorphologyAdaptiveDetect.")


def build_madh_yolo(size: str, variant: str, verbose: bool = False) -> ContrastRingYOLO:
    """Build E8 with fixed E2.1b supervision and standard pretrained YOLO26 weights."""
    architecture_yaml = madh_yaml_path(size, variant)
    model = ContrastRingYOLO(architecture_yaml, loss_config=E2_1B_CONFIG, verbose=verbose)
    if model.model.yaml.get("scale") != size:
        raise RuntimeError(f"Expected E8 scale {size!r}, received {model.model.yaml.get('scale')!r}.")
    if model.model.contrast_ring_loss_config != E2_1B_CONFIG:
        raise RuntimeError("E8 must use the fixed E2.1b loss configuration.")
    _validate_stock_architecture(model.model, size)
    _validate_madh_head(model.model, variant)
    _validate_criterion(model.model)

    pretrained = Path(f"yolo26{size}.pt")
    report = load_yolo26_madh_pretrained(model, pretrained, verbose=verbose)
    model.madh_variant = variant
    model.madh_transfer_report = report
    model.madh_pretrained = str(pretrained)
    model.ckpt_path = str(pretrained)
    model.ckpt = {"model": model.model}
    return model
