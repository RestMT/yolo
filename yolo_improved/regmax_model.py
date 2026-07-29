# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Construction and exact-shape pretrained transfer for E7 distributional box regression."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn

from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import load_checkpoint, yaml_model_load
from ultralytics.utils.loss import DFLoss, E2ELoss

from .contrast_ring_config import ContrastRingLossConfig
from .contrast_ring_loss import ContrastRingDetectionLoss, E2_LOCALIZATION_CONFIG
from .contrast_ring_model import ContrastRingDetectionModel, ContrastRingYOLO
from .residual_nwd_loss import ResidualNWDBboxLoss


MODEL_SIZES = ("n", "s", "m", "l", "x")
REG_MAX_VALUES = (4, 8)
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
class RegMaxTransferCoverage:
    """Pretrained parameter coverage for one E7 architecture region."""

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
class RegMaxTransferReport:
    """Exact-shape pretrained-transfer results for one E7 model."""

    source_checkpoint: str
    target_architecture: str
    target_reg_max: int
    exact_keys: tuple[str, ...]
    skipped_missing_keys: tuple[str, ...]
    skipped_shape_keys: tuple[str, ...]
    newly_initialized_keys: tuple[str, ...]
    transferred_parameter_elements: int
    target_parameter_elements: int
    coverages: tuple[RegMaxTransferCoverage, ...]

    @property
    def transfer_percentage(self) -> float:
        """Return target parameter-element coverage as a percentage."""
        return 100.0 * self.transferred_parameter_elements / self.target_parameter_elements

    def coverage(self, name: str) -> RegMaxTransferCoverage:
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
) -> RegMaxTransferCoverage:
    """Calculate tensor and parameter-element coverage for selected target keys."""
    target_keys = {key for key in target_state_dict if predicate(key)}
    transferred = target_keys & transferred_keys
    target_parameters = target_keys & parameter_keys
    transferred_parameters = transferred & parameter_keys
    return RegMaxTransferCoverage(
        name=name,
        transferred_tensors=len(transferred),
        target_tensors=len(target_keys),
        transferred_parameter_elements=sum(target_state_dict[key].numel() for key in transferred_parameters),
        target_parameter_elements=sum(target_state_dict[key].numel() for key in target_parameters),
    )


def remap_yolo26_regmax_state_dict(
    source_state_dict: Mapping[str, Tensor],
    target_state_dict: Mapping[str, Tensor],
    *,
    target_parameter_keys: Collection[str] | None = None,
    target_reg_max: int | None = None,
) -> tuple[dict[str, Tensor], RegMaxTransferReport]:
    """Select exact-name, exact-shape standard YOLO26 tensors for an E7 target.

    No tensor is cropped, padded, repeated, or interpolated.
    """
    parameter_keys = set(target_state_dict) if target_parameter_keys is None else set(target_parameter_keys)
    if target_reg_max is None:
        dfl_weight = target_state_dict.get("model.23.dfl.conv.weight")
        if dfl_weight is None or dfl_weight.ndim != 4 or dfl_weight.shape[0] != 1:
            raise ValueError("Cannot infer E7 reg_max from model.23.dfl.conv.weight.")
        target_reg_max = dfl_weight.shape[1]
    if target_reg_max not in REG_MAX_VALUES:
        raise ValueError(f"Unsupported E7 reg_max {target_reg_max!r}; expected 4 or 8.")

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
    newly_initialized_keys = tuple(sorted(set(target_state_dict) - transferred_keys))
    transferred_parameter_elements = sum(
        target_state_dict[key].numel() for key in transferred_keys & parameter_keys
    )
    target_parameter_elements = sum(target_state_dict[key].numel() for key in parameter_keys)
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
            "classification heads",
            target_state_dict,
            transferred_keys,
            parameter_keys,
            lambda key: key.startswith(("model.23.cv3.", "model.23.one2one_cv3.")),
        ),
        _coverage(
            "one-to-many box heads",
            target_state_dict,
            transferred_keys,
            parameter_keys,
            lambda key: key.startswith("model.23.cv2."),
        ),
        _coverage(
            "one-to-one box heads",
            target_state_dict,
            transferred_keys,
            parameter_keys,
            lambda key: key.startswith("model.23.one2one_cv2."),
        ),
    )
    return transferred, RegMaxTransferReport(
        source_checkpoint="<state_dict>",
        target_architecture="<state_dict>",
        target_reg_max=target_reg_max,
        exact_keys=tuple(sorted(transferred)),
        skipped_missing_keys=tuple(sorted(skipped_missing_keys)),
        skipped_shape_keys=tuple(sorted(skipped_shape_keys)),
        newly_initialized_keys=newly_initialized_keys,
        transferred_parameter_elements=transferred_parameter_elements,
        target_parameter_elements=target_parameter_elements,
        coverages=coverages,
    )


def print_regmax_transfer_report(report: RegMaxTransferReport) -> None:
    """Print the required E7 pretrained-transfer summary and regional coverage."""
    print(f"Source checkpoint: {report.source_checkpoint}")
    print(f"Target architecture: {report.target_architecture}")
    print(f"Target reg_max: {report.target_reg_max}")
    print(f"Exact transferred tensors: {len(report.exact_keys)}")
    print(f"Transferred parameter elements: {report.transferred_parameter_elements}")
    print(f"Skipped missing keys: {len(report.skipped_missing_keys)}")
    print(f"Skipped shape mismatches: {len(report.skipped_shape_keys)}")
    print(f"Newly initialized tensors: {len(report.newly_initialized_keys)}")
    print(f"Transfer percentage: {report.transfer_percentage:.6f}%")
    print("Transfer coverage:")
    for coverage in report.coverages:
        print(
            f"  {coverage.name}: {coverage.transferred_tensors}/{coverage.target_tensors} tensors, "
            f"{coverage.percentage:.6f}% parameter elements"
        )


def _validate_detect(model: nn.Module, reg_max: int) -> Detect:
    """Validate E7 Detect topology, DFL decoding, and both box heads."""
    detect = model.model[-1]
    if not isinstance(detect, Detect):
        raise RuntimeError("E7 must retain the standard Detect module.")
    if detect.f != [16, 19, 22] or detect.nl != 3:
        raise RuntimeError("E7 Detect must receive exactly P3/8, P4/16, and P5/32.")
    if detect.stride.tolist() != [8.0, 16.0, 32.0]:
        raise RuntimeError(f"E7 Detect strides must be [8, 16, 32], got {detect.stride.tolist()}.")
    if detect.reg_max != reg_max or detect.no != detect.nc + 4 * reg_max:
        raise RuntimeError("E7 Detect does not expose the requested distributional regression width.")
    if isinstance(detect.dfl, nn.Identity) or getattr(detect.dfl, "c1", None) != reg_max:
        raise RuntimeError("E7 must use the standard DFL decoder.")

    for name in ("cv2", "one2one_cv2"):
        head = getattr(detect, name, None)
        if head is None or len(head) != 3:
            raise RuntimeError(f"E7 Detect must contain three {name} levels.")
        if any(level[-1].out_channels != 4 * reg_max for level in head):
            raise RuntimeError(f"Every {name} output must have {4 * reg_max} regression channels.")
    return detect


def _validate_criterion(model: ContrastRingDetectionModel, reg_max: int) -> None:
    """Verify that both E2.1b end-to-end branches use stock DFL rather than normalized L1."""
    criterion = model.init_criterion()
    if not isinstance(criterion, E2ELoss):
        raise RuntimeError("E7 must retain the standard end-to-end loss wrapper.")
    for name, branch in (("one-to-many", criterion.one2many), ("one-to-one", criterion.one2one)):
        if not isinstance(branch, ContrastRingDetectionLoss):
            raise RuntimeError(f"E7 {name} must reuse ContrastRingDetectionLoss.")
        if branch.reg_max != reg_max or not branch.use_dfl:
            raise RuntimeError(f"E7 {name} did not activate distributional box regression.")
        if not isinstance(branch.bbox_loss, ResidualNWDBboxLoss):
            raise RuntimeError(f"E7 {name} must retain the E1.1 residual NWD box criterion.")
        if not isinstance(branch.bbox_loss.dfl_loss, DFLoss):
            raise RuntimeError(f"E7 {name} must use the stock DFLoss instead of normalized L1.")
        if branch.bbox_loss.dfl_loss.reg_max != reg_max:
            raise RuntimeError(f"E7 {name} DFLoss has the wrong reg_max.")
        if branch.bbox_loss.config != E2_LOCALIZATION_CONFIG:
            raise RuntimeError(f"E7 {name} changed the E1.1 localization configuration.")
        if branch.contrast_ring_config != E2_1B_CONFIG:
            raise RuntimeError(f"E7 {name} changed the E2.1b classification configuration.")


def _final_box_output_keys(detect: Detect) -> set[str]:
    """Return final one-to-many and one-to-one regression output parameter keys."""
    return {
        f"model.23.{head_name}.{level_index}.{len(level) - 1}.{parameter_name}"
        for head_name in ("cv2", "one2one_cv2")
        for level_index, level in enumerate(getattr(detect, head_name))
        for parameter_name in ("weight", "bias")
    }


def load_yolo26_regmax_pretrained(
    target: ContrastRingYOLO | nn.Module,
    pretrained: str | Path,
    *,
    verbose: bool = True,
) -> RegMaxTransferReport:
    """Load only exact-shape standard YOLO26 tensors into an E7 target."""
    target_model = _underlying_model(target)
    target_detect = target_model.model[-1]
    if not isinstance(target_detect, Detect) or target_detect.reg_max not in REG_MAX_VALUES:
        raise ValueError("Target must be an E7 Detect model with reg_max 4 or 8.")
    _validate_detect(target_model, target_detect.reg_max)

    source_model, _ = load_checkpoint(pretrained, device="cpu")
    source_detect = source_model.model[-1]
    if not isinstance(source_detect, Detect) or source_detect.reg_max != 1:
        raise ValueError("Pretrained source must be a standard YOLO26 checkpoint with reg_max=1.")
    if source_detect.f != [16, 19, 22] or source_detect.nl != 3:
        raise ValueError("Pretrained source does not have the standard YOLO26 P3/P4/P5 Detect topology.")
    if source_model.yaml.get("scale") != target_model.yaml.get("scale"):
        raise ValueError(
            f"Checkpoint scale {source_model.yaml.get('scale')!r} does not match "
            f"E7 target scale {target_model.yaml.get('scale')!r}."
        )

    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    transferred, report = remap_yolo26_regmax_state_dict(
        source_state_dict,
        target_state_dict,
        target_parameter_keys=dict(target_model.named_parameters()),
        target_reg_max=target_detect.reg_max,
    )
    target_architecture = Path(target_model.yaml["yaml_file"]).with_name(
        f"yolo26-regmax{target_detect.reg_max}.yaml"
    )
    report = replace(
        report,
        source_checkpoint=str(pretrained),
        target_architecture=str(target_architecture),
    )
    preserved_initialization = {
        key: target_state_dict[key].detach().clone() for key in report.newly_initialized_keys
    }

    incompatible = target_model.load_state_dict(transferred, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected E7 transfer keys: {incompatible.unexpected_keys}.")
    reported_missing = set(report.newly_initialized_keys)
    actual_missing = set(incompatible.missing_keys)
    unreported_missing = actual_missing - reported_missing
    if unreported_missing:
        raise RuntimeError(f"load_state_dict reported untracked E7 target keys: {sorted(unreported_missing)}.")
    silently_initialized = reported_missing - actual_missing
    if any(not key.endswith(".num_batches_tracked") for key in silently_initialized):
        raise RuntimeError("E7 transfer report does not match load_state_dict missing keys.")

    loaded_state_dict = target_model.state_dict()
    for key, initial_tensor in preserved_initialization.items():
        if not torch.equal(loaded_state_dict[key], initial_tensor):
            raise RuntimeError(f"New E7 tensor {key!r} was modified during pretrained transfer.")

    transferred_keys = set(report.exact_keys)
    for coverage_name in ("backbone", "neck", "classification heads"):
        coverage = report.coverage(coverage_name)
        if coverage.transferred_tensors != coverage.target_tensors or coverage.percentage != 100.0:
            raise RuntimeError(f"Pretrained transfer is incomplete for E7 {coverage_name}.")

    compatible_box_keys = {
        key
        for key, source_tensor in source_state_dict.items()
        if key.startswith(("model.23.cv2.", "model.23.one2one_cv2."))
        and key in target_state_dict
        and source_tensor.shape == target_state_dict[key].shape
    }
    if not compatible_box_keys <= transferred_keys:
        raise RuntimeError("Compatible E7 box-tower parameters were not fully transferred.")
    skipped_shape_keys = set(report.skipped_shape_keys)
    final_output_keys = _final_box_output_keys(target_detect)
    if not final_output_keys <= skipped_shape_keys:
        raise RuntimeError("E7 distributional output weights or biases were copied from reg_max=1.")
    if any(
        not key.startswith(("model.23.cv2.", "model.23.one2one_cv2."))
        for key in skipped_shape_keys
    ):
        raise RuntimeError("A non-regression tensor unexpectedly changed shape in E7.")

    dfl_weight_key = "model.23.dfl.conv.weight"
    if dfl_weight_key not in report.newly_initialized_keys:
        raise RuntimeError("The E7 DFL projection must use its own standard initialization.")
    expected_dfl = torch.arange(
        target_detect.reg_max,
        dtype=loaded_state_dict[dfl_weight_key].dtype,
        device=loaded_state_dict[dfl_weight_key].device,
    )
    if not torch.equal(loaded_state_dict[dfl_weight_key].flatten(), expected_dfl):
        raise RuntimeError("The E7 DFL projection does not have the standard bin initialization.")

    if verbose:
        print_regmax_transfer_report(report)
    return report


def regmax_yaml_path(size: str, reg_max: int) -> Path:
    """Return the virtual scale-specific E7 YAML path."""
    if size not in MODEL_SIZES:
        raise ValueError(f"Unsupported YOLO26 scale {size!r}; expected one of n, s, m, l, x.")
    if not isinstance(reg_max, int) or isinstance(reg_max, bool) or reg_max not in REG_MAX_VALUES:
        raise ValueError(f"Unsupported E7 reg_max {reg_max!r}; expected 4 or 8.")
    model_directory = Path(__file__).resolve().parents[1] / "research" / "models"
    unified_yaml = model_directory / f"yolo26-regmax{reg_max}.yaml"
    if not unified_yaml.is_file():
        raise FileNotFoundError(f"E7 architecture YAML not found: {unified_yaml}")
    return model_directory / f"yolo26{size}-regmax{reg_max}.yaml"


def _validate_stock_architecture(target_model: nn.Module, size: str, reg_max: int) -> None:
    """Ensure that the E7 YAML differs from stock YOLO26 only by reg_max."""
    repository_root = Path(__file__).resolve().parents[1]
    stock = yaml_model_load(repository_root / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{size}.yaml")
    target = target_model.yaml
    for key in ("nc", "end2end", "scales", "backbone", "head", "scale"):
        if target.get(key) != stock.get(key):
            raise RuntimeError(f"E7 architecture unexpectedly changes stock YOLO26 key {key!r}.")
    if stock.get("reg_max") != 1 or target.get("reg_max") != reg_max:
        raise RuntimeError("E7 architecture must change only stock reg_max=1 to the requested value.")


def build_regmax_yolo(size: str, reg_max: int, verbose: bool = False) -> ContrastRingYOLO:
    """Build E7 with fixed E2.1b supervision and standard pretrained YOLO26 weights."""
    architecture_yaml = regmax_yaml_path(size, reg_max)
    model = ContrastRingYOLO(architecture_yaml, loss_config=E2_1B_CONFIG, verbose=verbose)
    if model.model.yaml.get("scale") != size:
        raise RuntimeError(f"Expected E7 scale {size!r}, received {model.model.yaml.get('scale')!r}.")
    if model.model.contrast_ring_loss_config != E2_1B_CONFIG:
        raise RuntimeError("E7 must use the fixed E2.1b loss configuration.")
    _validate_stock_architecture(model.model, size, reg_max)
    _validate_detect(model.model, reg_max)
    _validate_criterion(model.model, reg_max)

    pretrained = Path(f"yolo26{size}.pt")
    report = load_yolo26_regmax_pretrained(model, pretrained, verbose=verbose)
    model.reg_max = reg_max
    model.regmax_transfer_report = report
    model.regmax_pretrained = str(pretrained)
    model.ckpt_path = str(pretrained)
    model.ckpt = {"model": model.model}
    return model
