# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""P2-guided Detail Injection construction and pretrained-weight transfer for E5."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor, nn

from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import load_checkpoint

from .contrast_ring_model import ContrastRingYOLO
from .mutual_distillation_config import E2_1B_CONFIG


@dataclass(frozen=True)
class P2DetailInjectionTransferCoverage:
    """Transfer coverage for one target-model region."""

    name: str
    transferred_tensors: int
    total_tensors: int
    transferred_parameter_elements: int
    total_parameter_elements: int


@dataclass(frozen=True)
class P2DetailInjectionTransferReport:
    """Exact-match transfer results for a P2 Detail Injection model."""

    transferred_keys: tuple[str, ...]
    skipped_absent_keys: tuple[str, ...]
    skipped_shape_keys: tuple[str, ...]
    newly_initialized_keys: tuple[str, ...]
    transferred_parameter_elements: int
    coverage: tuple[P2DetailInjectionTransferCoverage, ...]

    @property
    def transferred_tensors(self) -> int:
        """Return the number of transferred state tensors."""
        return len(self.transferred_keys)


def _layer_index(key: str) -> int:
    """Extract the integer layer index from a YOLO state-dict key."""
    match = re.fullmatch(r"model\.(\d+)\..+", key)
    if match is None:
        raise ValueError(f"Expected a state-dict key of the form 'model.<index>.<path>', received {key!r}.")
    return int(match.group(1))


def _validated_layer_index_map(layer_index_map: Mapping[int, int] | None) -> dict[int, int]:
    """Return the required stock-YOLO26 to E5 layer map and reject any deviation."""
    expected = {**{index: index for index in range(15)}, 16: 17, **{index: index + 1 for index in range(17, 24)}}
    if layer_index_map is None:
        return expected

    provided = dict(layer_index_map)
    if provided != expected:
        raise ValueError(
            "Invalid YOLO26 -> E5 layer mapping. Expected stock 0-14 -> E5 0-14, "
            "stock 16 -> E5 17, and stock 17-23 -> E5 18-24."
        )
    return provided


def remap_yolo26_p2di_state_dict(
    source_state_dict: Mapping[str, Tensor],
    target_state_dict: Mapping[str, Tensor],
    *,
    target_parameter_keys: Collection[str] | None = None,
    layer_index_map: Mapping[int, int] | None = None,
) -> tuple[dict[str, Tensor], P2DetailInjectionTransferReport]:
    """Remap compatible stock YOLO26 tensors to E5 using exact key and shape matches only.

    Args:
        source_state_dict (Mapping[str, Tensor]): State dictionary from a standard YOLO26 model.
        target_state_dict (Mapping[str, Tensor]): State dictionary from the E5 target model.
        target_parameter_keys (Collection[str] | None): Target parameter keys used for element counts.
        layer_index_map (Mapping[int, int] | None): Optional map accepted only when it exactly matches E5.

    Returns:
        tuple[dict[str, Tensor], P2DetailInjectionTransferReport]: Remapped tensors and transfer report.
    """
    index_map = _validated_layer_index_map(layer_index_map)
    source_layers = {_layer_index(key) for key in source_state_dict}
    target_layers = {_layer_index(key) for key in target_state_dict}
    required_source_layers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 17, 19, 20, 22, 23}
    required_target_layers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 17, 18, 20, 21, 23, 24}
    if missing := sorted(required_source_layers - source_layers):
        raise ValueError(f"Source state dict is not a standard YOLO26 model; missing parameterized layers: {missing}.")
    if missing := sorted(required_target_layers - target_layers):
        raise ValueError(f"Target state dict is not an E5 P2DI model; missing parameterized layers: {missing}.")
    if 15 in source_layers:
        raise ValueError(
            "Stock YOLO26 layer 15 must be a parameterless Concat; refusing to map it to E5 P2 projection."
        )
    if 16 in target_layers:
        raise ValueError("E5 layer 16 must be a parameterless Concat.")
    if unsupported := sorted(source_layers - set(index_map)):
        raise ValueError(f"Source state dict contains unmapped parameterized layers: {unsupported}.")

    parameter_keys = set(target_state_dict) if target_parameter_keys is None else set(target_parameter_keys)
    remapped: dict[str, Tensor] = {}
    skipped_absent: list[str] = []
    skipped_shape: list[str] = []
    for source_key, source_tensor in source_state_dict.items():
        source_index = _layer_index(source_key)
        target_index = index_map[source_index]
        target_key = re.sub(r"^model\.\d+\.", f"model.{target_index}.", source_key, count=1)
        target_tensor = target_state_dict.get(target_key)
        if target_tensor is None:
            skipped_absent.append(target_key)
        elif source_tensor.shape != target_tensor.shape:
            skipped_shape.append(target_key)
        else:
            remapped[target_key] = source_tensor

    transferred_keys = tuple(sorted(remapped))
    newly_initialized_keys = tuple(sorted(set(target_state_dict) - set(remapped)))
    transferred_parameter_elements = sum(
        target_state_dict[key].numel() for key in transferred_keys if key in parameter_keys
    )
    regions = (
        ("backbone 0-10", range(0, 11)),
        ("early neck 11-14", range(11, 15)),
        ("modified P3 fusion 15-17", range(15, 18)),
        ("lower PAN path 18-23", range(18, 24)),
        ("Detect 24", range(24, 25)),
    )
    coverage = []
    for name, indices in regions:
        region_indices = set(indices)
        region_keys = {key for key in target_state_dict if _layer_index(key) in region_indices}
        region_parameter_keys = region_keys & parameter_keys
        transferred_region_keys = region_keys & set(remapped)
        coverage.append(
            P2DetailInjectionTransferCoverage(
                name=name,
                transferred_tensors=len(transferred_region_keys),
                total_tensors=len(region_keys),
                transferred_parameter_elements=sum(
                    target_state_dict[key].numel() for key in transferred_region_keys & parameter_keys
                ),
                total_parameter_elements=sum(target_state_dict[key].numel() for key in region_parameter_keys),
            )
        )

    return remapped, P2DetailInjectionTransferReport(
        transferred_keys=transferred_keys,
        skipped_absent_keys=tuple(sorted(skipped_absent)),
        skipped_shape_keys=tuple(sorted(skipped_shape)),
        newly_initialized_keys=newly_initialized_keys,
        transferred_parameter_elements=transferred_parameter_elements,
        coverage=tuple(coverage),
    )


def print_p2_detail_injection_transfer_report(report: P2DetailInjectionTransferReport) -> None:
    """Print tensor counts and regional transfer coverage."""
    print(f"Transferred tensors: {report.transferred_tensors}")
    print(f"Transferred parameter elements: {report.transferred_parameter_elements}")
    print(f"Skipped because target key is absent: {len(report.skipped_absent_keys)}")
    print(f"Skipped because shape differs: {len(report.skipped_shape_keys)}")
    print(f"Newly initialized tensors: {len(report.newly_initialized_keys)}")
    print("Transfer coverage:")
    for region in report.coverage:
        print(
            f"  {region.name}: {region.transferred_tensors}/{region.total_tensors} tensors, "
            f"{region.transferred_parameter_elements}/{region.total_parameter_elements} parameter elements"
        )


def _model_module(model: ContrastRingYOLO | nn.Module) -> nn.Module:
    """Return the underlying PyTorch model from a facade or module."""
    return model.model if isinstance(model, ContrastRingYOLO) else model


def load_yolo26_p2di_pretrained(
    target: ContrastRingYOLO | nn.Module,
    pretrained: str | Path,
    *,
    verbose: bool = True,
) -> P2DetailInjectionTransferReport:
    """Load all exactly compatible tensors from a standard YOLO26 checkpoint into E5."""
    target_model = _model_module(target)
    source_model, _ = load_checkpoint(pretrained, device="cpu")
    source_scale = source_model.yaml.get("scale")
    target_scale = target_model.yaml.get("scale")
    if source_scale != target_scale:
        raise ValueError(f"Checkpoint scale {source_scale!r} does not match E5 target scale {target_scale!r}.")
    if (
        not isinstance(source_model.model[-1], Detect)
        or source_model.model[-1].i != 23
        or source_model.model[-1].f != [16, 19, 22]
    ):
        raise ValueError("Pretrained checkpoint must contain the standard YOLO26 Detect layer at index 23.")
    if (
        not isinstance(target_model.model[-1], Detect)
        or target_model.model[-1].i != 24
        or target_model.model[-1].f != [17, 20, 23]
        or target_model.model[15].f != 2
        or target_model.model[16].f != [14, 4, 15]
    ):
        raise ValueError("Target model must contain the required E5 P2 route and Detect layer at index 24.")

    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    remapped, report = remap_yolo26_p2di_state_dict(
        source_state_dict,
        target_state_dict,
        target_parameter_keys=dict(target_model.named_parameters()),
    )
    incompatible = target_model.load_state_dict(remapped, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected remapped target keys: {incompatible.unexpected_keys}.")
    reported_new = set(report.newly_initialized_keys)
    actual_missing = set(incompatible.missing_keys)
    if actual_missing - reported_new:
        raise RuntimeError("load_state_dict reported missing keys outside the newly initialized E5 tensors.")
    silently_initialized = reported_new - actual_missing
    if any(not key.endswith(".num_batches_tracked") for key in silently_initialized):
        raise RuntimeError("Reported newly initialized tensors do not match load_state_dict missing keys.")

    coverage = {region.name: region for region in report.coverage}
    for region_name in ("backbone 0-10", "early neck 11-14", "lower PAN path 18-23"):
        region = coverage[region_name]
        if region.transferred_tensors != region.total_tensors:
            raise RuntimeError(f"Incomplete compatible pretrained transfer in {region_name}.")
    p2_projection_keys = {key for key in target_state_dict if key.startswith("model.15.")}
    if not p2_projection_keys or p2_projection_keys & set(report.transferred_keys):
        raise RuntimeError("The new E5 P2 projection must remain newly initialized.")
    box_regression_keys = {
        key
        for key in target_state_dict
        if key.startswith(("model.24.cv2.", "model.24.one2one_cv2."))
    }
    if not box_regression_keys or not box_regression_keys <= set(report.transferred_keys):
        raise RuntimeError("Compatible Detect box-regression tensors were not fully transferred.")

    if verbose:
        print_p2_detail_injection_transfer_report(report)
    return report


def p2_detail_injection_yaml_path(size: str) -> Path:
    """Return the virtual scale-specific E5 YAML path resolved by yaml_model_load."""
    if size not in {"n", "s", "m", "l", "x"}:
        raise ValueError(f"Unsupported YOLO26 scale {size!r}; expected one of n, s, m, l, x.")
    model_dir = Path(__file__).resolve().parents[1] / "research" / "models"
    unified_yaml = model_dir / "yolo26-p2di.yaml"
    if not unified_yaml.is_file():
        raise FileNotFoundError(f"E5 architecture YAML not found: {unified_yaml}")
    return model_dir / f"yolo26{size}-p2di.yaml"


def build_p2_detail_injection_yolo(
    size: str,
    pretrained: str | Path,
    verbose: bool = False,
) -> ContrastRingYOLO:
    """Build an E5 model with fixed E2.1b supervision and remapped stock YOLO26 weights."""
    architecture_yaml = p2_detail_injection_yaml_path(size)
    model = ContrastRingYOLO(architecture_yaml, loss_config=E2_1B_CONFIG, verbose=verbose)
    if model.model.yaml.get("scale") != size:
        raise RuntimeError(f"Expected E5 scale {size!r}, received {model.model.yaml.get('scale')!r}.")
    if model.model.contrast_ring_loss_config != E2_1B_CONFIG:
        raise RuntimeError("E5 must use the fixed E2.1b loss configuration.")

    report = load_yolo26_p2di_pretrained(model, pretrained, verbose=verbose)
    model.p2di_transfer_report = report
    model.p2di_pretrained = str(pretrained)
    model.ckpt_path = str(pretrained)
    # Model.train() reuses a prepared model only when a checkpoint is present. This local checkpoint record
    # preserves the remapped E5 weights while the trainer adapts nc and builds its ordinary target model.
    model.ckpt = {"model": model.model}
    return model
