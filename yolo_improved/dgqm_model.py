# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Construction, training, diagnostics, and exact pretrained transfer for E10 DGQM."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.modules import (
    BoxClassAgreementQualityHead,
    Detect,
    DualGeometryQualityMorphologyDetect,
    MorphologyAdaptiveAdapter,
)
from ultralytics.nn.tasks import load_checkpoint, yaml_model_load
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.loss import E2ELoss

from .contrast_ring_loss import E2_LOCALIZATION_CONFIG
from .contrast_ring_model import ContrastRingDetectionModel
from .dgqm_config import DualGeometryQualityConfig, resolve_dual_geometry_quality_config
from .dgqm_loss import DualGeometryQualityDetectionLoss
from .madh_model import E2_1B_CONFIG, remap_yolo26_madh_state_dict
from .residual_nwd_loss import ResidualNWDBboxLoss


DGQM_VARIANTS = ("control", "trainable")
MODEL_SIZES = ("n", "s", "m", "l", "x")
_QUALITY_CONFIG_KEY = "_dual_geometry_quality_config"
_ADAPTER_PREFIXES = (
    "model.23.box_adapters.",
    "model.23.cls_adapters.",
    "model.23.one2one_box_adapters.",
    "model.23.one2one_cls_adapters.",
)
_QUALITY_PREFIXES = (
    "model.23.quality_heads.",
    "model.23.one2one_quality_heads.",
)
_NEW_DGQM_PREFIXES = (*_ADAPTER_PREFIXES, *_QUALITY_PREFIXES)


@dataclass(frozen=True)
class DGQMTransferCoverage:
    """Pretrained parameter coverage for one E10 architecture region."""

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
class DGQMTransferReport:
    """Exact-key pretrained-transfer results for one E10 model."""

    source_checkpoint: str
    target_architecture: str
    exact_keys: tuple[str, ...]
    skipped_missing_keys: tuple[str, ...]
    skipped_shape_keys: tuple[str, ...]
    new_dgqm_keys: tuple[str, ...]
    new_adapter_keys: tuple[str, ...]
    new_quality_keys: tuple[str, ...]
    transferred_parameter_elements: int
    target_parameter_elements: int
    new_adapter_parameter_elements: int
    new_quality_parameter_elements: int
    coverages: tuple[DGQMTransferCoverage, ...]

    @property
    def transfer_percentage(self) -> float:
        """Return target parameter-element coverage as a percentage."""
        return 100.0 * self.transferred_parameter_elements / self.target_parameter_elements

    def coverage(self, name: str) -> DGQMTransferCoverage:
        """Return one named architecture-region coverage record."""
        try:
            return next(item for item in self.coverages if item.name == name)
        except StopIteration as error:
            raise KeyError(name) from error


def _underlying_model(model: YOLO | nn.Module) -> nn.Module:
    """Return the PyTorch detection model from a facade or direct module."""
    return model.model if isinstance(model, YOLO) else model


def _quality_coverage(
    target_state_dict: Mapping[str, Tensor],
    transferred_keys: set[str],
    parameter_keys: set[str],
) -> DGQMTransferCoverage:
    """Calculate state and parameter coverage for both independent quality branches."""
    target_keys = {key for key in target_state_dict if key.startswith(_QUALITY_PREFIXES)}
    transferred = target_keys & transferred_keys
    target_parameters = target_keys & parameter_keys
    transferred_parameters = transferred & parameter_keys
    return DGQMTransferCoverage(
        name="quality_heads",
        transferred_tensors=len(transferred),
        target_tensors=len(target_keys),
        transferred_parameter_elements=sum(target_state_dict[key].numel() for key in transferred_parameters),
        target_parameter_elements=sum(target_state_dict[key].numel() for key in target_parameters),
    )


def remap_yolo26_dgqm_state_dict(
    source_state_dict: Mapping[str, Tensor],
    target_state_dict: Mapping[str, Tensor],
    *,
    target_parameter_keys: Collection[str] | None = None,
) -> tuple[dict[str, Tensor], DGQMTransferReport]:
    """Reuse E8 exact-key transfer and classify the new E10 adapter/quality state."""
    parameter_keys = set(target_state_dict) if target_parameter_keys is None else set(target_parameter_keys)
    transferred, madh_report = remap_yolo26_madh_state_dict(
        source_state_dict,
        target_state_dict,
        target_parameter_keys=parameter_keys,
    )
    transferred_keys = set(transferred)
    new_dgqm_keys = madh_report.new_adapter_keys
    new_adapter_keys = tuple(sorted(key for key in new_dgqm_keys if key.startswith(_ADAPTER_PREFIXES)))
    new_quality_keys = tuple(sorted(key for key in new_dgqm_keys if key.startswith(_QUALITY_PREFIXES)))
    coverages = tuple(
        DGQMTransferCoverage(
            name=coverage.name,
            transferred_tensors=coverage.transferred_tensors,
            target_tensors=coverage.target_tensors,
            transferred_parameter_elements=coverage.transferred_parameter_elements,
            target_parameter_elements=coverage.target_parameter_elements,
        )
        for coverage in madh_report.coverages
    ) + (_quality_coverage(target_state_dict, transferred_keys, parameter_keys),)
    return transferred, DGQMTransferReport(
        source_checkpoint=madh_report.source_checkpoint,
        target_architecture=madh_report.target_architecture,
        exact_keys=madh_report.exact_keys,
        skipped_missing_keys=madh_report.skipped_missing_keys,
        skipped_shape_keys=madh_report.skipped_shape_keys,
        new_dgqm_keys=new_dgqm_keys,
        new_adapter_keys=new_adapter_keys,
        new_quality_keys=new_quality_keys,
        transferred_parameter_elements=madh_report.transferred_parameter_elements,
        target_parameter_elements=madh_report.target_parameter_elements,
        new_adapter_parameter_elements=sum(
            target_state_dict[key].numel() for key in set(new_adapter_keys) & parameter_keys
        ),
        new_quality_parameter_elements=sum(
            target_state_dict[key].numel() for key in set(new_quality_keys) & parameter_keys
        ),
        coverages=coverages,
    )


def print_dgqm_transfer_report(report: DGQMTransferReport) -> None:
    """Print exact E10 pretrained-transfer totals and regional coverage."""
    print(f"Exact transferred tensors: {len(report.exact_keys)}")
    print(f"Transferred parameter elements: {report.transferred_parameter_elements}")
    print(f"Skipped shape mismatches: {len(report.skipped_shape_keys)}")
    print(
        f"New adapter tensors: {len(report.new_adapter_keys)} tensors, "
        f"{report.new_adapter_parameter_elements} parameter elements"
    )
    print(
        f"New quality tensors: {len(report.new_quality_keys)} tensors, "
        f"{report.new_quality_parameter_elements} parameter elements"
    )
    print(f"Transfer percentage: {report.transfer_percentage:.6f}%")
    print("Transfer coverage:")
    for coverage in report.coverages:
        print(
            f"  {coverage.name}: {coverage.transferred_tensors}/{coverage.target_tensors} tensors, "
            f"{coverage.percentage:.6f}% parameter elements"
        )


def collect_dgqm_diagnostics(
    model: YOLO | nn.Module,
    mean_absolute_corrections: Mapping[tuple[str, str], float] | None = None,
) -> tuple[dict[str, object], ...]:
    """Return JSON-ready P3/P4/P5 diagnostics for both assignment branches."""
    underlying = _underlying_model(model)
    head = underlying.model[-1]
    if not isinstance(head, DualGeometryQualityMorphologyDetect):
        raise ValueError("Model does not contain DualGeometryQualityMorphologyDetect.")
    corrections = mean_absolute_corrections or {}
    levels = ("P3", "P4", "P5")
    diagnostics = []
    for assignment, quality_name, box_name, cls_name in (
        ("one-to-many", "quality_heads", "box_adapters", "cls_adapters"),
        (
            "one-to-one",
            "one2one_quality_heads",
            "one2one_box_adapters",
            "one2one_cls_adapters",
        ),
    ):
        quality_heads = getattr(head, quality_name)
        box_adapters = getattr(head, box_name)
        cls_adapters = getattr(head, cls_name)
        for level, quality_head, box_adapter, cls_adapter in zip(
            levels,
            quality_heads,
            box_adapters,
            cls_adapters,
        ):
            diagnostics.append(
                {
                    "assignment": assignment,
                    "level": level,
                    "quality_output_weight_norm": float(
                        quality_head.output.weight.detach().float().norm().cpu().item()
                    ),
                    "quality_output_bias": float(
                        quality_head.output.bias.detach().float().cpu().item()
                    ),
                    "mean_absolute_quality_correction": corrections.get((assignment, level)),
                    "adapter_gate_raw": {
                        "box": float(box_adapter.gate_raw.detach().float().cpu().item()),
                        "classification": float(cls_adapter.gate_raw.detach().float().cpu().item()),
                    },
                    "adapter_alpha": {
                        "box": float(
                            (box_adapter.gate_max * torch.tanh(box_adapter.gate_raw.detach().float()))
                            .cpu()
                            .item()
                        ),
                        "classification": float(
                            (cls_adapter.gate_max * torch.tanh(cls_adapter.gate_raw.detach().float()))
                            .cpu()
                            .item()
                        ),
                    },
                }
            )
    return tuple(diagnostics)


def _validate_dgqm_head(
    model: nn.Module,
    variant: str,
    quality_config: DualGeometryQualityConfig,
) -> DualGeometryQualityMorphologyDetect:
    """Validate the full E8 topology and independent E10 quality branches."""
    head = model.model[-1]
    if not isinstance(head, DualGeometryQualityMorphologyDetect):
        raise RuntimeError("E10 must use DualGeometryQualityMorphologyDetect.")
    if head.f != [16, 19, 22] or head.nl != 3:
        raise RuntimeError("E10 must retain exactly the P3/8, P4/16, and P5/32 levels.")
    if head.stride.tolist() != [8.0, 16.0, 32.0]:
        raise RuntimeError(f"E10 Detect strides must be [8, 16, 32], got {head.stride.tolist()}.")
    if head.reg_max != 1 or head.no != head.nc + 4 or not isinstance(head.dfl, nn.Identity):
        raise RuntimeError("E10 must retain stock YOLO26 reg_max=1 without DFL.")
    if head.quality_scale != quality_config.quality_scale:
        raise RuntimeError("E10 head and quality-loss configurations must use the same quality_scale.")

    adapter_groups = (
        head.box_adapters,
        head.cls_adapters,
        head.one2one_box_adapters,
        head.one2one_cls_adapters,
    )
    if any(len(group) != 3 for group in adapter_groups):
        raise RuntimeError("E10 must retain all twelve independent E8 adapters.")
    for group in adapter_groups:
        for adapter in group:
            if not isinstance(adapter, MorphologyAdaptiveAdapter) or not adapter.trainable_adapter:
                raise RuntimeError("Both E10 variants must retain trainable E8 adapters.")
            if adapter.gate_max != 0.5 or torch.count_nonzero(adapter.gate_raw.detach()).item() != 0:
                raise RuntimeError("Every E10 adapter must start with gate_max=0.5 and a zero gate.")
            if torch.count_nonzero(adapter.router.weight.detach()).item() != 0:
                raise RuntimeError("Every E10 adapter router weight must start at zero.")
            if torch.count_nonzero(adapter.router.bias.detach()).item() != 0:
                raise RuntimeError("Every E10 adapter router bias must start at zero.")
            if any(not parameter.requires_grad for parameter in adapter.parameters()):
                raise RuntimeError("Both E10 variants must retain trainable E8 adapter parameters.")

    trainable_quality = variant == "trainable"
    quality_groups = (head.quality_heads, head.one2one_quality_heads)
    if any(len(group) != 3 for group in quality_groups):
        raise RuntimeError("E10 must contain independent quality heads for both assignments and all levels.")
    for group in quality_groups:
        for quality_head in group:
            if not isinstance(quality_head, BoxClassAgreementQualityHead):
                raise RuntimeError("E10 contains an unexpected quality-head type.")
            if quality_head.trainable_quality != trainable_quality:
                raise RuntimeError(f"E10 quality gradients do not match the requested {variant!r} variant.")
            if torch.count_nonzero(quality_head.output.weight.detach()).item() != 0:
                raise RuntimeError("Every E10 quality output weight must start at zero.")
            if torch.count_nonzero(quality_head.output.bias.detach()).item() != 0:
                raise RuntimeError("Every E10 quality output bias must start at zero.")
            if any(parameter.requires_grad != trainable_quality for parameter in quality_head.parameters()):
                raise RuntimeError(f"E10 quality parameters do not match the requested {variant!r} variant.")

    for first_group, second_group in (
        (head.box_adapters, head.cls_adapters),
        (head.box_adapters, head.one2one_box_adapters),
        (head.cls_adapters, head.one2one_cls_adapters),
        (head.quality_heads, head.one2one_quality_heads),
    ):
        for first, second in zip(first_group, second_group):
            if any(
                first_parameter.data_ptr() == second_parameter.data_ptr()
                for first_parameter, second_parameter in zip(first.parameters(), second.parameters())
            ):
                raise RuntimeError("E10 one-to-many/one-to-one and box/class modules must be independent.")
    return head


def _validate_criterion(
    model: ContrastRingDetectionModel,
    quality_config: DualGeometryQualityConfig,
) -> None:
    """Verify unchanged E2.1b components and the added fourth quality component."""
    criterion = model.init_criterion()
    if not isinstance(criterion, E2ELoss):
        raise RuntimeError("E10 must retain the local end-to-end loss wrapper.")
    for name, branch in (("one-to-many", criterion.one2many), ("one-to-one", criterion.one2one)):
        if not isinstance(branch, DualGeometryQualityDetectionLoss):
            raise RuntimeError(f"E10 {name} must use DualGeometryQualityDetectionLoss.")
        if branch.reg_max != 1 or branch.use_dfl or branch.bbox_loss.dfl_loss is not None:
            raise RuntimeError(f"E10 {name} must retain stock YOLO26 normalized L1 regression.")
        if not isinstance(branch.bbox_loss, ResidualNWDBboxLoss):
            raise RuntimeError(f"E10 {name} must retain the E1.1 residual NWD box criterion.")
        if branch.bbox_loss.config != E2_LOCALIZATION_CONFIG:
            raise RuntimeError(f"E10 {name} changed the E1.1 localization configuration.")
        if branch.contrast_ring_config != E2_1B_CONFIG:
            raise RuntimeError(f"E10 {name} changed the E2.1b classification configuration.")
        if branch.quality_config != quality_config:
            raise RuntimeError(f"E10 {name} changed the requested quality configuration.")
        if branch.loss_names != ("box_loss", "cls_loss", "l1_loss", "quality_loss"):
            raise RuntimeError(f"E10 {name} must expose quality_loss as the fourth component.")


class DualGeometryQualityDetectionModel(ContrastRingDetectionModel):
    """Detection model with full E8 architecture and isolated E10 quality supervision."""

    def __init__(
        self,
        cfg="yolo26n-dgqm.yaml",
        ch=3,
        nc=None,
        verbose=True,
        quality_config: DualGeometryQualityConfig | dict | None = None,
    ):
        """Initialize fixed E2.1b supervision and a local E10 quality configuration."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, loss_config=E2_1B_CONFIG)
        self.dual_geometry_quality_config = resolve_dual_geometry_quality_config(quality_config)
        head = self.model[-1]
        if not isinstance(head, DualGeometryQualityMorphologyDetect):
            raise ValueError("DualGeometryQualityDetectionModel requires a DGQM architecture YAML.")
        head.quality_scale = self.dual_geometry_quality_config.quality_scale

    def init_criterion(self):
        """Initialize independent one-to-many/one-to-one E2.1b plus quality losses."""
        loss_fn = partial(
            DualGeometryQualityDetectionLoss,
            contrast_config=E2_1B_CONFIG,
            quality_config=self.dual_geometry_quality_config,
        )
        return E2ELoss(self, loss_fn=loss_fn) if self.end2end else loss_fn(self)


class DualGeometryQualityDetectionTrainer(DetectionTrainer):
    """Detection trainer that constructs DualGeometryQualityDetectionModel."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the stock trainer after extracting the local E10 configuration."""
        overrides = dict(overrides or {})
        self.quality_config = resolve_dual_geometry_quality_config(overrides.pop(_QUALITY_CONFIG_KEY, None))
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        if self.ddp:
            setattr(self.args, _QUALITY_CONFIG_KEY, asdict(self.quality_config))

    def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
        """Return an E10 model and load provided training weights when present."""
        quality_config = getattr(weights, "dual_geometry_quality_config", self.quality_config)
        model = self.set_model_names_for_load(
            DualGeometryQualityDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                quality_config=quality_config,
            )
        )
        if weights:
            model.load(weights)
        return model


class DualGeometryQualityYOLO(YOLO):
    """YOLO facade that isolates the E10 model, trainer, and quality configuration."""

    def __init__(
        self,
        model: str | Path = "yolo26n-dgqm.yaml",
        task: str | None = None,
        verbose: bool = False,
        quality_config: DualGeometryQualityConfig | dict | None = None,
    ):
        """Initialize an E10 detection model or checkpoint."""
        super().__init__(model=model, task=task, verbose=verbose)
        if self.task != "detect" or not isinstance(self.model, DualGeometryQualityDetectionModel):
            raise ValueError("DualGeometryQualityYOLO supports only E10 PyTorch detection models.")
        checkpoint_config = getattr(self.model, "dual_geometry_quality_config", None)
        self.quality_config = resolve_dual_geometry_quality_config(
            quality_config if quality_config is not None else checkpoint_config
        )
        self.model.dual_geometry_quality_config = self.quality_config
        self.model.contrast_ring_loss_config = E2_1B_CONFIG
        self.model.model[-1].quality_scale = self.quality_config.quality_scale

    def train(self, trainer=None, **kwargs: Any):
        """Train with fixed E2.1b and the local E10 quality configuration."""
        kwargs[_QUALITY_CONFIG_KEY] = asdict(self.quality_config)
        return super().train(trainer=trainer, **kwargs)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map detection construction and training to isolated E10 classes."""
        task_map = super().task_map
        task_map["detect"] = {
            **task_map["detect"],
            "model": DualGeometryQualityDetectionModel,
            "trainer": DualGeometryQualityDetectionTrainer,
        }
        return task_map


def load_yolo26_dgqm_pretrained(
    target: YOLO | nn.Module,
    pretrained: str | Path,
    *,
    verbose: bool = True,
) -> DGQMTransferReport:
    """Load all standard YOLO26 tensors while preserving every new E10 tensor."""
    target_model = _underlying_model(target)
    target_head = target_model.model[-1]
    if not isinstance(target_head, DualGeometryQualityMorphologyDetect):
        raise ValueError("Target must contain DualGeometryQualityMorphologyDetect.")

    source_model, _ = load_checkpoint(pretrained, device="cpu")
    source_head = source_model.model[-1]
    if type(source_head) is not Detect or source_head.reg_max != 1:
        raise ValueError("Pretrained source must be a standard YOLO26 Detect checkpoint with reg_max=1.")
    if source_head.f != [16, 19, 22] or source_head.nl != 3:
        raise ValueError("Pretrained source does not have the stock YOLO26 P3/P4/P5 topology.")
    if source_model.yaml.get("scale") != target_model.yaml.get("scale"):
        raise ValueError(
            f"Checkpoint scale {source_model.yaml.get('scale')!r} does not match "
            f"E10 target scale {target_model.yaml.get('scale')!r}."
        )

    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    transferred, report = remap_yolo26_dgqm_state_dict(
        source_state_dict,
        target_state_dict,
        target_parameter_keys=dict(target_model.named_parameters()),
    )
    variant = "trainable" if target_head.trainable_quality else "control"
    suffix = "dgqm" if variant == "trainable" else "dgqm-control"
    target_architecture = Path(target_model.yaml["yaml_file"]).with_name(f"yolo26-{suffix}.yaml")
    report = replace(
        report,
        source_checkpoint=str(pretrained),
        target_architecture=str(target_architecture),
    )
    initial_new_state = {key: target_state_dict[key].detach().clone() for key in report.new_dgqm_keys}

    incompatible = target_model.load_state_dict(transferred, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected E10 transfer keys: {incompatible.unexpected_keys}.")
    reported_missing = set(report.new_dgqm_keys)
    actual_missing = set(incompatible.missing_keys)
    if actual_missing - reported_missing:
        raise RuntimeError(f"load_state_dict reported untracked E10 keys: {sorted(actual_missing - reported_missing)}.")
    silently_initialized = reported_missing - actual_missing
    if any(not key.endswith(".num_batches_tracked") for key in silently_initialized):
        raise RuntimeError("E10 transfer report does not match load_state_dict missing keys.")

    loaded_state_dict = target_model.state_dict()
    for key, initial_tensor in initial_new_state.items():
        if not torch.equal(loaded_state_dict[key], initial_tensor):
            raise RuntimeError(f"New E10 tensor {key!r} was modified during pretrained transfer.")
    if report.skipped_missing_keys or report.skipped_shape_keys:
        raise RuntimeError("Stock YOLO26 and E10 standard tensors must match by exact key and shape.")
    if not report.new_adapter_keys or not report.new_quality_keys:
        raise RuntimeError("E10 transfer must leave both adapters and quality heads newly initialized.")
    if any(not key.startswith(_NEW_DGQM_PREFIXES) for key in report.new_dgqm_keys):
        raise RuntimeError("Only E10 adapter and quality tensors may remain newly initialized.")
    for coverage_name in ("backbone", "neck", "cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        coverage = report.coverage(coverage_name)
        if coverage.transferred_tensors != coverage.target_tensors or coverage.percentage != 100.0:
            raise RuntimeError(f"Pretrained transfer is incomplete for E10 {coverage_name}.")
    for coverage_name in ("adapters", "quality_heads"):
        coverage = report.coverage(coverage_name)
        if coverage.transferred_tensors != 0 or coverage.transferred_parameter_elements != 0:
            raise RuntimeError(f"No standard YOLO26 tensor may be copied into E10 {coverage_name}.")
    quality_config = getattr(target_model, "dual_geometry_quality_config", DualGeometryQualityConfig())
    _validate_dgqm_head(target_model, variant, quality_config)

    if verbose:
        print_dgqm_transfer_report(report)
    return report


def dgqm_yaml_path(size: str, variant: str) -> Path:
    """Return the virtual scale-specific E10 YAML path."""
    if size not in MODEL_SIZES:
        raise ValueError(f"Unsupported YOLO26 scale {size!r}; expected one of n, s, m, l, x.")
    if variant not in DGQM_VARIANTS:
        raise ValueError(f"Unsupported DGQM variant {variant!r}; expected control or trainable.")
    model_directory = Path(__file__).resolve().parents[1] / "research" / "models"
    suffix = "dgqm-control" if variant == "control" else "dgqm"
    unified_yaml = model_directory / f"yolo26-{suffix}.yaml"
    if not unified_yaml.is_file():
        raise FileNotFoundError(f"E10 architecture YAML not found: {unified_yaml}")
    return model_directory / f"yolo26{size}-{suffix}.yaml"


def _validate_stock_architecture(target_model: nn.Module, size: str) -> None:
    """Ensure E10 changes only the final Detect class and its local arguments."""
    repository_root = Path(__file__).resolve().parents[1]
    stock = yaml_model_load(repository_root / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{size}.yaml")
    target = target_model.yaml
    for key in ("nc", "end2end", "reg_max", "scales", "backbone", "scale"):
        if target.get(key) != stock.get(key):
            raise RuntimeError(f"E10 architecture unexpectedly changes stock YOLO26 key {key!r}.")
    if target["head"][:-1] != stock["head"][:-1]:
        raise RuntimeError("E10 must retain every stock YOLO26 head layer before Detect.")
    if target["head"][-1][0:2] != stock["head"][-1][0:2]:
        raise RuntimeError("E10 must retain stock YOLO26 Detect inputs and repeat count.")
    if target["head"][-1][2] != "DualGeometryQualityMorphologyDetect":
        raise RuntimeError("E10 YAML must replace only Detect with DualGeometryQualityMorphologyDetect.")


def build_dgqm_yolo(
    size: str,
    variant: str,
    verbose: bool = False,
    quality_config: DualGeometryQualityConfig | dict | None = None,
) -> DualGeometryQualityYOLO:
    """Build E10 with full E8, fixed E2.1b, and a standard YOLO26 checkpoint."""
    resolved_quality_config = resolve_dual_geometry_quality_config(quality_config)
    architecture_yaml = dgqm_yaml_path(size, variant)
    model = DualGeometryQualityYOLO(
        architecture_yaml,
        quality_config=resolved_quality_config,
        verbose=verbose,
    )
    if model.model.yaml.get("scale") != size:
        raise RuntimeError(f"Expected E10 scale {size!r}, received {model.model.yaml.get('scale')!r}.")
    if model.model.contrast_ring_loss_config != E2_1B_CONFIG:
        raise RuntimeError("E10 must use the fixed E2.1b classification configuration.")
    _validate_stock_architecture(model.model, size)
    _validate_dgqm_head(model.model, variant, resolved_quality_config)
    _validate_criterion(model.model, resolved_quality_config)

    pretrained = Path(f"yolo26{size}.pt")
    report = load_yolo26_dgqm_pretrained(model, pretrained, verbose=verbose)
    model.dgqm_variant = variant
    model.dgqm_transfer_report = report
    model.dgqm_pretrained = str(pretrained)
    model.ckpt_path = str(pretrained)
    model.ckpt = {"model": model.model}
    return model
