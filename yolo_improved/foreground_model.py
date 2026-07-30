# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Construction, training, diagnostics, and exact pretrained transfer for E12 FF-DGQM."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.modules import (
    Detect,
    ForegroundnessAgreementHead,
    ForegroundnessFactorizedDGQMDetect,
)
from ultralytics.nn.tasks import load_checkpoint, yaml_model_load
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.loss import E2ELoss

from .contrast_ring_loss import E2_LOCALIZATION_CONFIG
from .dgqm_config import DualGeometryQualityConfig, resolve_dual_geometry_quality_config
from .dgqm_loss import DualGeometryQualityDetectionLoss
from .dgqm_model import (
    DGQMTransferCoverage,
    DualGeometryQualityDetectionModel,
    DualGeometryQualityDetectionTrainer,
    DualGeometryQualityYOLO,
    _validate_dgqm_head,
    dgqm_yaml_path,
    remap_yolo26_dgqm_state_dict,
)
from .foreground_config import (
    ForegroundnessFactorizationConfig,
    resolve_foregroundness_factorization_config,
)
from .foreground_loss import ForegroundnessFactorizedDetectionLoss
from .madh_model import E2_1B_CONFIG
from .residual_nwd_loss import ResidualNWDBboxLoss


FF_DGQM_VARIANTS = ("control", "trainable")
MODEL_SIZES = ("n", "s", "m", "l", "x")
_FOREGROUND_CONFIG_KEY = "_foregroundness_factorization_config"
_FOREGROUND_PREFIXES = (
    "model.23.foreground_heads.",
    "model.23.one2one_foreground_heads.",
)


@dataclass(frozen=True)
class FFDGQMTransferReport:
    """Exact-key pretrained-transfer results for one E12 model."""

    source_checkpoint: str
    target_architecture: str
    exact_keys: tuple[str, ...]
    skipped_missing_keys: tuple[str, ...]
    skipped_shape_keys: tuple[str, ...]
    new_ff_keys: tuple[str, ...]
    new_adapter_keys: tuple[str, ...]
    new_quality_keys: tuple[str, ...]
    new_foreground_keys: tuple[str, ...]
    transferred_parameter_elements: int
    target_parameter_elements: int
    new_adapter_parameter_elements: int
    new_quality_parameter_elements: int
    new_foreground_parameter_elements: int
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


def _foreground_coverage(
    target_state_dict: Mapping[str, Tensor],
    transferred_keys: set[str],
    parameter_keys: set[str],
) -> DGQMTransferCoverage:
    """Calculate state and parameter coverage for both independent foreground branches."""
    target_keys = {key for key in target_state_dict if key.startswith(_FOREGROUND_PREFIXES)}
    transferred = target_keys & transferred_keys
    target_parameters = target_keys & parameter_keys
    transferred_parameters = transferred & parameter_keys
    return DGQMTransferCoverage(
        name="foreground_heads",
        transferred_tensors=len(transferred),
        target_tensors=len(target_keys),
        transferred_parameter_elements=sum(target_state_dict[key].numel() for key in transferred_parameters),
        target_parameter_elements=sum(target_state_dict[key].numel() for key in target_parameters),
    )


def remap_yolo26_ff_dgqm_state_dict(
    source_state_dict: Mapping[str, Tensor],
    target_state_dict: Mapping[str, Tensor],
    *,
    target_parameter_keys: Collection[str] | None = None,
) -> tuple[dict[str, Tensor], FFDGQMTransferReport]:
    """Reuse E10 exact-key transfer and classify the new E12 foreground state."""
    parameter_keys = set(target_state_dict) if target_parameter_keys is None else set(target_parameter_keys)
    transferred, dgqm_report = remap_yolo26_dgqm_state_dict(
        source_state_dict,
        target_state_dict,
        target_parameter_keys=parameter_keys,
    )
    transferred_keys = set(transferred)
    new_ff_keys = dgqm_report.new_dgqm_keys
    new_foreground_keys = tuple(
        sorted(key for key in new_ff_keys if key.startswith(_FOREGROUND_PREFIXES))
    )
    coverages = (*dgqm_report.coverages, _foreground_coverage(target_state_dict, transferred_keys, parameter_keys))
    return transferred, FFDGQMTransferReport(
        source_checkpoint=dgqm_report.source_checkpoint,
        target_architecture=dgqm_report.target_architecture,
        exact_keys=dgqm_report.exact_keys,
        skipped_missing_keys=dgqm_report.skipped_missing_keys,
        skipped_shape_keys=dgqm_report.skipped_shape_keys,
        new_ff_keys=new_ff_keys,
        new_adapter_keys=dgqm_report.new_adapter_keys,
        new_quality_keys=dgqm_report.new_quality_keys,
        new_foreground_keys=new_foreground_keys,
        transferred_parameter_elements=dgqm_report.transferred_parameter_elements,
        target_parameter_elements=dgqm_report.target_parameter_elements,
        new_adapter_parameter_elements=dgqm_report.new_adapter_parameter_elements,
        new_quality_parameter_elements=dgqm_report.new_quality_parameter_elements,
        new_foreground_parameter_elements=sum(
            target_state_dict[key].numel() for key in set(new_foreground_keys) & parameter_keys
        ),
        coverages=coverages,
    )


def print_ff_dgqm_transfer_report(report: FFDGQMTransferReport) -> None:
    """Print exact E12 pretrained-transfer totals and regional coverage."""
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
    print(
        f"New foreground tensors: {len(report.new_foreground_keys)} tensors, "
        f"{report.new_foreground_parameter_elements} parameter elements"
    )
    print(f"Transfer percentage: {report.transfer_percentage:.6f}%")
    print("Transfer coverage:")
    for coverage in report.coverages:
        print(
            f"  {coverage.name}: {coverage.transferred_tensors}/{coverage.target_tensors} tensors, "
            f"{coverage.percentage:.6f}% parameter elements"
        )


def collect_ff_dgqm_diagnostics(
    model: YOLO | nn.Module,
    correction_statistics: Mapping[tuple[str, str], Mapping[str, float | None]] | None = None,
) -> tuple[dict[str, object], ...]:
    """Return JSON-ready E12 diagnostics for both assignments and P3/P4/P5."""
    underlying = _underlying_model(model)
    head = underlying.model[-1]
    if not isinstance(head, ForegroundnessFactorizedDGQMDetect):
        raise ValueError("Model does not contain ForegroundnessFactorizedDGQMDetect.")
    statistics = correction_statistics or {}
    levels = ("P3", "P4", "P5")
    diagnostics = []
    for assignment, quality_name, foreground_name, box_name, cls_name in (
        (
            "one-to-many",
            "quality_heads",
            "foreground_heads",
            "box_adapters",
            "cls_adapters",
        ),
        (
            "one-to-one",
            "one2one_quality_heads",
            "one2one_foreground_heads",
            "one2one_box_adapters",
            "one2one_cls_adapters",
        ),
    ):
        quality_heads = getattr(head, quality_name)
        foreground_heads = getattr(head, foreground_name)
        box_adapters = getattr(head, box_name)
        cls_adapters = getattr(head, cls_name)
        for level, quality_head, foreground_head, box_adapter, cls_adapter in zip(
            levels,
            quality_heads,
            foreground_heads,
            box_adapters,
            cls_adapters,
        ):
            runtime = statistics.get((assignment, level), {})
            diagnostics.append(
                {
                    "assignment": assignment,
                    "level": level,
                    "quality_output_weight_norm": float(
                        quality_head.output.weight.detach().float().norm().cpu().item()
                    ),
                    "mean_quality_correction": runtime.get("mean_quality_correction"),
                    "foreground_output_weight_norm": float(
                        foreground_head.output.weight.detach().float().norm().cpu().item()
                    ),
                    "foreground_output_bias": float(
                        foreground_head.output.bias.detach().float().cpu().item()
                    ),
                    "mean_foreground_correction": runtime.get("mean_foreground_correction"),
                    "mean_positive_foreground_correction": runtime.get(
                        "mean_positive_foreground_correction"
                    ),
                    "mean_background_foreground_correction": runtime.get(
                        "mean_background_foreground_correction"
                    ),
                    "fraction_corrections_above_positive_0_1": runtime.get(
                        "fraction_corrections_above_positive_0_1"
                    ),
                    "fraction_corrections_below_negative_0_1": runtime.get(
                        "fraction_corrections_below_negative_0_1"
                    ),
                    "madh_box_alpha": float(
                        (box_adapter.gate_max * torch.tanh(box_adapter.gate_raw.detach().float()))
                        .cpu()
                        .item()
                    ),
                    "madh_classification_alpha": float(
                        (cls_adapter.gate_max * torch.tanh(cls_adapter.gate_raw.detach().float()))
                        .cpu()
                        .item()
                    ),
                }
            )
    return tuple(diagnostics)


def _validate_ff_head(
    model: nn.Module,
    variant: str,
    quality_config: DualGeometryQualityConfig,
    foreground_config: ForegroundnessFactorizationConfig,
) -> ForegroundnessFactorizedDGQMDetect:
    """Validate the complete E10 topology and independent E12 foreground branches."""
    head = _validate_dgqm_head(model, "trainable", quality_config)
    if not isinstance(head, ForegroundnessFactorizedDGQMDetect):
        raise RuntimeError("E12 must use ForegroundnessFactorizedDGQMDetect.")
    if hasattr(head, "suppression_heads") or hasattr(head, "class_suppression"):
        raise RuntimeError("E12 must not contain E11 class-conditional suppression.")
    if head.foreground_scale != foreground_config.foreground_scale:
        raise RuntimeError("E12 head and foreground-loss configurations must use the same foreground_scale.")

    trainable_foreground = variant == "trainable"
    foreground_groups = (head.foreground_heads, head.one2one_foreground_heads)
    if any(len(group) != 3 for group in foreground_groups):
        raise RuntimeError("E12 must contain independent foreground heads for both assignments and all levels.")
    for group in foreground_groups:
        for foreground_head in group:
            if not isinstance(foreground_head, ForegroundnessAgreementHead):
                raise RuntimeError("E12 contains an unexpected foreground-head type.")
            if foreground_head.trainable_foreground != trainable_foreground:
                raise RuntimeError(f"E12 foreground gradients do not match the requested {variant!r} variant.")
            if torch.count_nonzero(foreground_head.output.weight.detach()).item() != 0:
                raise RuntimeError("Every E12 foreground output weight must start at zero.")
            if torch.count_nonzero(foreground_head.output.bias.detach()).item() != 0:
                raise RuntimeError("Every E12 foreground output bias must start at zero.")
            if any(
                parameter.requires_grad != trainable_foreground
                for parameter in foreground_head.parameters()
            ):
                raise RuntimeError(f"E12 foreground parameters do not match the requested {variant!r} variant.")

    for first, second in zip(head.foreground_heads, head.one2one_foreground_heads):
        if any(
            first_parameter.data_ptr() == second_parameter.data_ptr()
            for first_parameter, second_parameter in zip(first.parameters(), second.parameters())
        ):
            raise RuntimeError("E12 one-to-many and one-to-one foreground heads must be independent.")
    return head


def _validate_criterion(
    model: DualGeometryQualityDetectionModel,
    variant: str,
    quality_config: DualGeometryQualityConfig,
    foreground_config: ForegroundnessFactorizationConfig,
) -> None:
    """Verify unchanged E2.1b/E10 components and the fifth foreground component."""
    criterion = model.init_criterion()
    if not isinstance(criterion, E2ELoss):
        raise RuntimeError("E12 must retain the local end-to-end loss wrapper.")
    for name, branch in (("one-to-many", criterion.one2many), ("one-to-one", criterion.one2one)):
        if not isinstance(branch, ForegroundnessFactorizedDetectionLoss):
            raise RuntimeError(f"E12 {name} must use ForegroundnessFactorizedDetectionLoss.")
        if not isinstance(branch, DualGeometryQualityDetectionLoss):
            raise RuntimeError(f"E12 {name} must retain DualGeometryQualityDetectionLoss.")
        if branch.reg_max != 1 or branch.use_dfl or branch.bbox_loss.dfl_loss is not None:
            raise RuntimeError(f"E12 {name} must retain stock YOLO26 normalized L1 regression.")
        if not isinstance(branch.bbox_loss, ResidualNWDBboxLoss):
            raise RuntimeError(f"E12 {name} must retain the E1.1 residual NWD box criterion.")
        if branch.bbox_loss.config != E2_LOCALIZATION_CONFIG:
            raise RuntimeError(f"E12 {name} changed the E1.1 localization configuration.")
        if branch.contrast_ring_config != E2_1B_CONFIG:
            raise RuntimeError(f"E12 {name} changed the E2.1b classification configuration.")
        if branch.quality_config != quality_config or not branch.trainable_quality:
            raise RuntimeError(f"E12 {name} changed the active E10 quality loss.")
        if branch.foreground_config != foreground_config:
            raise RuntimeError(f"E12 {name} changed the requested foreground configuration.")
        if branch.trainable_foreground != (variant == "trainable"):
            raise RuntimeError(f"E12 {name} foreground state does not match {variant!r}.")
        if branch.loss_names != (
            "box_loss",
            "cls_loss",
            "l1_loss",
            "quality_loss",
            "foreground_loss",
        ):
            raise RuntimeError(f"E12 {name} must expose foreground_loss as the fifth component.")


class ForegroundnessFactorizedDetectionModel(DualGeometryQualityDetectionModel):
    """E10 DGQM detection model with isolated class-agnostic foregroundness."""

    def __init__(
        self,
        cfg="yolo26n-ff-dgqm.yaml",
        ch=3,
        nc=None,
        verbose=True,
        quality_config: DualGeometryQualityConfig | dict | None = None,
        foreground_config: ForegroundnessFactorizationConfig | dict | None = None,
    ):
        """Initialize fixed E10 supervision and a local E12 foreground configuration."""
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            quality_config=quality_config,
        )
        self.foregroundness_factorization_config = resolve_foregroundness_factorization_config(
            foreground_config
        )
        head = self.model[-1]
        if not isinstance(head, ForegroundnessFactorizedDGQMDetect):
            raise ValueError(
                "ForegroundnessFactorizedDetectionModel requires an FF-DGQM architecture YAML."
            )
        head.foreground_scale = self.foregroundness_factorization_config.foreground_scale

    def init_criterion(self):
        """Initialize independent one-to-many/one-to-one E10 plus foreground losses."""
        loss_fn = partial(
            ForegroundnessFactorizedDetectionLoss,
            contrast_config=E2_1B_CONFIG,
            quality_config=self.dual_geometry_quality_config,
            foreground_config=self.foregroundness_factorization_config,
        )
        return E2ELoss(self, loss_fn=loss_fn) if self.end2end else loss_fn(self)


class ForegroundnessFactorizedDetectionTrainer(DualGeometryQualityDetectionTrainer):
    """Detection trainer that constructs ForegroundnessFactorizedDetectionModel."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the E10 trainer after extracting the local E12 configuration."""
        overrides = dict(overrides or {})
        self.foreground_config = resolve_foregroundness_factorization_config(
            overrides.pop(_FOREGROUND_CONFIG_KEY, None)
        )
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        if self.ddp:
            setattr(self.args, _FOREGROUND_CONFIG_KEY, asdict(self.foreground_config))

    def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
        """Return an E12 model and load provided training weights when present."""
        quality_config = getattr(weights, "dual_geometry_quality_config", self.quality_config)
        foreground_config = getattr(
            weights,
            "foregroundness_factorization_config",
            self.foreground_config,
        )
        model = self.set_model_names_for_load(
            ForegroundnessFactorizedDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                quality_config=quality_config,
                foreground_config=foreground_config,
            )
        )
        if weights:
            model.load(weights)
        return model


class ForegroundnessFactorizedYOLO(DualGeometryQualityYOLO):
    """YOLO facade that isolates the E12 model, trainer, and foreground configuration."""

    def __init__(
        self,
        model: str | Path = "yolo26n-ff-dgqm.yaml",
        task: str | None = None,
        verbose: bool = False,
        quality_config: DualGeometryQualityConfig | dict | None = None,
        foreground_config: ForegroundnessFactorizationConfig | dict | None = None,
    ):
        """Initialize an E12 detection model or checkpoint."""
        super().__init__(
            model=model,
            task=task,
            verbose=verbose,
            quality_config=quality_config,
        )
        if not isinstance(self.model, ForegroundnessFactorizedDetectionModel):
            raise ValueError("ForegroundnessFactorizedYOLO supports only E12 PyTorch detection models.")
        checkpoint_config = getattr(self.model, "foregroundness_factorization_config", None)
        self.foreground_config = resolve_foregroundness_factorization_config(
            foreground_config if foreground_config is not None else checkpoint_config
        )
        self.model.foregroundness_factorization_config = self.foreground_config
        self.model.model[-1].foreground_scale = self.foreground_config.foreground_scale

    def train(self, trainer=None, **kwargs: Any):
        """Train with fixed E10 and the local E12 foreground configuration."""
        kwargs[_FOREGROUND_CONFIG_KEY] = asdict(self.foreground_config)
        return super().train(trainer=trainer, **kwargs)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map detection construction and training to isolated E12 classes."""
        task_map = super().task_map
        task_map["detect"] = {
            **task_map["detect"],
            "model": ForegroundnessFactorizedDetectionModel,
            "trainer": ForegroundnessFactorizedDetectionTrainer,
        }
        return task_map


def load_yolo26_ff_dgqm_pretrained(
    target: YOLO | nn.Module,
    pretrained: str | Path,
    *,
    verbose: bool = True,
) -> FFDGQMTransferReport:
    """Load all standard YOLO26 tensors while preserving every new E10/E12 tensor."""
    target_model = _underlying_model(target)
    target_head = target_model.model[-1]
    if not isinstance(target_head, ForegroundnessFactorizedDGQMDetect):
        raise ValueError("Target must contain ForegroundnessFactorizedDGQMDetect.")

    source_model, _ = load_checkpoint(pretrained, device="cpu")
    source_head = source_model.model[-1]
    if type(source_head) is not Detect or source_head.reg_max != 1:
        raise ValueError("Pretrained source must be a standard YOLO26 Detect checkpoint with reg_max=1.")
    if source_head.f != [16, 19, 22] or source_head.nl != 3:
        raise ValueError("Pretrained source does not have the stock YOLO26 P3/P4/P5 topology.")
    if source_model.yaml.get("scale") != target_model.yaml.get("scale"):
        raise ValueError(
            f"Checkpoint scale {source_model.yaml.get('scale')!r} does not match "
            f"E12 target scale {target_model.yaml.get('scale')!r}."
        )

    target_state_dict = target_model.state_dict()
    transferred, report = remap_yolo26_ff_dgqm_state_dict(
        source_model.state_dict(),
        target_state_dict,
        target_parameter_keys=dict(target_model.named_parameters()),
    )
    variant = "trainable" if target_head.trainable_foreground else "control"
    suffix = "ff-dgqm" if variant == "trainable" else "ff-dgqm-control"
    target_architecture = Path(target_model.yaml["yaml_file"]).with_name(f"yolo26-{suffix}.yaml")
    report = replace(
        report,
        source_checkpoint=str(pretrained),
        target_architecture=str(target_architecture),
    )
    initial_new_state = {key: target_state_dict[key].detach().clone() for key in report.new_ff_keys}

    incompatible = target_model.load_state_dict(transferred, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected E12 transfer keys: {incompatible.unexpected_keys}.")
    reported_missing = set(report.new_ff_keys)
    actual_missing = set(incompatible.missing_keys)
    if actual_missing - reported_missing:
        raise RuntimeError(f"load_state_dict reported untracked E12 keys: {sorted(actual_missing - reported_missing)}.")
    silently_initialized = reported_missing - actual_missing
    if any(not key.endswith(".num_batches_tracked") for key in silently_initialized):
        raise RuntimeError("E12 transfer report does not match load_state_dict missing keys.")

    loaded_state_dict = target_model.state_dict()
    for key, initial_tensor in initial_new_state.items():
        if not torch.equal(loaded_state_dict[key], initial_tensor):
            raise RuntimeError(f"New E12 tensor {key!r} was modified during pretrained transfer.")
    if report.skipped_missing_keys or report.skipped_shape_keys:
        raise RuntimeError("Stock YOLO26 and E12 standard tensors must match by exact key and shape.")
    classified_new_keys = {
        *report.new_adapter_keys,
        *report.new_quality_keys,
        *report.new_foreground_keys,
    }
    if classified_new_keys != set(report.new_ff_keys):
        raise RuntimeError("Only E10 adapters/quality and E12 foreground tensors may remain newly initialized.")
    if not report.new_adapter_keys or not report.new_quality_keys or not report.new_foreground_keys:
        raise RuntimeError("E12 transfer must leave adapters, quality heads, and foreground heads new.")
    for coverage_name in ("backbone", "neck", "cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        coverage = report.coverage(coverage_name)
        if coverage.transferred_tensors != coverage.target_tensors or coverage.percentage != 100.0:
            raise RuntimeError(f"Pretrained transfer is incomplete for E12 {coverage_name}.")
    for coverage_name in ("adapters", "quality_heads", "foreground_heads"):
        coverage = report.coverage(coverage_name)
        if coverage.transferred_tensors != 0 or coverage.transferred_parameter_elements != 0:
            raise RuntimeError(f"No standard YOLO26 tensor may be copied into E12 {coverage_name}.")
    for foreground_head in (*target_head.foreground_heads, *target_head.one2one_foreground_heads):
        if torch.count_nonzero(foreground_head.output.weight.detach()).item() != 0:
            raise RuntimeError("E12 foreground output weights changed during pretrained transfer.")
        if torch.count_nonzero(foreground_head.output.bias.detach()).item() != 0:
            raise RuntimeError("E12 foreground output biases changed during pretrained transfer.")

    quality_config = getattr(target_model, "dual_geometry_quality_config", DualGeometryQualityConfig())
    foreground_config = getattr(
        target_model,
        "foregroundness_factorization_config",
        ForegroundnessFactorizationConfig(),
    )
    _validate_ff_head(target_model, variant, quality_config, foreground_config)
    if verbose:
        print_ff_dgqm_transfer_report(report)
    return report


def ff_dgqm_yaml_path(size: str, variant: str) -> Path:
    """Return the virtual scale-specific E12 YAML path."""
    if size not in MODEL_SIZES:
        raise ValueError(f"Unsupported YOLO26 scale {size!r}; expected one of n, s, m, l, x.")
    if variant not in FF_DGQM_VARIANTS:
        raise ValueError(f"Unsupported FF-DGQM variant {variant!r}; expected control or trainable.")
    model_directory = Path(__file__).resolve().parents[1] / "research" / "models"
    suffix = "ff-dgqm-control" if variant == "control" else "ff-dgqm"
    unified_yaml = model_directory / f"yolo26-{suffix}.yaml"
    if not unified_yaml.is_file():
        raise FileNotFoundError(f"E12 architecture YAML not found: {unified_yaml}")
    return model_directory / f"yolo26{size}-{suffix}.yaml"


def _validate_e10_architecture(target_model: nn.Module, size: str) -> None:
    """Ensure E12 extends only the final E10 Detect with foreground arguments."""
    repository_root = Path(__file__).resolve().parents[1]
    stock = yaml_model_load(repository_root / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{size}.yaml")
    e10 = yaml_model_load(dgqm_yaml_path(size, "trainable"))
    target = target_model.yaml
    for key in ("nc", "end2end", "reg_max", "scales", "backbone", "scale"):
        if target.get(key) != stock.get(key) or target.get(key) != e10.get(key):
            raise RuntimeError(f"E12 architecture unexpectedly changes E10 key {key!r}.")
    if target["head"][:-1] != e10["head"][:-1]:
        raise RuntimeError("E12 must retain every E10 head layer before Detect.")
    if target["head"][-1][0:2] != e10["head"][-1][0:2]:
        raise RuntimeError("E12 must retain E10 Detect inputs and repeat count.")
    if target["head"][-1][2] != "ForegroundnessFactorizedDGQMDetect":
        raise RuntimeError("E12 YAML must replace only E10 Detect with ForegroundnessFactorizedDGQMDetect.")
    e10_args = e10["head"][-1][3]
    if target["head"][-1][3][: len(e10_args)] != e10_args:
        raise RuntimeError("E12 must retain every E10 Detect argument unchanged.")


def build_ff_dgqm_yolo(
    size: str,
    variant: str,
    verbose: bool = False,
    quality_config: DualGeometryQualityConfig | dict | None = None,
    foreground_config: ForegroundnessFactorizationConfig | dict | None = None,
) -> ForegroundnessFactorizedYOLO:
    """Build E12 with fixed E10, local foregroundness, and a standard YOLO26 checkpoint."""
    resolved_quality_config = resolve_dual_geometry_quality_config(quality_config)
    if resolved_quality_config != DualGeometryQualityConfig():
        raise ValueError("E12 must retain the default E10 dual-geometry quality configuration.")
    resolved_foreground_config = resolve_foregroundness_factorization_config(foreground_config)
    architecture_yaml = ff_dgqm_yaml_path(size, variant)
    model = ForegroundnessFactorizedYOLO(
        architecture_yaml,
        quality_config=resolved_quality_config,
        foreground_config=resolved_foreground_config,
        verbose=verbose,
    )
    if model.model.yaml.get("scale") != size:
        raise RuntimeError(f"Expected E12 scale {size!r}, received {model.model.yaml.get('scale')!r}.")
    if model.model.contrast_ring_loss_config != E2_1B_CONFIG:
        raise RuntimeError("E12 must use the fixed E2.1b classification configuration.")
    _validate_e10_architecture(model.model, size)
    _validate_ff_head(
        model.model,
        variant,
        resolved_quality_config,
        resolved_foreground_config,
    )
    _validate_criterion(
        model.model,
        variant,
        resolved_quality_config,
        resolved_foreground_config,
    )

    pretrained = Path(f"yolo26{size}.pt")
    report = load_yolo26_ff_dgqm_pretrained(model, pretrained, verbose=verbose)
    model.ff_dgqm_variant = variant
    model.ff_dgqm_transfer_report = report
    model.ff_dgqm_pretrained = str(pretrained)
    model.ckpt_path = str(pretrained)
    model.ckpt = {"model": model.model}
    return model
