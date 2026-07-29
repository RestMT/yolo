from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.utils import DATASETS_DIR


CLASS_BALANCED_POSITIVE_MODES = ("control", "effective-099", "uplift-025")


@dataclass(frozen=True)
class ClassBalancedPositiveConfig:
    """Configuration for E4 class-balanced positive classification elements."""

    mode: str = "effective-099"
    beta: float = 0.99
    uplift_strength: float = 0.25
    min_weight: float = 1.0
    max_weight: float = 1.25
    eps: float = 1e-12

    def __post_init__(self) -> None:
        """Validate the fixed E4 modes and effective-number coefficients."""
        if self.mode not in CLASS_BALANCED_POSITIVE_MODES:
            raise ValueError(f"mode must be one of {CLASS_BALANCED_POSITIVE_MODES}, got {self.mode!r}.")
        for name in ("beta", "uplift_strength", "min_weight", "max_weight", "eps"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a finite number, got {value!r}.")
            try:
                finite = math.isfinite(value)
            except TypeError as error:
                raise ValueError(f"{name} must be a finite number, got {value!r}.") from error
            if not finite:
                raise ValueError(f"{name} must be finite, got {value!r}.")
        if not 0 <= self.beta < 1:
            raise ValueError(f"beta must be in [0, 1), got {self.beta}.")
        if self.uplift_strength < 0:
            raise ValueError(f"uplift_strength must be nonnegative, got {self.uplift_strength}.")
        if self.min_weight <= 0:
            raise ValueError(f"min_weight must be positive, got {self.min_weight}.")
        if self.max_weight < self.min_weight:
            raise ValueError(
                f"max_weight must be at least min_weight, got {self.max_weight} < {self.min_weight}."
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")
        if self.mode in {"effective-099", "uplift-025"} and self.beta != 0.99:
            raise ValueError(f"{self.mode} requires beta=0.99, got {self.beta}.")
        if self.mode == "uplift-025":
            fixed_values = {
                "uplift_strength": 0.25,
                "min_weight": 1.0,
                "max_weight": 1.25,
            }
            for name, expected in fixed_values.items():
                value = getattr(self, name)
                if value != expected:
                    raise ValueError(f"uplift-025 requires {name}={expected}, got {value}.")


def resolve_class_balanced_positive_config(
    config: ClassBalancedPositiveConfig | dict | None,
) -> ClassBalancedPositiveConfig:
    """Return a validated E4 configuration."""
    if config is None:
        return ClassBalancedPositiveConfig()
    if isinstance(config, ClassBalancedPositiveConfig):
        return config
    if isinstance(config, dict):
        return ClassBalancedPositiveConfig(**config)
    raise TypeError(
        "class_balanced_config must be ClassBalancedPositiveConfig, dict, or None, "
        f"got {type(config).__name__}."
    )


def get_class_balanced_positive_config(mode: str) -> ClassBalancedPositiveConfig:
    """Return one of the fixed E4 experiment configurations."""
    return ClassBalancedPositiveConfig(mode=mode)


def _parse_class_names(data: dict, data_yaml: Path) -> tuple[str, ...]:
    """Return names ordered by class index and validate them against nc."""
    names_value = data.get("names")
    nc_value = data.get("nc")
    if names_value is None and nc_value is None:
        raise ValueError(f"{data_yaml} must define names or nc.")

    if nc_value is not None:
        if isinstance(nc_value, bool):
            raise ValueError(f"nc must be a positive integer, got {nc_value!r}.")
        try:
            numeric_nc = float(nc_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"nc must be a positive integer, got {nc_value!r}.") from error
        if not math.isfinite(numeric_nc) or not numeric_nc.is_integer() or numeric_nc <= 0:
            raise ValueError(f"nc must be a positive integer, got {nc_value!r}.")
        nc = int(numeric_nc)
    else:
        nc = None

    if names_value is None:
        return tuple(f"class_{class_id}" for class_id in range(nc))
    if isinstance(names_value, list):
        class_names = tuple(str(name) for name in names_value)
    elif isinstance(names_value, dict):
        try:
            indexed_names = {int(class_id): str(name) for class_id, name in names_value.items()}
        except (TypeError, ValueError) as error:
            raise ValueError(f"class-name keys in {data_yaml} must be integer indices.") from error
        if len(indexed_names) != len(names_value):
            raise ValueError(f"class-name keys in {data_yaml} must map to unique integer indices.")
        expected_indices = set(range(len(indexed_names)))
        if set(indexed_names) != expected_indices:
            raise ValueError(
                f"class-name indices in {data_yaml} must be contiguous from 0 to {len(indexed_names) - 1}."
            )
        class_names = tuple(indexed_names[class_id] for class_id in range(len(indexed_names)))
    else:
        raise ValueError(f"names in {data_yaml} must be a list or dictionary.")

    if not class_names:
        raise ValueError(f"{data_yaml} must define at least one class.")
    if nc is not None and len(class_names) != nc:
        raise ValueError(f"names length {len(class_names)} does not match nc={nc} in {data_yaml}.")
    return class_names


def _resolve_dataset_root(data: dict, data_yaml: Path) -> Path:
    """Resolve the dataset root using the current Ultralytics path convention."""
    root_value = data.get("path")
    if not root_value:
        return data_yaml.parent

    root = Path(str(root_value)).expanduser()
    if root.is_absolute() or root.exists():
        return root.resolve()
    return (DATASETS_DIR / root).resolve()


def _resolve_train_source(root: Path, source: str) -> Path:
    """Resolve one train entry, including Ultralytics' fallback for leading ../."""
    source_path = Path(source).expanduser()
    if source_path.is_absolute():
        return source_path.resolve()

    resolved = (root / source_path).resolve()
    normalized_source = source.replace("\\", "/")
    if not resolved.exists() and normalized_source.startswith("../"):
        fallback = (root / normalized_source[3:]).resolve()
        if fallback.exists():
            return fallback
    return resolved


def _read_image_list(list_path: Path) -> list[Path]:
    """Resolve image paths from a text file without loading image contents."""
    image_files = []
    for line_number, raw_line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("./", ".\\")):
            image_path = (list_path.parent / line[2:]).expanduser().resolve()
        else:
            image_path = Path(line).expanduser().resolve()
        if image_path.suffix[1:].lower() not in IMG_FORMATS:
            continue
        if not image_path.is_file():
            raise FileNotFoundError(f"image listed at {list_path}:{line_number} does not exist: {image_path}")
        image_files.append(image_path)
    return image_files


def _collect_training_images(train_sources: Sequence[Path]) -> list[Path]:
    """Collect image files from directories and text lists like the detection dataset loader."""
    image_files = []
    for source in train_sources:
        if source.is_dir():
            image_files.extend(
                path.resolve()
                for path in source.rglob("*")
                if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS
            )
        elif source.is_file():
            image_files.extend(_read_image_list(source))
        else:
            raise FileNotFoundError(f"training image source does not exist: {source}")

    image_files = sorted(image_files, key=str)
    if not image_files:
        raise FileNotFoundError(f"no training images found in: {', '.join(map(str, train_sources))}")
    return image_files


def count_yolo_class_instances(data_yaml: str | Path) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Count YOLO annotation rows per class using only the train split from data.yaml."""
    data_yaml = Path(data_yaml).expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    if not isinstance(data, dict):
        raise ValueError(f"data.yaml must contain a mapping: {data_yaml}")

    class_names = _parse_class_names(data, data_yaml)
    train_value = data.get("train")
    if isinstance(train_value, str) and train_value.strip():
        train_entries = [train_value]
    elif (
        isinstance(train_value, list)
        and train_value
        and all(isinstance(entry, str) and entry.strip() for entry in train_value)
    ):
        train_entries = train_value
    else:
        raise ValueError(f"train in {data_yaml} must be a path or a nonempty list of paths.")

    dataset_root = _resolve_dataset_root(data, data_yaml)
    train_sources = [_resolve_train_source(dataset_root, entry) for entry in train_entries]
    image_files = _collect_training_images(train_sources)
    label_files = [Path(path) for path in img2label_paths([str(path) for path in image_files])]

    class_counts = [0] * len(class_names)
    for label_path in label_files:
        if not label_path.is_file():
            continue
        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            class_token = line.split(maxsplit=1)[0]
            try:
                numeric_class_id = float(class_token)
            except ValueError as error:
                raise ValueError(
                    f"class id at {label_path}:{line_number} must be an integer, got {class_token!r}."
                ) from error
            if not math.isfinite(numeric_class_id) or not numeric_class_id.is_integer():
                raise ValueError(
                    f"class id at {label_path}:{line_number} must be an integer, got {class_token!r}."
                )
            class_id = int(numeric_class_id)
            if class_id < 0:
                raise ValueError(f"class id at {label_path}:{line_number} must be nonnegative, got {class_id}.")
            if class_id >= len(class_names):
                raise ValueError(
                    f"class id {class_id} at {label_path}:{line_number} is outside nc={len(class_names)}."
                )
            class_counts[class_id] += 1

    for class_id, (class_name, count) in enumerate(zip(class_names, class_counts)):
        if count <= 0:
            raise ValueError(f"training class {class_id} ({class_name!r}) has no annotated objects.")
    return tuple(class_counts), class_names


def validate_positive_class_weights(
    weights: torch.Tensor | Sequence[float],
    number_of_classes: int | None = None,
    eps: float = 1e-12,
    require_mean_one: bool = True,
) -> torch.Tensor:
    """Return detached CPU float64 positive weights after shape and optional normalization checks."""
    if weights is None:
        raise ValueError("positive class weights are required.")
    try:
        weights_tensor = torch.as_tensor(weights, dtype=torch.float64, device="cpu").detach()
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("positive class weights must be a one-dimensional numeric vector.") from error
    if weights_tensor.ndim != 1 or weights_tensor.numel() == 0:
        raise ValueError(
            f"positive class weights must be a nonempty one-dimensional vector, got {tuple(weights_tensor.shape)}."
        )
    if number_of_classes is not None and weights_tensor.numel() != number_of_classes:
        raise ValueError(
            f"positive class weights contain {weights_tensor.numel()} values, expected {number_of_classes}."
        )
    if not torch.isfinite(weights_tensor).all() or not torch.all(weights_tensor > 0):
        raise ValueError("all positive class weights must be finite and greater than zero.")
    if require_mean_one and not math.isclose(
        weights_tensor.mean().item(),
        1.0,
        rel_tol=0.0,
        abs_tol=max(eps, 1e-12),
    ):
        raise ValueError(f"positive class weights must have mean 1, got {weights_tensor.mean().item()}.")
    return weights_tensor.clone()


def validate_class_balanced_positive_weights(
    weights: torch.Tensor | Sequence[float],
    config: ClassBalancedPositiveConfig | dict | None = None,
    number_of_classes: int | None = None,
) -> torch.Tensor:
    """Validate E4 weights while preserving the constraints of each fixed mode."""
    config = resolve_class_balanced_positive_config(config)
    weights_tensor = validate_positive_class_weights(
        weights,
        number_of_classes=number_of_classes,
        eps=config.eps,
        require_mean_one=config.mode != "uplift-025",
    )
    if config.mode == "control" and not torch.allclose(
        weights_tensor,
        torch.ones_like(weights_tensor),
        rtol=0.0,
        atol=config.eps,
    ):
        raise ValueError("E4 control mode requires all positive class weights to equal one.")
    if config.mode == "uplift-025":
        if not torch.all(weights_tensor >= config.min_weight):
            raise ValueError(f"uplift-025 positive class weights must be at least {config.min_weight}.")
        if not torch.all(weights_tensor <= config.max_weight):
            raise ValueError(f"uplift-025 positive class weights must be at most {config.max_weight}.")
    return weights_tensor


def _validate_class_counts(class_counts: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Return a CPU float64 vector of finite positive integer counts."""
    try:
        counts = torch.as_tensor(class_counts).detach().to(device="cpu", dtype=torch.float64)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("class counts must be a one-dimensional numeric vector.") from error
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError(f"class counts must be a nonempty one-dimensional vector, got {tuple(counts.shape)}.")
    if not torch.isfinite(counts).all() or not torch.all(counts > 0) or not torch.equal(counts, counts.round()):
        raise ValueError("all class counts must be finite positive integers.")
    return counts


def calculate_effective_number_weights(
    class_counts: Sequence[int] | torch.Tensor,
    beta: float = 0.99,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Calculate mean-one effective-number class weights in float64."""
    if isinstance(beta, bool):
        raise ValueError(f"beta must be a finite number in [0, 1), got {beta!r}.")
    if isinstance(eps, bool):
        raise ValueError(f"eps must be a finite positive number, got {eps!r}.")
    try:
        valid_beta = math.isfinite(beta) and 0 <= beta < 1
    except TypeError as error:
        raise ValueError(f"beta must be a finite number in [0, 1), got {beta!r}.") from error
    try:
        valid_eps = math.isfinite(eps) and eps > 0
    except TypeError as error:
        raise ValueError(f"eps must be a finite positive number, got {eps!r}.") from error
    if not valid_beta:
        raise ValueError(f"beta must be a finite number in [0, 1), got {beta!r}.")
    if not valid_eps:
        raise ValueError(f"eps must be a finite positive number, got {eps!r}.")

    counts = _validate_class_counts(class_counts)

    unnormalized = (1.0 - beta) / (1.0 - torch.pow(beta, counts)).clamp_min(eps)
    normalized = unnormalized / unnormalized.mean()
    return validate_positive_class_weights(normalized, number_of_classes=counts.numel(), eps=eps)


def calculate_positive_uplift_weights(
    effective_weights: torch.Tensor | Sequence[float],
    uplift_strength: float = 0.25,
    min_weight: float = 1.0,
    max_weight: float = 1.25,
) -> torch.Tensor:
    """Apply bounded positive uplift to normalized effective-number weights."""
    for name, value in (
        ("uplift_strength", uplift_strength),
        ("min_weight", min_weight),
        ("max_weight", max_weight),
    ):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number, got {value!r}.")
        try:
            finite = math.isfinite(value)
        except TypeError as error:
            raise ValueError(f"{name} must be a finite number, got {value!r}.") from error
        if not finite:
            raise ValueError(f"{name} must be finite, got {value!r}.")
    if uplift_strength < 0:
        raise ValueError(f"uplift_strength must be nonnegative, got {uplift_strength}.")
    if min_weight <= 0:
        raise ValueError(f"min_weight must be positive, got {min_weight}.")
    if max_weight < min_weight:
        raise ValueError(f"max_weight must be at least min_weight, got {max_weight} < {min_weight}.")

    effective_weights = validate_positive_class_weights(effective_weights)
    final_weights = 1.0 + uplift_strength * torch.clamp(effective_weights - 1.0, min=0.0)
    final_weights = final_weights.clamp(min=min_weight, max=max_weight)
    return validate_positive_class_weights(
        final_weights,
        number_of_classes=effective_weights.numel(),
        require_mean_one=False,
    )


def calculate_class_balanced_positive_weights(
    class_counts: Sequence[int] | torch.Tensor,
    config: ClassBalancedPositiveConfig | dict | None = None,
) -> torch.Tensor:
    """Return positive-only class weights for one fixed E4 mode."""
    config = resolve_class_balanced_positive_config(config)
    counts = _validate_class_counts(class_counts)
    if config.mode == "control":
        weights = torch.ones(counts.numel(), dtype=torch.float64)
    else:
        effective_weights = calculate_effective_number_weights(counts, beta=config.beta, eps=config.eps)
        weights = (
            effective_weights
            if config.mode == "effective-099"
            else calculate_positive_uplift_weights(
                effective_weights,
                uplift_strength=config.uplift_strength,
                min_weight=config.min_weight,
                max_weight=config.max_weight,
            )
        )
    return validate_class_balanced_positive_weights(weights, config=config, number_of_classes=counts.numel())
