from __future__ import annotations

from dataclasses import dataclass

from .mutual_distillation_config import MutualDistillationConfig


ONE_WAY_DISTILLATION_VARIANTS = ("control", "cls-005", "cls-0025")
_VARIANT_CLASSIFICATION_GAINS = {
    "control": 0.0,
    "cls-005": 0.05,
    "cls-0025": 0.025,
}


@dataclass(frozen=True)
class OneWayDistillationConfig:
    """Configuration for E3.1 one-to-many to one-to-one classification distillation."""

    classification_gain: float = 0.05
    temperature: float = 2.0
    confidence_temperature: float = 0.10
    start_epoch: int = 5
    warmup_epochs: int = 5
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate E3.1 values through the established E3 validation rules."""
        MutualDistillationConfig(
            classification_gain=self.classification_gain,
            box_gain=0.0,
            temperature=self.temperature,
            confidence_temperature=self.confidence_temperature,
            start_epoch=self.start_epoch,
            warmup_epochs=self.warmup_epochs,
            eps=self.eps,
        )


def resolve_one_way_distillation_config(
    config: OneWayDistillationConfig | dict | None,
) -> OneWayDistillationConfig:
    """Return a validated E3.1 configuration."""
    if config is None:
        return OneWayDistillationConfig()
    if isinstance(config, OneWayDistillationConfig):
        return config
    if isinstance(config, dict):
        return OneWayDistillationConfig(**config)
    raise TypeError(
        "distillation_config must be OneWayDistillationConfig, dict, or None, "
        f"got {type(config).__name__}."
    )


def get_one_way_distillation_config(variant: str) -> OneWayDistillationConfig:
    """Return the fixed configuration for an E3.1 experiment variant."""
    if variant not in _VARIANT_CLASSIFICATION_GAINS:
        raise ValueError(f"variant must be one of {ONE_WAY_DISTILLATION_VARIANTS}, got {variant!r}.")
    return OneWayDistillationConfig(classification_gain=_VARIANT_CLASSIFICATION_GAINS[variant])
