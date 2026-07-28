from __future__ import annotations

import math
from dataclasses import dataclass

from .contrast_ring_config import ContrastRingLossConfig


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
class MutualDistillationConfig:
    """Configuration for E3 adaptive mutual branch distillation."""

    classification_gain: float = 0.10
    box_gain: float = 0.05
    temperature: float = 2.0
    confidence_temperature: float = 0.10
    start_epoch: int = 3
    warmup_epochs: int = 5
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate finite coefficients, positive temperatures, and epoch bounds."""
        for name in ("classification_gain", "box_gain", "temperature", "confidence_temperature", "eps"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a finite number, got {value!r}.")
            try:
                finite = math.isfinite(value)
            except TypeError as error:
                raise ValueError(f"{name} must be a finite number, got {value!r}.") from error
            if not finite:
                raise ValueError(f"{name} must be finite, got {value!r}.")

        for name in ("classification_gain", "box_gain"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}.")
        for name in ("temperature", "confidence_temperature"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        for name in ("start_epoch", "warmup_epochs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer, got {value!r}.")
        if self.start_epoch < 0:
            raise ValueError(f"start_epoch must be nonnegative, got {self.start_epoch}.")
        if self.warmup_epochs <= 0:
            raise ValueError(f"warmup_epochs must be positive, got {self.warmup_epochs}.")
        if not 0 < self.eps < 0.5:
            raise ValueError(f"eps must be in (0, 0.5), got {self.eps}.")


def resolve_mutual_distillation_config(
    config: MutualDistillationConfig | dict | None,
) -> MutualDistillationConfig:
    """Return a validated E3 mutual-distillation configuration."""
    if config is None:
        return MutualDistillationConfig()
    if isinstance(config, MutualDistillationConfig):
        return config
    if isinstance(config, dict):
        return MutualDistillationConfig(**config)
    raise TypeError(
        "distillation_config must be MutualDistillationConfig, dict, or None, "
        f"got {type(config).__name__}."
    )
