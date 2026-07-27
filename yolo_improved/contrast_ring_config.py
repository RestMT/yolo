from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ContrastRingLossConfig:
    """Configuration for the E2 contrast-ring classification loss."""

    inner_kernel: int = 3
    outer_kernel: int = 7
    contrast_tau: float = 0.25
    positive_gain: float = 0.25
    negative_gain: float = 0.25
    negative_gamma: float = 3.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate kernel geometry and finite nonnegative loss coefficients."""
        for name in ("inner_kernel", "outer_kernel"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer, got {value!r}.")
        if self.outer_kernel <= self.inner_kernel:
            raise ValueError(
                f"outer_kernel must be greater than inner_kernel, got {self.outer_kernel} <= {self.inner_kernel}."
            )

        for name in ("contrast_tau", "positive_gain", "negative_gain", "negative_gamma", "eps"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a finite number, got {value!r}.")
            try:
                finite = math.isfinite(value)
            except TypeError as error:
                raise ValueError(f"{name} must be a finite number, got {value!r}.") from error
            if not finite:
                raise ValueError(f"{name} must be finite, got {value!r}.")

        if self.contrast_tau <= 0:
            raise ValueError(f"contrast_tau must be positive, got {self.contrast_tau}.")
        for name in ("positive_gain", "negative_gain", "negative_gamma"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")


def resolve_contrast_ring_config(
    config: ContrastRingLossConfig | dict | None,
) -> ContrastRingLossConfig:
    """Return a validated E2 configuration."""
    if config is None:
        return ContrastRingLossConfig()
    if isinstance(config, ContrastRingLossConfig):
        return config
    if isinstance(config, dict):
        return ContrastRingLossConfig(**config)
    raise TypeError(f"loss_config must be ContrastRingLossConfig, dict, or None, got {type(config).__name__}.")
