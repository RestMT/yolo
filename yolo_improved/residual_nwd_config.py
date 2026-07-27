from __future__ import annotations

import math
from dataclasses import dataclass


RESIDUAL_NWD_MODES = ("control", "constant-005", "constant-010", "adaptive-010")


@dataclass(frozen=True)
class ResidualNWDLossConfig:
    """Configuration for the E1.1 residual NWD localization loss."""

    mode: str = "adaptive-010"
    nwd_scale: float = 0.10
    alpha_max: float = 0.10
    area_threshold: float = 0.01
    area_slope: float = 2.0
    eps: float = 1e-9

    def __post_init__(self) -> None:
        """Validate the selected mode and values used in logarithms and divisions."""
        if self.mode not in RESIDUAL_NWD_MODES:
            raise ValueError(f"mode must be one of {RESIDUAL_NWD_MODES}, got {self.mode!r}.")
        for name in ("nwd_scale", "area_threshold", "area_slope", "eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number, got {value}.")
        if self.area_threshold > 1:
            raise ValueError(f"area_threshold must not exceed normalized image area 1, got {self.area_threshold}.")
        if not math.isfinite(self.alpha_max) or not 0 <= self.alpha_max <= 1:
            raise ValueError(f"alpha_max must be a finite number in [0, 1], got {self.alpha_max}.")
