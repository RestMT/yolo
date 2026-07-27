from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HybridLossConfig:
    """Configuration for the scale-adaptive hybrid localization loss."""

    area_threshold: float = 0.01
    area_slope: float = 8.0
    nwd_scale: float = 0.10
    shape_weight: float = 0.05
    eps: float = 1e-9

    def __post_init__(self) -> None:
        """Validate values that are used in logarithms and divisions."""
        for name in ("area_threshold", "area_slope", "nwd_scale", "eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number, got {value}.")
        if self.area_threshold > 1:
            raise ValueError(f"area_threshold must not exceed normalized image area 1, got {self.area_threshold}.")
        if not math.isfinite(self.shape_weight) or self.shape_weight < 0:
            raise ValueError(f"shape_weight must be a finite non-negative number, got {self.shape_weight}.")
