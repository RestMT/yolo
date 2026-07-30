# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Configuration for E10 dual-geometry quality calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DualGeometryQualityConfig:
    """Configuration for the isolated E10 quality target, loss, and score correction."""

    iou_weight: float = 0.75
    nwd_weight: float = 0.25
    quality_gain: float = 0.25
    negative_neutral_weight: float = 0.05
    quality_scale: float = 1.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate finite coefficients and the normalized dual-geometry mixture."""
        for name in (
            "iou_weight",
            "nwd_weight",
            "quality_gain",
            "negative_neutral_weight",
            "quality_scale",
            "eps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a finite number, got {value!r}.")
            try:
                finite = math.isfinite(value)
            except TypeError as error:
                raise ValueError(f"{name} must be a finite number, got {value!r}.") from error
            if not finite:
                raise ValueError(f"{name} must be finite, got {value!r}.")

        if self.iou_weight < 0:
            raise ValueError(f"iou_weight must be nonnegative, got {self.iou_weight}.")
        if self.nwd_weight < 0:
            raise ValueError(f"nwd_weight must be nonnegative, got {self.nwd_weight}.")
        if self.iou_weight + self.nwd_weight != 1.0:
            raise ValueError(
                f"iou_weight + nwd_weight must equal 1, got {self.iou_weight + self.nwd_weight}."
            )
        if self.quality_gain < 0:
            raise ValueError(f"quality_gain must be nonnegative, got {self.quality_gain}.")
        if self.negative_neutral_weight < 0:
            raise ValueError(
                f"negative_neutral_weight must be nonnegative, got {self.negative_neutral_weight}."
            )
        if self.quality_scale <= 0:
            raise ValueError(f"quality_scale must be positive, got {self.quality_scale}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")


def resolve_dual_geometry_quality_config(
    config: DualGeometryQualityConfig | dict | None,
) -> DualGeometryQualityConfig:
    """Return a validated E10 quality configuration."""
    if config is None:
        return DualGeometryQualityConfig()
    if isinstance(config, DualGeometryQualityConfig):
        return config
    if isinstance(config, dict):
        return DualGeometryQualityConfig(**config)
    raise TypeError(
        f"quality_config must be DualGeometryQualityConfig, dict, or None, got {type(config).__name__}."
    )
