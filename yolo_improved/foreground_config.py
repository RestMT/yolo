# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Configuration for E12 class-agnostic foregroundness factorization."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ForegroundnessFactorizationConfig:
    """Configuration for the isolated E12 foreground target, loss, and score correction."""

    foreground_gain: float = 0.10
    foreground_scale: float = 0.5
    hard_negative_gamma: float = 2.0
    hard_negative_min_probability: float = 0.05
    negative_weight: float = 0.5
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate finite foreground coefficients and probability bounds."""
        for name in (
            "foreground_gain",
            "foreground_scale",
            "hard_negative_gamma",
            "hard_negative_min_probability",
            "negative_weight",
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

        if self.foreground_gain < 0:
            raise ValueError(f"foreground_gain must be nonnegative, got {self.foreground_gain}.")
        if self.foreground_scale <= 0:
            raise ValueError(f"foreground_scale must be positive, got {self.foreground_scale}.")
        if self.hard_negative_gamma < 0:
            raise ValueError(f"hard_negative_gamma must be nonnegative, got {self.hard_negative_gamma}.")
        if not 0 <= self.hard_negative_min_probability < 1:
            raise ValueError(
                "hard_negative_min_probability must be in [0, 1), "
                f"got {self.hard_negative_min_probability}."
            )
        if self.negative_weight < 0:
            raise ValueError(f"negative_weight must be nonnegative, got {self.negative_weight}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")


def resolve_foregroundness_factorization_config(
    config: ForegroundnessFactorizationConfig | dict | None,
) -> ForegroundnessFactorizationConfig:
    """Return a validated E12 foregroundness configuration."""
    if config is None:
        return ForegroundnessFactorizationConfig()
    if isinstance(config, ForegroundnessFactorizationConfig):
        return config
    if isinstance(config, dict):
        return ForegroundnessFactorizationConfig(**config)
    raise TypeError(
        "foreground_config must be ForegroundnessFactorizationConfig, dict, or None, "
        f"got {type(config).__name__}."
    )
