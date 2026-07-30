# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Configuration for E11 class-conditional score suppression."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassConditionalSuppressionConfig:
    """Configuration for the isolated E11 suppression target, loss, and score correction."""

    suppression_gain: float = 0.10
    suppression_scale: float = 1.0
    hard_negative_gamma: float = 2.0
    hard_negative_min_probability: float = 0.05
    positive_neutral_weight: float = 1.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate finite suppression coefficients and probability bounds."""
        for name in (
            "suppression_gain",
            "suppression_scale",
            "hard_negative_gamma",
            "hard_negative_min_probability",
            "positive_neutral_weight",
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

        if self.suppression_gain < 0:
            raise ValueError(f"suppression_gain must be nonnegative, got {self.suppression_gain}.")
        if self.suppression_scale <= 0:
            raise ValueError(f"suppression_scale must be positive, got {self.suppression_scale}.")
        if self.hard_negative_gamma < 0:
            raise ValueError(f"hard_negative_gamma must be nonnegative, got {self.hard_negative_gamma}.")
        if not 0 <= self.hard_negative_min_probability < 1:
            raise ValueError(
                "hard_negative_min_probability must be in [0, 1), "
                f"got {self.hard_negative_min_probability}."
            )
        if self.positive_neutral_weight < 0:
            raise ValueError(
                f"positive_neutral_weight must be nonnegative, got {self.positive_neutral_weight}."
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")


def resolve_class_conditional_suppression_config(
    config: ClassConditionalSuppressionConfig | dict | None,
) -> ClassConditionalSuppressionConfig:
    """Return a validated E11 suppression configuration."""
    if config is None:
        return ClassConditionalSuppressionConfig()
    if isinstance(config, ClassConditionalSuppressionConfig):
        return config
    if isinstance(config, dict):
        return ClassConditionalSuppressionConfig(**config)
    raise TypeError(
        "suppression_config must be ClassConditionalSuppressionConfig, dict, or None, "
        f"got {type(config).__name__}."
    )
