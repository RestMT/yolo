from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import unwrap_model

from .class_balanced_config import (
    ClassBalancedPositiveConfig,
    resolve_class_balanced_positive_config,
    validate_class_balanced_positive_weights,
)
from .class_balanced_loss import ClassBalancedContrastRingDetectionLoss
from .contrast_ring_model import (
    ContrastRingDetectionModel,
    ContrastRingDetectionTrainer,
    ContrastRingYOLO,
)
from .mutual_distillation_config import E2_1B_CONFIG


_CLASS_BALANCED_CONFIG_KEY = "_class_balanced_positive_config"
_POSITIVE_CLASS_WEIGHTS_KEY = "_positive_class_weights"


class ClassBalancedDetectionModel(ContrastRingDetectionModel):
    """Detection model with unchanged E2.1b architecture and isolated E4 training weights."""

    def __init__(
        self,
        cfg="yolo26n.yaml",
        ch=3,
        nc=None,
        verbose=True,
        positive_class_weights: torch.Tensor | Sequence[float] | None = None,
        class_balanced_config: ClassBalancedPositiveConfig | dict | None = None,
    ):
        """Initialize fixed E2.1b and store normalized positive-only class weights."""
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            loss_config=E2_1B_CONFIG,
        )
        self.class_balanced_positive_config = resolve_class_balanced_positive_config(class_balanced_config)
        self.positive_class_weights = validate_class_balanced_positive_weights(
            positive_class_weights,
            config=self.class_balanced_positive_config,
            number_of_classes=self.model[-1].nc,
        )

    def init_criterion(self):
        """Initialize E4 for both YOLO26 end-to-end branches without altering their assignment."""
        loss_fn = partial(
            ClassBalancedContrastRingDetectionLoss,
            positive_class_weights=self.positive_class_weights,
            class_balanced_config=self.class_balanced_positive_config,
            config=E2_1B_CONFIG,
        )
        return E2ELoss(self, loss_fn=loss_fn) if self.end2end else loss_fn(self)


class ClassBalancedDetectionTrainer(ContrastRingDetectionTrainer):
    """Detection trainer that passes E4 configuration and weights without globals."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Extract E4-only values before the stock configuration parser runs."""
        overrides = dict(overrides or {})
        self.class_balanced_config = resolve_class_balanced_positive_config(
            overrides.pop(_CLASS_BALANCED_CONFIG_KEY, None)
        )
        self.positive_class_weights = validate_class_balanced_positive_weights(
            overrides.pop(_POSITIVE_CLASS_WEIGHTS_KEY, None),
            config=self.class_balanced_config,
        )
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        if self.loss_config != E2_1B_CONFIG:
            raise ValueError("E4 requires the fixed E2.1b contrast-ring configuration.")
        self.positive_class_weights = validate_class_balanced_positive_weights(
            self.positive_class_weights,
            config=self.class_balanced_config,
            number_of_classes=self.data["nc"],
        )
        if self.ddp:
            setattr(self.args, _CLASS_BALANCED_CONFIG_KEY, asdict(self.class_balanced_config))
            setattr(self.args, _POSITIVE_CLASS_WEIGHTS_KEY, self.positive_class_weights.tolist())

    def set_class_weights(self):
        """Prevent the stock global class-weight route from weighting E4 negative elements."""
        if self.args.cls_pw != 0.0:
            raise ValueError("E4 requires cls_pw=0 because global class weights would also weight negative elements.")
        model = unwrap_model(self.model)
        if getattr(model, "class_weights", None) is not None:
            raise RuntimeError("model.class_weights must be disabled for E4.")
        model.class_weights = None

    def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
        """Return an E4 model with the stock detection architecture."""
        class_balanced_config = getattr(
            weights,
            "class_balanced_positive_config",
            self.class_balanced_config,
        )
        positive_class_weights = getattr(
            weights,
            "positive_class_weights",
            self.positive_class_weights,
        )
        model = self.set_model_names_for_load(
            ClassBalancedDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                positive_class_weights=positive_class_weights,
                class_balanced_config=class_balanced_config,
            )
        )
        if weights:
            model.load(weights)
        return model


class ClassBalancedYOLO(ContrastRingYOLO):
    """YOLO facade that confines E4 class balancing to the training criterion."""

    def __init__(
        self,
        model: str | Path = "yolo26n.pt",
        task: str | None = None,
        verbose: bool = False,
        positive_class_weights: torch.Tensor | Sequence[float] | None = None,
        class_balanced_config: ClassBalancedPositiveConfig | dict | None = None,
    ):
        """Initialize fixed E2.1b supervision and E4 positive-only class weights."""
        super().__init__(
            model=model,
            task=task,
            verbose=verbose,
            loss_config=E2_1B_CONFIG,
        )
        if self.task != "detect" or not isinstance(self.model, DetectionModel):
            raise ValueError("ClassBalancedYOLO supports only PyTorch detection models.")
        checkpoint_config = getattr(self.model, "class_balanced_positive_config", None)
        self.class_balanced_config = resolve_class_balanced_positive_config(
            class_balanced_config if class_balanced_config is not None else checkpoint_config
        )
        checkpoint_weights = getattr(self.model, "positive_class_weights", None)
        self.positive_class_weights = validate_class_balanced_positive_weights(
            positive_class_weights if positive_class_weights is not None else checkpoint_weights,
            config=self.class_balanced_config,
        )
        if getattr(self.model, "class_weights", None) is not None:
            raise ValueError("E4 cannot be combined with global model.class_weights.")
        self.model.class_balanced_positive_config = self.class_balanced_config
        self.model.positive_class_weights = self.positive_class_weights
        self.model.class_weights = None

    def train(self, trainer=None, **kwargs: Any):
        """Train with fixed E2.1b and positive-only E4 class balancing."""
        kwargs[_CLASS_BALANCED_CONFIG_KEY] = asdict(self.class_balanced_config)
        kwargs[_POSITIVE_CLASS_WEIGHTS_KEY] = self.positive_class_weights.tolist()
        return super().train(trainer=trainer, **kwargs)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map E4 detection construction and training to isolated classes."""
        task_map = super().task_map
        task_map["detect"] = {
            **task_map["detect"],
            "model": ClassBalancedDetectionModel,
            "trainer": ClassBalancedDetectionTrainer,
        }
        return task_map
