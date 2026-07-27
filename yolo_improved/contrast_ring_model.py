from __future__ import annotations

from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.loss import E2ELoss

from .contrast_ring_config import ContrastRingLossConfig, resolve_contrast_ring_config
from .contrast_ring_loss import ContrastRingDetectionLoss, E2_LOCALIZATION_CONFIG
from .residual_nwd_model import ResidualNWDDetectionModel


_LOSS_CONFIG_KEY = "_contrast_ring_loss_config"


class ContrastRingDetectionModel(ResidualNWDDetectionModel):
    """Detection model with fixed E1.1 constant-010 localization and E2 classification."""

    def __init__(
        self,
        cfg="yolo26n.yaml",
        ch=3,
        nc=None,
        verbose=True,
        loss_config: ContrastRingLossConfig | dict | None = None,
    ):
        """Initialize the unchanged detection architecture and local E2 configuration."""
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            loss_config=E2_LOCALIZATION_CONFIG,
        )
        self.contrast_ring_loss_config = resolve_contrast_ring_config(loss_config)

    def init_criterion(self):
        """Initialize E2 losses for both end-to-end YOLO26 branches."""
        loss_fn = partial(ContrastRingDetectionLoss, config=self.contrast_ring_loss_config)
        return E2ELoss(self, loss_fn=loss_fn) if self.end2end else loss_fn(self)


class ContrastRingDetectionTrainer(DetectionTrainer):
    """Detection trainer that constructs ContrastRingDetectionModel."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the stock trainer after extracting the local E2 configuration."""
        overrides = dict(overrides or {})
        self.loss_config = resolve_contrast_ring_config(overrides.pop(_LOSS_CONFIG_KEY, None))
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        if self.ddp:
            setattr(self.args, _LOSS_CONFIG_KEY, asdict(self.loss_config))

    def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
        """Return an E2 model with the stock detection architecture."""
        loss_config = getattr(weights, "contrast_ring_loss_config", self.loss_config)
        model = self.set_model_names_for_load(
            ContrastRingDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                loss_config=loss_config,
            )
        )
        if weights:
            model.load(weights)
        return model


class ContrastRingYOLO(YOLO):
    """YOLO facade that isolates E2 contrast-ring classification training."""

    def __init__(
        self,
        model: str | Path = "yolo26n.pt",
        task: str | None = None,
        verbose: bool = False,
        loss_config: ContrastRingLossConfig | dict | None = None,
    ):
        """Initialize a detection model with a local E2 loss configuration."""
        super().__init__(model=model, task=task, verbose=verbose)
        if self.task != "detect" or not isinstance(self.model, DetectionModel):
            raise ValueError("ContrastRingYOLO supports only PyTorch detection models.")
        checkpoint_config = getattr(self.model, "contrast_ring_loss_config", None)
        self.loss_config = resolve_contrast_ring_config(loss_config if loss_config is not None else checkpoint_config)
        self.model.contrast_ring_loss_config = self.loss_config

    def train(self, trainer=None, **kwargs: Any):
        """Train with fixed E1.1 constant-010 localization and local E2 classification."""
        kwargs[_LOSS_CONFIG_KEY] = asdict(self.loss_config)
        return super().train(trainer=trainer, **kwargs)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map detection model construction and training to the isolated E2 classes."""
        task_map = super().task_map
        task_map["detect"] = {
            **task_map["detect"],
            "model": ContrastRingDetectionModel,
            "trainer": ContrastRingDetectionTrainer,
        }
        return task_map
