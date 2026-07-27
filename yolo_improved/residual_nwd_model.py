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

from .residual_nwd_config import ResidualNWDLossConfig
from .residual_nwd_loss import ResidualNWDDetectionLoss


_LOSS_CONFIG_KEY = "_residual_nwd_loss_config"


def _resolve_loss_config(config: ResidualNWDLossConfig | dict | None) -> ResidualNWDLossConfig:
    """Return a validated residual NWD configuration."""
    if config is None:
        return ResidualNWDLossConfig()
    if isinstance(config, ResidualNWDLossConfig):
        return config
    if isinstance(config, dict):
        return ResidualNWDLossConfig(**config)
    raise TypeError(f"loss_config must be ResidualNWDLossConfig, dict, or None, got {type(config).__name__}.")


class ResidualNWDDetectionModel(DetectionModel):
    """Detection model with an isolated residual NWD training criterion."""

    def __init__(
        self,
        cfg="yolo26n.yaml",
        ch=3,
        nc=None,
        verbose=True,
        loss_config: ResidualNWDLossConfig | dict | None = None,
    ):
        """Initialize the unchanged detection architecture and local loss configuration."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.residual_nwd_loss_config = _resolve_loss_config(loss_config)

    def init_criterion(self):
        """Initialize residual NWD losses for both end-to-end YOLO26 branches."""
        loss_fn = partial(ResidualNWDDetectionLoss, config=self.residual_nwd_loss_config)
        return E2ELoss(self, loss_fn=loss_fn) if self.end2end else loss_fn(self)


class ResidualNWDDetectionTrainer(DetectionTrainer):
    """Detection trainer that constructs ResidualNWDDetectionModel."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the stock trainer after extracting the local loss configuration."""
        overrides = dict(overrides or {})
        self.loss_config = _resolve_loss_config(overrides.pop(_LOSS_CONFIG_KEY, None))
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        setattr(self.args, _LOSS_CONFIG_KEY, asdict(self.loss_config))

    def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
        """Return a residual NWD model with the stock detection architecture."""
        loss_config = getattr(weights, "residual_nwd_loss_config", self.loss_config)
        model = self.set_model_names_for_load(
            ResidualNWDDetectionModel(
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


class ResidualNWDYOLO(YOLO):
    """YOLO facade that isolates E1-control and E1.1 detection training."""

    def __init__(
        self,
        model: str | Path = "yolo26n.pt",
        task: str | None = None,
        verbose: bool = False,
        loss_config: ResidualNWDLossConfig | dict | None = None,
    ):
        """Initialize a detection model with a local residual NWD configuration."""
        loss_config = _resolve_loss_config(loss_config)
        super().__init__(model=model, task=task, verbose=verbose)
        if self.task != "detect" or not isinstance(self.model, DetectionModel):
            raise ValueError("ResidualNWDYOLO supports only PyTorch detection models.")
        self.loss_config = loss_config
        self.model.residual_nwd_loss_config = loss_config

    def train(self, trainer=None, **kwargs: Any):
        """Train with the local residual NWD configuration."""
        kwargs[_LOSS_CONFIG_KEY] = asdict(self.loss_config)
        return super().train(trainer=trainer, **kwargs)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map detection model construction and training to the E1.1 classes."""
        task_map = super().task_map
        task_map["detect"] = {
            **task_map["detect"],
            "model": ResidualNWDDetectionModel,
            "trainer": ResidualNWDDetectionTrainer,
        }
        return task_map
