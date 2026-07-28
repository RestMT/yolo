from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, RANK

from .contrast_ring_model import (
    ContrastRingDetectionModel,
    ContrastRingDetectionTrainer,
    ContrastRingYOLO,
)
from .mutual_distillation_config import (
    E2_1B_CONFIG,
    MutualDistillationConfig,
    resolve_mutual_distillation_config,
)
from .mutual_distillation_loss import MutualDistillationE2ELoss


_DISTILLATION_CONFIG_KEY = "_mutual_distillation_config"


class MutualDistillationDetectionModel(ContrastRingDetectionModel):
    """Detection model with fixed E2.1b supervision and an isolated E3 criterion."""

    def __init__(
        self,
        cfg="yolo26n.yaml",
        ch=3,
        nc=None,
        verbose=True,
        distillation_config: MutualDistillationConfig | dict | None = None,
    ):
        """Initialize the unchanged E2 architecture and local distillation configuration."""
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            loss_config=E2_1B_CONFIG,
        )
        self.mutual_distillation_config = resolve_mutual_distillation_config(distillation_config)

    def init_criterion(self):
        """Initialize E3 for end-to-end YOLO26 and fixed E2.1b otherwise."""
        if self.end2end:
            return MutualDistillationE2ELoss(self, config=self.mutual_distillation_config)
        return super().init_criterion()


class MutualDistillationDetectionTrainer(ContrastRingDetectionTrainer):
    """Detection trainer that constructs MutualDistillationDetectionModel."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the E2 trainer after extracting the local E3 configuration."""
        overrides = dict(overrides or {})
        self.distillation_config = resolve_mutual_distillation_config(overrides.pop(_DISTILLATION_CONFIG_KEY, None))
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        if self.loss_config != E2_1B_CONFIG:
            raise ValueError("E3 requires the fixed E2.1b contrast-ring configuration.")
        if self.ddp:
            setattr(self.args, _DISTILLATION_CONFIG_KEY, asdict(self.distillation_config))

    def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
        """Return an E3 model with the stock detection architecture."""
        distillation_config = getattr(weights, "mutual_distillation_config", self.distillation_config)
        model = self.set_model_names_for_load(
            MutualDistillationDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                distillation_config=distillation_config,
            )
        )
        if weights:
            model.load(weights)
        return model


class MutualDistillationYOLO(ContrastRingYOLO):
    """YOLO facade that isolates E3 training while retaining the stock inference path."""

    def __init__(
        self,
        model: str | Path = "yolo26n.pt",
        task: str | None = None,
        verbose: bool = False,
        distillation_config: MutualDistillationConfig | dict | None = None,
    ):
        """Initialize fixed E2.1b supervision and local E3 distillation."""
        super().__init__(
            model=model,
            task=task,
            verbose=verbose,
            loss_config=E2_1B_CONFIG,
        )
        if self.task != "detect" or not isinstance(self.model, DetectionModel):
            raise ValueError("MutualDistillationYOLO supports only PyTorch detection models.")
        checkpoint_config = getattr(self.model, "mutual_distillation_config", None)
        self.distillation_config = resolve_mutual_distillation_config(
            distillation_config if distillation_config is not None else checkpoint_config
        )
        self.model.mutual_distillation_config = self.distillation_config

    def train(self, trainer=None, **kwargs: Any):
        """Train with fixed E2.1b supervision and local E3 distillation."""
        kwargs[_DISTILLATION_CONFIG_KEY] = asdict(self.distillation_config)
        return super().train(trainer=trainer, **kwargs)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map detection model construction and training to the isolated E3 classes."""
        task_map = super().task_map
        task_map["detect"] = {
            **task_map["detect"],
            "model": MutualDistillationDetectionModel,
            "trainer": MutualDistillationDetectionTrainer,
        }
        return task_map
