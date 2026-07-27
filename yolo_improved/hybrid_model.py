from __future__ import annotations

from functools import partial
from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import E2ELoss

from .hybrid_config import HybridLossConfig
from .hybrid_loss import HybridDetectionLoss


class HybridDetectionModel(DetectionModel):
    """Detection model with an isolated hybrid training criterion."""

    def __init__(
        self,
        cfg="yolo26n.yaml",
        ch=3,
        nc=None,
        verbose=True,
        hybrid_loss_config: HybridLossConfig | None = None,
    ):
        """Initialize the unchanged detection architecture and local loss configuration."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.hybrid_loss_config = hybrid_loss_config or HybridLossConfig()

    def init_criterion(self):
        """Initialize hybrid losses for both branches of end-to-end YOLO26 training."""
        loss_fn = partial(HybridDetectionLoss, config=self.hybrid_loss_config)
        return E2ELoss(self, loss_fn=loss_fn) if self.end2end else loss_fn(self)


class HybridDetectionTrainer(DetectionTrainer):
    """Detection trainer that constructs HybridDetectionModel instead of DetectionModel."""

    def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
        """Return a hybrid-criterion model with the stock detection architecture."""
        hybrid_loss_config = getattr(weights, "hybrid_loss_config", None)
        model = self.set_model_names_for_load(
            HybridDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                hybrid_loss_config=hybrid_loss_config,
            )
        )
        if weights:
            model.load(weights)
        return model


class HybridYOLO(YOLO):
    """YOLO facade that selects the isolated hybrid model and trainer only for detection training."""

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map detection model construction and training to the isolated E1 classes."""
        task_map = super().task_map
        task_map["detect"] = {
            **task_map["detect"],
            "model": HybridDetectionModel,
            "trainer": HybridDetectionTrainer,
        }
        return task_map
