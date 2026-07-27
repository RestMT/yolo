from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .contrast_ring_config import ContrastRingLossConfig, resolve_contrast_ring_config
from .residual_nwd_config import ResidualNWDLossConfig
from .residual_nwd_loss import ResidualNWDDetectionLoss


E2_LOCALIZATION_CONFIG = ResidualNWDLossConfig(mode="constant-010")


def _local_moments(image: torch.Tensor, kernel: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local mean and second moment using replicated boundary padding."""
    padding = kernel // 2
    padded = F.pad(image, (padding, padding, padding, padding), mode="replicate")
    mean = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
    second_moment = F.avg_pool2d(padded.square(), kernel_size=kernel, stride=1)
    return mean, second_moment


def calculate_contrast_ring_map(
    images: torch.Tensor,
    features: Sequence[torch.Tensor],
    config: ContrastRingLossConfig | dict | None = None,
) -> torch.Tensor:
    """Calculate normalized contrast-ring values ordered like flattened detection features."""
    config = resolve_contrast_ring_config(config)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must have shape (batch, 3, height, width), got {tuple(images.shape)}.")
    if not features:
        raise ValueError("features must contain at least one detection level.")

    image = images.detach().float()
    if not torch.is_floating_point(images):
        image = image / 255.0
    image = image.clamp(0.0, 1.0)
    rgb_weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    grayscale = (image * rgb_weights).sum(1, keepdim=True)

    inner_count = config.inner_kernel**2
    outer_count = config.outer_kernel**2
    ring_count = outer_count - inner_count
    contrast_levels = []

    for feature in features:
        if feature.ndim != 4 or feature.shape[0] != images.shape[0]:
            raise ValueError(
                "each feature must have shape (batch, channels, height, width) with the same batch size as images."
            )
        level_gray = F.interpolate(grayscale, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        inner_mean, inner_second = _local_moments(level_gray, config.inner_kernel)
        outer_mean, outer_second = _local_moments(level_gray, config.outer_kernel)

        ring_mean = (outer_count * outer_mean - inner_count * inner_mean) / ring_count
        ring_second = (outer_count * outer_second - inner_count * inner_second) / ring_count
        inner_sigma = torch.sqrt((inner_second - inner_mean.square()).clamp_min(0.0) + config.eps)
        ring_sigma = torch.sqrt((ring_second - ring_mean.square()).clamp_min(0.0) + config.eps)

        contrast = (inner_mean - ring_mean).abs() / (inner_sigma + ring_sigma + config.eps)
        normalized_contrast = (contrast / (contrast + config.contrast_tau)).clamp(0.0, 1.0)
        contrast_levels.append(normalized_contrast.flatten(2).transpose(1, 2).contiguous())

    return torch.cat(contrast_levels, dim=1)


class ContrastRingBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """Elementwise BCE with detached asymmetric contrast-ring weights."""

    def __init__(self, config: ContrastRingLossConfig | dict | None = None):
        """Initialize unreduced BCE and local E2 coefficients."""
        super().__init__(reduction="none")
        self.config = resolve_contrast_ring_config(config)
        self._contrast_map: torch.Tensor | None = None

    def set_contrast_map(self, contrast_map: torch.Tensor) -> None:
        """Set the image-derived per-anchor contrast map for the next BCE call."""
        self._contrast_map = contrast_map.detach()

    def clear_contrast_map(self) -> None:
        """Release the per-batch contrast map after BCE computation."""
        self._contrast_map = None

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """Return contrast-ring-weighted elementwise BCE."""
        bce_loss = super().forward(pred_scores, target_scores)
        if self.config.positive_gain == 0 and self.config.negative_gain == 0:
            return bce_loss
        if self._contrast_map is None:
            raise RuntimeError("contrast map must be set before computing E2 classification loss.")
        if self._contrast_map.shape != (*pred_scores.shape[:2], 1):
            raise ValueError(
                f"contrast map shape {tuple(self._contrast_map.shape)} does not match "
                f"prediction shape {tuple(pred_scores.shape)}."
            )

        contrast = self._contrast_map.to(device=pred_scores.device, dtype=pred_scores.dtype)
        probability = pred_scores.detach().sigmoid()
        positive_mask = target_scores > 0
        positive_weight = 1.0 + self.config.positive_gain * (1.0 - contrast)
        negative_weight = (
            1.0
            + self.config.negative_gain
            * contrast
            * probability.pow(self.config.negative_gamma)
        )
        return bce_loss * torch.where(positive_mask, positive_weight, negative_weight)


class ContrastRingDetectionLoss(ResidualNWDDetectionLoss):
    """E1.1 constant-010 localization plus contrast-ring classification."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        config: ContrastRingLossConfig | dict | None = None,
    ):
        """Initialize fixed E1.1 localization and the isolated E2 BCE criterion."""
        if config is None:
            config = getattr(model, "contrast_ring_loss_config", None)
        self.contrast_ring_config = resolve_contrast_ring_config(config)
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2, config=E2_LOCALIZATION_CONFIG)
        self.bce = ContrastRingBCEWithLogitsLoss(self.contrast_ring_config).to(self.device)

    def calculate_contrast_map(
        self,
        images: torch.Tensor,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Return image-derived contrast values in detection-anchor order."""
        return calculate_contrast_ring_map(images, features, self.contrast_ring_config)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict) -> tuple:
        """Use the stock assignment/loss path with an image-derived BCE weight map."""
        contrast_map = self.calculate_contrast_map(batch["img"], preds["feats"])
        expected_anchors = preds["scores"].shape[-1]
        if contrast_map.shape != (preds["scores"].shape[0], expected_anchors, 1):
            raise RuntimeError(
                f"contrast map shape {tuple(contrast_map.shape)} does not match {expected_anchors} prediction anchors."
            )

        self.bce.set_contrast_map(contrast_map)
        try:
            return super().get_assigned_targets_and_loss(preds, batch)
        finally:
            self.bce.clear_contrast_map()
