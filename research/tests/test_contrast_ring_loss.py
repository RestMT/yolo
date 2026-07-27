from __future__ import annotations

import gc

import pytest
import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss
from yolo_improved import (
    ContrastRingDetectionLoss,
    ContrastRingDetectionModel,
    ContrastRingLossConfig,
    ContrastRingYOLO,
    ResidualNWDBboxLoss,
    ResidualNWDDetectionLoss,
    ResidualNWDDetectionModel,
    calculate_contrast_ring_map,
)
from yolo_improved.contrast_ring_loss import (
    E2_LOCALIZATION_CONFIG,
    ContrastRingBCEWithLogitsLoss,
)


def test_contrast_ring_config_and_invalid_values() -> None:
    """Validate E2 defaults and reject invalid kernel or coefficient values."""
    config = ContrastRingLossConfig()
    assert (config.inner_kernel, config.outer_kernel) == (3, 7)
    assert config.contrast_tau == 0.25
    assert config.positive_gain == 0.25
    assert config.negative_gain == 0.50
    assert config.negative_gamma == 2.0

    for kwargs in (
        {"inner_kernel": 2},
        {"outer_kernel": 3},
        {"contrast_tau": 0},
        {"positive_gain": -0.1},
        {"negative_gain": float("inf")},
        {"negative_gamma": -1},
        {"eps": 0},
    ):
        with pytest.raises(ValueError):
            ContrastRingLossConfig(**kwargs)


def test_contrast_map_shape_anchor_order_and_bounds() -> None:
    """Check per-level flattening, anchor count, finite values, and normalized bounds."""
    torch.manual_seed(7)
    images = torch.rand(2, 3, 32, 40)
    features = [
        torch.empty(2, 16, 8, 10),
        torch.empty(2, 32, 4, 5),
        torch.empty(2, 64, 2, 3),
    ]

    contrast = calculate_contrast_ring_map(images, features)
    expected_anchors = sum(feature.shape[-2] * feature.shape[-1] for feature in features)

    assert contrast.shape == (2, expected_anchors, 1)
    assert torch.isfinite(contrast).all()
    assert torch.all((0 <= contrast) & (contrast <= 1))


def test_contrast_ring_classification_loss_and_gradients() -> None:
    """Check finite asymmetric classification loss and logits gradients."""
    torch.manual_seed(11)
    pred_scores = torch.randn(2, 9, 3, requires_grad=True)
    target_scores = torch.zeros_like(pred_scores)
    target_scores[0, 2, 1] = 0.8
    target_scores[1, 5, 2] = 0.6
    contrast = torch.rand(2, 9, 1)
    class_weights = torch.tensor([[[1.0, 1.5, 0.75]]])

    criterion = ContrastRingBCEWithLogitsLoss()
    criterion.set_contrast_map(contrast)
    target_scores_sum = max(target_scores.sum(), 1)
    cls_loss = (criterion(pred_scores, target_scores) * class_weights).sum() / target_scores_sum
    cls_loss.backward()

    assert torch.isfinite(cls_loss)
    assert pred_scores.grad is not None and torch.isfinite(pred_scores.grad).all()


def test_zero_gains_match_stock_bce_with_class_weights() -> None:
    """Ensure disabled E2 gains reproduce the E1.1 BCE component."""
    torch.manual_seed(13)
    pred_scores = torch.randn(2, 7, 4)
    target_scores = torch.rand(2, 7, 4)
    target_scores[target_scores < 0.75] = 0
    class_weights = torch.tensor([[[0.5, 1.0, 1.5, 2.0]]])
    target_scores_sum = max(target_scores.sum(), 1)

    e2_bce = ContrastRingBCEWithLogitsLoss(
        ContrastRingLossConfig(positive_gain=0, negative_gain=0)
    )
    stock_bce = nn.BCEWithLogitsLoss(reduction="none")
    e2_loss = (e2_bce(pred_scores, target_scores) * class_weights).sum() / target_scores_sum
    stock_loss = (stock_bce(pred_scores, target_scores) * class_weights).sum() / target_scores_sum

    torch.testing.assert_close(e2_loss, stock_loss, rtol=0, atol=0)


def test_e2_reuses_e1_1_localization_without_changing_stock_classes() -> None:
    """Check class isolation and the fixed constant-010 localization configuration."""
    assert issubclass(ContrastRingDetectionLoss, ResidualNWDDetectionLoss)
    assert E2_LOCALIZATION_CONFIG.mode == "constant-010"
    assert E2_LOCALIZATION_CONFIG.nwd_scale == 0.10
    assert BboxLoss.forward.__module__ == "ultralytics.utils.loss"
    assert v8DetectionLoss.get_assigned_targets_and_loss.__module__ == "ultralytics.utils.loss"
    assert ContrastRingYOLO is not YOLO


def test_e2_and_e1_1_architectures_match() -> None:
    """Build models from YAML without weights and compare all parameter and state shapes."""
    e1_1 = ResidualNWDDetectionModel(
        "yolo26n.yaml",
        nc=2,
        verbose=False,
        loss_config=E2_LOCALIZATION_CONFIG,
    )
    e2 = ContrastRingDetectionModel("yolo26n.yaml", nc=2, verbose=False)

    e1_1_parameters = [(name, tuple(parameter.shape), parameter.numel()) for name, parameter in e1_1.named_parameters()]
    e2_parameters = [(name, tuple(parameter.shape), parameter.numel()) for name, parameter in e2.named_parameters()]
    e1_1_state = {name: tuple(value.shape) for name, value in e1_1.state_dict().items()}
    e2_state = {name: tuple(value.shape) for name, value in e2.state_dict().items()}

    assert e2_parameters == e1_1_parameters
    assert e2_state == e1_1_state

    e2.args = get_cfg()
    criterion = e2.init_criterion()
    assert isinstance(criterion, E2ELoss)
    for branch in (criterion.one2many, criterion.one2one):
        assert isinstance(branch, ContrastRingDetectionLoss)
        assert isinstance(branch.bbox_loss, ResidualNWDBboxLoss)
        assert branch.bbox_loss.config == E2_LOCALIZATION_CONFIG
        assert isinstance(branch.bce, ContrastRingBCEWithLogitsLoss)

    del e1_1
    del e2
    gc.collect()
