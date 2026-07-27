from __future__ import annotations

import pytest
import torch

from ultralytics.utils.loss import BboxLoss
from yolo_improved import RESIDUAL_NWD_MODES, ResidualNWDBboxLoss, ResidualNWDLossConfig


def synthetic_inputs() -> tuple[torch.Tensor, ...]:
    """Create a small deterministic bounding-box batch without loading a model or dataset."""
    pred_dist = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32).reshape(1, 2, 16).requires_grad_()
    pred_bboxes = torch.tensor(
        [[[1.0, 1.1, 3.1, 3.2], [3.8, 3.9, 5.4, 5.5]]], dtype=torch.float32, requires_grad=True
    )
    anchor_points = torch.tensor([[2.0, 2.0], [4.5, 4.5]], dtype=torch.float32)
    target_bboxes = torch.tensor([[[1.1, 1.0, 3.2, 3.0], [4.0, 4.0, 5.2, 5.3]]], dtype=torch.float32)
    target_scores = torch.tensor([[[0.8, 0.1], [0.7, 0.2]]], dtype=torch.float32)
    target_scores_sum = target_scores.sum()
    fg_mask = torch.tensor([[True, True]])
    imgsz = torch.tensor([64.0, 96.0])
    stride = torch.tensor([[8.0], [16.0]])
    return (
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride,
    )


@pytest.mark.parametrize(
    ("mode", "expected_alpha"),
    (("control", 0.0), ("constant-005", 0.05), ("constant-010", 0.10), ("adaptive-010", None)),
)
def test_residual_nwd_config_modes(mode: str, expected_alpha: float | None) -> None:
    """Validate every public E1.1 mode and each fixed coefficient."""
    config = ResidualNWDLossConfig(mode=mode)
    assert config.mode in RESIDUAL_NWD_MODES
    if expected_alpha is not None:
        alpha = ResidualNWDBboxLoss(config=config).calculate_alpha(torch.tensor([[0.01]]))
        torch.testing.assert_close(alpha, torch.tensor([[expected_alpha]]), rtol=0, atol=0)


def test_control_matches_stock_bbox_loss() -> None:
    """Ensure the isolated control route reproduces stock CIoU and DFL."""
    inputs = synthetic_inputs()
    stock = BboxLoss(reg_max=4)(*inputs)
    control = ResidualNWDBboxLoss(reg_max=4, config=ResidualNWDLossConfig(mode="control"))(*inputs)

    torch.testing.assert_close(control[0], stock[0], rtol=0, atol=1e-7)
    torch.testing.assert_close(control[1], stock[1], rtol=0, atol=1e-7)


def test_adaptive_residual_nwd_is_finite_and_small_object_weighted() -> None:
    """Check loss bounds, gradients, and adaptive small-object weighting."""
    config = ResidualNWDLossConfig(mode="adaptive-010")
    criterion = ResidualNWDBboxLoss(reg_max=4, config=config)
    inputs = synthetic_inputs()

    box_loss, dfl_loss = criterion(*inputs)
    total_loss = box_loss + dfl_loss
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert inputs[0].grad is not None and torch.isfinite(inputs[0].grad).all()
    assert inputs[1].grad is not None and torch.isfinite(inputs[1].grad).all()

    areas = torch.tensor([[1e-4], [0.25]], dtype=torch.float32)
    alpha = criterion.calculate_alpha(areas)
    assert torch.all((0 <= alpha) & (alpha <= config.alpha_max))
    assert alpha[0] > alpha[1]

    pred_xywh = torch.tensor([[0.5, 0.5, 1e-12, 1e-12]], dtype=torch.float32, requires_grad=True)
    target_xywh = torch.tensor([[0.500001, 0.499999, 2e-12, 2e-12]], dtype=torch.float32)
    nwd_loss = criterion.calculate_nwd_loss(pred_xywh, target_xywh)
    assert torch.isfinite(nwd_loss).all()
    assert torch.all(nwd_loss >= 0)

    nwd_loss.sum().backward()
    assert pred_xywh.grad is not None and torch.isfinite(pred_xywh.grad).all()

    pred_known = torch.tensor([[0.55, 0.5, 0.2, 0.2]])
    target_known = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    expected = -torch.expm1(-torch.sqrt(torch.tensor(0.05**2 + config.eps)) / config.nwd_scale)
    torch.testing.assert_close(criterion.calculate_nwd_loss(pred_known, target_known).squeeze(), expected)
