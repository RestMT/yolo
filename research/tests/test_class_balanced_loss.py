from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import torch
import yaml

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import img2label_paths
from ultralytics.utils.loss import E2ELoss
from yolo_improved import (
    CLASS_BALANCED_POSITIVE_MODES,
    E2_1B_CONFIG,
    ClassBalancedContrastRingBCEWithLogitsLoss,
    ClassBalancedContrastRingDetectionLoss,
    ClassBalancedDetectionModel,
    ClassBalancedDetectionTrainer,
    ClassBalancedPositiveConfig,
    ClassBalancedYOLO,
    ContrastRingDetectionModel,
    ContrastRingLossConfig,
    MutualDistillationYOLO,
    OneWayDistillationYOLO,
    calculate_class_balanced_positive_weights,
    calculate_effective_number_weights,
    calculate_positive_uplift_weights,
    count_yolo_class_instances,
    get_class_balanced_positive_config,
)
from yolo_improved.class_balanced_config import validate_class_balanced_positive_weights
from yolo_improved.contrast_ring_loss import ContrastRingBCEWithLogitsLoss, E2_LOCALIZATION_CONFIG


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _create_image_and_label(images_directory: Path, image_name: str, rows: list[str]) -> None:
    """Create empty image and YOLO label files for path/counting checks."""
    image_path = images_directory / image_name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.touch()
    label_path = Path(img2label_paths([str(image_path)])[0])
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _synthetic_predictions(nc: int, reg_max: int) -> dict[str, dict[str, torch.Tensor]]:
    """Create small aligned end-to-end predictions without model inference."""
    torch.manual_seed(43)
    feature_shapes = ((8, 8), (4, 4), (2, 2))
    number_of_anchors = sum(height * width for height, width in feature_shapes)
    features = [torch.rand(1, 8, height, width) for height, width in feature_shapes]
    return {
        "one2many": {
            "boxes": torch.randn(1, 4 * reg_max, number_of_anchors, requires_grad=True),
            "scores": torch.randn(1, nc, number_of_anchors, requires_grad=True),
            "feats": features,
        },
        "one2one": {
            "boxes": torch.randn(1, 4 * reg_max, number_of_anchors, requires_grad=True),
            "scores": torch.randn(1, nc, number_of_anchors, requires_grad=True),
            "feats": [feature.detach() for feature in features],
        },
    }


def _synthetic_batch() -> dict[str, torch.Tensor]:
    """Create one small detection batch."""
    return {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }


def test_class_balanced_config_modes_and_validation() -> None:
    """Check fixed E4 modes and reject invalid values."""
    assert CLASS_BALANCED_POSITIVE_MODES == ("control", "effective-099", "uplift-025")
    assert get_class_balanced_positive_config("control") == ClassBalancedPositiveConfig(mode="control")
    assert get_class_balanced_positive_config("effective-099") == ClassBalancedPositiveConfig()
    assert get_class_balanced_positive_config("uplift-025") == ClassBalancedPositiveConfig(mode="uplift-025")
    for kwargs in (
        {"mode": "unknown"},
        {"beta": -0.1},
        {"beta": 1.0},
        {"beta": float("inf")},
        {"mode": "effective-099", "beta": 0.98},
        {"mode": "uplift-025", "beta": 0.98},
        {"uplift_strength": -0.1},
        {"uplift_strength": float("inf")},
        {"min_weight": 0},
        {"min_weight": float("nan")},
        {"min_weight": 1.2, "max_weight": 1.1},
        {"max_weight": float("inf")},
        {"mode": "uplift-025", "uplift_strength": 0.5},
        {"mode": "uplift-025", "min_weight": 0.9},
        {"mode": "uplift-025", "max_weight": 1.5},
        {"eps": 0},
        {"eps": float("nan")},
    ):
        with pytest.raises(ValueError):
            ClassBalancedPositiveConfig(**kwargs)


def test_count_yolo_class_instances_supports_relative_directory_list_and_absolute_path(tmp_path: Path) -> None:
    """Count train-only labels from relative directories, a list file, and an absolute directory."""
    dataset = tmp_path / "dataset"
    _create_image_and_label(dataset / "images" / "train", "a.jpg", ["0 0.5 0.5 0.2 0.2", "2 0.4 0.4 0.1 0.1"])
    _create_image_and_label(dataset / "extra" / "images", "b.png", ["1 0.5 0.5 0.2 0.2", "1 0.4 0.4 0.1 0.1"])
    _create_image_and_label(dataset / "valid" / "images", "ignored.jpg", ["99 0.5 0.5 0.2 0.2"])
    image_list = dataset / "train.txt"
    image_list.write_text("./extra/images/b.png\n", encoding="utf-8")
    relative_yaml = dataset / "relative.yaml"
    relative_yaml.write_text(
        yaml.safe_dump(
            {
                "train": ["images/train", "train.txt"],
                "val": "valid/images",
                "nc": 3,
                "names": ["first", "second", "third"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    counts, names = count_yolo_class_instances(relative_yaml)
    assert counts == (1, 2, 1)
    assert names == ("first", "second", "third")

    absolute_images = dataset / "absolute" / "images"
    _create_image_and_label(
        absolute_images,
        "c.jpeg",
        ["0 0.5 0.5 0.2 0.2", "1 0.4 0.4 0.1 0.1", "2 0.3 0.3 0.1 0.1"],
    )
    absolute_yaml = dataset / "absolute.yaml"
    absolute_yaml.write_text(
        yaml.safe_dump(
            {
                "train": str(absolute_images),
                "val": str(absolute_images),
                "names": {2: "third", 0: "first", 1: "second"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert count_yolo_class_instances(absolute_yaml) == ((1, 1, 1), names)

    fallback_images = dataset / "fallback" / "images"
    _create_image_and_label(
        fallback_images,
        "d.jpg",
        ["0 0.5 0.5 0.2 0.2", "1 0.4 0.4 0.1 0.1", "2 0.3 0.3 0.1 0.1"],
    )
    fallback_yaml = dataset / "fallback.yaml"
    fallback_yaml.write_text(
        yaml.safe_dump(
            {
                "train": "../fallback/images",
                "val": "../fallback/images",
                "nc": 3,
                "names": list(names),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert count_yolo_class_instances(fallback_yaml) == ((1, 1, 1), names)


def test_count_yolo_class_instances_rejects_invalid_or_empty_classes(tmp_path: Path) -> None:
    """Reject out-of-range class ids and report empty classes with index and name."""
    dataset = tmp_path / "dataset"
    _create_image_and_label(dataset / "images", "a.jpg", ["2 0.5 0.5 0.2 0.2"])
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"train": "images", "val": "images", "nc": 2, "names": ["first", "second"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside nc=2"):
        count_yolo_class_instances(data_yaml)

    label_path = dataset / "labels" / "a.txt"
    label_path.write_text("-1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nonnegative"):
        count_yolo_class_instances(data_yaml)

    label_path.write_text("0.5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integer"):
        count_yolo_class_instances(data_yaml)

    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"class 1 \('second'\)"):
        count_yolo_class_instances(data_yaml)


def test_effective_number_weights_are_float64_normalized_and_favor_rare_classes() -> None:
    """Check the requested formula, mean-one normalization, and rarity ordering."""
    counts = torch.tensor([10, 100, 1000])
    weights = calculate_effective_number_weights(counts, beta=0.99)
    raw = (1.0 - 0.99) / (1.0 - torch.pow(torch.tensor(0.99, dtype=torch.float64), counts.double()))
    expected = raw / raw.mean()

    assert weights.dtype == torch.float64
    torch.testing.assert_close(weights, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12, rtol=0)
    assert weights[0] > weights[1] > weights[2]
    assert torch.equal(
        calculate_class_balanced_positive_weights(counts, ClassBalancedPositiveConfig(mode="control")),
        torch.ones(3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        calculate_class_balanced_positive_weights(
            counts,
            ClassBalancedPositiveConfig(mode="effective-099"),
        ),
        weights,
        atol=0,
        rtol=0,
    )


def test_positive_uplift_weights_match_expected_values_and_bounds() -> None:
    """Check E4.1a uplift values, common-class floor, rare-class ceiling, and automatic calculation."""
    counts = [62, 16, 11]
    effective_weights = calculate_effective_number_weights(counts, beta=0.99)
    final_weights = calculate_positive_uplift_weights(
        effective_weights,
        uplift_strength=0.25,
        min_weight=1.0,
        max_weight=1.25,
    )
    calculated_weights = calculate_class_balanced_positive_weights(
        counts,
        ClassBalancedPositiveConfig(mode="uplift-025"),
    )

    torch.testing.assert_close(final_weights, calculated_weights, atol=0, rtol=0)
    torch.testing.assert_close(
        final_weights,
        torch.tensor([1.0, 1.0237646638, 1.1385432358], dtype=torch.float64),
        atol=5e-10,
        rtol=0,
    )
    assert effective_weights[0] < 1
    assert final_weights[0].item() == 1.0
    assert torch.all(final_weights >= 1.0)
    assert torch.all(final_weights <= 1.25)

    capped_weights = calculate_positive_uplift_weights([0.01, 0.01, 2.98])
    assert capped_weights[-1].item() == 1.25
    uplift_config = ClassBalancedPositiveConfig(mode="uplift-025")
    with pytest.raises(ValueError, match="at least"):
        validate_class_balanced_positive_weights([0.99, 1.01], config=uplift_config)
    with pytest.raises(ValueError, match="at most"):
        validate_class_balanced_positive_weights([1.0, 1.26], config=uplift_config)


def test_positive_weights_preserve_float64_precision_across_trainer_list_transport() -> None:
    """Keep mean-one weights exact when the model passes them to the trainer as a Python list."""
    weights = calculate_effective_number_weights([9577, 1728, 1100])
    restored = validate_class_balanced_positive_weights(
        weights.tolist(),
        config=ClassBalancedPositiveConfig(mode="effective-099"),
        number_of_classes=3,
    )

    assert restored.dtype == torch.float64
    assert restored.mean().item() == 1.0
    torch.testing.assert_close(restored, weights, rtol=0.0, atol=0.0)


def test_positive_weights_change_only_positive_e2_1b_elements_and_support_amp() -> None:
    """Verify buffer shape, soft targets, unchanged negatives, finite loss, and gradients."""
    weights = torch.tensor([0.5, 1.5], dtype=torch.float64)
    base = ContrastRingBCEWithLogitsLoss(E2_1B_CONFIG)
    balanced = ClassBalancedContrastRingBCEWithLogitsLoss(weights, E2_1B_CONFIG)
    logits = torch.tensor([[[1.5, -0.5], [-1.0, 2.0]]], requires_grad=True)
    targets = torch.tensor([[[0.8, 0.0], [0.0, 0.6]]])
    contrast = torch.tensor([[[0.2], [0.7]]])
    base.set_contrast_map(contrast)
    balanced.set_contrast_map(contrast)

    base_loss = base(logits, targets)
    balanced_loss = balanced(logits, targets)
    positive_mask = targets > 0
    expected_multiplier = torch.where(positive_mask, weights.view(1, 1, -1).float(), torch.tensor(1.0))
    raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = logits.detach().sigmoid()
    positive_weight = 1.0 + 0.25 * (1.0 - contrast)
    negative_weight = 1.0 + 0.25 * contrast * probability.pow(3.0)
    expected_base_loss = raw_bce * torch.where(positive_mask, positive_weight, negative_weight)

    assert balanced.positive_class_weights.shape == (1, 1, 2)
    assert balanced.positive_class_weights.dtype == torch.float64
    torch.testing.assert_close(base_loss, expected_base_loss)
    torch.testing.assert_close(balanced_loss, base_loss * expected_multiplier)
    torch.testing.assert_close(balanced_loss[~positive_mask], base_loss[~positive_mask])

    amp_logits = logits.detach().clone().requires_grad_()
    balanced.set_contrast_map(contrast)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_loss = balanced(amp_logits, targets).sum()
    amp_loss.backward()
    assert torch.isfinite(amp_loss)
    assert amp_logits.grad is not None and torch.isfinite(amp_logits.grad).all()


def test_control_bce_exactly_matches_e2_1b() -> None:
    """Check elementwise E2.1b equivalence when every positive class weight is one."""
    base = ContrastRingBCEWithLogitsLoss(E2_1B_CONFIG)
    control = ClassBalancedContrastRingBCEWithLogitsLoss(torch.ones(3), E2_1B_CONFIG)
    logits = torch.randn(2, 5, 3)
    targets = torch.rand(2, 5, 3)
    targets[targets < 0.7] = 0
    contrast = torch.rand(2, 5, 1)
    base.set_contrast_map(contrast)
    control.set_contrast_map(contrast)
    torch.testing.assert_close(control(logits, targets), base(logits, targets), atol=0, rtol=0)


def test_e4_disables_stock_global_class_weights_and_reuses_fixed_e2_1b() -> None:
    """Check both E4 branches and the double-weighting guard."""
    model = ClassBalancedDetectionModel(
        "yolo26n.yaml",
        nc=2,
        verbose=False,
        positive_class_weights=torch.ones(2),
        class_balanced_config=ClassBalancedPositiveConfig(mode="control"),
    )
    model.args = get_cfg()
    criterion = model.init_criterion()
    assert isinstance(criterion, E2ELoss)
    assert E2_LOCALIZATION_CONFIG.mode == "constant-010"
    assert E2_LOCALIZATION_CONFIG.nwd_scale == 0.10
    assert E2_1B_CONFIG == ContrastRingLossConfig(
        inner_kernel=3,
        outer_kernel=7,
        contrast_tau=0.25,
        positive_gain=0.25,
        negative_gain=0.25,
        negative_gamma=3.0,
        eps=1e-6,
    )
    for branch in (criterion.one2many, criterion.one2one):
        assert isinstance(branch, ClassBalancedContrastRingDetectionLoss)
        assert branch.class_weights is None
        assert branch.contrast_ring_config == E2_1B_CONFIG
        assert branch.bbox_loss.config == E2_LOCALIZATION_CONFIG
    assert criterion.one2many.assigner.topk == 10
    assert criterion.one2one.assigner.topk == 7
    assert criterion.one2one.assigner.topk2 == 1

    model.class_weights = torch.ones(2)
    with pytest.raises(ValueError, match="negative"):
        model.init_criterion()
    trainer_source = inspect.getsource(ClassBalancedDetectionTrainer.set_class_weights)
    assert "cls_pw != 0.0" in trainer_source
    assert "model.class_weights = None" in trainer_source


def test_e4_control_reproduces_full_e2_1b_loss() -> None:
    """Compare the complete end-to-end optimization vector in control mode."""
    e2_model = ContrastRingDetectionModel("yolo26n.yaml", nc=2, verbose=False, loss_config=E2_1B_CONFIG)
    e4_model = ClassBalancedDetectionModel(
        "yolo26n.yaml",
        nc=2,
        verbose=False,
        positive_class_weights=torch.ones(2),
        class_balanced_config=ClassBalancedPositiveConfig(mode="control"),
    )
    e2_model.args = get_cfg()
    e4_model.args = get_cfg()
    e2_criterion = e2_model.init_criterion()
    e4_criterion = e4_model.init_criterion()
    predictions = _synthetic_predictions(nc=2, reg_max=e4_criterion.one2many.reg_max)
    batch = _synthetic_batch()

    e2_loss, e2_items = e2_criterion(predictions, batch)
    e4_loss, e4_items = e4_criterion(predictions, batch)
    torch.testing.assert_close(e4_loss, e2_loss, atol=1e-6, rtol=1e-6)
    assert e4_items.keys() == e2_items.keys()
    for name in e2_items:
        torch.testing.assert_close(e4_items[name], e2_items[name], atol=1e-6, rtol=1e-6)


def test_e2_1b_e4_and_e4_1a_architectures_and_standard_loading_match(tmp_path: Path) -> None:
    """Compare architectures and load E4.1a weights through the standard YOLO facade."""
    standard_facade = YOLO(str(REPOSITORY_ROOT / "yolo26n.pt"))
    standard = standard_facade.model
    channels = standard.yaml.get("channels") or 3
    number_of_classes = standard.model[-1].nc
    model_yaml = copy.deepcopy(standard.yaml)
    e2_model = ContrastRingDetectionModel(
        copy.deepcopy(model_yaml),
        ch=channels,
        nc=number_of_classes,
        verbose=False,
        loss_config=E2_1B_CONFIG,
    )
    e4_model = ClassBalancedDetectionModel(
        copy.deepcopy(model_yaml),
        ch=channels,
        nc=number_of_classes,
        verbose=False,
        positive_class_weights=torch.ones(number_of_classes),
        class_balanced_config=ClassBalancedPositiveConfig(mode="control"),
    )
    uplift_weights = calculate_class_balanced_positive_weights(
        torch.arange(1, number_of_classes + 1),
        ClassBalancedPositiveConfig(mode="uplift-025"),
    )
    e4_1a_model = ClassBalancedDetectionModel(
        copy.deepcopy(model_yaml),
        ch=channels,
        nc=number_of_classes,
        verbose=False,
        positive_class_weights=uplift_weights,
        class_balanced_config=ClassBalancedPositiveConfig(mode="uplift-025"),
    )

    e2_parameters = [(name, tuple(parameter.shape)) for name, parameter in e2_model.named_parameters()]
    e4_parameters = [(name, tuple(parameter.shape)) for name, parameter in e4_model.named_parameters()]
    e4_1a_parameters = [(name, tuple(parameter.shape)) for name, parameter in e4_1a_model.named_parameters()]
    e2_state = {name: tuple(tensor.shape) for name, tensor in e2_model.state_dict().items()}
    e4_state = {name: tuple(tensor.shape) for name, tensor in e4_model.state_dict().items()}
    e4_1a_state = {name: tuple(tensor.shape) for name, tensor in e4_1a_model.state_dict().items()}
    assert e4_parameters == e2_parameters
    assert e4_1a_parameters == e2_parameters
    assert e4_state == e2_state
    assert e4_1a_state == e2_state

    for experimental_model in (e4_model, e4_1a_model):
        incompatible = experimental_model.load_state_dict(standard.state_dict(), strict=True)
        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []
    facade = ClassBalancedYOLO(
        str(REPOSITORY_ROOT / "yolo26n.pt"),
        positive_class_weights=torch.ones(number_of_classes),
        class_balanced_config=ClassBalancedPositiveConfig(mode="control"),
    )
    assert facade.task_map["detect"]["trainer"] is ClassBalancedDetectionTrainer
    assert facade.model.class_weights is None
    assert MutualDistillationYOLO is not ClassBalancedYOLO
    assert OneWayDistillationYOLO is not ClassBalancedYOLO

    checkpoint = tmp_path / "e4-1a.pt"
    standard_facade.model = e4_1a_model
    standard_facade.save(checkpoint)
    loaded = YOLO(checkpoint)
    assert isinstance(loaded.model, ClassBalancedDetectionModel)
    assert loaded.model.class_balanced_positive_config.mode == "uplift-025"
    torch.testing.assert_close(loaded.model.positive_class_weights, uplift_weights)
