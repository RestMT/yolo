# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train the isolated E12 Foregroundness-Factorized DGQM experiment."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ultralytics.nn.modules import ForegroundnessFactorizedDGQMDetect  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402
from yolo_improved import (  # noqa: E402
    FF_DGQM_VARIANTS,
    DualGeometryQualityConfig,
    ForegroundnessFactorizationConfig,
    build_ff_dgqm_yolo,
    collect_ff_dgqm_diagnostics,
    dgqm_yaml_path,
    ff_dgqm_yaml_path,
    print_ff_dgqm_transfer_report,
)
from yolo_improved.dgqm_model import DualGeometryQualityDetectionModel  # noqa: E402


MODEL_SIZES = ("n", "s", "m", "l", "x")
DEFAULT_DATA_YAML = REPOSITORY_ROOT / "datasets" / "roboflow" / "mpi-detection-v1" / "data.yaml"
PROJECT_DIRECTORIES = {
    "control": REPOSITORY_ROOT / "runs" / "e12-ff-dgqm-control" / "roboflow-v1",
    "trainable": REPOSITORY_ROOT / "runs" / "e12-ff-dgqm-trainable" / "roboflow-v1",
}
QUALITY_CONFIG = DualGeometryQualityConfig()
FOREGROUND_CONFIG = ForegroundnessFactorizationConfig()


def _mean_or_none(values: torch.Tensor) -> float | None:
    """Return a finite tensor mean or None when no anchors were selected."""
    return float(values.mean().cpu().item()) if values.numel() else None


def _next_validation_batch(model) -> dict:
    """Reset the validation loader exhausted by training and return its first batch."""
    validation_loader = model.trainer.validator.dataloader
    reset = getattr(validation_loader, "reset", None)
    if callable(reset):
        reset()
    return next(iter(validation_loader))


def _measure_assignment_statistics(
    model,
    validation_batch: dict,
) -> dict[tuple[str, str], dict[str, float | None]]:
    """Measure per-level quality/foreground corrections using validation assignments."""
    underlying = model.model
    head = underlying.model[-1]
    if not isinstance(head, ForegroundnessFactorizedDGQMDetect):
        raise RuntimeError("E12 diagnostics require ForegroundnessFactorizedDGQMDetect.")

    parameter = next(underlying.parameters())
    diagnostic_batch = {
        key: (
            value.to(device=parameter.device, non_blocking=parameter.device.type == "cuda")
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in validation_batch.items()
    }
    images = diagnostic_batch["img"].to(dtype=parameter.dtype) / 255.0
    diagnostic_batch["img"] = images

    was_training = underlying.training
    try:
        underlying.eval()
        with torch.inference_mode():
            output = underlying(images)
            if not isinstance(output, tuple) or not isinstance(output[1], dict):
                raise RuntimeError("E12 diagnostics expected raw end-to-end predictions.")
            raw_predictions = output[1]
            criterion = underlying.init_criterion()
            statistics: dict[tuple[str, str], dict[str, float | None]] = {}
            levels = ("P3", "P4", "P5")
            for assignment, branch_name, branch_criterion in (
                ("one-to-many", "one2many", criterion.one2many),
                ("one-to-one", "one2one", criterion.one2one),
            ):
                predictions = raw_predictions[branch_name]
                assigned, _, _ = branch_criterion.get_assigned_targets_and_loss(
                    predictions,
                    diagnostic_batch,
                )
                fg_mask = assigned[0].bool()
                quality_correction = (
                    head.quality_scale
                    * torch.tanh(predictions["quality"].permute(0, 2, 1).float())
                ).squeeze(-1)
                foreground_correction = (
                    head.foreground_scale
                    * torch.tanh(predictions["foreground"].permute(0, 2, 1).float())
                ).squeeze(-1)
                anchor_counts = [
                    feature.shape[-2] * feature.shape[-1] for feature in predictions["feats"]
                ]
                if sum(anchor_counts) != foreground_correction.shape[1]:
                    raise RuntimeError("E12 diagnostic level sizes do not match flattened anchors.")

                start = 0
                for level, count in zip(levels, anchor_counts):
                    stop = start + count
                    level_foreground = foreground_correction[:, start:stop]
                    level_quality = quality_correction[:, start:stop]
                    level_positive = fg_mask[:, start:stop]
                    level_background = ~level_positive
                    statistics[(assignment, level)] = {
                        "mean_quality_correction": _mean_or_none(level_quality),
                        "mean_foreground_correction": _mean_or_none(level_foreground),
                        "mean_positive_foreground_correction": _mean_or_none(
                            level_foreground[level_positive]
                        ),
                        "mean_background_foreground_correction": _mean_or_none(
                            level_foreground[level_background]
                        ),
                        "fraction_corrections_above_positive_0_1": float(
                            (level_foreground > 0.1).float().mean().cpu().item()
                        ),
                        "fraction_corrections_below_negative_0_1": float(
                            (level_foreground < -0.1).float().mean().cpu().item()
                        ),
                    }
                    start = stop
    finally:
        underlying.train(was_training)
    return statistics


def parse_arguments() -> argparse.Namespace:
    """Parse fixed E12 variants and standard YOLO26 training settings."""
    parser = argparse.ArgumentParser(
        description="Послідовне навчання E12 YOLO26n/s/m/l/x із Foregroundness-Factorized DGQM."
    )
    parser.add_argument(
        "--variant",
        choices=FF_DGQM_VARIANTS,
        required=True,
        help="Варіант E12: control або trainable.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_SIZES,
        default=list(MODEL_SIZES),
        help="Масштаби моделей: n s m l x.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="Шлях до data.yaml.")
    parser.add_argument("--epochs", type=int, default=100, help="Максимальна кількість епох.")
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Кількість епох без покращення перед раннім припиненням.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Розмір вхідного зображення.")
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Розмір пакета; -1 вмикає автоматичний вибір приблизно для 60%% відеопам'яті.",
    )
    parser.add_argument("--device", type=str, default="0", help="Пристрій: 0 для першої відеокарти або cpu.")
    parser.add_argument("--workers", type=int, default=4, help="Кількість процесів завантаження даних.")
    parser.add_argument("--seed", type=int, default=42, help="Початкове число генератора випадкових чисел.")
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Періодичність збереження проміжних контрольних точок.",
    )
    return parser.parse_args()


def _clear_memory() -> None:
    """Release Python and CUDA cache memory between model scales."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_name(size: str, variant: str, imgsz: int, epochs: int, batch: int, seed: int) -> str:
    """Return the fixed E12 run name."""
    suffix = "-ff-dgqm-control" if variant == "control" else "-ff-dgqm"
    batch_tag = "auto" if batch == -1 else str(batch)
    return f"yolo26{size}{suffix}_img{imgsz}_e{epochs}_b{batch_tag}_seed{seed}"


def _cost_report(model, size: str, imgsz: int) -> dict[str, float | int]:
    """Return E12 parameter and FLOPs changes relative to E10-DGQM."""
    dgqm = DualGeometryQualityDetectionModel(
        dgqm_yaml_path(size, "trainable"),
        verbose=False,
    )
    try:
        dgqm_parameters = get_num_params(dgqm)
        target_parameters = get_num_params(model.model)
        dgqm_flops = get_flops(dgqm, imgsz=imgsz)
        target_flops = get_flops(model.model, imgsz=imgsz)
    finally:
        del dgqm
    return {
        "dgqm_parameters": dgqm_parameters,
        "target_parameters": target_parameters,
        "parameter_increase_percent": 100.0 * (target_parameters - dgqm_parameters) / dgqm_parameters,
        "dgqm_gflops": dgqm_flops,
        "target_gflops": target_flops,
        "gflops_increase_percent": 100.0 * (target_flops - dgqm_flops) / dgqm_flops,
    }


def _print_experiment_header(variant: str, result_directory: Path) -> None:
    """Print the fixed E12 experiment definition."""
    print("E12: Foregroundness-Factorized DGQM")
    print(f"Variant: {variant}")
    print("Base architecture: E10-DGQM")
    print("Base loss: E2.1b")
    print("Dual-geometry quality: enabled")
    print(f"Foregroundness: {'enabled' if variant == 'trainable' else 'disabled'}")
    print(f"Foreground gain: {FOREGROUND_CONFIG.foreground_gain}")
    print(f"Foreground scale: {FOREGROUND_CONFIG.foreground_scale}")
    print(f"Hard-negative gamma: {FOREGROUND_CONFIG.hard_negative_gamma}")
    print(
        "Hard-negative minimum probability: "
        f"{FOREGROUND_CONFIG.hard_negative_min_probability}"
    )
    print(f"Negative weight: {FOREGROUND_CONFIG.negative_weight}")
    print(f"Result directory: {result_directory}")


def train_model(
    size: str,
    variant: str,
    data_yaml: Path,
    project_directory: Path,
    epochs: int,
    patience: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    seed: int,
    save_period: int,
) -> None:
    """Train one E12 scale after guarded result-directory checks."""
    architecture_yaml = ff_dgqm_yaml_path(size, variant)
    pretrained = Path(f"yolo26{size}.pt")
    run_name = _run_name(size, variant, imgsz, epochs, batch, seed)
    run_directory = project_directory / run_name
    best_weights = run_directory / "weights" / "best.pt"
    last_weights = run_directory / "weights" / "last.pt"
    diagnostic_output = run_directory / "ff_dgqm_diagnostics.json"

    if best_weights.exists():
        print()
        print(f"Пропущено {run_name}: завершений результат E12 уже існує.")
        print(f"Ваги: {best_weights}")
        return
    if last_weights.exists():
        raise RuntimeError(
            f"Знайдено незавершене навчання E12 {run_name}: {last_weights}\n"
            "Каталог не буде перезаписано; явно продовжте або приберіть попередній запуск."
        )
    if run_directory.exists():
        raise RuntimeError(
            f"Каталог запуску E12 уже існує, але best.pt і last.pt відсутні: {run_directory}. "
            "Каталог не буде перезаписано."
        )

    print()
    print("=" * 80)
    _print_experiment_header(variant, run_directory)
    print(f"Model scale: {size}")
    suffix = "ff-dgqm-control" if variant == "control" else "ff-dgqm"
    print(f"Architecture YAML: {architecture_yaml.with_name(f'yolo26-{suffix}.yaml')}")
    print(f"Virtual scale YAML: {architecture_yaml.name}")
    print(f"Pretrained checkpoint: {pretrained}")
    print("=" * 80)

    model = None
    results = None
    try:
        model = build_ff_dgqm_yolo(
            size=size,
            variant=variant,
            verbose=False,
            quality_config=QUALITY_CONFIG,
            foreground_config=FOREGROUND_CONFIG,
        )
        print("Pretrained transfer coverage:")
        print_ff_dgqm_transfer_report(model.ff_dgqm_transfer_report)
        costs = _cost_report(model, size, imgsz)
        print(
            f"Parameters: {costs['target_parameters']} vs E10 {costs['dgqm_parameters']} "
            f"({costs['parameter_increase_percent']:+.6f}%)"
        )
        print(
            f"GFLOPs: {costs['target_gflops']:.6f} vs E10 {costs['dgqm_gflops']:.6f} "
            f"({costs['gflops_increase_percent']:+.6f}%)"
        )
        if costs["parameter_increase_percent"] > 2.0:
            print("WARNING: E12 foreground-head parameter increase relative to E10 exceeds 2%.")
        if costs["gflops_increase_percent"] > 2.0:
            print("WARNING: E12 foreground-head GFLOPs increase relative to E10 exceeds 2%.")

        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            optimizer="auto",
            seed=seed,
            deterministic=True,
            amp=True,
            val=True,
            plots=True,
            save=True,
            save_period=save_period,
            cache=False,
            project=str(project_directory),
            name=run_name,
            exist_ok=False,
        )
        validation_batch = _next_validation_batch(model)
        correction_statistics = _measure_assignment_statistics(model, validation_batch)
        diagnostics = {
            "experiment": "E12",
            "variant": variant,
            "model_scale": size,
            "diagnostic_split": "val",
            "levels": collect_ff_dgqm_diagnostics(model, correction_statistics),
        }
        diagnostic_output.write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print()
        print(f"Навчання E12 {run_name} завершено.")
        print(f"Найкращі ваги: {best_weights}")
        print(f"Діагностика FF-DGQM: {diagnostic_output}")
    finally:
        del results
        del model
        _clear_memory()


def main() -> None:
    """Validate common inputs and train selected E12 scales sequentially."""
    args = parse_arguments()
    data_yaml = args.data.resolve()
    project_directory = PROJECT_DIRECTORIES[args.variant].resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Файл data.yaml не знайдено: {data_yaml}")
    project_directory.mkdir(parents=True, exist_ok=True)

    _print_experiment_header(args.variant, project_directory)
    print(f"Моделі: {', '.join(args.models)}")
    print(f"Епохи: {args.epochs}")
    print(f"Розмір зображення: {args.imgsz}")
    print(f"Розмір пакета: {args.batch}")
    print(f"Початкове число: {args.seed}")

    for size in args.models:
        train_model(
            size=size,
            variant=args.variant,
            data_yaml=data_yaml,
            project_directory=project_directory,
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            save_period=args.save_period,
        )

    print()
    print("Навчання всіх вибраних моделей E12 завершено.")


if __name__ == "__main__":
    main()
