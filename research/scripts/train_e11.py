# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train the isolated E11 Class-Conditional Suppression-Calibrated DGQM experiment."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ultralytics.nn.modules import ClassConditionalSuppressionDGQMDetect  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402
from yolo_improved import (  # noqa: E402
    CCS_DGQM_VARIANTS,
    ClassConditionalSuppressionConfig,
    DualGeometryQualityConfig,
    build_ccs_dgqm_yolo,
    ccs_dgqm_yaml_path,
    collect_ccs_dgqm_diagnostics,
    dgqm_yaml_path,
    print_ccs_dgqm_transfer_report,
)
from yolo_improved.dgqm_model import DualGeometryQualityDetectionModel  # noqa: E402


MODEL_SIZES = ("n", "s", "m", "l", "x")
DEFAULT_DATA_YAML = REPOSITORY_ROOT / "datasets" / "roboflow" / "mpi-detection-v1" / "data.yaml"
PROJECT_DIRECTORIES = {
    "control": REPOSITORY_ROOT / "runs" / "e11-ccs-dgqm-control" / "roboflow-v1",
    "trainable": REPOSITORY_ROOT / "runs" / "e11-ccs-dgqm-trainable" / "roboflow-v1",
}
QUALITY_CONFIG = DualGeometryQualityConfig()
SUPPRESSION_CONFIG = ClassConditionalSuppressionConfig()


def _measure_correction_statistics(
    model,
    images: torch.Tensor,
) -> dict[tuple[str, str], dict[str, float]]:
    """Measure shared-quality and suppression corrections on validation images."""
    underlying = model.model
    head = underlying.model[-1]
    if not isinstance(head, ClassConditionalSuppressionDGQMDetect):
        raise RuntimeError("E11 diagnostics require ClassConditionalSuppressionDGQMDetect.")

    accumulators: dict[tuple[str, str], dict[str, float | int]] = {}
    handles = []
    levels = ("P3", "P4", "P5")
    for assignment, quality_heads, suppression_heads in (
        ("one-to-many", head.quality_heads, head.suppression_heads),
        ("one-to-one", head.one2one_quality_heads, head.one2one_suppression_heads),
    ):
        for level, quality_head, suppression_head in zip(levels, quality_heads, suppression_heads):
            key = (assignment, level)
            accumulators[key] = {
                "shared_output_squared_sum": 0.0,
                "shared_sum": 0.0,
                "shared_count": 0,
                "suppression_sum": 0.0,
                "suppression_count": 0,
                "below_negative_0_1_count": 0,
                "clamped_zero_count": 0,
            }

            def capture_quality(_, __, output, *, diagnostic_key=key):
                raw_output = output.detach().float()
                correction = head.quality_scale * torch.tanh(raw_output)
                accumulator = accumulators[diagnostic_key]
                accumulator["shared_output_squared_sum"] += float(
                    raw_output.square().sum().cpu().item()
                )
                accumulator["shared_sum"] += float(correction.sum().cpu().item())
                accumulator["shared_count"] += correction.numel()

            def capture_suppression(_, __, output, *, diagnostic_key=key):
                correction = head.suppression_scale * torch.tanh(output.detach().float()).clamp(max=0.0)
                accumulator = accumulators[diagnostic_key]
                accumulator["suppression_sum"] += float(correction.sum().cpu().item())
                accumulator["suppression_count"] += correction.numel()
                accumulator["below_negative_0_1_count"] += int((correction < -0.1).sum().cpu().item())
                accumulator["clamped_zero_count"] += int((correction == 0).sum().cpu().item())

            handles.append(quality_head.register_forward_hook(capture_quality))
            handles.append(suppression_head.register_forward_hook(capture_suppression))

    was_training = underlying.training
    try:
        underlying.eval()
        with torch.inference_mode():
            underlying(images)
    finally:
        underlying.train(was_training)
        for handle in handles:
            handle.remove()

    statistics = {}
    for key, accumulator in accumulators.items():
        shared_count = int(accumulator["shared_count"])
        suppression_count = int(accumulator["suppression_count"])
        if not shared_count or not suppression_count:
            raise RuntimeError(f"E11 diagnostic hooks did not observe outputs for {key}.")
        statistics[key] = {
            "shared_quality_output_norm": float(accumulator["shared_output_squared_sum"]) ** 0.5,
            "mean_shared_quality_correction": float(accumulator["shared_sum"]) / shared_count,
            "mean_suppression_correction": float(accumulator["suppression_sum"]) / suppression_count,
            "fraction_corrections_below_negative_0_1": (
                int(accumulator["below_negative_0_1_count"]) / suppression_count
            ),
            "fraction_corrections_clamped_to_zero": (
                int(accumulator["clamped_zero_count"]) / suppression_count
            ),
        }
    return statistics


def parse_arguments() -> argparse.Namespace:
    """Parse fixed E11 variants and standard YOLO26 training settings."""
    parser = argparse.ArgumentParser(
        description="Послідовне навчання E11 YOLO26n/s/m/l/x із Class-Conditional Suppression-Calibrated DGQM."
    )
    parser.add_argument(
        "--variant",
        choices=CCS_DGQM_VARIANTS,
        required=True,
        help="Варіант E11: control або trainable.",
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
    """Return the fixed E11 run name."""
    suffix = "-ccs-dgqm-control" if variant == "control" else "-ccs-dgqm"
    batch_tag = "auto" if batch == -1 else str(batch)
    return f"yolo26{size}{suffix}_img{imgsz}_e{epochs}_b{batch_tag}_seed{seed}"


def _cost_report(model, size: str, imgsz: int) -> dict[str, float | int]:
    """Return E11 parameter and FLOPs changes relative to E10-DGQM."""
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
    """Print the fixed E11 experiment definition."""
    print("E11: Class-Conditional Suppression-Calibrated DGQM")
    print(f"Variant: {variant}")
    print("Base architecture: E10-DGQM")
    print("Base loss: E2.1b")
    print("Shared quality: enabled")
    print(f"Class suppression: {'enabled' if variant == 'trainable' else 'disabled'}")
    print(f"Suppression gain: {SUPPRESSION_CONFIG.suppression_gain}")
    print(f"Suppression scale: {SUPPRESSION_CONFIG.suppression_scale}")
    print(f"Hard-negative gamma: {SUPPRESSION_CONFIG.hard_negative_gamma}")
    print(
        "Hard-negative minimum probability: "
        f"{SUPPRESSION_CONFIG.hard_negative_min_probability}"
    )
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
    """Train one E11 scale after guarded result-directory checks."""
    architecture_yaml = ccs_dgqm_yaml_path(size, variant)
    pretrained = Path(f"yolo26{size}.pt")
    run_name = _run_name(size, variant, imgsz, epochs, batch, seed)
    run_directory = project_directory / run_name
    best_weights = run_directory / "weights" / "best.pt"
    last_weights = run_directory / "weights" / "last.pt"
    diagnostic_output = run_directory / "ccs_dgqm_diagnostics.json"

    if best_weights.exists():
        print()
        print(f"Пропущено {run_name}: завершений результат E11 уже існує.")
        print(f"Ваги: {best_weights}")
        return
    if last_weights.exists():
        raise RuntimeError(
            f"Знайдено незавершене навчання E11 {run_name}: {last_weights}\n"
            "Каталог не буде перезаписано; явно продовжте або приберіть попередній запуск."
        )
    if run_directory.exists():
        raise RuntimeError(
            f"Каталог запуску E11 уже існує, але best.pt і last.pt відсутні: {run_directory}. "
            "Каталог не буде перезаписано."
        )

    print()
    print("=" * 80)
    _print_experiment_header(variant, run_directory)
    print(f"Model scale: {size}")
    suffix = "ccs-dgqm-control" if variant == "control" else "ccs-dgqm"
    print(f"Architecture YAML: {architecture_yaml.with_name(f'yolo26-{suffix}.yaml')}")
    print(f"Virtual scale YAML: {architecture_yaml.name}")
    print(f"Pretrained checkpoint: {pretrained}")
    print("=" * 80)

    model = None
    results = None
    try:
        model = build_ccs_dgqm_yolo(
            size=size,
            variant=variant,
            verbose=False,
            quality_config=QUALITY_CONFIG,
            suppression_config=SUPPRESSION_CONFIG,
        )
        print("Pretrained transfer coverage:")
        print_ccs_dgqm_transfer_report(model.ccs_dgqm_transfer_report)
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
            print("WARNING: E11 suppression-head parameter increase relative to E10 exceeds 2%.")
        if costs["gflops_increase_percent"] > 2.0:
            print("WARNING: E11 suppression-head GFLOPs increase relative to E10 exceeds 2%.")

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
        validation_batch = next(iter(model.trainer.validator.dataloader))
        parameter = next(model.model.parameters())
        validation_images = validation_batch["img"][:1].to(
            device=parameter.device,
            dtype=parameter.dtype,
            non_blocking=parameter.device.type == "cuda",
        )
        validation_images = validation_images / 255.0
        correction_statistics = _measure_correction_statistics(model, validation_images)
        diagnostics = {
            "experiment": "E11",
            "variant": variant,
            "model_scale": size,
            "diagnostic_split": "val",
            "levels": collect_ccs_dgqm_diagnostics(model, correction_statistics),
        }
        diagnostic_output.write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print()
        print(f"Навчання E11 {run_name} завершено.")
        print(f"Найкращі ваги: {best_weights}")
        print(f"Діагностика CCS-DGQM: {diagnostic_output}")
    finally:
        del results
        del model
        _clear_memory()


def main() -> None:
    """Validate common inputs and train selected E11 scales sequentially."""
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
    print("Навчання всіх вибраних моделей E11 завершено.")


if __name__ == "__main__":
    main()
