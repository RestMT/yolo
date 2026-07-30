# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train the isolated E10 Dual-Geometry Quality-Calibrated MADH experiment."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ultralytics.nn.modules import DualGeometryQualityMorphologyDetect  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402
from yolo_improved import (  # noqa: E402
    DGQM_VARIANTS,
    DualGeometryQualityConfig,
    build_dgqm_yolo,
    collect_dgqm_diagnostics,
    dgqm_yaml_path,
    madh_yaml_path,
    print_dgqm_transfer_report,
)
from yolo_improved.contrast_ring_model import ContrastRingDetectionModel  # noqa: E402
from yolo_improved.madh_model import E2_1B_CONFIG  # noqa: E402


MODEL_SIZES = ("n", "s", "m", "l", "x")
DEFAULT_DATA_YAML = REPOSITORY_ROOT / "datasets" / "roboflow" / "mpi-detection-v1" / "data.yaml"
PROJECT_DIRECTORIES = {
    "control": REPOSITORY_ROOT / "runs" / "e10-dgqm-control" / "roboflow-v1",
    "trainable": REPOSITORY_ROOT / "runs" / "e10-dgqm-trainable" / "roboflow-v1",
}
QUALITY_CONFIG = DualGeometryQualityConfig()


def _measure_quality_corrections(model, images: torch.Tensor) -> dict[tuple[str, str], float]:
    """Measure per-head corrections on validation images without running evaluation metrics."""
    underlying = model.model
    head = underlying.model[-1]
    if not isinstance(head, DualGeometryQualityMorphologyDetect):
        raise RuntimeError("E10 diagnostics require DualGeometryQualityMorphologyDetect.")
    totals: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    handles = []
    levels = ("P3", "P4", "P5")
    for assignment, quality_heads in (
        ("one-to-many", head.quality_heads),
        ("one-to-one", head.one2one_quality_heads),
    ):
        for level, quality_head in zip(levels, quality_heads):
            key = (assignment, level)

            def capture(_, __, output, *, diagnostic_key=key):
                correction = head.quality_scale * torch.tanh(output.detach().float())
                totals[diagnostic_key] = totals.get(diagnostic_key, 0.0) + float(
                    correction.abs().sum().cpu().item()
                )
                counts[diagnostic_key] = counts.get(diagnostic_key, 0) + correction.numel()

            handles.append(quality_head.register_forward_hook(capture))

    was_training = underlying.training
    try:
        underlying.eval()
        with torch.inference_mode():
            underlying(images)
    finally:
        underlying.train(was_training)
        for handle in handles:
            handle.remove()
    return {key: totals[key] / counts[key] for key in totals if counts.get(key, 0)}


def parse_arguments() -> argparse.Namespace:
    """Parse fixed E10 variants and standard YOLO26 training settings."""
    parser = argparse.ArgumentParser(
        description="Послідовне навчання E10 YOLO26n/s/m/l/x із Dual-Geometry Quality-Calibrated MADH."
    )
    parser.add_argument("--variant", choices=DGQM_VARIANTS, required=True, help="Варіант E10: control або trainable.")
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_SIZES, default=list(MODEL_SIZES), help="Масштаби моделей: n s m l x."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="Шлях до data.yaml.")
    parser.add_argument("--epochs", type=int, default=100, help="Максимальна кількість епох.")
    parser.add_argument(
        "--patience", type=int, default=20, help="Кількість епох без покращення перед раннім припиненням."
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
        "--save-period", type=int, default=10, help="Періодичність збереження проміжних контрольних точок."
    )
    return parser.parse_args()


def _clear_memory() -> None:
    """Release Python and CUDA cache memory between model scales."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_name(size: str, variant: str, imgsz: int, epochs: int, batch: int, seed: int) -> str:
    """Return the fixed E10 run name."""
    suffix = "-dgqm-control" if variant == "control" else "-dgqm"
    batch_tag = "auto" if batch == -1 else str(batch)
    return f"yolo26{size}{suffix}_img{imgsz}_e{epochs}_b{batch_tag}_seed{seed}"


def _cost_report(model, size: str, imgsz: int) -> dict[str, float | int]:
    """Return E10 parameter and FLOPs changes relative to E8-MADH."""
    madh = ContrastRingDetectionModel(
        madh_yaml_path(size, "trainable"),
        verbose=False,
        loss_config=E2_1B_CONFIG,
    )
    try:
        madh_parameters = get_num_params(madh)
        target_parameters = get_num_params(model.model)
        madh_flops = get_flops(madh, imgsz=imgsz)
        target_flops = get_flops(model.model, imgsz=imgsz)
    finally:
        del madh
    return {
        "madh_parameters": madh_parameters,
        "target_parameters": target_parameters,
        "parameter_increase_percent": 100.0 * (target_parameters - madh_parameters) / madh_parameters,
        "madh_gflops": madh_flops,
        "target_gflops": target_flops,
        "gflops_increase_percent": 100.0 * (target_flops - madh_flops) / madh_flops,
    }


def _print_experiment_header(variant: str, result_directory: Path) -> None:
    """Print the fixed E10 experiment definition."""
    print("E10: Dual-Geometry Quality-Calibrated MADH")
    print(f"Variant: {variant}")
    print("Base architecture: E8-MADH")
    print("Base loss: E2.1b")
    print(f"IoU quality weight: {QUALITY_CONFIG.iou_weight}")
    print(f"NWD quality weight: {QUALITY_CONFIG.nwd_weight}")
    print(f"Quality gain: {QUALITY_CONFIG.quality_gain}")
    print(f"Negative neutral weight: {QUALITY_CONFIG.negative_neutral_weight}")
    print(f"Quality score scale: {QUALITY_CONFIG.quality_scale}")
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
    """Train one E10 scale after guarded result-directory checks."""
    architecture_yaml = dgqm_yaml_path(size, variant)
    pretrained = Path(f"yolo26{size}.pt")
    run_name = _run_name(size, variant, imgsz, epochs, batch, seed)
    run_directory = project_directory / run_name
    best_weights = run_directory / "weights" / "best.pt"
    last_weights = run_directory / "weights" / "last.pt"
    diagnostic_output = run_directory / "dgqm_diagnostics.json"

    if best_weights.exists():
        print()
        print(f"Пропущено {run_name}: завершений результат E10 уже існує.")
        print(f"Ваги: {best_weights}")
        return
    if last_weights.exists():
        raise RuntimeError(
            f"Знайдено незавершене навчання E10 {run_name}: {last_weights}\n"
            "Каталог не буде перезаписано; явно продовжте або приберіть попередній запуск."
        )
    if run_directory.exists():
        raise RuntimeError(
            f"Каталог запуску E10 уже існує, але best.pt і last.pt відсутні: {run_directory}. "
            "Каталог не буде перезаписано."
        )

    print()
    print("=" * 80)
    _print_experiment_header(variant, run_directory)
    print(f"Model scale: {size}")
    suffix = "dgqm-control" if variant == "control" else "dgqm"
    print(f"Architecture YAML: {architecture_yaml.with_name(f'yolo26-{suffix}.yaml')}")
    print(f"Virtual scale YAML: {architecture_yaml.name}")
    print(f"Pretrained checkpoint: {pretrained}")
    print("=" * 80)

    model = None
    results = None
    try:
        model = build_dgqm_yolo(
            size=size,
            variant=variant,
            verbose=False,
            quality_config=QUALITY_CONFIG,
        )
        print("Pretrained transfer coverage:")
        print_dgqm_transfer_report(model.dgqm_transfer_report)
        costs = _cost_report(model, size, imgsz)
        print(
            f"Parameters: {costs['target_parameters']} vs E8 {costs['madh_parameters']} "
            f"({costs['parameter_increase_percent']:+.6f}%)"
        )
        print(
            f"GFLOPs: {costs['target_gflops']:.6f} vs E8 {costs['madh_gflops']:.6f} "
            f"({costs['gflops_increase_percent']:+.6f}%)"
        )
        if costs["parameter_increase_percent"] > 2.0:
            print("WARNING: E10 quality-head parameter increase relative to E8 exceeds 2%.")
        if costs["gflops_increase_percent"] > 2.0:
            print("WARNING: E10 quality-head GFLOPs increase relative to E8 exceeds 2%.")

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
        mean_corrections = _measure_quality_corrections(model, validation_images)
        diagnostics = {
            "experiment": "E10",
            "variant": variant,
            "model_scale": size,
            "diagnostic_split": "val",
            "levels": collect_dgqm_diagnostics(model, mean_corrections),
        }
        diagnostic_output.write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print()
        print(f"Навчання E10 {run_name} завершено.")
        print(f"Найкращі ваги: {best_weights}")
        print(f"Діагностика DGQM: {diagnostic_output}")
    finally:
        del results
        del model
        _clear_memory()


def main() -> None:
    """Validate common inputs and train selected E10 scales sequentially."""
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
    print("Навчання всіх вибраних моделей E10 завершено.")


if __name__ == "__main__":
    main()
