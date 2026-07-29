# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train the isolated E9 Geometry-Preserving Spatial Morphology experiment."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402
from yolo_improved import (  # noqa: E402
    GPSM_VARIANTS,
    build_gpsm_yolo,
    collect_gpsm_parameters,
    gpsm_yaml_path,
    print_gpsm_transfer_report,
)


MODEL_SIZES = ("n", "s", "m", "l", "x")
DEFAULT_DATA_YAML = REPOSITORY_ROOT / "datasets" / "roboflow" / "mpi-detection-v1" / "data.yaml"
PROJECT_DIRECTORIES = {
    "control": REPOSITORY_ROOT / "runs" / "e9-gpsm-control" / "roboflow-v1",
    "trainable": REPOSITORY_ROOT / "runs" / "e9-gpsm-trainable" / "roboflow-v1",
}


def parse_arguments() -> argparse.Namespace:
    """Parse fixed E9 variants and standard YOLO26 training settings."""
    parser = argparse.ArgumentParser(
        description="Послідовне навчання E9 YOLO26n/s/m/l/x із Geometry-Preserving Spatial Morphology Head."
    )
    parser.add_argument("--variant", choices=GPSM_VARIANTS, required=True, help="Варіант E9: control або trainable.")
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
    """Return the fixed E9 run name, preserving negative batch values other than -1."""
    suffix = "-gpsm-control" if variant == "control" else "-gpsm"
    batch_tag = "auto" if batch == -1 else str(batch)
    return f"yolo26{size}{suffix}_img{imgsz}_e{epochs}_b{batch_tag}_seed{seed}"


def _cost_report(model, size: str, imgsz: int) -> dict[str, float | int]:
    """Return parameter and FLOPs changes against stock YOLO26 at one scale."""
    stock_yaml = REPOSITORY_ROOT / "ultralytics" / "cfg" / "models" / "26" / f"yolo26{size}.yaml"
    stock = DetectionModel(stock_yaml, verbose=False)
    try:
        stock_parameters = get_num_params(stock)
        target_parameters = get_num_params(model.model)
        stock_flops = get_flops(stock, imgsz=imgsz)
        target_flops = get_flops(model.model, imgsz=imgsz)
    finally:
        del stock
    return {
        "stock_parameters": stock_parameters,
        "target_parameters": target_parameters,
        "parameter_increase_percent": 100.0 * (target_parameters - stock_parameters) / stock_parameters,
        "stock_gflops": stock_flops,
        "target_gflops": target_flops,
        "gflops_increase_percent": 100.0 * (target_flops - stock_flops) / stock_flops,
    }


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
    """Train one E9 scale after guarded result-directory checks."""
    architecture_yaml = gpsm_yaml_path(size, variant)
    pretrained = Path(f"yolo26{size}.pt")
    run_name = _run_name(size, variant, imgsz, epochs, batch, seed)
    run_directory = project_directory / run_name
    best_weights = run_directory / "weights" / "best.pt"
    last_weights = run_directory / "weights" / "last.pt"
    diagnostic_output = run_directory / "gpsm_parameters.json"

    if best_weights.exists():
        print()
        print(f"Пропущено {run_name}: завершений результат E9 уже існує.")
        print(f"Ваги: {best_weights}")
        return
    if last_weights.exists():
        raise RuntimeError(
            f"Знайдено незавершене навчання E9 {run_name}: {last_weights}\n"
            "Каталог не буде перезаписано; явно продовжте або приберіть попередній запуск."
        )
    if run_directory.exists():
        raise RuntimeError(
            f"Каталог запуску E9 уже існує, але best.pt і last.pt відсутні: {run_directory}. "
            "Каталог не буде перезаписано."
        )

    print()
    print("=" * 80)
    print("E9: Geometry-Preserving Spatial Morphology Head")
    print(f"Variant: {variant}")
    print(f"Model scale: {size}")
    print("Base loss: E2.1b")
    print("Box path: stock")
    print("Classification path: GPSM")
    print("Spatial gate maximum: 0.5")
    print("Morphology branches: local 3x3, horizontal 1x7, vertical 7x1, context 3x3 dilation=2")
    suffix = "gpsm-control" if variant == "control" else "gpsm"
    print(f"Architecture YAML: {architecture_yaml.with_name(f'yolo26-{suffix}.yaml')}")
    print(f"Virtual scale YAML: {architecture_yaml.name}")
    print(f"Pretrained checkpoint: {pretrained}")
    print(f"Result directory: {run_directory}")
    print("=" * 80)

    model = None
    results = None
    try:
        model = build_gpsm_yolo(size=size, variant=variant, verbose=False)
        print("Pretrained transfer coverage:")
        print_gpsm_transfer_report(model.gpsm_transfer_report)
        costs = _cost_report(model, size, imgsz)
        print(
            f"Parameters: {costs['target_parameters']} vs {costs['stock_parameters']} "
            f"({costs['parameter_increase_percent']:+.6f}%)"
        )
        print(
            f"GFLOPs: {costs['target_gflops']:.6f} vs {costs['stock_gflops']:.6f} "
            f"({costs['gflops_increase_percent']:+.6f}%)"
        )
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
        diagnostics = {
            "experiment": "E9",
            "variant": variant,
            "model_scale": size,
            "levels": collect_gpsm_parameters(model),
        }
        diagnostic_output.write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print()
        print(f"Навчання E9 {run_name} завершено.")
        print(f"Найкращі ваги: {best_weights}")
        print(f"Параметрична діагностика GPSM: {diagnostic_output}")
    finally:
        del results
        del model
        _clear_memory()


def main() -> None:
    """Validate common inputs and train selected E9 scales sequentially."""
    args = parse_arguments()
    data_yaml = args.data.resolve()
    project_directory = PROJECT_DIRECTORIES[args.variant].resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Файл data.yaml не знайдено: {data_yaml}")
    project_directory.mkdir(parents=True, exist_ok=True)

    print("E9: Geometry-Preserving Spatial Morphology Head")
    print(f"Variant: {args.variant}")
    print("Base loss: E2.1b")
    print("Box path: stock")
    print("Classification path: GPSM")
    print(f"Result directory: {project_directory}")
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
    print("Навчання всіх вибраних моделей E9 завершено.")


if __name__ == "__main__":
    main()
