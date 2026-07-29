# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train the isolated E7 distributional box-regression experiment."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from yolo_improved import (  # noqa: E402
    REG_MAX_VALUES,
    build_regmax_yolo,
    print_regmax_transfer_report,
    regmax_yaml_path,
)


MODEL_SIZES = ("n", "s", "m", "l", "x")
DEFAULT_DATA_YAML = REPOSITORY_ROOT / "datasets" / "roboflow" / "mpi-detection-v1" / "data.yaml"
PROJECT_DIRECTORIES = {
    reg_max: REPOSITORY_ROOT / "runs" / f"e7-regmax{reg_max}" / "roboflow-v1"
    for reg_max in REG_MAX_VALUES
}


def parse_arguments() -> argparse.Namespace:
    """Parse the fixed E7 variant and standard YOLO26 training settings."""
    parser = argparse.ArgumentParser(
        description="Послідовне навчання E7 YOLO26n/s/m/l/x із distributional box regression."
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_SIZES, default=list(MODEL_SIZES), help="Масштаби моделей: n s m l x."
    )
    parser.add_argument(
        "--reg-max",
        type=int,
        choices=REG_MAX_VALUES,
        required=True,
        help="Кількість інтервалів DFL: 4 або 8.",
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


def _run_name(size: str, reg_max: int, imgsz: int, epochs: int, batch: int, seed: int) -> str:
    """Return the fixed E7 run name, preserving negative batch values other than -1."""
    batch_tag = "auto" if batch == -1 else str(batch)
    return f"yolo26{size}-regmax{reg_max}_img{imgsz}_e{epochs}_b{batch_tag}_seed{seed}"


def train_model(
    size: str,
    reg_max: int,
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
    """Train one E7 scale after guarded result-directory checks."""
    architecture_yaml = regmax_yaml_path(size, reg_max)
    pretrained = Path(f"yolo26{size}.pt")
    run_name = _run_name(size, reg_max, imgsz, epochs, batch, seed)
    run_directory = project_directory / run_name
    best_weights = run_directory / "weights" / "best.pt"
    last_weights = run_directory / "weights" / "last.pt"

    if best_weights.exists():
        print()
        print(f"Пропущено {run_name}: завершений результат E7 уже існує.")
        print(f"Ваги: {best_weights}")
        return
    if last_weights.exists():
        raise RuntimeError(
            f"Знайдено незавершене навчання E7 {run_name}: {last_weights}\n"
            "Каталог не буде перезаписано; явно продовжте або приберіть попередній запуск."
        )
    if run_directory.exists():
        raise RuntimeError(
            f"Каталог запуску E7 уже існує, але best.pt і last.pt відсутні: {run_directory}. "
            "Каталог не буде перезаписано."
        )

    print()
    print("=" * 80)
    print("E7: distributional box regression")
    print("Base model: YOLO26")
    print("Base loss: E2.1b")
    print(f"reg_max: {reg_max}")
    print(f"Regression channels per anchor: {4 * reg_max}")
    print("DFL enabled: True")
    print("Detect levels: P3/8, P4/16, P5/32")
    print(f"Architecture YAML: {architecture_yaml.with_name(f'yolo26-regmax{reg_max}.yaml')}")
    print(f"Virtual scale YAML: {architecture_yaml.name}")
    print(f"Pretrained checkpoint: {pretrained}")
    print(f"Result directory: {run_directory}")
    print("=" * 80)

    model = None
    results = None
    try:
        model = build_regmax_yolo(size=size, reg_max=reg_max, verbose=False)
        print("Pretrained transfer coverage:")
        print_regmax_transfer_report(model.regmax_transfer_report)
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
        print()
        print(f"Навчання E7 {run_name} завершено.")
        print(f"Найкращі ваги: {best_weights}")
    finally:
        del results
        del model
        _clear_memory()


def main() -> None:
    """Validate common inputs and train selected E7 scales sequentially."""
    args = parse_arguments()
    data_yaml = args.data.resolve()
    project_directory = PROJECT_DIRECTORIES[args.reg_max].resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Файл data.yaml не знайдено: {data_yaml}")
    project_directory.mkdir(parents=True, exist_ok=True)

    print("E7: distributional box regression")
    print("Base model: YOLO26")
    print("Base loss: E2.1b")
    print(f"reg_max: {args.reg_max}")
    print(f"Regression channels per anchor: {4 * args.reg_max}")
    print("DFL enabled: True")
    print("Detect levels: P3/8, P4/16, P5/32")
    print(f"Result directory: {project_directory}")
    print(f"Моделі: {', '.join(args.models)}")
    print(f"Епохи: {args.epochs}")
    print(f"Розмір зображення: {args.imgsz}")
    print(f"Розмір пакета: {args.batch}")
    print(f"Початкове число: {args.seed}")

    for size in args.models:
        train_model(
            size=size,
            reg_max=args.reg_max,
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
    print("Навчання всіх вибраних моделей E7 завершено.")


if __name__ == "__main__":
    main()
