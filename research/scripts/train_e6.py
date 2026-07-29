# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train the isolated E6 Residual Multi-Scale Refinement experiment."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ultralytics.nn.modules import C3k2RMSR  # noqa: E402
from yolo_improved import (  # noqa: E402
    RMSR_VARIANTS,
    build_rmsr_yolo,
    print_rmsr_transfer_report,
    rmsr_yaml_path,
)


MODEL_SIZES = ("n", "s", "m", "l", "x")
DEFAULT_DATA_YAML = REPOSITORY_ROOT / "datasets" / "roboflow" / "mpi-detection-v1" / "data.yaml"
PROJECT_DIRECTORIES = {
    "control": REPOSITORY_ROOT / "runs" / "e6-rmsr-control" / "roboflow-v1",
    "trainable": REPOSITORY_ROOT / "runs" / "e6-rmsr-trainable" / "roboflow-v1",
}


def parse_arguments() -> argparse.Namespace:
    """Parse fixed E6 variants and standard YOLO26 training settings."""
    parser = argparse.ArgumentParser(
        description="Послідовне навчання моделей E6 YOLO26n/s/m/l/x із Residual Multi-Scale Refinement."
    )
    parser.add_argument("--variant", choices=RMSR_VARIANTS, required=True, help="Варіант E6: control або trainable.")
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
    """Return the fixed E6 run name, preserving negative batch values other than -1."""
    suffix = "-rmsr-control" if variant == "control" else "-rmsr"
    batch_tag = "auto" if batch == -1 else str(batch)
    return f"yolo26{size}{suffix}_img{imgsz}_e{epochs}_b{batch_tag}_seed{seed}"


def _gate_values(model) -> tuple[float, float]:
    """Return gate_raw and bounded alpha from an E6 facade."""
    rmsr_layer = model.model.model[16]
    if not isinstance(rmsr_layer, C3k2RMSR):
        raise RuntimeError("E6 checkpoint does not contain C3k2RMSR at layer 16.")
    gate_raw = float(rmsr_layer.gate_raw.detach().float().cpu().item())
    alpha = float((rmsr_layer.gate_max * torch.tanh(rmsr_layer.gate_raw.detach().float())).cpu().item())
    return gate_raw, alpha


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
    """Train one E6 scale after guarded result-directory checks."""
    architecture_yaml = rmsr_yaml_path(size, variant)
    pretrained = Path(f"yolo26{size}.pt")
    run_name = _run_name(size, variant, imgsz, epochs, batch, seed)
    run_directory = project_directory / run_name
    best_weights = run_directory / "weights" / "best.pt"
    last_weights = run_directory / "weights" / "last.pt"

    if best_weights.exists():
        print()
        print(f"Пропущено {run_name}: завершений результат E6 уже існує.")
        print(f"Ваги: {best_weights}")
        return
    if last_weights.exists():
        raise RuntimeError(
            f"Знайдено незавершене навчання E6 {run_name}: {last_weights}\n"
            "Каталог не буде перезаписано; явно продовжте або приберіть попередній запуск."
        )
    if run_directory.exists():
        raise RuntimeError(
            f"Каталог запуску E6 уже існує, але best.pt і last.pt відсутні: {run_directory}. "
            "Каталог не буде перезаписано."
        )

    print()
    print("=" * 80)
    print("E6: Residual Multi-Scale Refinement")
    print(f"Variant: {variant}")
    unified_name = "yolo26-rmsr-control.yaml" if variant == "control" else "yolo26-rmsr.yaml"
    print(f"Architecture YAML: {architecture_yaml.with_name(unified_name)}")
    print(f"Virtual scale YAML: {architecture_yaml.name}")
    print("Base loss: E2.1b")
    print("Refined level: P3/8 only")
    print("Local dilation: 1")
    print("Context dilation: 2")
    print("Gate maximum: 1.0")
    print(f"Pretrained checkpoint: {pretrained}")
    print(f"Result directory: {run_directory}")
    print("=" * 80)

    model = None
    results = None
    try:
        model = build_rmsr_yolo(size=size, variant=variant, verbose=False)
        initial_gate, initial_alpha = _gate_values(model)
        print(f"Initial gate: gate_raw={initial_gate:.10f}, alpha={initial_alpha:.10f}")
        print("Pretrained transfer coverage:")
        print_rmsr_transfer_report(model.rmsr_transfer_report)
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
        gate_raw, alpha = _gate_values(model)
        print()
        print(f"Навчання E6 {run_name} завершено.")
        print(f"Найкращі ваги: {best_weights}")
        print(f"gate_raw: {gate_raw:.10f}")
        print(f"alpha = tanh(gate_raw): {alpha:.10f}")
    finally:
        del results
        del model
        _clear_memory()


def main() -> None:
    """Validate common inputs and train selected E6 scales sequentially."""
    args = parse_arguments()
    data_yaml = args.data.resolve()
    project_directory = PROJECT_DIRECTORIES[args.variant].resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Файл data.yaml не знайдено: {data_yaml}")
    project_directory.mkdir(parents=True, exist_ok=True)

    print("E6: Residual Multi-Scale Refinement")
    print(f"Variant: {args.variant}")
    print("Base loss: E2.1b")
    print("Refined level: P3/8 only")
    print("Local dilation: 1")
    print("Context dilation: 2")
    print("Gate maximum: 1.0")
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
    print("Навчання всіх вибраних моделей E6 завершено.")


if __name__ == "__main__":
    main()
