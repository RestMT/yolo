# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train the isolated E5 P2-guided Detail Injection experiment."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from yolo_improved import (  # noqa: E402
    E2_1B_CONFIG,
    build_p2_detail_injection_yolo,
    p2_detail_injection_yaml_path,
    print_p2_detail_injection_transfer_report,
)


MODEL_SIZES = ("n", "s", "m", "l", "x")
DEFAULT_DATA_YAML = REPOSITORY_ROOT / "datasets" / "roboflow" / "mpi-detection-v1" / "data.yaml"
DEFAULT_PROJECT_DIRECTORY = REPOSITORY_ROOT / "runs" / "e5-p2-detail-injection" / "roboflow-v1"


def parse_arguments() -> argparse.Namespace:
    """Parse E5 training arguments."""
    parser = argparse.ArgumentParser(
        description="Послідовне навчання моделей E5 YOLO26n/s/m/l/x із P2-guided Detail Injection Neck."
    )
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


def train_model(
    size: str,
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
    """Train one E5 scale after guarded result-directory checks."""
    pretrained = Path(f"yolo26{size}.pt")
    architecture_yaml = p2_detail_injection_yaml_path(size)
    batch_tag = "auto" if batch == -1 else str(batch)
    run_name = f"yolo26{size}-p2di_img{imgsz}_e{epochs}_b{batch_tag}_seed{seed}"
    run_directory = project_directory / run_name
    best_weights = run_directory / "weights" / "best.pt"
    last_weights = run_directory / "weights" / "last.pt"

    if best_weights.exists():
        print()
        print(f"Пропущено yolo26{size}-p2di: завершений результат E5 уже існує.")
        print(f"Ваги: {best_weights}")
        return
    if last_weights.exists():
        raise RuntimeError(
            f"Знайдено незавершене навчання E5 yolo26{size}-p2di: {last_weights}\n"
            "Каталог не буде перезаписано; явно продовжте або приберіть попередній запуск."
        )
    if run_directory.exists():
        raise RuntimeError(
            f"Каталог запуску E5 уже існує, але best.pt і last.pt відсутні: {run_directory}. "
            "Каталог не буде перезаписано."
        )

    print()
    print("=" * 80)
    print("E5: P2-guided Detail Injection Neck")
    print("Detection outputs: P3/8, P4/16, P5/32")
    print("Loss: E2.1b")
    print(f"Architecture YAML: {architecture_yaml.with_name('yolo26-p2di.yaml')}")
    print(f"Virtual scale YAML: {architecture_yaml.name}")
    print(f"Pretrained checkpoint: {pretrained}")
    print(f"Result directory: {run_directory}")
    print("=" * 80)

    model = None
    results = None
    try:
        model = build_p2_detail_injection_yolo(size=size, pretrained=pretrained, verbose=False)
        print_p2_detail_injection_transfer_report(model.p2di_transfer_report)
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
        print(f"Навчання E5 yolo26{size}-p2di завершено.")
        print(f"Найкращі ваги: {best_weights}")
    finally:
        del results
        del model
        _clear_memory()


def main() -> None:
    """Validate common inputs and train the selected E5 scales sequentially."""
    args = parse_arguments()
    data_yaml = args.data.resolve()
    project_directory = DEFAULT_PROJECT_DIRECTORY.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Файл data.yaml не знайдено: {data_yaml}")
    project_directory.mkdir(parents=True, exist_ok=True)

    print("E5: P2-guided Detail Injection Neck")
    print("Detection outputs: P3/8, P4/16, P5/32")
    print("Loss: E2.1b")
    print("Localization: E1.1 constant-010 (CIoU + 0.10 NWD, S=0.10)")
    print(
        "Contrast ring: "
        f"inner={E2_1B_CONFIG.inner_kernel}, outer={E2_1B_CONFIG.outer_kernel}, "
        f"tau={E2_1B_CONFIG.contrast_tau}, positive_gain={E2_1B_CONFIG.positive_gain}, "
        f"negative_gain={E2_1B_CONFIG.negative_gain}, gamma={E2_1B_CONFIG.negative_gamma}, "
        f"eps={E2_1B_CONFIG.eps}"
    )
    print(f"Result directory: {project_directory}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA у PyTorch: {torch.version.cuda}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Відеокарта: {torch.cuda.get_device_name(0)}")
    print(f"Моделі: {', '.join(args.models)}")
    print(f"Епохи: {args.epochs}")
    print(f"Розмір зображення: {args.imgsz}")
    print(f"Розмір пакета: {args.batch}")
    print(f"Початкове число: {args.seed}")

    for size in args.models:
        train_model(
            size=size,
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
    print("Навчання всіх вибраних моделей E5 завершено.")


if __name__ == "__main__":
    main()
