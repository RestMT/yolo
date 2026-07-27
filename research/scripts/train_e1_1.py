from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from yolo_improved import RESIDUAL_NWD_MODES, ResidualNWDLossConfig, ResidualNWDYOLO  # noqa: E402


MODEL_SIZES = ("n", "s", "m", "l", "x")

DEFAULT_DATA_YAML = (
    REPOSITORY_ROOT
    / "datasets"
    / "roboflow"
    / "mpi-detection-v1"
    / "data.yaml"
)

PROJECT_DIRECTORIES = {
    "control": REPOSITORY_ROOT / "runs" / "e1-control" / "roboflow-v1",
    "constant-005": REPOSITORY_ROOT / "runs" / "e1-1-constant-005" / "roboflow-v1",
    "constant-010": REPOSITORY_ROOT / "runs" / "e1-1-constant-010" / "roboflow-v1",
    "adaptive-010": REPOSITORY_ROOT / "runs" / "e1-1-adaptive-010" / "roboflow-v1",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Послідовне донавчання моделей YOLO26n/s/m/l/x "
            "для експериментів E1-control та E1.1."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_SIZES,
        default=list(MODEL_SIZES),
        help="Масштаби моделей: n s m l x.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Шлях до data.yaml.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Максимальна кількість епох.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Кількість епох без покращення перед раннім припиненням.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Розмір вхідного зображення.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help=(
            "Розмір пакета. Значення -1 вмикає автоматичний вибір "
            "приблизно для 60%% відеопам'яті."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Пристрій: 0 для першої відеокарти або cpu.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Кількість процесів завантаження даних.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Початкове число генератора випадкових чисел.",
    )

    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Періодичність проміжних контрольних точок.",
    )

    parser.add_argument(
        "--variant",
        choices=RESIDUAL_NWD_MODES,
        required=True,
        help="Варіант: control, constant-005, constant-010 або adaptive-010.",
    )

    return parser.parse_args()


def train_model(
    size: str,
    data_yaml: Path,
    project_directory: Path,
    experiment_name: str,
    config: ResidualNWDLossConfig,
    epochs: int,
    patience: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    seed: int,
    save_period: int,
) -> None:
    model_name = f"yolo26{size}.pt"
    batch_tag = "auto" if batch == -1 else str(batch)

    run_name = (
        f"yolo26{size}"
        f"_img{imgsz}"
        f"_e{epochs}"
        f"_b{batch_tag}"
        f"_seed{seed}"
    )

    run_directory = project_directory / run_name
    best_weights = run_directory / "weights" / "best.pt"
    last_weights = run_directory / "weights" / "last.pt"

    if best_weights.exists():
        print()
        print(f"Пропущено {model_name}: результат {experiment_name} уже існує.")
        print(f"Ваги: {best_weights}")
        return

    if last_weights.exists():
        raise RuntimeError(
            f"Знайдено незавершене навчання {experiment_name} {model_name}: "
            f"{last_weights}\n"
            "Для продовження використайте режим resume."
        )

    if run_directory.exists():
        raise RuntimeError(
            f"Каталог запуску {experiment_name} вже існує, але best.pt і last.pt відсутні: "
            f"{run_directory}"
        )

    print()
    print("=" * 80)
    print(f"ЕКСПЕРИМЕНТ {experiment_name}: ЗАЛИШКОВА NWD-ВТРАТА")
    print(f"Варіант: {config.mode}")
    print(f"Початок навчання: {model_name}")
    print(f"Набір даних: {data_yaml}")
    print(f"Результати: {run_directory}")
    print("=" * 80)

    model = ResidualNWDYOLO(model_name, loss_config=config)

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
    print(f"Навчання {experiment_name} {model_name} завершено.")
    print(f"Найкращі ваги: {best_weights}")

    del results
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_arguments()

    data_yaml = args.data.resolve()
    project_directory = PROJECT_DIRECTORIES[args.variant].resolve()
    experiment_name = "E1-control" if args.variant == "control" else "E1.1"
    config = ResidualNWDLossConfig(mode=args.variant)

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Файл data.yaml не знайдено: {data_yaml}"
        )

    project_directory.mkdir(parents=True, exist_ok=True)

    print(f"Експеримент: {experiment_name}")
    print(f"Активний варіант: {config.mode}")
    print(f"S: {config.nwd_scale}")
    print(f"alpha_max: {config.alpha_max}")
    print(f"A0: {config.area_threshold}")
    print(f"k: {config.area_slope}")
    print(f"Каталог результатів: {project_directory}")
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
            experiment_name=experiment_name,
            config=config,
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
    print(f"Навчання всіх вибраних моделей {experiment_name} завершено.")


if __name__ == "__main__":
    main()
