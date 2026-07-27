from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from yolo_improved import ContrastRingLossConfig, ContrastRingYOLO  # noqa: E402


MODEL_SIZES = ("n", "s", "m", "l", "x")

DEFAULT_DATA_YAML = (
    REPOSITORY_ROOT
    / "datasets"
    / "roboflow"
    / "mpi-detection-v1"
    / "data.yaml"
)

DEFAULT_PROJECT_DIRECTORY = (
    REPOSITORY_ROOT
    / "runs"
    / "e2-contrast-ring"
    / "roboflow-v1"
)

DEFAULT_LOSS_CONFIG = ContrastRingLossConfig()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Послідовне донавчання моделей E2 YOLO26n/s/m/l/x "
            "з контрастно-кільцевою класифікаційною втратою."
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
        "--contrast-tau",
        type=float,
        default=DEFAULT_LOSS_CONFIG.contrast_tau,
        help="Параметр насичення нормованої контрастної характеристики.",
    )

    parser.add_argument(
        "--positive-gain",
        type=float,
        default=DEFAULT_LOSS_CONFIG.positive_gain,
        help="Коефіцієнт підсилення слабкоконтрастних позитивних елементів.",
    )

    parser.add_argument(
        "--negative-gain",
        type=float,
        default=DEFAULT_LOSS_CONFIG.negative_gain,
        help="Коефіцієнт підсилення контрастних хибнопозитивних елементів.",
    )

    parser.add_argument(
        "--negative-gamma",
        type=float,
        default=DEFAULT_LOSS_CONFIG.negative_gamma,
        help="Степінь упевненості для ваги негативних елементів.",
    )

    return parser.parse_args()


def train_model(
    size: str,
    data_yaml: Path,
    project_directory: Path,
    loss_config: ContrastRingLossConfig,
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
        print(f"Пропущено {model_name}: результат E2 уже існує.")
        print(f"Ваги: {best_weights}")
        return

    if last_weights.exists():
        raise RuntimeError(
            f"Знайдено незавершене навчання E2 {model_name}: "
            f"{last_weights}\n"
            "Для продовження використайте режим resume."
        )

    if run_directory.exists():
        raise RuntimeError(
            f"Каталог запуску E2 вже існує, але best.pt і last.pt відсутні: "
            f"{run_directory}"
        )

    print()
    print("=" * 80)
    print("E2: contrast-ring classification")
    print("Локалізаційна основа: E1.1 constant-010")
    print(f"Початок навчання: {model_name}")
    print(f"Набір даних: {data_yaml}")
    print(f"Результати: {run_directory}")
    print("=" * 80)

    model = ContrastRingYOLO(model_name, loss_config=loss_config)

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
    print(f"Навчання E2 {model_name} завершено.")
    print(f"Найкращі ваги: {best_weights}")

    del results
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_arguments()

    data_yaml = args.data.resolve()
    project_directory = DEFAULT_PROJECT_DIRECTORY.resolve()
    loss_config = ContrastRingLossConfig(
        contrast_tau=args.contrast_tau,
        positive_gain=args.positive_gain,
        negative_gain=args.negative_gain,
        negative_gamma=args.negative_gamma,
    )

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Файл data.yaml не знайдено: {data_yaml}"
        )

    project_directory.mkdir(parents=True, exist_ok=True)

    print("E2: contrast-ring classification")
    print("Локалізаційна основа: E1.1 constant-010 (CIoU + 0.10 NWD, S=0.10)")
    print(f"Внутрішнє ядро: {loss_config.inner_kernel} × {loss_config.inner_kernel}")
    print(f"Зовнішнє ядро: {loss_config.outer_kernel} × {loss_config.outer_kernel}")
    print(f"contrast_tau: {loss_config.contrast_tau}")
    print(f"positive_gain: {loss_config.positive_gain}")
    print(f"negative_gain: {loss_config.negative_gain}")
    print(f"negative_gamma: {loss_config.negative_gamma}")
    print(f"eps: {loss_config.eps}")
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
            loss_config=loss_config,
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
    print("Навчання всіх вибраних моделей E2 завершено.")


if __name__ == "__main__":
    main()
