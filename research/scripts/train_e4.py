from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from yolo_improved import (  # noqa: E402
    CLASS_BALANCED_POSITIVE_MODES,
    E2_1B_CONFIG,
    ClassBalancedPositiveConfig,
    ClassBalancedYOLO,
    calculate_class_balanced_positive_weights,
    count_yolo_class_instances,
    get_class_balanced_positive_config,
)


MODEL_SIZES = ("n", "s", "m", "l", "x")

DEFAULT_DATA_YAML = (
    REPOSITORY_ROOT
    / "datasets"
    / "roboflow"
    / "mpi-detection-v1"
    / "data.yaml"
)

PROJECT_DIRECTORIES = {
    "control": REPOSITORY_ROOT / "runs" / "e4-control" / "roboflow-v1",
    "effective-099": REPOSITORY_ROOT / "runs" / "e4-class-balanced-positive-beta099" / "roboflow-v1",
}


def parse_arguments() -> argparse.Namespace:
    """Parse fixed E4 variants and standard YOLO26 training settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Послідовне донавчання моделей E4 YOLO26n/s/m/l/x "
            "із класово-збалансованими позитивними елементами."
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
        choices=CLASS_BALANCED_POSITIVE_MODES,
        required=True,
        help="Варіант E4: control або effective-099.",
    )
    return parser.parse_args()


def print_class_balance_table(
    class_names: tuple[str, ...],
    class_counts: tuple[int, ...],
    positive_class_weights: torch.Tensor,
) -> None:
    """Print training counts and display-only rounded positive weights."""
    print("class_id | class_name | instance_count | positive_weight")
    for class_id, (class_name, count, weight) in enumerate(
        zip(class_names, class_counts, positive_class_weights.tolist())
    ):
        print(f"{class_id} | {class_name} | {count} | {weight:.10f}")


def train_model(
    size: str,
    variant: str,
    data_yaml: Path,
    project_directory: Path,
    class_balanced_config: ClassBalancedPositiveConfig,
    positive_class_weights: torch.Tensor,
    epochs: int,
    patience: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    seed: int,
    save_period: int,
) -> None:
    """Train one E4 scale from its standard pretrained YOLO26 checkpoint."""
    model_name = f"yolo26{size}.pt"
    batch_tag = "auto" if batch == -1 else str(batch)
    run_name = f"yolo26{size}_img{imgsz}_e{epochs}_b{batch_tag}_seed{seed}"

    run_directory = project_directory / run_name
    best_weights = run_directory / "weights" / "best.pt"
    last_weights = run_directory / "weights" / "last.pt"

    if best_weights.exists():
        print()
        print(f"Пропущено {model_name}: результат E4 {variant} уже існує.")
        print(f"Ваги: {best_weights}")
        return

    if last_weights.exists():
        raise RuntimeError(
            f"Знайдено незавершене навчання E4 {variant} {model_name}: "
            f"{last_weights}\n"
            "Для продовження використайте режим resume."
        )

    if run_directory.exists():
        raise RuntimeError(
            f"Каталог запуску E4 {variant} вже існує, але best.pt і last.pt відсутні: "
            f"{run_directory}"
        )

    print()
    print("=" * 80)
    print("E4: class-balanced positive classification")
    print("Локалізація: E1.1 constant-010")
    print("Класифікація: E2.1b")
    print("Class balancing: positive elements only")
    print(f"Варіант: {variant}")
    print(f"Початок навчання: {model_name}")
    print(f"Набір даних: {data_yaml}")
    print(f"Результати: {run_directory}")
    print("=" * 80)

    model = ClassBalancedYOLO(
        model_name,
        positive_class_weights=positive_class_weights,
        class_balanced_config=class_balanced_config,
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

    print()
    print(f"Навчання E4 {variant} {model_name} завершено.")
    print(f"Найкращі ваги: {best_weights}")

    del results
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    """Count train instances, calculate E4 weights, and train selected scales sequentially."""
    args = parse_arguments()

    data_yaml = args.data.resolve()
    project_directory = PROJECT_DIRECTORIES[args.variant].resolve()
    class_balanced_config = get_class_balanced_positive_config(args.variant)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Файл data.yaml не знайдено: {data_yaml}")

    class_counts, class_names = count_yolo_class_instances(data_yaml)
    positive_class_weights = calculate_class_balanced_positive_weights(
        class_counts,
        config=class_balanced_config,
    )
    project_directory.mkdir(parents=True, exist_ok=True)

    print("E4: class-balanced positive classification")
    print("Локалізація: E1.1 constant-010")
    print("Класифікація: E2.1b")
    print("Class balancing: positive elements only")
    print(f"Варіант: {args.variant}")
    print(f"beta: {class_balanced_config.beta}")
    print(f"eps: {class_balanced_config.eps}")
    print("Масштаб NWD E1.1: 0.10")
    print(f"E2.1b contrast_tau: {E2_1B_CONFIG.contrast_tau}")
    print(f"E2.1b positive_gain: {E2_1B_CONFIG.positive_gain}")
    print(f"E2.1b negative_gain: {E2_1B_CONFIG.negative_gain}")
    print(f"E2.1b negative_gamma: {E2_1B_CONFIG.negative_gamma}")
    print_class_balance_table(class_names, class_counts, positive_class_weights)
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
            variant=args.variant,
            data_yaml=data_yaml,
            project_directory=project_directory,
            class_balanced_config=class_balanced_config,
            positive_class_weights=positive_class_weights,
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
    print(f"Навчання всіх вибраних моделей E4 {args.variant} завершено.")


if __name__ == "__main__":
    main()
