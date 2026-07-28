from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from yolo_improved import (  # noqa: E402
    ONE_WAY_DISTILLATION_VARIANTS,
    OneWayDistillationConfig,
    OneWayDistillationYOLO,
    get_one_way_distillation_config,
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
    "control": REPOSITORY_ROOT / "runs" / "e3-1-control" / "roboflow-v1",
    "cls-005": REPOSITORY_ROOT / "runs" / "e3-1-oneway-cls-005" / "roboflow-v1",
    "cls-0025": REPOSITORY_ROOT / "runs" / "e3-1-oneway-cls-0025" / "roboflow-v1",
}


def parse_arguments() -> argparse.Namespace:
    """Parse fixed E3.1 variants and standard YOLO26 training settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Послідовне донавчання моделей E3.1 YOLO26n/s/m/l/x "
            "з односторонньою класифікаційною дистиляцією."
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
        choices=ONE_WAY_DISTILLATION_VARIANTS,
        required=True,
        help="Варіант E3.1: control, cls-005 або cls-0025.",
    )
    return parser.parse_args()


def train_model(
    size: str,
    variant: str,
    data_yaml: Path,
    project_directory: Path,
    distillation_config: OneWayDistillationConfig,
    epochs: int,
    patience: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    seed: int,
    save_period: int,
) -> None:
    """Train one E3.1 scale from its standard pretrained YOLO26 checkpoint."""
    model_name = f"yolo26{size}.pt"
    batch_tag = "auto" if batch == -1 else str(batch)
    run_name = f"yolo26{size}_img{imgsz}_e{epochs}_b{batch_tag}_seed{seed}"

    run_directory = project_directory / run_name
    best_weights = run_directory / "weights" / "best.pt"
    last_weights = run_directory / "weights" / "last.pt"

    if best_weights.exists():
        print()
        print(f"Пропущено {model_name}: результат E3.1 {variant} уже існує.")
        print(f"Ваги: {best_weights}")
        return

    if last_weights.exists():
        raise RuntimeError(
            f"Знайдено незавершене навчання E3.1 {variant} {model_name}: "
            f"{last_weights}\n"
            "Для продовження використайте режим resume."
        )

    if run_directory.exists():
        raise RuntimeError(
            f"Каталог запуску E3.1 {variant} вже існує, але best.pt і last.pt відсутні: "
            f"{run_directory}"
        )

    print()
    print("=" * 80)
    print("E3.1: one-to-many to one-to-one classification distillation")
    print("Teacher: one-to-many")
    print("Student: one-to-one")
    print("Box distillation: disabled")
    print("Base supervision: E1.1 constant-010 + E2.1b")
    print(f"Варіант: {variant}")
    print(f"Початок навчання: {model_name}")
    print(f"Набір даних: {data_yaml}")
    print(f"Результати: {run_directory}")
    print("=" * 80)

    model = OneWayDistillationYOLO(
        model_name,
        distillation_config=distillation_config,
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
    print(f"Навчання E3.1 {variant} {model_name} завершено.")
    print(f"Найкращі ваги: {best_weights}")

    del results
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    """Validate arguments and train the selected E3.1 scales sequentially."""
    args = parse_arguments()

    data_yaml = args.data.resolve()
    project_directory = PROJECT_DIRECTORIES[args.variant].resolve()
    distillation_config = get_one_way_distillation_config(args.variant)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Файл data.yaml не знайдено: {data_yaml}")

    project_directory.mkdir(parents=True, exist_ok=True)

    print("E3.1: one-to-many to one-to-one classification distillation")
    print("Teacher: one-to-many")
    print("Student: one-to-one")
    print("Box distillation: disabled")
    print("Base supervision: E1.1 constant-010 + E2.1b")
    print(f"Варіант: {args.variant}")
    print(f"classification_gain: {distillation_config.classification_gain}")
    print("box_gain: 0.0")
    print(f"temperature: {distillation_config.temperature}")
    print(f"confidence_temperature: {distillation_config.confidence_temperature}")
    print(f"start_epoch: {distillation_config.start_epoch}")
    print(f"warmup_epochs: {distillation_config.warmup_epochs}")
    print(f"eps: {distillation_config.eps}")
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
            distillation_config=distillation_config,
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
    print(f"Навчання всіх вибраних моделей E3.1 {args.variant} завершено.")


if __name__ == "__main__":
    main()
