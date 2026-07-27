from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


MODEL_SIZES = ("n", "s", "m", "l", "x")
MODEL_ORDER = {
    "n": 0,
    "s": 1,
    "m": 2,
    "l": 3,
    "x": 4,
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_YAML = (
    REPOSITORY_ROOT
    / "datasets"
    / "roboflow"
    / "mpi-detection-v1"
    / "data.yaml"
)

DEFAULT_TRAINING_PROJECT = (
    REPOSITORY_ROOT
    / "runs"
    / "baselines"
    / "roboflow-v1"
)

DEFAULT_EVALUATION_PROJECT = (
    REPOSITORY_ROOT
    / "runs"
    / "baseline-evaluation"
    / "roboflow-v1"
)

CSV_FIELDS = [
    "model",
    "run_name",
    "weights",
    "split",
    "data",
    "imgsz",
    "evaluation_batch",
    "parameters",
    "weights_mb",
    "precision",
    "recall",
    "mAP50",
    "mAP75",
    "mAP50-95",
    "preprocess_ms_per_image",
    "inference_ms_per_image",
    "postprocess_ms_per_image",
    "evaluated_at_utc",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Оцінювання однієї навченої базової моделі YOLO26 "
            "на валідаційній або тестовій вибірці."
        )
    )

    model_source = parser.add_mutually_exclusive_group(required=True)

    model_source.add_argument(
        "--model",
        choices=MODEL_SIZES,
        help=(
            "Масштаб моделі: n, s, m, l або x. "
            "Сценарій автоматично знайде найновіший файл best.pt "
            "для цього масштабу."
        ),
    )

    model_source.add_argument(
        "--weights",
        type=Path,
        help="Точний шлях до файла best.pt.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Шлях до data.yaml.",
    )

    parser.add_argument(
        "--training-project",
        type=Path,
        default=DEFAULT_TRAINING_PROJECT,
        help=(
            "Каталог із результатами навчання. "
            "Використовується під час запуску через --model."
        ),
    )

    parser.add_argument(
        "--evaluation-project",
        type=Path,
        default=DEFAULT_EVALUATION_PROJECT,
        help="Каталог для збереження результатів оцінювання.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help=(
            "Шлях до підсумкового CSV. За замовчуванням створюється "
            "summary_test.csv або summary_val.csv."
        ),
    )

    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Частина набору даних для оцінювання.",
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
        default=4,
        help="Розмір пакета під час оцінювання.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Обчислювальний пристрій: 0, 1, cpu тощо.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Кількість процесів завантаження даних. "
            "Для Windows безпечне початкове значення — 0."
        ),
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """
    Перетворює відносний шлях на абсолютний відносно кореня репозиторію.
    """
    expanded_path = path.expanduser()

    if expanded_path.is_absolute():
        return expanded_path.resolve()

    return (REPOSITORY_ROOT / expanded_path).resolve()


def find_latest_best_weights(
    training_project: Path,
    model_size: str,
) -> Path:
    """
    Знаходить найновіший best.pt для вказаного масштабу моделі.
    """
    if not training_project.exists():
        raise FileNotFoundError(
            "Каталог із результатами навчання не знайдено:\n"
            f"{training_project}"
        )

    expected_prefix = f"yolo26{model_size}"

    candidates: list[Path] = []

    for candidate in training_project.rglob("best.pt"):
        run_directory = candidate.parent.parent

        if run_directory.name.lower().startswith(expected_prefix):
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"Не знайдено best.pt для YOLO26{model_size} у каталозі:\n"
            f"{training_project}\n\n"
            "Передайте точний шлях через параметр --weights."
        )

    latest_weights = max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )

    return latest_weights.resolve()


def determine_run_name(weights_path: Path) -> str:
    """
    Визначає назву навчального запуску за шляхом до best.pt.
    """
    if weights_path.parent.name.lower() == "weights":
        return weights_path.parent.parent.name

    return weights_path.stem


def determine_model_name(
    run_name: str,
    weights_path: Path,
    requested_size: str | None,
) -> str:
    """
    Формує читабельну назву моделі для підсумкової таблиці.
    """
    if requested_size:
        return f"YOLO26{requested_size}"

    searchable_text = f"{run_name} {weights_path}".lower()
    match = re.search(r"yolo26([nsmlx])", searchable_text)

    if match:
        return f"YOLO26{match.group(1)}"

    return run_name


def validate_dataset_split(
    data_yaml: Path,
    split: str,
) -> None:
    """
    Перевіряє, чи визначена потрібна частина набору в data.yaml.
    """
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Файл data.yaml не знайдено:\n{data_yaml}"
        )

    with data_yaml.open("r", encoding="utf-8") as yaml_file:
        data_config = yaml.safe_load(yaml_file)

    if not isinstance(data_config, dict):
        raise ValueError(
            f"Файл data.yaml має неправильну структуру:\n{data_yaml}"
        )

    if split not in data_config or not data_config[split]:
        raise ValueError(
            f"У файлі data.yaml не визначено частину '{split}'.\n"
            f"Файл: {data_yaml}\n\n"
            "Для оцінювання на валідаційній вибірці "
            "використайте параметр --split val."
        )


def sanitize_directory_name(name: str) -> str:
    """
    Видаляє із назви символи, непридатні для каталогу.
    """
    sanitized = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        name,
    ).strip("._-")

    return sanitized or "evaluation"


def load_existing_rows(summary_path: Path) -> list[dict[str, str]]:
    """
    Читає раніше створену підсумкову таблицю.
    """
    if not summary_path.exists():
        return []

    with summary_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def model_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    """
    Сортує YOLO26n, s, m, l, x у логічному порядку.
    """
    model_name = str(row.get("model", "")).lower()
    match = re.search(r"yolo26([nsmlx])", model_name)

    if match:
        return MODEL_ORDER[match.group(1)], str(
            row.get("run_name", "")
        )

    return 99, str(row.get("run_name", ""))


def update_summary_csv(
    summary_path: Path,
    new_row: dict[str, Any],
) -> None:
    """
    Додає новий результат або оновлює наявний результат того самого запуску.
    """
    existing_rows = load_existing_rows(summary_path)

    updated_rows: list[dict[str, Any]] = []

    for existing_row in existing_rows:
        same_run = (
            existing_row.get("run_name") == new_row["run_name"]
            and existing_row.get("split") == new_row["split"]
        )

        legacy_same_model = (
            not existing_row.get("run_name")
            and existing_row.get("model") == new_row["model"]
            and existing_row.get("split") == new_row["split"]
        )

        if not same_run and not legacy_same_model:
            updated_rows.append(existing_row)

    updated_rows.append(new_row)
    updated_rows.sort(key=model_sort_key)

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(updated_rows)


def main() -> None:
    args = parse_arguments()

    data_yaml = resolve_path(args.data)
    training_project = resolve_path(args.training_project)
    evaluation_project = resolve_path(args.evaluation_project)

    validate_dataset_split(
        data_yaml=data_yaml,
        split=args.split,
    )

    if args.weights is not None:
        weights_path = resolve_path(args.weights)
    else:
        weights_path = find_latest_best_weights(
            training_project=training_project,
            model_size=args.model,
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Файл ваг не знайдено:\n{weights_path}"
        )

    if weights_path.suffix.lower() != ".pt":
        raise ValueError(
            f"Очікувався файл ваг із розширенням .pt:\n{weights_path}"
        )

    run_name = determine_run_name(weights_path)

    model_name = determine_model_name(
        run_name=run_name,
        weights_path=weights_path,
        requested_size=args.model,
    )

    evaluation_name = (
        f"{sanitize_directory_name(run_name)}_{args.split}"
    )

    if args.summary is not None:
        summary_path = resolve_path(args.summary)
    else:
        summary_path = (
            evaluation_project
            / f"summary_{args.split}.csv"
        )

    print()
    print("=" * 80)
    print("ОЦІНЮВАННЯ МОДЕЛІ")
    print("=" * 80)
    print(f"Модель: {model_name}")
    print(f"Навчальний запуск: {run_name}")
    print(f"Ваги: {weights_path}")
    print(f"Набір даних: {data_yaml}")
    print(f"Частина набору: {args.split}")
    print(f"Розмір зображення: {args.imgsz}")
    print(f"Розмір пакета: {args.batch}")
    print(f"Пристрій: {args.device}")
    print(f"Результати: {evaluation_project / evaluation_name}")
    print("=" * 80)
    print()

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        verbose=True,
        project=str(evaluation_project),
        name=evaluation_name,
        exist_ok=True,
    )

    precision, recall, map50, map50_95 = (
        float(value)
        for value in metrics.box.mean_results()
    )

    map75 = float(metrics.box.map75)

    speed = metrics.speed or {}

    parameter_count = sum(
        parameter.numel()
        for parameter in model.model.parameters()
    )

    weights_size_mb = (
        weights_path.stat().st_size
        / (1024 ** 2)
    )

    result_row: dict[str, Any] = {
        "model": model_name,
        "run_name": run_name,
        "weights": str(weights_path),
        "split": args.split,
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "evaluation_batch": args.batch,
        "parameters": parameter_count,
        "weights_mb": f"{weights_size_mb:.3f}",
        "precision": f"{precision:.6f}",
        "recall": f"{recall:.6f}",
        "mAP50": f"{map50:.6f}",
        "mAP75": f"{map75:.6f}",
        "mAP50-95": f"{map50_95:.6f}",
        "preprocess_ms_per_image": (
            f"{float(speed.get('preprocess', 0.0) or 0.0):.3f}"
        ),
        "inference_ms_per_image": (
            f"{float(speed.get('inference', 0.0) or 0.0):.3f}"
        ),
        "postprocess_ms_per_image": (
            f"{float(speed.get('postprocess', 0.0) or 0.0):.3f}"
        ),
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
    }

    update_summary_csv(
        summary_path=summary_path,
        new_row=result_row,
    )

    print()
    print("=" * 80)
    print("ОЦІНЮВАННЯ ЗАВЕРШЕНО")
    print("=" * 80)
    print(f"Модель: {model_name}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"mAP50: {map50:.6f}")
    print(f"mAP75: {map75:.6f}")
    print(f"mAP50-95: {map50_95:.6f}")
    print(
        "Час висновування: "
        f"{float(speed.get('inference', 0.0) or 0.0):.3f} мс/зображення"
    )
    print(f"Параметри моделі: {parameter_count:,}")
    print(f"Розмір файла ваг: {weights_size_mb:.3f} МіБ")
    print(f"Підсумкова таблиця: {summary_path}")
    print(
        "Графіки та матриця помилок: "
        f"{evaluation_project / evaluation_name}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()