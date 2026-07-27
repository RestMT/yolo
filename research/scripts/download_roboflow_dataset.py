from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow


WORKSPACE_ID = "mpi-gpotm"
PROJECT_ID = "mpi-detection-nqpjc"
VERSION_NUMBER = 1
EXPORT_FORMAT = "yolo26"


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    env_path = repository_root / ".env"

    load_dotenv(env_path)

    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Не знайдено ROBOFLOW_API_KEY. "
            "Створіть файл .env у корені проєкту та додайте до нього ключ."
        )

    output_directory = (
        repository_root
        / "datasets"
        / "roboflow"
        / "mpi-detection-v1"
    )

    output_directory.parent.mkdir(parents=True, exist_ok=True)

    print(f"Робоча область Roboflow: {WORKSPACE_ID}")
    print(f"Проєкт Roboflow: {PROJECT_ID}")
    print(f"Версія набору даних: {VERSION_NUMBER}")
    print(f"Каталог завантаження: {output_directory}")

    rf = Roboflow(api_key=api_key)

    project = (
        rf.workspace(WORKSPACE_ID)
        .project(PROJECT_ID)
    )

    version = project.version(VERSION_NUMBER)

    dataset = version.download(
        model_format=EXPORT_FORMAT,
        location=str(output_directory),
        overwrite=False,
    )

    dataset_directory = Path(dataset.location).resolve()
    data_yaml = dataset_directory / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Набір завантажено, але файл data.yaml не знайдено: "
            f"{data_yaml}"
        )

    print()
    print("Набір даних успішно завантажено.")
    print(f"Каталог набору: {dataset_directory}")
    print(f"Конфігурація: {data_yaml}")


if __name__ == "__main__":
    main()