$ErrorActionPreference = "Stop"

$directories = @(
    "research",
    "research\configs",
    "research\configs\datasets",
    "research\configs\experiments",
    "research\scripts",
    "research\tests",
    "research\notes"
)

foreach ($directory in $directories) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

$emptyFiles = @(
    "research\configs\datasets\.gitkeep",
    "research\configs\experiments\.gitkeep",
    "research\tests\.gitkeep",
    "research\notes\.gitkeep"
)

foreach ($file in $emptyFiles) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType File -Path $file | Out-Null
    }
}

$readme = @"
# YOLO26 Research

This directory contains custom experiments and scripts for modifying the YOLO26 architecture.

## Directories

- configs/datasets — dataset configuration files.
- configs/experiments — experiment configuration files.
- scripts — training, prediction, and environment-check scripts.
- tests — tests for custom modules and model configurations.
- notes — research notes and experiment observations.
"@

$checkEnvironment = @'
from pathlib import Path

import torch
import ultralytics


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    ultralytics_path = Path(ultralytics.__file__).resolve()

    print(f"Repository root: {repository_root}")
    print(f"Ultralytics path: {ultralytics_path}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
'@

$predictBaseline = @'
import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS


def main() -> None:
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("yolo26n.pt")

    model.predict(
        source=str(ASSETS / "bus.jpg"),
        imgsz=640,
        device=device,
        save=True,
        project="runs/predict",
        name="yolo26n_baseline",
    )


if __name__ == "__main__":
    main()
'@

$trainBaseline = @'
import torch
from ultralytics import YOLO


def main() -> None:
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("yolo26n.pt")

    model.train(
        data="coco8.yaml",
        epochs=1,
        imgsz=640,
        batch=4,
        device=device,
        workers=0,
        seed=42,
        deterministic=True,
        project="runs/baseline",
        name="yolo26n_coco8",
    )


if __name__ == "__main__":
    main()
'@

$trainCustom = @'
import torch
from ultralytics import YOLO


def main() -> None:
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("yolo26n-custom.yaml")

    model.train(
        data="coco8.yaml",
        epochs=1,
        imgsz=640,
        batch=4,
        device=device,
        workers=0,
        seed=42,
        deterministic=True,
        project="runs/custom",
        name="yolo26n_custom_coco8",
    )


if __name__ == "__main__":
    main()
'@

$files = @{
    "research\README.md" = $readme
    "research\scripts\check_environment.py" = $checkEnvironment
    "research\scripts\predict_baseline.py" = $predictBaseline
    "research\scripts\train_baseline.py" = $trainBaseline
    "research\scripts\train_custom.py" = $trainCustom
}

foreach ($entry in $files.GetEnumerator()) {
    if (-not (Test-Path $entry.Key)) {
        Set-Content -Path $entry.Key -Value $entry.Value -Encoding UTF8
        Write-Host "Created: $($entry.Key)"
    }
    else {
        Write-Host "Skipped existing file: $($entry.Key)"
    }
}

Write-Host ""
Write-Host "Research structure created successfully."
Write-Host ""
tree research /F