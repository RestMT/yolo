from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ASSETS


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_path = Path(ultralytics.__file__).resolve()

    print(f"Repository root: {repo_root}")
    print(f"Ultralytics source: {package_path}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if repo_root not in package_path.parents:
        raise RuntimeError(
            "Ultralytics is not loaded from the current repository. "
            "Check the selected PyCharm interpreter and editable installation."
        )

    model = YOLO("yolo26n.yaml")
    model.info()

    device = 0 if torch.cuda.is_available() else "cpu"

    results = model.predict(
        source=str(ASSETS / "bus.jpg"),
        imgsz=640,
        device=device,
        save=True,
        project=str(repo_root / "runs"),
        name="environment_check",
    )

    print(f"Processed images: {len(results)}")


if __name__ == "__main__":
    main()