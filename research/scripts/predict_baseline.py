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
