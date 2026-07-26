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
