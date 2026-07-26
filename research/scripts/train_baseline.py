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
