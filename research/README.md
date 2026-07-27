# YOLO26 Research

This directory contains custom experiments and scripts for modifying the YOLO26 architecture.

## Directories

- configs/datasets — dataset configuration files.
- configs/experiments — experiment configuration files.
- scripts — training, prediction, and environment-check scripts.
- tests — tests for custom modules and model configurations.
- notes — research notes and experiment observations.

## E1 hybrid localization

The E1 loss parameters are isolated in `yolo_improved/hybrid_config.py`.

The baseline and E1 model can coexist in one Python process:

```python
from ultralytics import YOLO
from yolo_improved import HybridYOLO

baseline = YOLO("yolo26n.pt")
modified = HybridYOLO("yolo26n.pt")
```

Run the unchanged E0 baseline or the isolated E1 experiment:

```bash
python research/scripts/train_baseline.py
python research/scripts/train_e1.py
```

Evaluate E1 checkpoints with the existing evaluator and standard `YOLO` loader:

```bash
python research/scripts/evaluate_baselines.py --model n \
    --training-project runs/e1-hybrid-localization/roboflow-v1 \
    --evaluation-project runs/e1-hybrid-localization-evaluation/roboflow-v1
```
