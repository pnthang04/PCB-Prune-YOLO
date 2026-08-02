---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - yolo
  - yolov8
  - depgraph
  - torch-pruning
  - structured-pruning
  - deeppcb
datasets:
  - thangkt/PCB-Prune-YOLO-DeepPCB
metrics:
  - map
---

# PCB-Prune-YOLO P10 Direct

Validation-selected YOLOv8n checkpoint produced by direct DepGraph structured
pruning followed by matched fine-tuning on DeepPCB. This model does not use
sparse learning or knowledge distillation.

## Validation results

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.96479 | 0.95706 | 0.98273 | 0.77736 |

At seed 42 this direct P10 control exceeded the matched sparse-learning P10 by
1.42 mAP50-95 percentage points. This is a single-seed observation and the
DeepPCB test split was not used for model selection.

## Compression and Tesla T4 benchmark

| Parameters | MACs | Size | Latency batch 1 | FPS |
|---:|---:|---:|---:|---:|
| 2,416,871 | 3.2695G | 4.854 MiB | 10.433 ms | 95.85 |

Input size is 640. Latency uses 50 warm-up and 200 synchronized CUDA iterations.

## Training configuration

- Direct local group-magnitude pruning, ratio 0.10, one step, no channel rounding
- AdamW, `lr0=0.001`, `lrf=0.01`, momentum 0.9, weight decay 0.0005
- 50 epochs, batch 64, patience 10, seed 42, AMP and deterministic mode
- Six classes: open, short, mousebite, spur, copper, pin-hole

## Loading

Structured pruning changes the serialized architecture. Install the project so
the `PrunableC2f` class is importable before loading:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("pcb.jpg", imgsz=640)
```

Project: https://github.com/pnthang04/PCB-Prune-YOLO

The checkpoint was verified by loading in a new process and running CUDA
inference with decoded output shape `[1, 10, 8400]`.
