---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags: [yolo, yolov8, depgraph, torch-pruning, structured-pruning, deeppcb]
datasets: [thangkt/PCB-Prune-YOLO-DeepPCB]
metrics: [map]
---

# PCB-Prune-YOLO P20 Direct

Validation-selected YOLOv8n checkpoint produced by direct DepGraph structured
pruning followed by 50-epoch matched fine-tuning on DeepPCB. No sparse learning,
knowledge distillation, or test-set model selection was used.

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.96214 | 0.96186 | 0.98184 | 0.76710 |

The model has 1,913,971 parameters and 2.5722 GMACs, reductions of 36.46% and
36.85% from baseline. Its static batch-1 TensorRT FP16 engine reaches 1.933 ms
(517.45 FPS) on Tesla T4 with validation mAP50-95 0.76931.

Training used direct local group-magnitude pruning at ratio 0.20 without channel
rounding, followed by AdamW (`lr0=0.001`, `lrf=0.01`, weight decay 0.0005),
batch 64, patience 10, seed 42, AMP, and deterministic mode.

Structured pruning changes the architecture. Clone and install the
[project](https://github.com/pnthang04/PCB-Prune-YOLO) before loading:

```python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model("pcb.jpg", imgsz=640)
```

New-process CUDA inference was verified with output `[1,10,8400]`.
