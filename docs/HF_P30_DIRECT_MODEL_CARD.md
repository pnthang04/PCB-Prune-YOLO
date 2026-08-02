---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags: [yolo, yolov8, depgraph, torch-pruning, structured-pruning, deeppcb]
datasets: [thangkt/PCB-Prune-YOLO-DeepPCB]
metrics: [map]
---

# PCB-Prune-YOLO P30 Direct

Strongest compression candidate in the direct DepGraph study. This
validation-selected YOLOv8n checkpoint uses structured pruning followed by the
same 50-epoch fine-tuning configuration as P10/P20. No knowledge distillation
or test-set model selection was used.

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.95324 | 0.94374 | 0.97788 | 0.75030 |

The model has 1,452,562 parameters and 1.9619 GMACs, reductions of 51.77% and
51.83% from baseline. Its static batch-1 TensorRT FP16 engine reaches 1.754 ms
(569.97 FPS) on Tesla T4, about 1.05x the TensorRT baseline throughput, with
validation mAP50-95 0.75610.

Training used direct local group-magnitude pruning at ratio 0.30 without channel
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
