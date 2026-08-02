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

# PCB-Prune-YOLO P40-A8 Direct

YOLOv8n checkpoint produced by DepGraph structured pruning at a hardware
latency-first target ratio of 0.40 (`round_to=8`, "A8" candidate), then
fine-tuned with plain AdamW. This is the standard-fine-tune control for the
matched P40-A8 knowledge-distillation checkpoint,
[thangkt/PCB-Prune-YOLO-P40-A8-KD](https://huggingface.co/thangkt/PCB-Prune-YOLO-P40-A8-KD).
This model does not use sparse learning or knowledge distillation.

## Validation results

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.867 | 0.805 | 0.893 | 0.634 |

The matched knowledge-distillation checkpoint reaches mAP50-95 0.660 under an
otherwise identical fine-tune recipe (+2.7 percentage points). Both remain
well below P10/P20/P30 direct (0.77736 / 0.76710 / 0.75030) because this
architecture is a substantially more aggressive compression point, selected
for a TensorRT hardware-latency gate rather than for accuracy. The DeepPCB
test split was not used for model selection.

## Compression and Tesla T4 benchmark

| Parameters | MACs | Size | Latency batch 1 (PyTorch) | FPS |
|---:|---:|---:|---:|---:|
| 903,466 | 1.1212G | 1.959 MiB | 7.818 ms | 127.92 |

Versus baseline (3,012,018 params, 4.0733G MACs): -70.00% parameters, -72.47%
MACs. Input size is 640. Latency uses 50 warm-up and 200 synchronized CUDA
iterations. On the same hardware-latency gate that selected this architecture,
a same-session rebuilt TensorRT FP16 engine measured 1.423 ms forward
(50 warm-up / 200 iterations, batch 1), about 1.21x faster than a
same-session baseline TensorRT engine (1.716 ms); TensorRT engines are not
included in this repository.

## Training configuration

- DepGraph local group-magnitude pruning, target ratio 0.40, `round_to=8`
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
