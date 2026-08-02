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
  - knowledge-distillation
  - deeppcb
datasets:
  - thangkt/PCB-Prune-YOLO-DeepPCB
metrics:
  - map
---

# PCB-Prune-YOLO P40-A8 KD

YOLOv8n checkpoint produced by DepGraph structured pruning at a hardware
latency-first target ratio of 0.40 (`round_to=8`, "A8" candidate), then
fine-tuned with Ultralytics' native knowledge distillation, teacher =
[thangkt/PCB-Prune-YOLO-Baseline](https://huggingface.co/thangkt/PCB-Prune-YOLO-Baseline).
This is the strongest-compression checkpoint in the project so far. Its
matched standard-fine-tune control (identical hyperparameters, no
distillation) is
[thangkt/PCB-Prune-YOLO-P40-A8-Direct](https://huggingface.co/thangkt/PCB-Prune-YOLO-P40-A8-Direct).

## Validation results

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.930 | 0.898 | 0.957 | 0.712 |

An initial 50-epoch fine-tune reached only mAP50-95 0.660, with the best
validation epoch landing on the very last epoch (not converged), so training
was extended to 100 epochs with cosine LR annealing, after which the curve
plateaus. Under an otherwise identical fine-tune recipe, this checkpoint
beats the matched standard fine-tune by +1.1 mAP50-95 percentage points
(0.712 vs 0.701). A `dis` sweep (3.0 and 10.0) confirmed the default `dis=6.0`
used here was already near-optimal (0.711 / 0.712 / 0.709, within noise).
Versus P30 direct (0.75030, -51.77%/-51.83% params/MACs), this checkpoint is
now only 3.83 points lower for 37.8% fewer parameters and 42.85% fewer MACs —
a substantially more competitive trade-off than the initial 50-epoch result.
The DeepPCB test split was not used for model selection.

## Compression and Tesla T4 benchmark

| Parameters | MACs | Size | Latency batch 1 (PyTorch) | FPS |
|---:|---:|---:|---:|---:|
| 903,466 | 1.1212G | 1.960 MiB | 8.018 ms | 124.72 |

Input size is 640. Latency uses 50 warm-up and 200 synchronized CUDA
iterations. Fine-tuning does not change channel counts, so a same-session
rebuilt TensorRT FP16 engine for this checkpoint measured 1.497 ms forward
(50 warm-up / 200 iterations, batch 1), still clearly ahead of a same-session
baseline TensorRT engine (1.716 ms); TensorRT engines are not included in
this repository.

## Training configuration

- DepGraph local group-magnitude pruning, target ratio 0.40, `round_to=8`
- AdamW, `lr0=0.001`, `lrf=0.01`, momentum 0.9, weight decay 0.0005, cosine LR
- 100 epochs, batch 64, patience 20, seed 42, AMP and deterministic mode
- Knowledge distillation: Ultralytics 8.4.115 native `distill_model` +
  `dis=6.0` (framework default, confirmed near-optimal by a 3.0/10.0 sweep) —
  a score-weighted feature L2 loss between teacher and student at the Detect
  head's input layers, on top of the normal detection loss. Teacher
  checkpoint: baseline `best.pt`, frozen.
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
