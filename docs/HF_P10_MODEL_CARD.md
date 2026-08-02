---
license: mit
library_name: ultralytics
pipeline_tag: object-detection
tags:
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

# PCB-Prune-YOLO P10 DepGraph

Fine-tuned structurally pruned YOLOv8n detector for the six DeepPCB defect
classes: `open`, `short`, `mousebite`, `spur`, `copper`, and `pin-hole`.

## Pipeline

The checkpoint was produced with:

```text
YOLOv8n baseline
→ DepGraph GroupNormPruner sparse training (reg=5e-4, alpha=4, 30 epochs)
→ 10% no-round group-magnitude structured pruning
→ fine-tuning (best epoch 27, stopped epoch 37)
```

Model selection used the validation split only. The DeepPCB test split was not
used to select sparse-training, pruning, or fine-tuning settings.

## Validation results

| Model | Params | MACs | mAP50 | mAP50-95 | T4 latency | FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n baseline | 3,012,018 | 4.0733G | 0.98630 | 0.78524 | 8.289 ms | 120.64 |
| P10 before fine-tuning | 2,415,613 | 3.2328G | 0.00243 | 0.00035 | 10.127 ms | 98.75 |
| P10 after fine-tuning | 2,415,613 | 3.2328G | 0.98124 | 0.76318 | 9.719 ms | 102.89 |

Relative to the baseline, the fine-tuned P10 checkpoint has 19.80% fewer
parameters and 20.63% fewer MACs, with a 2.21-point mAP50-95 drop. It does not
provide a batch-1 speedup on Tesla T4: measured latency is 17.25% higher.

The sparse regularizer changed gradients in every sparse-training epoch without
introducing non-finite values. However, mean and median group norm moved only
-0.0042% and -0.0181%, and the measured near-zero fraction remained zero. This
checkpoint therefore demonstrates post-pruning accuracy recovery, not strong
group sparsification or deployment acceleration.

## Loading

Structured pruning changes the architecture. Clone and install the project so
the serialized `PrunableC2f` class is available, then load the complete model:

```bash
git clone https://github.com/pnthang04/PCB-Prune-YOLO.git
cd PCB-Prune-YOLO
pip install -e . --no-deps
```

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

checkpoint = hf_hub_download(
    repo_id="thangkt/PCB-Prune-YOLO-P10-DepGraph",
    filename="best.pt",
)
model = YOLO(checkpoint)
results = model.predict("pcb.jpg", imgsz=640)
```

## Artifacts

- `best.pt`: complete fine-tuned structurally pruned model.
- `depgraph_sparse_reg5e4.yaml`: sparse-training configuration.
- `summary.json` / `summary.csv`: comparison and provenance.
- `metrics_val.json` / `metrics_val.csv`: validation metrics including classes.
- `benchmark.json` / `benchmark.csv`: synchronized batch-1 Tesla T4 benchmark.

Project repository: https://github.com/pnthang04/PCB-Prune-YOLO
