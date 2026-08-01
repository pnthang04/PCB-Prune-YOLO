# Project state

Last verified: 2026-08-01

## Purpose

Develop a reproducible YOLOv8n pipeline for six-class PCB defect detection on DeepPCB, then study DepGraph structured pruning at channel ratios P10, P20, and P30 without knowledge distillation.

Classes: `open`, `short`, `mousebite`, `spur`, `copper`, `pin-hole`.

## Environment and data

- Python 3.12.12.
- PyTorch 2.10.0+cu128; CUDA 12.8; cuDNN 91002.
- Ultralytics 8.4.115.
- Vendored Torch-Pruning 1.6.0 from `src/torch_pruning`.
- Two Tesla T4 GPUs, each 14.56 GiB.
- Train: 800 images, 5,485 boxes.
- Validation: 200 images, 1,388 boxes.
- Test: 500 images, 3,140 boxes.
- Dataset validation found no content duplicates across splits.

## Baseline

Checkpoint: `outputs/train/baseline/weights/best.pt`.

Training used YOLOv8n, image size 640, global batch 128 on two GPUs, seed 42, AMP, deterministic mode, 100 epochs, and patience 20. The run completed all 100 epochs. Best validation checkpoint was epoch 98.

Validation:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.96545 | 0.97221 | 0.98630 | 0.78524 |

Test, for final baseline reporting only:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.95094 | 0.93944 | 0.96965 | 0.73793 |

Test per class:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| open | 0.97069 | 0.95473 | 0.97940 | 0.66044 |
| short | 0.93134 | 0.84728 | 0.91361 | 0.61358 |
| mousebite | 0.98593 | 0.95734 | 0.98849 | 0.73041 |
| spur | 0.98252 | 0.93095 | 0.97586 | 0.72587 |
| copper | 0.97817 | 0.96552 | 0.98080 | 0.86682 |
| pin-hole | 0.85698 | 0.98085 | 0.97974 | 0.83044 |

Baseline benchmark, batch 1 at 640 on Tesla T4:

| Params | MACs | Estimated FLOPs | Size | Mean/median/p95 latency | FPS | Peak GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 3,012,018 | 4.0733G | 8.1465G | 5.968 MiB | 8.289/8.196/9.171 ms | 120.64 | 47.40 MiB |

Reports:

- `outputs/evaluation/metrics_test.{json,csv}`
- `outputs/benchmark/benchmark.{json,csv}`

Public model: `https://huggingface.co/thangkt/PCB-Prune-YOLO-Baseline`.

## DepGraph integration

Implemented under `src/pcb_prune_yolo/pruning/` and `scripts/prune_model.py`.

- Replaces eight Ultralytics C2f blocks with the equivalent `PrunableC2f` two-branch form. This avoids split/chunk coupling that collapsed the dependency graph into one unusable group.
- Verified conversion output shape is unchanged; observed maximum numerical difference was about `3.7e-4` on a random input.
- Dry-run finds 57 pruning groups with input `[1,3,640,640]`.
- Uses `GroupMagnitudeImportance(p=2)` with `BasePruner` and local pruning by default.
- Protects the six final regression/classification convolutions and `model.22.dfl.conv` as output-pruning roots.
- Checks six class names and decoded prediction shape `[1,10,8400]`.
- Saves the complete changed model in an Ultralytics checkpoint dictionary.
- Launches `scripts/verify_pruned_model.py` in a new process to test load and CUDA inference.

## P10 before fine-tuning

Canonical `round_to=8` artifact: `outputs/pruning/p10/pruned.pt`.

| Params | MACs | Estimated FLOPs | Size | Mean latency | FPS | Peak GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 2,289,938 | 2.9733G | 5.9467G | 9.035 MiB | 10.063 ms | 99.38 | 36.65 MiB |

- Parameter reduction: about 24.0%.
- MAC reduction: about 27.0%.
- Validation precision, recall, mAP50, and mAP50-95 are all zero before fine-tuning.
- Raw class confidence collapsed below 0.001.
- Save, new-process load, and CUDA inference succeeded.
- Latency became worse despite lower MACs.

Diagnostic P10 without channel rounding: `outputs/pruning_no_round/p10/pruned.pt`.

- Params: 2,416,871.
- MACs: 3.2695G.
- Validation mAP50: 0.00586.
- Validation mAP50-95: 0.000923.
- This is slightly better than round-to-8 according to validation, but still unusable without fine-tuning.

## Verification state

- Unit tests: 11 passed.
- Compileall: passed.
- Ruff on changed pruning/benchmark code: passed.
- Baseline test and benchmark: complete.
- DepGraph dry-run: complete.
- P10 pre-fine-tune validation, complexity, benchmark, and save/load inference: complete.
- P10 fine-tune: not run.
- P20/P30: not run.

## Group-level sparse training

Implemented `GroupNormPruner` sparse-gradient training from the unpruned baseline.
The custom trainer unscales AMP gradients, calls the vendored TP 1.6.0
`regularize(model, alpha=2**4)`, then clips and steps. It refreshes groups each
epoch and protects the same six Detect output convolutions plus DFL.

Accepted smoke artifact: `outputs/sparse/depgraph_sparse_smoke_final/`.

- 1 epoch, 10% of train (80 images), full 200-image validation, batch 32, GPU 0.
- Train losses: box 0.94178, cls 0.65621, DFL 0.89616.
- Validation: precision 0.96472, recall 0.97079, mAP50 0.98624,
  mAP50-95 0.78532.
- 59 dependency groups and 4,864 channel/group-norm values regularized; zero
  degenerate groups skipped; near-zero fraction at threshold 0.001 was 0.
- Regularizer gradient delta L2 was 0.0397833, nonzero, with no newly introduced
  non-finite gradient values.
- New-process CUDA load and inference succeeded with six classes and output
  `[1, 10, 8400]`.

This is only a hook smoke test, not a completed sparse-training or P10 result.
