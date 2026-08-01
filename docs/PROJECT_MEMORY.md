# PCB-Prune-YOLO project memory

Last verified: 2026-08-01

The authoritative agent context lives in `.codex/skills/pcb-prune-yolo/`:

- `SKILL.md`: operating rules and context routing.
- `references/project-state.md`: environment, dataset, measured results, and artifacts.
- `references/workflows.md`: safe commands and implementation constraints.
- `references/roadmap.md`: goals, completed work, backlog, and known issues.

## Goal

Train a reproducible YOLOv8n baseline for six DeepPCB defect classes, then compare DepGraph group-magnitude structured pruning at P10, P20, and P30 before and after low-learning-rate fine-tuning. Select configurations only on validation; reserve test for final reporting. Knowledge distillation is out of scope.

## Current results

Baseline checkpoint: `outputs/train/baseline/weights/best.pt`.

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| validation | 0.96545 | 0.97221 | 0.98630 | 0.78524 |
| test | 0.95094 | 0.93944 | 0.96965 | 0.73793 |

Baseline batch-1 benchmark on Tesla T4: 3,012,018 params, 4.0733 GMACs, 5.968 MiB, 8.289 ms mean latency, 120.64 FPS, and 47.40 MiB peak GPU memory.

DepGraph dry-run succeeds with 57 dependency groups and output `[1,10,8400]`. Eight C2f blocks are converted to equivalent explicit two-branch blocks before tracing. Fixed regression/classification output convolutions and DFL are protected.

P10 `round_to=8` reduces params to 2,289,938 and MACs to 2.9733G. Save, new-process load, and CUDA inference pass, but validation metrics are zero before fine-tuning and latency worsens to 10.063 ms. A no-round diagnostic retains 2,416,871 params and 3.2695 GMACs, with validation mAP50-95 0.000923. Neither P10 candidate is usable before fine-tuning.

## Next gate

Fine-tune the no-round P10 candidate for up to 50 epochs with learning rate 0.001, batch 64, image size 640, seed 42, AMP, and validation-based early stopping. Verify the pruned dimensions are preserved and meaningful validation accuracy recovers before starting P20/P30.

## Verification

- Dataset: 800 train, 200 validation, 500 test; no cross-split duplicates.
- Tests: 11 passed.
- Compileall and Ruff: passed.
- Baseline test/benchmark: complete.
- P10 dry-run, pruning, validation, benchmark, and save/load inference: complete.
- P10 fine-tune and P20/P30: not run.
