# PCB-Prune-YOLO project memory

Group-level sparse training is now implemented using vendored Torch-Pruning 1.6.0.
The hook runs after backward/unscale and before optimizer step, preserves YOLO's
detection loss, and logs group norms plus direct evidence that the regularizer
changes gradients. The accepted one-epoch smoke run is under
`outputs/sparse/depgraph_sparse_smoke_final`. See `docs/DEPGRAPH_SPARSE.md` for
provenance and commands.

The first full sparse run (`reg=1e-4`) later stopped at epoch 20 with best
validation mAP50-95 0.78752, but its group near-zero fraction remained zero.
No-round sparse P10 reached only mAP50 0.002615 and mAP50-95 0.000375 before
fine-tuning, worse than direct P10. Its save/load and benchmark checks passed;
do not fine-tune it until sparse regularization is tuned to create measurable
group sparsity.

A second sparse run used `reg=5e-4` for all 30 epochs. Its best unpruned
validation mAP50-95 was 0.78938, but mean/median group norm changed by only
-0.0042%/-0.0181% and near-zero fraction stayed zero. P10 again collapsed before
fine-tuning, then recovered to mAP50-95 0.76318 after fine-tuning (best epoch 27,
stopped epoch 37). It reduces params by 19.80% and MACs by 20.63%, but remains
17.25% slower than baseline on T4 batch 1.

The fine-tuned P10 checkpoint is public at
`https://huggingface.co/thangkt/PCB-Prune-YOLO-P10-DepGraph`, together with its
model card, sparse config, validation metrics, benchmark, and summary. Anonymous
access was verified. Treat it as the current P10 candidate, not the final model
before P20/P30 comparison.

The matched direct-P10 control was then fine-tuned for all 50 epochs with
explicit AdamW, lr0 0.001, lrf 0.01, momentum 0.9, weight decay 0.0005, batch
64, patience 10, and seed 42. It reached validation mAP50 0.98273 and mAP50-95
0.77736, versus 0.98124/0.76318 for sparse P10. Direct is therefore +1.42
mAP50-95 percentage points at this seed. Save/load CUDA inference passed; its
benchmark is 2,416,871 params, 3.2695 GMACs, 10.433 ms, and 95.85 FPS. Continue
the P20/P30 accuracy-compression curve with direct pruning and retain sparse P10
as an ablation. Reports are in `outputs/experiments/direct_vs_sparse_p10/`.
The direct checkpoint is public at
`https://huggingface.co/thangkt/PCB-Prune-YOLO-P10-Direct`; anonymous ranged
download and `private=false` were verified.

Direct P20 used the same no-round local group-magnitude pruning and explicit
AdamW fine-tune as direct P10. It reduces params/MACs by 36.46%/36.85%.
Validation was zero before fine-tuning, then recovered after all 50 epochs to
precision 0.96214, recall 0.96186, mAP50 0.98184, and mAP50-95 0.76710. This is
1.03 points below direct P10 and 1.81 points below baseline. T4 latency is
11.717 ms (85.35 FPS), so MAC reduction still does not produce acceleration.
Save/load CUDA inference passed. Next gate is direct P30.

Last verified: 2026-08-02

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
