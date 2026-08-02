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

Direct P30 completed the same pipeline and all 50 fine-tune epochs. It reduces
params/MACs by 51.77%/51.83% and reaches validation precision 0.95324, recall
0.94374, mAP50 0.97788, and mAP50-95 0.75030. This is 3.49 points below
baseline, 2.71 below direct P10, and 1.68 below P20. T4 latency is 9.863 ms
(101.39 FPS), still 18.99% slower than baseline despite the model shrinking to
3.014 MiB. Save/load CUDA inference passed. P30 is the strongest compression
point, while P20 remains the more balanced accuracy-compression candidate.

TensorRT FP16 deployment evaluation is complete for baseline and direct
P10/P20/P30 on the same Tesla T4, TensorRT 10.16.1.11, CUDA 12.8, static batch
1 `[1,3,640,640]`. Pure-forward means are 1.837, 2.023, 1.933, and 1.754 ms;
validation mAP50-95 values are 0.78716, 0.77842, 0.76931, and 0.75610. P30 is
the only pruned engine faster than TensorRT baseline in this measurement
(1.05x), making it the compression/speed deployment candidate rather than the
accuracy candidate. All engines passed new-process inference with
`[1,10,8400]`. Public artifacts:

- `https://huggingface.co/thangkt/PCB-Prune-YOLO-P20-Direct`
- `https://huggingface.co/thangkt/PCB-Prune-YOLO-P30-Direct`
- `https://huggingface.co/thangkt/PCB-Prune-YOLO-TensorRT-FP16`

HALP Stage 1 is complete, but HALP pruning is not. The official NeurIPS 2022
paper/supplement and `NVlabs/HALP` commit
`dfee297d55d1638b968359e7ffff878be846ec02` were reviewed and classified in
`docs/HALP_ADAPTATION_PLAN.md`. A TensorRT 10.16.1.11 FP16 LUT was measured on
Tesla T4 for 27 YOLOv8n backbone conv names (19 unique signatures): 598/598
configurations succeeded using 50 warm-ups and 200 timed iterations. Analysis
found 56 cliffs, 98 plateaus, and layer-specific candidate steps from 8 through
64 channels. Stage 2 then averaged true detection-loss BN Taylor saliency over
8 train minibatches and ran a non-mutating 5% augmented-knapsack dry-run: 13 of
25 backbone roots were eligible, 12 were protected for lack of a reliable
cliff, no exact LUT pair was missing, and output remained `[1,10,8400]`.
Artifacts are under `outputs/halp/lut/` and `outputs/halp/stage2/`. At the end
of Stage 2, structural pruning and save/load verification had not yet run.

HALP Stage 3 applied the first structural 5% milestone after correcting signed
Taylor aggregation against the official implementation. Eight missing exact
LUT pairs were measured on T4; save/new-process-load/inference passed. The
checkpoint has 2.691M params and 3.918G MACs, but before fine-tuning validation
mAP50-95 is 0.67681 and PyTorch latency is 10.115 ms, worse than baseline.
Fine-tuning and full TensorRT measurement are the next gate; test was not used.

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

Select the final operating point on validation: P10 for best pruned accuracy,
P20 for balanced compression, or P30 for maximum compression and measured
TensorRT speed. Only the selected checkpoint should then receive final test-set
evaluation.

## Verification

- Dataset: 800 train, 200 validation, 500 test; no cross-split duplicates.
- Tests: 11 passed.
- Compileall and Ruff: passed.
- Baseline test/benchmark: complete.
- P10 dry-run, pruning, validation, benchmark, and save/load inference: complete.
- P10/P20/P30 direct fine-tuning: complete.
- TensorRT FP16 export, validation, and benchmark for all four models: complete.
