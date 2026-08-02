# Project state

Last verified: 2026-08-02

## Purpose

Develop a reproducible YOLOv8n pipeline for six-class PCB defect detection on DeepPCB, then study DepGraph structured pruning at channel ratios P10, P20, and P30 without knowledge distillation.

Classes: `open`, `short`, `mousebite`, `spur`, `copper`, `pin-hole`.

## Environment and data

- Python 3.12.12.
- PyTorch 2.10.0+cu128; CUDA 12.8; cuDNN 91002.
- Ultralytics 8.4.115.
- TensorRT 10.16.1.11 for the FP16 deployment benchmark.
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

- Unit tests: see the latest verification run below; this count is historical.
- Compileall: passed.
- Ruff on changed pruning/benchmark code: passed.
- Baseline test and benchmark: complete.
- DepGraph dry-run: complete.
- P10 pre-fine-tune validation, complexity, benchmark, and save/load inference: complete.
- P10 direct and sparse fine-tunes: complete.
- P20/P30 direct prune, fine-tune, validation, and benchmark: complete.
- Baseline/P30 FP16 `trtexec` + Nsight profile and reusable-buffer/CUDA-Graph
  runtime ablation: complete. CUDA Graph helps P30 forward mean but not E2E.
- P30 INT8 PTQ calibration from 500 train-only images, engine layer audit,
  validation and latency benchmark: complete. mAP50-95 is 0.61119 versus
  0.75610 FP16, so the PTQ engine is rejected and QAT remains pending.
- P30 ModelOpt explicit-Q/DQ QAT smoke (3 epochs) is complete. Validation
  mAP50-95 recovered to 0.72462, but forward/E2E remain 4.20%/8.68% slower than
  P30 FP16. Q/DQ count is 133 pairs; coverage is 38/68 INT8-output convs versus
  PTQ 35/61, with more reformats and kernel launches. Decision: FIX_GRAPH_FIRST;
  full QAT and distillation are stopped.
- P40-HW FP16 architecture gate is complete for A8, A16 and BLOCK, all rebuilt
  from baseline at target ratio 0.40. All save/load/CUDA/TRT invariants pass.
  A8 is fastest at 1.3540 ms versus a matched P30 rerun at 1.7575 ms (1.298x),
  with 903,466 params and 1.1212G MACs. A8 pre-fine-tune validation is zero for
  all metrics/classes.
- P40-A8 standard fine-tune and a matched knowledge-distillation fine-tune
  (Ultralytics native `distill_model`/`dis`, teacher = baseline) are both
  complete; see "P40-A8: fine-tuning and knowledge distillation" below. KD
  beats standard FT by 2.7 mAP50-95 points under identical hyperparameters.
- TensorRT FP16 export/validation/benchmark for baseline and direct P10/P20/P30:
  complete.

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

## Full sparse training and sparse P10

Full run: `outputs/sparse/depgraph_sparse_p10/`. Early stopping ended the
configured 30-epoch run at epoch 20; the best validation checkpoint was epoch 10.

- Sparse best validation: precision 0.97628, recall 0.95690, mAP50 0.98548,
  mAP50-95 0.78752.
- Regularizer gradient was nonzero in all 20 epochs and introduced no non-finite
  gradients, but near-zero fraction remained 0 at threshold 0.001 and group-norm
  statistics barely moved.

Sparse P10 before fine-tuning: `outputs/pruning_sparse/p10/pruned.pt`.

- DepGraph: 59 groups; seven fixed-width output layers protected; no rounding.
- Params: 2,415,613; MACs: 3.2328G; estimated FLOPs: 6.4656G.
- Validation: precision 0.004666, recall 0.051948, mAP50 0.002615,
  mAP50-95 0.000375.
- Mean latency: 9.737 ms; FPS: 102.71; peak GPU memory: 40.02 MiB.
- New-process CUDA load and inference succeeded with output `[1,10,8400]`.

Sparse P10 is worse than the preserved no-round direct-P10 ablation on both
mAP50 and mAP50-95. Do not fine-tune or present it as a successful paper-path
result yet. The next experiment must first produce measurable group sparsity.

## Sparse reg=5e-4, P10 and fine-tuning

Sparse run: `outputs/sparse/depgraph_sparse_p10_reg5e4/`.

- Completed 30 epochs; best epoch 10.
- Sparse validation: precision 0.97646, recall 0.95673, mAP50 0.98539,
  mAP50-95 0.78938.
- Regularizer gradient was nonzero in all epochs with no new non-finite values.
- Epoch 1 to 30 group-norm change: min -0.0954%, mean -0.0042%, median
  -0.0181%, max +0.0498%; near-zero fraction remained 0. The distribution did
  not shift down materially.

No-round P10 before fine-tuning: `outputs/pruning_sparse_reg5e4/p10/pruned.pt`.

- Params 2,415,613; MACs 3.2328G.
- Validation mAP50 0.002427 and mAP50-95 0.000351.
- New-process CUDA inference passed.

Fine-tuned P10: `outputs/finetune_sparse_reg5e4/p10/weights/best.pt`.

- Training stopped at epoch 37; best epoch 27.
- Validation precision 0.95909, recall 0.94869, mAP50 0.98124,
  mAP50-95 0.76318.
- Mean latency 9.719 ms, FPS 102.89, size 4.850 MiB, peak memory 40.02 MiB.
- Versus baseline: params -19.80%, MACs -20.63%, mAP50-95 -2.21 points,
  latency +17.25%, FPS -14.71%.
- New-process CUDA load and output `[1,10,8400]` passed.

Fine-tuning recovers useful accuracy, but group sparsity remains weak and the
pruned model does not accelerate T4 batch-1 inference.

Public checkpoint and model card:
`https://huggingface.co/thangkt/PCB-Prune-YOLO-P10-DepGraph`. Anonymous model
page access and partial checkpoint download were verified; the Hub API reports
`private=false`.

## Matched direct-P10 control

Direct checkpoint: `outputs/pruning_no_round/p10/pruned.pt`. Its executed
pruning settings match the sparse P10 artifact: local group-magnitude pruning,
ratio 0.10, one step, no rounding, and the same protected detection outputs.

Matched fine-tune:
`outputs/finetune_direct_fair/p10_adamw_exact/weights/best.pt`.

- Completed all 50 epochs with AdamW, lr0 0.001, lrf 0.01, momentum 0.9,
  weight decay 0.0005, batch 64, patience 10, and seed 42.
- Validation precision 0.96479, recall 0.95706, mAP50 0.98273, and mAP50-95
  0.77736.
- Params 2,416,871; MACs 3.2695G; latency 10.433 ms; FPS 95.85; size 4.854 MiB.
- New-process CUDA load and output `[1,10,8400]` passed.
- Direct exceeds sparse reg=5e-4 by 0.01418 mAP50-95 (1.42 percentage points)
  at seed 42. Sparse learning did not improve P10 accuracy in this matched,
  single-seed experiment.
- Comparison JSON/CSV and detailed direct reports:
  `outputs/experiments/direct_vs_sparse_p10/`.

Proceed with direct P20/P30 for the accuracy-compression curve; retain sparse
P10 as an ablation.

Public direct-P10 checkpoint and model card:
`https://huggingface.co/thangkt/PCB-Prune-YOLO-P10-Direct`. Anonymous ranged
download returned HTTP 206 and the Hub API reports `private=false`.

## Direct P20

Pruned checkpoint: `outputs/pruning_direct/p20/pruned.pt`.

- Local group-magnitude pruning, ratio 0.20, one step, no rounding; 59 groups.
- Params 1,913,971 (-36.46%); MACs 2.5722G (-36.85%).
- Seven fixed-width Detect outputs protected; forward `[1,10,8400]` and
  new-process CUDA reload passed.
- Before fine-tune validation precision, recall, mAP50, and mAP50-95 were 0.
- Before fine-tune latency 10.802 ms and FPS 92.57.

Fine-tuned checkpoint:
`outputs/finetune_direct/p20_adamw_exact/weights/best.pt`.

- Completed all 50 epochs with the matched direct-P10 AdamW configuration.
- Validation precision 0.96214, recall 0.96186, mAP50 0.98184, mAP50-95
  0.76710.
- Mean latency 11.717 ms, FPS 85.35, size 3.897 MiB, peak memory 34.14 MiB.
- New-process CUDA load and output `[1,10,8400]` passed.
- Versus baseline: mAP50-95 -1.81 points, params -36.46%, MACs -36.85%, but
  latency +41.35%. Versus direct P10: mAP50-95 -1.03 points.
- Reports: `outputs/pruning_direct/p20/`,
  `outputs/finetune_direct/p20_adamw_exact/`, and
  `outputs/experiments/direct_p20_summary.{json,csv}`.

## Direct P30

Pruned checkpoint: `outputs/pruning_direct/p30/pruned.pt`.

- Local group-magnitude pruning, ratio 0.30, one step, no rounding; 59 groups.
- Params 1,452,562 (-51.77%); MACs 1.9619G (-51.83%).
- Seven fixed-width Detect outputs protected; forward `[1,10,8400]` and
  new-process CUDA reload passed.
- Before fine-tune validation precision, recall, mAP50, and mAP50-95 were 0.
- Before fine-tune latency 10.011 ms and FPS 99.89.

Fine-tuned checkpoint:
`outputs/finetune_direct/p30_adamw_exact/weights/best.pt`.

- Completed all 50 epochs with the matched P10/P20 AdamW configuration.
- Validation precision 0.95324, recall 0.94374, mAP50 0.97788, mAP50-95
  0.75030.
- Mean latency 9.863 ms, FPS 101.39, size 3.014 MiB, peak memory 29.18 MiB.
- New-process CUDA load and output `[1,10,8400]` passed.
- Versus baseline: mAP50-95 -3.49 points, params -51.77%, MACs -51.83%, and
  latency +18.99%. Versus direct P20: mAP50-95 -1.68 points.
- Reports: `outputs/pruning_direct/p30/`,
  `outputs/finetune_direct/p30_adamw_exact/`, and
  `outputs/experiments/direct_p30_summary.{json,csv}`.

## TensorRT FP16 deployment benchmark

Built baseline, direct P10, direct P20, and direct P30 engines on Tesla T4 with
TensorRT 10.16.1.11, CUDA 12.8, Ultralytics 8.4.115, FP16, batch 1, and static
`[1,3,640,640]`. No engine includes NMS. Every exact pruned source was checked
for its expected parameter count and raw output `[1,10,8400]`; new-process
engine load and inference passed for all four models.

Pure forward results after 50 warm-ups and 200 synchronized measurements:

| Model | TRT mAP50-95 | Mean/median/p95 ms | FPS | Engine MiB | vs own PyTorch | vs TRT baseline |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.78716 | 1.837/1.501/2.740 | 544.28 | 7.477 | 4.51x | 1.00x |
| P10 direct | 0.77842 | 2.023/1.728/3.489 | 494.34 | 7.627 | 5.16x | 0.91x |
| P20 direct | 0.76931 | 1.933/1.761/2.254 | 517.45 | 7.378 | 6.06x | 0.95x |
| P30 direct | 0.75610 | 1.754/1.721/3.318 | 569.97 | 5.482 | 5.62x | 1.05x |

Validation uses only the validation split. Detailed export provenance,
validation metrics (including per-class and preprocess/inference/postprocess
timings), benchmarks, and comparison reports are under
`outputs/tensorrt_fp16/`. TensorRT peak memory in benchmark JSON is the PyTorch
CUDA allocator observation; execution-context allocation is separately emitted
by TensorRT during engine load (9-10 MiB for these engines).

Public deployment/model artifacts:

- `https://huggingface.co/thangkt/PCB-Prune-YOLO-P20-Direct`
- `https://huggingface.co/thangkt/PCB-Prune-YOLO-P30-Direct`
- `https://huggingface.co/thangkt/PCB-Prune-YOLO-TensorRT-FP16`

All three repositories were created public. P20/P30 contain the validation-best
PyTorch checkpoint, model card, training args, validation metrics, and benchmark.
The TensorRT repository contains all four engines plus exact export, validation,
benchmark, and comparison JSON/CSV.

## P40-A8: fine-tuning and knowledge distillation

`outputs/pruning_hw/p40_a8_v2/p40/pruned.pt` was gitignored and lost between
sessions. It was rebuilt byte-for-byte from `configs/prune/p40_hw_a8.yaml`
against the restored baseline (`outputs/pruning_hw/p40_a8_restore/p40/pruned.pt`):
params 903,466 and MACs 1,121,237,600 match the original report exactly, and
pre-fine-tune validation again collapsed to zero for every metric/class,
confirming the reconstruction is deterministic (group-magnitude pruning
depends only on baseline weights, not RNG state). Restoring the environment
also required downloading the baseline checkpoint and processed dataset from
the public Hugging Face repositories, since `outputs/` and `data/processed/`
are both gitignored.

Ultralytics 8.4.115 (the project's pinned version) ships **native knowledge
distillation** via `distill_model` (teacher checkpoint path) and `dis`
(distillation loss weight, framework default 6.0): `ultralytics/nn/distill_model.py`
wraps a frozen teacher and the trainable student, auto-detects the Detect
head's input feature layers, and computes a score-weighted L2 feature loss
between teacher and student (with a small Conv-ReLU-Conv projector aligning
channel widths) alongside the normal detection loss. This resolves the
previously recorded blocker ("missing teacher/loss/weight from a truncated
request") without inventing a custom distillation design. `scripts/finetune_pruned.py`
gained optional `--distill-model`/`--dis` passthrough flags (default `None`,
no effect on existing calls) so `train_pruned()` forwards them straight into
Ultralytics' trainer overrides.

Two branches were fine-tuned from the identical restored `pruned.pt`, with
every other hyperparameter matched exactly (AdamW, lr0 0.001, lrf 0.01,
momentum 0.9, weight decay 0.0005, batch 64, imgsz 640, patience 10, seed 42,
50 epochs, one T4 each, no early stop triggered on either):

| Branch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Standard FT (`outputs/finetune_direct/p40_a8_adamw_exact`) | 0.867 | 0.805 | 0.893 | 0.634 |
| KD, teacher=baseline (`outputs/finetune_direct/p40_a8_kd_baseline_teacher`) | 0.895 | 0.828 | 0.913 | 0.660 |

KD wins on precision, recall, mAP50 and mAP50-95 (+0.026 mAP50-95, +2.0 mAP50
points) under a fair, single-difference comparison. Both checkpoints passed
new-process CUDA load/inference (`[1,10,8400]`, 6 classes) and batch-1
PyTorch benchmark: identical architecture (903,466 params, 1.1212G MACs,
~1.96 MiB) since fine-tuning does not change channel counts. Both accuracy
figures remain well below P30 direct (mAP50-95 0.75030) at a much more
aggressive compression point (-70.00% params, -72.47% MACs vs P30's
-51.77%/-51.83%), so P40-A8 (even with KD) is a faster-but-less-accurate
alternative to P30, not a strict improvement. Test split was not touched.

Both checkpoints were published publicly on Hugging Face with model card,
training args, validation metrics, and benchmark reports:

- `https://huggingface.co/thangkt/PCB-Prune-YOLO-P40-A8-Direct`
- `https://huggingface.co/thangkt/PCB-Prune-YOLO-P40-A8-KD`

Anonymous download of `best.pt` and public API visibility (`private: false`)
were verified for both repositories.

TensorRT FP16 engines were rebuilt for both fine-tuned checkpoints, plus a
fresh baseline engine in the same session (a same-instance rebuild is
required because absolute PyTorch/TensorRT latency drifts noticeably, roughly
10-20%, across different cloud T4 instances even with identical declared
GPU/software specs — confirmed by re-measuring baseline PyTorch latency
across sessions). Same-session, 50 warm-up/200 iteration, batch-1 FP16
results:

| Model | TensorRT latency | FPS | Speedup vs baseline |
|---|---:|---:|---:|
| Baseline (rebuilt) | 1.716 ms | 582.75 | 1.00x |
| P40-A8 standard FT | 1.423 ms | 702.91 | 1.206x |
| P40-A8 KD | 1.417 ms | 705.96 | 1.211x |

Both fine-tuned engines passed new-process load/inference. This confirms
fine-tuning (weights only, no channel-count change) preserves the
architecture-level TensorRT speedup measured before fine-tuning. Reports:
`outputs/finetune_direct/p40_a8_adamw_exact/{evaluation_val,benchmark}`,
`outputs/finetune_direct/p40_a8_kd_baseline_teacher/{evaluation_val,benchmark}`,
and `outputs/tensorrt_recheck/`.

## HALP Stage 1 latency LUT

Reviewed the 26-page official HALP paper/supplement and official `NVlabs/HALP`
repository at commit `dfee297d55d1638b968359e7ffff878be846ec02`. The NVIDIA
license restricts official code/derivatives to non-commercial research or
evaluation, so the project uses it as a reference and implements independent
AGPL-compatible adaptation code. Full provenance and PAPER/OFFICIAL
CODE/ADAPTATION/TODO classification are in `docs/HALP_ADAPTATION_PLAN.md`.

## HALP Stage 2 dry-run

`scripts/run_halp_stage2.py` starts from the unpruned baseline and performs no
optimizer step or structural pruning. The verified run used 8 train minibatches
at batch 8, accumulated the official BN Taylor term, enumerated backbone-only
DepGraph roots, and solved a 5% prefix-constrained latency milestone.

- Artifacts: `outputs/halp/stage2/dry_run.json` and `groups.csv`.
- 25 backbone roots; 13 eligible and 12 protected for missing latency cliffs.
- No exact LUT pair was missing.
- Eligible dense latency: 0.447758 ms; budget: 0.425370 ms; selected: 0.419882 ms.
- Detect/DFL fixed outputs remained protected; 6-class output was `[1,10,8400]`.
- C2f `cv0` and `cv1` branches map to the original Stage 1 `cv1` operator
  surface. Structural Stage 3 must rebuild costs after every milestone so
  downstream `Cin` changes are never silently approximated.

## HALP Stage 3 M05

The official Taylor detail was corrected before pruning: sum signed
`gamma*dL/dgamma + beta*dL/dbeta` terms across a dependency group, then take
the absolute value. Seven roots were structurally pruned. Eight newly required
2D LUT pairs were profiled on the same T4/TensorRT setup; the final audit is
exact with no interpolation.

- Checkpoint: `outputs/halp/stage3_m05/pruned.pt` (ignored by Git).
- Reports: `outputs/halp/stage3_m05/{report,summary}.json`, validation and
  benchmark JSON/CSV; refined LUT under `outputs/halp/lut_stage3/`.
- Params 2,690,674 (-10.67%); MACs 3.9181G (-3.81%).
- Pre-fine-tune validation: P 0.94135, R 0.90165, mAP50 0.96418,
  mAP50-95 0.67681.
- PyTorch latency 10.115 ms; FPS 98.86; slower than baseline.
- New-process load and output `[1,10,8400]` passed.

This is a structurally valid intermediate checkpoint, not a successful final
HALP model. Fine-tuning and full TensorRT engine measurement remain TODO. Test
was not used.

Matched TensorRT FP16 gate under `outputs/halp/tensorrt_m05_comparison/`:

- Baseline forward 1.7797 ms, M05 1.8385 ms: M05 speedup 0.968x (slower).
- `trtexec` per-layer total 1.6741 vs 1.6768 ms: no acceleration.
- E2E excluding disk but including preprocess/H2D/NMS: 4.9587 vs 4.6080 ms,
  yet M05 loses 0.12971 mAP50-95 and 0.07235 recall, so the NMS/content effect
  is not accepted as architecture speedup.
- Engine layers 159 → 166, reformats 28 → 34, pointwise nodes 14 → 30;
  pointwise time 0.1087 → 0.2405 ms.
- Decision: stop further milestones and final fine-tuning. Fix C2f export fusion
  and account for full-engine graph overhead in cost/grouping first.

Measured TensorRT FP16 LUT on Tesla T4:

- Baseline checkpoint, static batch 1, full-model input `[1,3,640,640]`.
- 27 leaf Conv2d names in backbone `model.0`–`model.9`; 19 unique signatures.
- 37 convolution names in neck/Detect/DFL excluded and recorded.
- 598 two-dimensional sampled `Cin×Cout` records; 598 success, zero failure.
- Each entry: 50 warm-ups, 200 synchronized iterations, reused buffers, build
  and image I/O excluded, exact TensorRT tactic recorded.
- Dense native configurations repeated; maximum median relative error 13.90%,
  under the declared 20% tolerance.
- 56 latency cliffs and 98 plateaus. Resolved group steps span 8, 16, 24, 32,
  40, 48, and 64 channels; 14 layers remain unresolved due to insufficient
  distinct cliffs.
- Strong operator-level latency potential: stem/downsample and selected fusion
  convs (`model.0`, `model.1`, `model.3`, `model.4.cv2`, `model.9.cv1`).
- Little/no extreme-sweep benefit: `model.2.m.0.cv1/cv2`, `model.8.cv1`, and
  `model.8.cv2`.

Artifacts: `outputs/halp/lut/t4_fp16_backbone.{json,csv}`,
`environment.json`, and `latency_steps.{json,csv}`. This is a LUT only; Taylor
saliency, latency-aware DepGraph grouping, augmented knapsack, iterative pruning,
fine-tuning, and test evaluation are not implemented.
