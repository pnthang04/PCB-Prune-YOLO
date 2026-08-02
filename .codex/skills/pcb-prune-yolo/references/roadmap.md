# Roadmap

## Project goals

1. Establish a reproducible YOLOv8n DeepPCB baseline.
2. Apply DepGraph group-magnitude structured channel pruning at P10, P20, and P30.
3. Recover accuracy with low-learning-rate fine-tuning and no knowledge distillation.
4. Select the best compression/accuracy trade-off using validation mAP50-95.
5. Evaluate the selected model once on test.
6. Report accuracy, parameters, MACs, latency, FPS, memory, and model size for every comparison row.
7. Publish reproducible code, reports, and chosen checkpoints.

## Completed

- Deterministic DeepPCB conversion and train/validation split.
- Dataset pairing, coordinate, class-range, and cross-split duplicate validation.
- YOLOv8n baseline training on two T4 GPUs.
- Baseline validation and official test evaluation with per-class JSON/CSV.
- Baseline complexity and synchronized batch-1 GPU benchmark.
- Public baseline checkpoint and model card on Hugging Face.
- Torch-Pruning 1.6.0 API inspection and DepGraph dry-run.
- Pruning-friendly C2f conversion without editing Ultralytics source.
- Fixed output-layer protection and forward invariants.
- P10 pruning with and without channel rounding.
- P10 save, new-process load, CUDA inference, pre-fine-tune validation, and benchmark.
- Dedicated single-GPU fine-tune path that preserves the changed architecture.
- Paper/official-code review for group-level sparse learning and a custom safe
  optimizer-boundary hook using vendored `GroupNormPruner`.
- One-epoch sparse hook smoke test with nonzero regularizer gradient, stable
  validation, and new-process CUDA inference.
- Full sparse-training run with `reg=1e-4`, followed by no-round sparse P10
  validation, complexity, benchmark, and new-process inference verification.
- Fixed 30-epoch sparse run with `reg=5e-4`, P10 pruning, 37-epoch fine-tuning,
  validation, benchmark, and new-process inference verification.
- Published the fine-tuned P10 checkpoint, model card, config, validation,
  benchmark, and summary artifacts in a public Hugging Face model repository.
- Completed the matched direct-P10 control with explicit AdamW settings. It
  reached validation mAP50-95 0.77736 versus 0.76318 for sparse P10 at seed 42;
  save/load inference, JSON/CSV evaluation, and benchmark passed.
- Completed direct P20 pruning, pre-fine-tune validation/benchmark, matched
  50-epoch fine-tune, post-fine-tune validation/benchmark, and new-process CUDA
  reload. P20 reached validation mAP50-95 0.76710 with 36.46% fewer params and
  36.85% fewer MACs, but latency remained worse than baseline.
- Completed direct P30 pruning, pre/post-fine-tune validation and benchmark,
  matched 50-epoch fine-tune, and new-process CUDA reload. P30 reached
  validation mAP50-95 0.75030 with 51.77% fewer params and 51.83% fewer MACs;
  latency remained 18.99% slower than baseline.
- Completed static batch-1 TensorRT FP16 export, new-process inference,
  validation-only evaluation, and synchronized 50/200 latency benchmarks for
  baseline and direct P10/P20/P30 on Tesla T4. P30 was the only pruned engine
  faster than TensorRT baseline in this run (about 1.05x).
- Published public P20/P30 validation-best checkpoints and all four TensorRT
  FP16 engines with model cards and measured reports on Hugging Face.
- Reviewed HALP paper/supplement and official code, documented the YOLOv8n +
  DepGraph adaptation, and completed Stage 1 TensorRT FP16 T4 LUT profiling:
  27 backbone conv names, 19 unique signatures, 598/598 successful sampled 2D
  configurations, 56 cliffs, and 98 plateaus.
- Implemented HALP Stage 2 dry-run: averaged BN Taylor saliency over 8 train
  minibatches, enumerated 25 backbone DepGraph roots, formed measured
  latency-step prefixes for 13 eligible roots, protected 12 roots without a
  reliable cliff, and solved the 5% augmented-knapsack milestone with zero
  missing exact LUT pairs. The model was not mutated and output stayed
  `[1,10,8400]`.
- Corrected Taylor dependency aggregation against official HALP, applied the
  first 5% structural milestone across seven roots, measured eight missing
  exact LUT pairs, and passed save/new-process-load/inference. M05 has 10.67%
  fewer params and 3.81% fewer MACs, but pre-fine-tune mAP50-95 is 0.67681 and
  PyTorch latency is slower at 10.115 ms.
- Completed the matched TensorRT FP16 gate. M05 forward-only is 1.838 ms versus
  1.780 ms baseline (0.968x); `trtexec` per-layer is also slightly slower.
  Although E2E including NMS is 1.076x faster, severe accuracy/recall loss makes
  it content-dependent rather than accepted architecture acceleration.
- Completed the independent P40-HW structural latency gate from baseline. A8,
  A16 and BLOCK all beat matched P30 forward latency; A8 is selected at 1.3540
  ms versus 1.7575 ms P30 with 0.903M params and 1.1212G MACs. Its pre-FT
  validation is zero, so no long training or KD was launched.

## Not completed

- Complete the truncated P40-HW training specification (teacher checkpoint,
  distillation outputs/loss, temperature, weights, epochs and early stopping),
  then run standard FT and KD from the exact same A8 pre-FT checkpoint. Rebuild
  and benchmark FP16 engines after training before accepting deployment speed.

- Fix the P30 explicit-Q/DQ graph before any full QAT: fuse Conv-BN, reduce
  Q/DQ and reformat boundaries around SiLU/residual/concat, then require a
  full-engine forward win over FP16. The completed 3-epoch QAT smoke recovered
  accuracy but failed coverage and latency gates; distillation remains stopped.

- Fix C2f conversion/export fusion and include full-engine reformat/pointwise
  overhead in the HALP cost/grouping adaptation. Re-run the M05 TensorRT
  forward gate before training or another milestone.

- Tune sparse learning on validation until group norms show measurable sparsity;
  the first `reg=1e-4` run retained baseline accuracy but produced no near-zero
  groups and its P10 result was worse than direct pruning.

- Decide whether `round_to=8` or no rounding is the standard policy. Current validation favors no rounding before fine-tuning, but both are extremely weak.
- Implement safe multi-GPU DDP fine-tuning for changed model objects.
- Produce the final comparison table covering baseline and P10/P20/P30 before/after fine-tuning.
- Select the best pruned model using validation mAP50-95.
- Evaluate only that selected pruned model on test.
- Publish only the final validation-selected model after P20/P30 comparisons;
  the current P10 candidate is already available as an explicitly provisional
  public artifact.

## Known issues and decisions

- Directly tracing stock C2f produced one unusable dependency group. Replace C2f with explicit two-branch blocks first.
- Ignoring the entire Detect module expands to all submodules in vendored Torch-Pruning 1.6.0 and can suppress all safe groups. Ignore only the fixed-width terminal output convolutions and DFL convolution.
- `round_to=8` makes nominal P10 substantially stronger in small layers: observed parameter reduction is 24%, not 10%.
- Both tested P10 variants lose nearly all validation accuracy before fine-tuning. Do not present them as successful compressed detectors yet.
- Lower MACs did not improve T4 latency for P10. Always measure deployment latency.
- Pruned FP32 checkpoint size can exceed the stripped FP16 baseline checkpoint even with fewer parameters.
- Fine-tuning must preserve the in-memory changed architecture; ordinary Ultralytics rebuilding from model YAML is unsafe.

## Next decision gate

Build the final comparison table and select the operating point on validation.
P10 has the highest pruned accuracy, P20 is the current balanced compression
candidate, and P30 is the strongest compression candidate. Evaluate only the
selected model on test, then publish that final checkpoint and reports.
