
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

- Implemented the DiariZen-derived Hard-Concrete gate, differentiable expected
  L0 constraint, linear sparsity schedule, augmented-Lagrangian optimizer
  groups, DepGraph group registry, and learned-index physical pruning for YOLO.
  Added P10 config/CLIs and tests. Synthetic YOLOv8n forward, deepcopy and
  one-channel materialization pass; no DeepPCB training result is claimed.

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
- Restored the gitignored P40-A8 pre-FT checkpoint deterministically from the
  public baseline checkpoint, and ran matched 50-epoch standard-FT and KD
  fine-tunes from it. KD used Ultralytics 8.4.115's native `distill_model`/`dis`
  (teacher = baseline) rather than a custom design, resolving the earlier
  missing-specification blocker. KD reached validation mAP50-95 0.660 versus
  0.634 for standard FT (+2.7 points) with every other hyperparameter matched;
  both passed save/new-process-load/inference and batch-1 benchmark. Test was
  not used.
- Rebuilt TensorRT FP16 engines for both fine-tuned P40-A8 checkpoints plus a
  fresh same-session baseline engine (needed because absolute latency drifts
  ~10-20% across different cloud T4 instances even with identical declared
  specs). Confirmed both fine-tuned engines beat the same-session baseline by
  ~1.21x (1.716 ms to 1.417-1.423 ms), matching the pre-FT architecture-level
  expectation. Both passed new-process load/inference.
- Published both P40-A8 checkpoints (standard FT and KD) publicly on Hugging
  Face with model card, training args, validation metrics, and benchmark;
  verified anonymous download and `private=false` for both.
- Re-ran both P40-A8 branches for 100 epochs with cosine LR (the 50-epoch
  runs were still improving, not converged) and confirmed plateau/convergence.
  Standard FT reached mAP50-95 0.701 (+6.7 points) and KD 0.712 (+5.2 points).
  A `dis` sweep (3.0/10.0) confirmed the default 6.0 was already near-optimal.
  Rebuilt and verified TensorRT engines for the 100-epoch checkpoints
  (architecture unchanged, latency confirmed consistent). The gap to P30
  narrowed from 9.03 to 3.83 mAP50-95 points while keeping much stronger
  compression and a TensorRT speed advantage. Re-published the improved
  checkpoints to the same two Hugging Face repositories.

- Ran the dataset-backed gated-pruning smoke and then full training for both
  `cost_type` values at two target sparsities (P10 and P30), materialized,
  new-process load/inference verified, and evaluated all four on validation
  before any post-materialize fine-tune. Found and fixed three real bugs
  along the way (`expected_sparsity` unit mismatch for `cost_type="macs"`,
  Ultralytics' `strip_optimizer` silently replacing the saved gate state with
  an EMA-lagged copy that once caused a full run to materialize zero pruned
  channels, and a `train_args` namespace-vs-dict crash on reload), plus a
  `results.csv` plotting crash (col name collision) and a
  `min_hold_epochs` mechanism to stop patience triggering on the validation
  fitness peak before the sparsity constraint converges. Headline result: all
  four checkpoints hold baseline-level validation mAP50-95 immediately after
  physical pruning with no further fine-tuning, but MACs only drop ~3.3% and
  latency is 16-27% worse than baseline in every case, for both `cost_type`
  values and both target levels — see `docs/PROJECT_MEMORY.md` for full
  numbers and the channel-level explanation (pruning concentrates in C2f
  `cv1` branches and the Detect P5 branch; the stem, `model.0`/`model.1`, has
  a gate but was never pruned in any run despite being where HALP profiling
  found the strongest latency potential).

## Not completed

- Proposed, not implemented: add multi-depth backbone feature distillation
  (`model.2/4/6/9`, DiariZen-style `L1 + (1 - cosine)` at several depths
  instead of only the existing Detect-input native KD) to gated training, as
  an ablation against today's single-point KD. See
  `docs/GATED_KD_MULTI_DEPTH_PLAN.md` for the full design, hook-ordering
  decision, and required verification steps before any full run.

- Decide how to make gated pruning actually reduce MACs/latency instead of
  concentrating cuts in cheap-per-channel C2f `cv1` branches while leaving
  the expensive stem untouched. Raising `target_sparsity` (0.10 → 0.30) and
  forcing more training epochs before early stop (`min_hold_epochs`) both
  failed to change this pattern — the augmented-Lagrangian multipliers
  climbed to roughly ±30 without meaningfully moving the stem, so this looks
  structural (the detection-loss gradient judges stem channels unsafe to
  lose regardless of the sparsity loss's cost weighting or target), not a
  training-budget shortfall. Candidate fixes, not yet implemented or agreed:
  a per-group maximum prune fraction (forcing cuts to spread beyond the
  easiest groups once a cap is hit) or a per-layer MAC-cost multiplier large
  enough to outweigh `L_detect`'s resistance. An open question raised but not
  yet tested: whether an even higher target (e.g. 0.50) would eventually
  exhaust the cheap `cv1`/Detect-P5 capacity and force stem pruning on its
  own, since `cv1` branches were cut by exactly 50% in every run so far
  regardless of target level, hinting at a per-group ceiling rather than a
  global one.

- Decide whether P40-A8 KD is added to the main README accuracy-compression
  table as a fourth, more aggressive operating point alongside P10/P20/P30.
  After the 100-epoch re-run this is a much more favorable trade-off (only
  3.83 mAP50-95 points behind P30 for 37.8% fewer params / 42.85% fewer MACs
  / ~1.3x TensorRT speed), so worth revisiting, but the P40-HW gate remains
  documented as an independent experiment from the main P10/P20/P30 line
  unless the user asks to merge them.

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
