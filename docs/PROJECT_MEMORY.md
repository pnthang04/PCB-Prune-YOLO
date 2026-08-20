# PCB-Prune-YOLO project memory

DiariZen-style learnable gated pruning is implemented but not yet trained on
DeepPCB. It uses the official paper code's Hard-Concrete sampling, expected L0,
linear sparsity ramp and augmented-Lagrangian multipliers, while the existing
YOLO DepGraph owns physical dependency pruning. Native Ultralytics feature KD
and the unchanged detection loss remain active. Unit tests, synthetic YOLOv8n
forward/deepcopy, and one-channel physical DepGraph pruning passed; the local
baseline checkpoint is currently absent, so the dataset-backed one-epoch smoke
has not run. See `docs/DIARIZEN_GATED_PRUNING_DESIGN.md`.

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

The TensorRT gate failed for architecture speed: forward-only M05 is 1.838 ms
versus 1.780 ms baseline, and `trtexec` per-layer is also slightly slower. M05
adds seven engine layers, six reformat nodes, and doubles pointwise layer count
despite slightly lower convolution time. E2E including NMS is faster, but the
large accuracy/recall loss makes it content-dependent. Do not continue pruning
or final fine-tuning until C2f fusion and the full-engine cost model are fixed.

The independent direct-P30 deployment optimization branch profiled baseline
and P30 FP16 without touching HALP or test. P30 has lower `trtexec` GPU compute
but 3,571 kernel launches versus baseline's 2,616. A reusable context/stream,
pinned-buffer and async-H2D runtime is implemented. CUDA Graph improves P30
forward mean by 5.32% but changes E2E by -0.16%, so it is not enabled by
default. P30 INT8 PTQ used 500 deterministic training images only and produced
35/61 INT8 convolutions; validation mAP50-95 fell from 0.75610 to 0.61119 and
latency worsened. Do not deploy this PTQ engine. The next controlled gate is
P30 QAT with explicit Q/DQ and no distillation initially; it has not run.

P30 explicit-Q/DQ QAT smoke is now complete using NVIDIA ModelOpt 0.45.0. Three
epochs restored validation mAP50-95 from PTQ's 0.61119 to 0.72462, but the
strongly-typed TensorRT engine is still 4.20% slower forward and 8.68% slower
E2E than P30 FP16. ONNX retains 133 Q/DQ pairs. Full-FP32 convolutions fall
from 13 to 7, yet total conv/reformat/kernel-launch counts rise and INT8-output
coverage is only 38/68 versus PTQ's 35/61. Decision: `FIX_GRAPH_FIRST`; do not
run full QAT or distillation until Q/DQ placement and Conv-BN/SiLU fusion are
fixed and full-engine latency passes.

The Hard-Concrete gated pruning registry now supports targeting expected MAC
reduction, not just expected parameter reduction. Reviewing DiariZen's own
recipe and the `asappresearch/flop` code it descends from confirmed both
released trainers optimize parameter sparsity only, despite the "flop"
package name — MACs are computed only for post-prune reporting, never inside
either codebase's training loss. Since this project's own history already
shows parameter reduction alone never produced a T4 latency win (see the
TensorRT results above), `GatedGroupRegistry` gained `cost_type="macs"`: one
extra forward hook pass records each convolution's own output spatial size,
then the existing parameter-cost formula is multiplied by that size (BatchNorm
counts 0 MACs, being elementwise). This is unit-tested with an exact
hand-computed example (45 params vs. 7936 MACs per channel on a small
strided toy model) and the full non-Ultralytics-dependent suite passes
(27/27), but it has not run against the real YOLOv8n graph or DeepPCB data.
Matched configs `configs/prune/gated_p10.yaml` (`cost_type: params`) and
`configs/prune/gated_p10_macs.yaml` (`cost_type: macs`) are ready for that
comparison once the environment is restored. See
`docs/DIARIZEN_GATED_PRUNING_DESIGN.md` ("Cost accounting: parameter vs. MAC
target").

A companion plan, not yet implemented, proposes multi-depth backbone feature
distillation (DiariZen-style `L1 + (1 - cosine)` at `model.2/4/6/9` instead of
only the existing Detect-input native KD) for the same gated trainer; see
`docs/GATED_KD_MULTI_DEPTH_PLAN.md`.

Three real bugs surfaced only once gated pruning was actually run end-to-end
on the server (GPU + real DeepPCB data), all now fixed and confirmed with a
fresh 1-epoch smoke for both `cost_type` values (train → materialize →
new-process load/inference all passed, 4 channels physically pruned each):

1. `expected_sparsity()` divided cost-weighted numerator by total *parameter*
   count regardless of `cost_type`, so a `cost_type="macs"` run's sparsity
   ratio mixed MAC and parameter units. Observed impact: a live run logged
   `expected_sparsity=0.664` against `target=0.10` (6.6x apparent overshoot).
   Fixed: `GatedGroupRegistry.original_cost` now uses the matching total
   (total params, or total model MACs via `tp.utils.count_ops_and_params`).
2. Ultralytics' end-of-training `strip_optimizer` step replaces the saved
   checkpoint's `model` with the EMA-smoothed shadow copy. `log_alpha` drives
   a hard top-k threshold at materialize time, not a smooth inference weight,
   so the EMA copy left every gate clustered within its init range even after
   the raw model's gates had genuinely separated — one finished 100-epoch run
   materialized **zero** pruned channels despite logging 66% expected
   sparsity throughout training. Fixed: `GatedDetectionTrainerMixin.final_eval`
   now no-ops to skip the strip step, the same pattern
   `QATDetectionTrainerMixin` already used for an analogous reason.
3. `YOLODepGraphPruner.save_pruned_model()` stored `train_args` as whatever
   `getattr(saved_model, "args", {})` returned; after gated+distill training
   that is an Ultralytics `IterableSimpleNamespace`, not a dict, and
   `YOLO(checkpoint)` reloading a materialized model crashed with
   `TypeError: 'IterableSimpleNamespace' object is not a mapping`. Fixed:
   convert with `vars(train_args)` when it isn't already a dict.

All three fixes and their smoke verification landed together; the prior
100-epoch runs (both `cost_type` values) predate the fixes and were
discarded (`outputs/gated_pruning/_stale_discard/`) rather than trusted.

**First real gated-training results (P10 and P30, both `cost_type` values),
materialized and evaluated on validation, before any post-materialize
fine-tune.** All four checkpoints passed materialize/new-process-load/
inference; latency was re-benchmarked against a same-session baseline
(8.338 ms) for a fair comparison:

| Model | Params | MACs | val mAP50-95 | Latency |
|---|---:|---:|---:|---:|
| Baseline (same session) | 3,012,018 | 4.0733G | — | 8.338 ms |
| Gated P10, `cost_type=params` | 2,679,454 (−11.03%) | 3.9402G (−3.27%) | 0.775 | 9.806 ms |
| Gated P10, `cost_type=macs` | 2,681,344 (−10.97%) | 3.9410G (−3.25%) | 0.774 | 9.669 ms |
| Gated P30, `cost_type=params` | 2,677,296 (−11.10%) | 3.9394G (−3.29%) | 0.778 | 10.590 ms |
| Gated P30, `cost_type=macs` | 2,680,414 (−10.99%) | 3.9406G (−3.26%) | 0.775 | 10.074 ms |

The standout positive result: unlike direct/sparse pruning (validation near
zero before fine-tuning), every gated checkpoint above holds baseline-level
validation accuracy **immediately after physical pruning, with zero
post-materialize fine-tuning epochs** — the jointly-trained gate has already
adapted the network to its own pruning noise. The standout negative result:
MACs reduction is small (~3.3%) and essentially identical between P10 and
P30 targets and between `params`/`macs` cost types; latency is 16-27% worse
than baseline in every case.

A channel-level diff (`p30_v3_physical/pruned.pt` vs baseline) explains why:
every changed convolution is either a C2f `cv1` branch cut by exactly 50%
(`model.2/4/6/8/12/15/18/21.cv1`) or a Detect P5 branch (`model.22.cv2.2`/
`cv3.2`, 55-87% cut); `model.7.conv` lost one channel (rounding noise). The
stem (`model.0`, `model.1`) has a gate — it is not excluded, both are in the
59-root gated set — but was never pruned at all in any of the four runs, even
though it is exactly where prior HALP LUT profiling found the strongest
operator-level latency potential
(`docs/HALP_ADAPTATION_PLAN.md`: "model.0, model.1, model.3, model.4.cv2,
model.9.cv1"). The mechanism: which channels are safe to remove is decided by
the detection-loss gradient, not by how the sparsity loss weights cost;
`cv1`'s channels sit deep in the network at small spatial resolution (cheap
per-channel MACs, many of them), while the stem sits at full/near-full
resolution (expensive per-channel MACs, few of them) and evidently costs
detection accuracy more to touch. Reweighting `L_sparse` by MAC cost does not
change which channels `L_detect` tolerates losing, so `cost_type="macs"`
produced almost the same realized MAC reduction as `cost_type="params"`.

**A "just train longer" fix did not solve this.** Two more bugs/gaps were
found and fixed in this cycle, on top of the three below:

- Ultralytics' `plot_results()` classifies any results.csv column containing
  the substring "loss" as a loss subplot; the added `gated/sparsity_loss`
  column made that count odd (9 instead of 8), so `len(columns)//2` subplots
  were built for 13 total columns and plotting crashed with `index 12 is out
  of bounds for axis 0 with size 12`. Training itself was unaffected (the
  exception is caught), only `results.png` failed to render. Fixed by
  renaming the metric key to `gated/sparsity_penalty`.
- Training always stopped at the same epoch (patience=20 from the validation
  fitness peak, epoch 39 in every run) regardless of `target_sparsity`,
  because fitness plateaus long before the sparsity constraint has had time
  to converge toward a higher target. Added `_MinHoldEpochsStopper`
  (`GatedDetectionTrainerMixin._setup_train`, config key
  `gated.min_hold_epochs`) to forbid early stopping before a configured
  epoch while still feeding the real `EarlyStopping` every epoch so its
  bookkeeping stays correct. Unit-tested (3 tests) and confirmed to work
  mechanically: a P30 rerun with `sparsity_warmup_epochs=10`, `reg_lr=0.05`,
  `min_hold_epochs=60` did run to epoch 60 instead of 59. It did not fix the
  underlying problem: `weights/best.pt` is still selected by validation
  fitness, which peaked at epoch 39 in this run too, so the checkpoint
  actually used for materialize was unaffected by the extra epochs. Reading
  the *last* epoch (60) instead showed only a modest sparsity gain (params
  0.174→0.195, macs 0.046→0.052 expected_sparsity) at a real ~2-point mAP50-95
  cost, with `lambda1`/`lambda2` climbing to roughly ±30 — the augmented
  Lagrangian pushing hard without reaching the target, not a training-time
  shortfall. This confirms the channel-level explanation above: the fix is a
  structural/loss-weighting change (a per-group max-prune-fraction cap, or a
  per-layer MAC-cost multiplier strong enough to outweigh `L_detect`'s
  resistance), not a longer schedule. Not yet decided or implemented.

Last verified: 2026-08-20 (gated-pruning P10/P30 results, min_hold_epochs, plotting fix)

The independent P40-HW latency gate created three FP16 candidates from the
same baseline without touching HALP/QAT/INT8/test. A8/A16/BLOCK have
0.903M/0.800M/1.132M parameters. Under a fresh matched 50/200 graph-off run,
their forward means are 1.3540/1.4903/1.5590 ms versus 1.7575 ms for P30.
A8 is selected provisionally (1.298x P30 speedup), but its validation metrics
are all zero before fine-tuning. See `docs/P40_HW_LATENCY_GATE.md`.

P40-A8's pre-FT checkpoint is gitignored and did not survive the session
break; it was rebuilt deterministically from baseline via the same
`configs/prune/p40_hw_a8.yaml` and matched the original report exactly.
Ultralytics 8.4.115 ships native knowledge distillation (`distill_model`/`dis`,
score-weighted feature L2 loss), which supplied the previously missing
teacher/loss/weight specification without a custom design. A standard-FT
branch and a KD branch (teacher = baseline) were fine-tuned from the identical
checkpoint with every other hyperparameter matched (AdamW, lr0 0.001, lrf
0.01, momentum 0.9, weight decay 0.0005, batch 64, patience 10, seed 42, 50
epochs, no early stop). KD reached validation mAP50-95 0.660 versus 0.634 for
standard FT (+2.7 points), and also won precision, recall and mAP50; both
passed save/new-process-load/inference and batch-1 benchmark with identical
architecture (903,466 params, 1.1212G MACs). Both remain well below P30
direct's 0.75030 at a much more aggressive compression point (-70.00% params,
-72.47% MACs vs P30's -51.77%/-51.83%), so P40-A8 KD is a faster/smaller but
less accurate alternative to P30, not a replacement. Test split was not used.

Both P40-A8 checkpoints (standard FT and KD) were published publicly on
Hugging Face (`thangkt/PCB-Prune-YOLO-P40-A8-Direct`,
`thangkt/PCB-Prune-YOLO-P40-A8-KD`) with model card, args, validation, and
benchmark; anonymous download and `private=false` were verified for both.

The 50-epoch runs above were still improving at the final epoch (KD's best
epoch was epoch 50 itself), so both were re-run with `epochs=100`,
`patience=20`, `cos_lr=True`, everything else unchanged. Both now plateau
under cosine annealing (converged, not epoch-budget-limited). Results:
standard FT mAP50-95 0.701 (+6.7 points over 50 epochs), KD 0.712 (+5.2
points); KD's lead over standard FT narrowed from +2.7 to +1.1 points with
enough training. A `dis` sweep (3.0/10.0 vs the default 6.0) confirmed 6.0
is already near-optimal (0.711/0.712/0.709 — within noise). Versus P30
(0.75030 mAP50-95, -51.77%/-51.83% params/MACs), the 100-epoch KD checkpoint
is now only 3.83 points behind (down from 9.03 at 50 epochs) while keeping
much stronger compression (-70.00%/-72.47%). TensorRT engines rebuilt for the
100-epoch checkpoints initially measured 1.422/1.497 ms vs a same-session
baseline of 1.716 ms (single sample each). A follow-up repeated-sampling
check (4 measurements each, no rebuild) found ~5% run-to-run latency noise
even on an unmodified engine: KD averaged 1.494 ms (σ 0.052), baseline
averaged 1.702 ms (σ 0.011). The reliable KD speedup is therefore **~1.14x
(≈14%)**, not the ~1.21x a single sample suggested; do not quote a
single-measurement ratio as the deployment speedup again — always average
several repeated measurements on an unmodified engine first. The 100-epoch
checkpoints supersede the 50-epoch ones and were re-published to the same two
Hugging Face repositories. Test split was not used.

TensorRT FP16 engines were subsequently rebuilt for both fine-tuned
checkpoints, plus a fresh baseline engine in the same session (absolute
PyTorch/TensorRT latency numbers can drift ~10-20% across different cloud T4
instances even with identical declared specs, so same-session rebuilds are
required for a trustworthy relative comparison; do not mix historical and
new-session absolute numbers). Result: baseline 1.716 ms, P40-A8 standard FT
1.423 ms (1.206x), P40-A8 KD 1.417 ms (1.211x). Both fine-tuned engines passed
new-process load/inference. This confirms fine-tuning preserves the
architecture-level TensorRT speedup measured pre-FT. See
`docs/P40_HW_LATENCY_GATE.md` and `outputs/tensorrt_recheck/`.

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
