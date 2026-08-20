# DiariZen-style gated pruning for YOLOv8n

Status: core implementation complete; no gated training result is claimed here.

## Source reviewed

DiariZen was inspected at commit `844f5555b0a98acd0931511fc641a8c5b8ba92c7`.
The executable pruning path is:

1. `recipes/diar_ssl_pruning/run_stage.sh`
2. `run_distill_prune.py`
3. `diarizen/models/pruning/model_distill_prune.py`
4. `diarizen/models/module/wav2vec2/hardconcrete.py`
5. `recipes/diar_ssl_pruning/trainer_distill_prune.py`
6. `recipes/diar_ssl_pruning/apply_pruning.py`

The implementation is associated with arXiv:2505.24111 and arXiv:2506.18623.

## What DiariZen actually does

DiariZen does not discover dependency groups. It defines pruning units directly
from WavLM structure:

- one Hard-Concrete variable per CNN output channel;
- one variable per attention head, optionally one for the whole attention layer;
- one variable per FFN intermediate feature, optionally one for the whole FFN;
- default recipe: `conv,head,interm` (not whole-layer pruning).

During training, a sampled gate is multiplied into the corresponding activation.
`HardConcrete.l0_norm()` returns the differentiable expected number of retained
units:

`sum(sigmoid(log_alpha + bias))`, where
`bias = -temperature * log(-limit_l / limit_r)`.

The default limits are `[-0.1, 1.1]`, temperature is `2/3`, CNN/head gates start
near a 1% drop rate, and FFN intermediate gates start near 50%.

The student and frozen teacher start from the same fine-tuned checkpoint. The
teacher and student hidden states at layers `0,4,8,12` are compared. The base
recipe uses:

`L_KD = L1(student, teacher) - mean(cosine(student, teacher))`

(`l2_weight=0`, `l1_weight=1`, `cos_weight=1`, raw cosine).

The pruning term is an augmented-Lagrangian constraint, not a standalone sum of
gate probabilities:

`d = expected_sparsity - scheduled_target_sparsity`

`L_sparse = lambda1 * d + lambda2 * d^2`

`L = L_KD + L_sparse`

Model weights use LR `2e-4`; `log_alpha` uses LR `2e-2`; `lambda1/lambda2` use
negative LR `-2e-2`, implementing descent for the model/gates and ascent for the
constraint multipliers. Expected sparsity is computed from differentiable
expected parameter count, not from thresholded masks.

The target ramps linearly from zero to the configured target over five sparsity
warm-up epochs (after optional pre-training epochs), then stays constant. The
S80 recipe trains 30 epochs with target parameter sparsity 0.8.

Physical pruning is a separate post-training step. DiariZen averages five
post-peak checkpoints with the lowest validation distillation loss, switches to
evaluation mode, compiles deterministic masks, deletes zeroed channels/heads/FFN
features, removes gate modules, saves a new architecture, then performs further
distillation and downstream diarization training.

In evaluation mode, a gate rounds its expected zero count, ranks
`sigmoid(log_alpha / temperature * 0.8)`, and sets exactly the lowest-ranked
entries to zero. This is the point where learned masking becomes real structural
pruning.

## Mapping to YOLOv8n

| DiariZen unit | YOLO unit | Gate placement | Physical operation |
|---|---|---|---|
| WavLM CNN channel | Conv/BN/SiLU output channel | after the Conv block activation | DepGraph group prune from the Conv output root |
| MHA head | C2f/Bottleneck hidden channel | on eligible `cv0`, `cv1`, Bottleneck and `cv2` channel roots | DepGraph propagates through concat/residual dependencies |
| FFN intermediate | Detect intermediate channel | on `cv2.*.0/.1` and `cv3.*.0/.1` outputs | DepGraph propagates into the fixed-width terminal conv inputs |
| fixed model output | Detect regression/class outputs and DFL | no gate | always protected |

The eight stock C2f modules must first be converted to the existing
`PrunableC2f` form. This is already required by the current project because
`split/chunk` otherwise collapses DepGraph into an unusable dependency group.

### Gate ownership

Create one gate vector for each eligible DepGraph output-pruning root, not one
unrelated gate for every module touched by the dependency. The root gate is
applied to its output activation during learning. Its DepGraph group is retained
as the authoritative physical-pruning plan for the same channel indices.

Reject a root when:

- it is one of the six fixed Detect output convolutions or DFL;
- DepGraph cannot validate its group;
- pruning it can remove all channels;
- its channel-index mapping is ambiguous after C2f conversion;
- it violates a configured minimum width.

Detect intermediate convolutions remain eligible; only their terminal outputs
are protected. This preserves `[1, 10, 8400]` while allowing the head's internal
width to shrink.

## Proposed training objective

Use the frozen validation-best YOLO baseline as teacher and an identical,
unpruned, gated copy as student:

`L_total = L_detect + w_kd * L_kd + L_sparse`

- `L_detect`: the unchanged Ultralytics box, classification and DFL loss.
- `L_kd`: reuse Ultralytics 8.4.115 native score-weighted feature L2
  distillation at the three feature tensors entering Detect. Gates do not change
  tensor shapes during this phase, so no projector beyond the native path is
  needed.
- `L_sparse`: DiariZen's augmented-Lagrangian constraint.

For YOLO, expected sparsity must be computed over unique effective parameters,
not by naively summing DepGraph group costs because dependency groups overlap.
Build a channel ownership map once from DepGraph and compute differentiable
expected parameters per module:

- Conv: kernel/group factor times expected active input channels times expected
  active output channels, plus bias when present;
- BatchNorm: two trainable parameters per expected active channel;
- protected Detect outputs: output width is constant, input width remains gated;
- other parameters: constant.

Start with parameter sparsity, matching the released DiariZen S80 recipe. Do not
mix MAC or latency constraints into the first experiment; measure them after
physical pruning. A hardware-aligned follow-up may add `round_to=8` only after
the faithful no-round control works.

### Cost accounting: parameter vs. MAC target

Neither DiariZen's own recipe nor the `asappresearch/flop` code it is built on
(reviewed directly, both at their current default branch) actually train
against a differentiable MAC/FLOP objective, despite the latter's package name:
both compute `expected_sparsity` from parameter counts only
(`flop/examples/wt103/train_agp_struct.py`, `diarizen/trainer_distill_prune.py`);
`get_num_macs()` in the reference model is called only for post-prune reporting
in `apply_pruning.py`, never inside a training loss. Parameter-count sparsity
is simply the tractable default both codebases ship, not evidence that a
MAC-weighted cost is unsound — the underlying L0-regularization framework
(Louizos et al. 2018; Wang et al. 2020, the FLOP paper) is agnostic to what
"expected cost" measures, as long as it is a differentiable function of each
gate's retention probability.

This project measured, and this project's own history shows why both are
worth trying: parameter and MAC reduction track closely for direct DepGraph
pruning (P10 19.80%/20.63%, P20 36.46%/36.85%, P30 51.77%/51.83% — within about
1 point), but parameter reduction alone has never translated into lower T4
latency for any pruned checkpoint so far (see `references/project-state.md`).
A gate trained to remove the cheapest *parameters* is not the same as a gate
trained to remove the cheapest *computation*, and only the latter has any
chance of the compute reduction the current experiment is actually after.

`GatedGroupRegistry` (`src/pcb_prune_yolo/pruning/gated_groups.py`) therefore
supports `cost_type="params"` (default, unchanged behavior) or
`cost_type="macs"`. The MAC variant runs one extra forward pass with the same
`example_input` already used to build DepGraph, hooking every `Conv2d` to
record its own output `(height, width)`, then reuses the existing
parameter-cost formula per module and multiplies by that spatial size — MACs
of a convolution scale with `Cout * Cin/groups * k^2 * H_out * W_out`, and the
parameter-cost helper already computes the `Cout * Cin/groups * k^2` part.
BatchNorm is treated as contributing 0 MACs (elementwise, negligible next to
its owning convolution) even though it still counts toward parameter cost.
This is implemented and unit-tested
(`tests/test_gated_pruning.py::test_mac_cost_scales_by_spatial_size_unlike_param_cost`
asserts exact hand-computed param and MAC costs on a 2-conv toy model with a
strided second layer). `configs/prune/gated_p10.yaml` (`cost_type: params`)
and `configs/prune/gated_p10_macs.yaml` (`cost_type: macs`, otherwise
identical) are the matched pair for that ablation, and both have now run to
completion on real DeepPCB data at two target sparsities (P10 and P30): see
`docs/PROJECT_MEMORY.md` for full numbers. Result: `cost_type="macs"`
produced essentially the same realized MAC reduction as `cost_type="params"`
in every run (~3.3% either way). A channel-level diff explains why —
`expected_sparsity()`'s cost weighting only changes how the aggregate
sparsity target is measured; it does not change which specific channels
`L_detect`'s gradient tolerates losing, and that gradient concentrates every
run's cuts in the same cheap-per-channel C2f `cv1` branches while leaving the
expensive stem (`model.0`, `model.1`) untouched regardless of cost_type. A
MAC-aware cost is therefore necessary but not sufficient; redirecting
*which* channels get cut likely needs a per-group prune-fraction cap or a
per-layer cost multiplier strong enough to outweigh `L_detect`, neither of
which is implemented yet.

## Schedule and state machine

1. **Gate warm start (optional, 0-1 epoch):** target sparsity 0; confirm the gated
   student matches the baseline closely and all gate gradients are finite.
2. **Sparsity ramp (default 5 epochs):** linearly increase target parameter
   sparsity from 0 to the chosen target.
3. **Target hold:** keep the target fixed until expected sparsity and validation
   mAP50-95 stabilize. Save every epoch.
4. **Checkpoint selection:** use validation only. Select the highest mAP50-95
   among epochs after the target was reached and held; distillation loss is a
   diagnostic, not the project selection metric.
5. **Compile gates:** switch to eval, rank deterministic gate scores globally,
   and choose channel indices up to the requested parameter budget while
   enforcing minimum widths. Record expected and realized sparsity separately.
6. **Physical prune:** call each saved DepGraph group's prune operation for its
   selected indices. Do not call the current magnitude `BasePruner.step()`;
   magnitude must not overwrite the learned gate ranking.
7. **Verify immediately:** class names/count, raw output `[1,10,8400]`, parameter
   and MAC counts, complete-model save, new-process load, and CUDA inference.
8. **Final fine-tune:** remove all gates and fine-tune the physically changed
   model with detection loss plus optional frozen-teacher KD. Select on
   validation; do not touch test.
9. **Deployment gate:** rebuild baseline and candidate TensorRT engines in the
   same session and repeat latency measurements. Lower parameters/MACs alone are
   not evidence of speedup.

## Minimal implementation shape

Reuse the current entry points and add only these project-owned pieces:

- `src/pcb_prune_yolo/pruning/hard_concrete.py`: the small gate module;
- `src/pcb_prune_yolo/pruning/gated_groups.py`: DepGraph group registry, forward
  hooks, expected-parameter accounting, and learned-index materialization;
- one gated trainer mixin beside the existing sparse trainer;
- one config and one script that delegates to the existing training wrapper;
- focused tests for deterministic masks, target schedule, protected Detect
  outputs, learned-index pruning, and new-process reload.

Do not modify Ultralytics or vendored Torch-Pruning. Do not create a second C2f
converter, checkpoint format, evaluator, benchmarker, or fine-tune pipeline.

Implemented files:

- `src/pcb_prune_yolo/pruning/hard_concrete.py`
- `src/pcb_prune_yolo/pruning/gated_groups.py`
- `scripts/train_gated_pruning.py`
- `scripts/materialize_gated_pruning.py`
- `configs/prune/gated_p10.yaml`

Run the required smoke first with
`python scripts/train_gated_pruning.py --epochs 1 --name p10_smoke`. Start the
full gated run with `python scripts/train_gated_pruning.py`. Convert a
learned checkpoint with:

```bash
python scripts/materialize_gated_pruning.py \
  --checkpoint outputs/gated_pruning/p10/weights/best.pt \
  --output outputs/gated_pruning/p10_physical/pruned.pt
```

The baseline checkpoint and processed dataset are absent from the current local
workspace, so only unit, synthetic YOLOv8n, DepGraph materialization and
serialization/deepcopy checks have run. A dataset-backed one-epoch smoke remains
the next gate.

## First controlled experiment

Use a low target first (P10 by realized parameter reduction), one GPU, seed 42:

1. one-batch forward/backward gate test;
2. one-epoch smoke with target ramp compressed into the smoke;
3. compile and physically prune a temporary checkpoint;
4. new-process CUDA inference and validation-only evaluation;
5. only after all invariants pass, authorize the full gate-training run.

Acceptance requires nonzero finite gate gradients, target/expected/realized
sparsity logs, no zero-width group, unchanged detection loss path, valid
`[1,10,8400]`, and a reloadable structurally pruned checkpoint.
