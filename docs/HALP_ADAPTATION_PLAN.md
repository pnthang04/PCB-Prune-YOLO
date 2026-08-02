# HALP adaptation plan for PCB-Prune-YOLO

## Scope and provenance

This document separates four kinds of claims:

- **PAPER**: stated in *Structural Pruning via Latency-Saliency Knapsack*,
  NeurIPS 2022, arXiv v2 `2210.06659` (26 pages, including supplementary
  appendices).
- **OFFICIAL CODE**: observed in `NVlabs/HALP` at commit
  `dfee297d55d1638b968359e7ffff878be846ec02`.
- **ADAPTATION**: project-specific design for YOLOv8n, DepGraph, TensorRT
  10.16.1.11, CUDA 12.8, FP16, batch 1, and Tesla T4.
- **TODO**: intentionally deferred beyond the latency-LUT phase.

The attachment available in this workspace contained the task specification but
not a PDF. The official arXiv v2 PDF was therefore used; its latter pages contain
the supplementary material referenced by the task.

Sources read:

- Paper Sections 3.1–3.4, 4.2, and 4.5; supplementary Sections A, B, G, H, M,
  O, P, Q, and R: <https://arxiv.org/pdf/2210.06659>
- Official repository README, LICENSE, `main.py`, `profile.py`, every file under
  `prune/`, `configs/exp_configs/rn50_imagenet_prune.yaml`, and every file under
  `configs/prune_configs/`:
  <https://github.com/NVlabs/HALP/tree/dfee297d55d1638b968359e7ffff878be846ec02>

The official repository uses the NVIDIA Source Code License with a
non-commercial research/evaluation restriction. We use it as a design reference
and do not copy its implementation into this AGPL project.

## 1. Latency-constrained objective

**PAPER.** For layer `l`, let `p_l` be the number of output neurons/channels
kept, `I_l(p_l)` the cumulative importance of the `p_l` highest-ranked channels,
and `T_l(p_{l-1}, p_l)` the measured layer latency with `p_{l-1}` input channels
and `p_l` output channels. HALP solves

```text
maximize    sum_l I_l(p_l)
subject to  sum_l T_l(p_{l-1}, p_l) <= C
            0 <= p_l <= N_l.
```

The dependence on both input and output width is essential. For the first layer,
`p_0` is fixed (three RGB channels). Layer latency is decomposed into ordered
incremental contributions

```text
c_l^j = T_l(p_{l-1}, j) - T_l(p_{l-1}, j-1),
T_l(p_{l-1}, p_l) = sum_{j=1..p_l} c_l^j.
```

**OFFICIAL CODE.** `prune/cost.py` keys the LUT by batch size, input channels,
output channels, feature size, kernel, stride, and groups, and recomputes latency
contributions as masks change.

## 2. Taylor importance

**PAPER.** HALP estimates the first-order loss change on a BN channel:

```text
I_l^n = |g_gamma * gamma + g_beta * beta|.
```

It uses the absolute value rather than the squared form of the referenced Taylor
criterion. Scores are averaged over minibatches, sorted descending within each
layer, and cumulatively summed so keeping `p_l` always means keeping the most
important prefix.

**OFFICIAL CODE.** `prune/importance.py` implements the BN expression above,
accumulates it across pruning intervals, and falls back to `|sum(W * grad_W)|`
for groups without BN.

**ADAPTATION.** YOLOv8 `Conv` blocks have BatchNorm, but dependency groups can
span multiple producers/consumers. Stage 2 will aggregate Taylor terms over the
entire DepGraph group rather than treating aliases as independent channels.

## 3. Latency lookup table

**PAPER.** The lookup surface is `T_l(input_channels, output_channels)` measured
on the target device. Supplementary Section M varies channels by eight, warms
up, averages repeated profiles, and reports that layer-wise estimates correlate
strongly with whole-network latency while still omitting nonlinear, pooling,
cache, and cross-branch effects.

**OFFICIAL CODE.** `prune/cost.py` consumes a pre-generated pickle LUT; the
released repository does not contain the LUT generator. `profile.py` benchmarks
whole networks, not individual LUT entries.

**ADAPTATION.** Stage 1 generates JSON/CSV directly with TensorRT FP16 on T4.
Each record includes `(Cin, Cout, H, W, kernel, stride, groups, dtype)`, tactic
name/value, runtime versions, mean/median/p95, warm-up/measurement counts, and
success/failure. Build time and image I/O are excluded, and one execution
context plus fixed input/output buffers are reused for each measurement.

## 4. Latency steps and cliffs

**PAPER.** GPU convolution latency is staircase-shaped. A latency cliff is a
channel transition whose latency reduction is materially larger than the noisy
plateaus around it; the difference in channel counts between adjacent cliffs is
the latency step size.

**ADAPTATION.** For every layer/signature and fixed sampled `Cin`, Stage 1 sorts
samples by `Cout`. Adjacent reductions are compared against both an absolute
noise floor and a robust relative threshold derived from the median positive
drop. A plateau is a width reduction below the noise floor; a cliff exceeds the
threshold. The proposed layer group size is the robust median distance between
detected cliffs, capped by the layer width. If there are too few cliffs, the
result remains unresolved rather than defaulting silently to eight.

## 5. Latency-aware grouping

**PAPER.** Channels are first sorted by importance, then consecutive ranked
channels are grouped using the layer's measured latency step. Group importance
and latency contributions are summed. For cross-layer dependencies, HALP uses
the largest group size among linked layers.

**ADAPTATION.** Candidate output widths include an eight-channel sweep at the
dense input width because T4 Tensor Cores commonly expose aligned cliffs. This
is only a sampling grid, not an assumed pruning group. DepGraph groups will use
the maximum measured step of coupled layers in Stage 2, after validating that
the requested widths exist in the two-dimensional LUT.

## 6. Augmented knapsack

**PAPER.** Ordinary 0/1 knapsack is invalid because a channel's latency cost
depends on how many more-important channels in the same layer are already kept.
Algorithm 1 therefore adds a preceding constraint: ranked group `j` can be kept
only if groups `1..j-1` are kept. Dynamic programming maximizes total grouped
importance under latency capacity and backtracks to obtain the kept prefix per
layer.

**OFFICIAL CODE.** `_knapsack` in `prune/pruner.py` implements the greedy
augmented DP, including negative latency contributions and optional prevention
of whole-layer pruning. `CostCalculator.get_group_latency_contribute` supplies
the ordered costs.

**TODO.** No knapsack code is implemented in Stage 1.

## 7. Iterative pruning milestones

**PAPER.** HALP prunes every `r` minibatches over `k` steps. Each interval
collects/averages Taylor importance, updates latency contributions for current
widths, groups channels, solves the knapsack for the next latency milestone, and
masks/removes channels. Milestones decrease exponentially from dense latency to
the final budget. Fine-tuning follows pruning.

**OFFICIAL CODE.** The released ResNet configuration uses 30 pruning steps with
an interval of 40 iterations. `set_latency_prune_target` creates the exponential
schedule.

**TODO.** Iterative pruning and fine-tuning are Stage 2+ work.

## 8. Reusable concepts and code boundaries

Reusable as specifications/reference:

- Objective and ordered latency contribution from the paper.
- Taylor score behavior from `prune/importance.py`.
- Two-dimensional key semantics and dynamic contribution logic from
  `prune/cost.py`.
- Preceding-constrained DP and exponential milestones from `prune/pruner.py`.
- Layer/dependency metadata concepts from `prune/prune_config.py`.

Not reused verbatim:

- Python 3.6, PyTorch 1.4, torchvision 0.5, APEX, and DataParallel training.
- Static ResNet-specific layer JSON, masks/gates, and pickle LUT format.
- PyTorch/cuDNN whole-model timing in `profile.py`.
- Any code under the official non-commercial license.

## 9. Components rewritten for YOLOv8

**ADAPTATION.** Required project-native components are: model graph discovery,
leaf-convolution shape capture, two-dimensional TensorRT LUT profiling, tactic
metadata extraction, staircase analysis, DepGraph group-to-LUT mapping, YOLO
detection-loss Taylor accumulation, augmented knapsack over dependency groups,
physical structural pruning, serialization, and Ultralytics fine-tuning that
preserves the changed model object.

Stage 1 implements only discovery, LUT profiling, reporting, validation, and
staircase analysis.

## 10. Combining HALP with DepGraph

**ADAPTATION.** HALP decides how many latency-aware, saliency-ranked channel
groups to retain. DepGraph remains responsible for applying that choice to all
coupled producers, consumers, concatenations, BatchNorm layers, and residual
paths. The knapsack item will therefore be a DepGraph pruning group, not an
isolated convolution channel. Group cost is the change in the sum of affected
operator LUT entries when both upstream and downstream widths change.

## 11. Protected YOLOv8 layers

The following remain protected:

- All six fixed-width Detect regression/classification output convolutions.
- `model.22.dfl.conv` and DFL dimensional semantics.
- Class count, class names, and output width `4 + nc = 10`.
- The entire detection head during the initial backbone-only HALP study.
- The RGB input width of the stem.

The raw model output must remain `[1,10,8400]` at image size 640 after any later
structural pruning.

## 12. Why the paper prunes only the detection backbone

**PAPER.** Section 4.5 transfers an ImageNet-pretrained SSD backbone, prunes the
backbone using HALP, and then fine-tunes the detector on VOC. This isolates a
well-defined classification backbone with available pruning groups/LUT behavior,
preserves detector predictors whose output widths are tied to anchors/classes,
and avoids destabilizing task-specific output semantics. It also makes the
reported detection experiment a transfer/generalization test of HALP rather
than a new detector-head pruning formulation.

## 13. SSD versus YOLOv8n

- SSD uses explicit multi-scale predictor heads attached to a VGG/ResNet-style
  backbone; YOLOv8n has CSP/C2f concatenations, SPPF, a PAN/FPN-like neck, and a
  decoupled anchor-free Detect head.
- SSD predictor output widths are tied to anchors and classes; YOLOv8 regression
  outputs are tied to distribution bins and DFL, while classification outputs
  are tied to six classes.
- YOLOv8 C2f split/concat and neck concatenations produce dependency coupling
  absent from a simple sequential LUT view. DepGraph is therefore mandatory.
- The paper's fixed ResNet skip-group JSON cannot represent YOLOv8 dynamically;
  dependencies must be traced from the live model.
- TensorRT 10 on T4 selects different tactics from the paper's TensorRT 7.2 on
  Titan V/RTX 3080, so every latency table and cliff must be remeasured.

## Stage 1 measurement design

The baseline checkpoint is traced once with `[1,3,640,640]`. All leaf
`Conv2d` operators under backbone modules `model.0` through `model.9` are
included; modules `model.10+`, Detect outputs, and DFL are excluded.

To control LUT size without pretending it is dense:

1. Sweep `Cout` in steps of eight at the dense `Cin` for cliff discovery.
2. Add a sparse two-dimensional calibration grid at approximately 25%, 50%,
   75%, and 100% of both `Cin` and `Cout`, aligned to valid channel counts.
3. Fix the RGB stem input to three.
4. Deduplicate identical operator signatures for engine builds, but emit a
   record for every layer name that uses a measurement.
5. Never interpolate a missing pair silently. Stage 2 must either request an
   exact measured entry or explicitly add measurements/interpolation with error
   calibration.

This sampling captures output staircases densely at the current upstream width
and measures genuine two-dimensional dependence at representative widths. It
does not cover every possible `(Cin,Cout)` pair; that limitation and the need
for targeted refinement after DepGraph group enumeration are explicit.

## Stage 1 measured results

Stage 1 was run on the baseline checkpoint and Tesla T4 with TensorRT 10.16.1.11,
CUDA 12.8, FP16, batch 1, and static shapes. Each configuration used 50 warm-ups
and 200 synchronized measurements with reused buffers. Engine build and image
I/O are outside the latency samples.

- 27 backbone convolution names were profiled (`model.0` through `model.9`).
- 19 unique operator signatures were built after exact-signature deduplication.
- 37 neck/Detect/DFL convolution names were explicitly excluded.
- 598 layer/configuration records were emitted; 598 succeeded and zero failed.
- No recorded TensorRT tactic used an FP32 convolution fallback.
- Dense native configurations were measured twice; maximum relative median
  difference was 13.90%, below the declared 20% reproducibility bound.
- Staircase analysis found 56 cliff transitions and 98 plateau transitions.
- Thirteen layers had enough distinct cliffs to estimate a step. Proposed steps
  were not universally eight: observed values were 8, 16, 24, 32, 40, 48, and
  64 channels. Fourteen layers remain `insufficient_cliffs` and require targeted
  refinement rather than an invented group size.

At dense input width, the largest absolute latency reductions between native
`Cout` and the smallest sampled `Cout` occurred in `model.0.conv`,
`model.3.conv`, `model.1.conv`, `model.4.cv2.conv`, `model.2.cv1.conv`,
`model.2.cv2.conv`, and `model.9.cv1.conv`. This is an operator-level screening
signal, not an architecture speedup prediction; DepGraph coupling and full-engine
validation are still required.

Layers that showed essentially no benefit across the same extreme output sweep
were `model.2.m.0.cv1.conv`, `model.2.m.0.cv2.conv`, `model.8.cv1.conv`, and
`model.8.cv2.conv`. Some became slightly slower, illustrating tactic changes and
measurement noise and reinforcing that MAC reduction alone is not a latency
criterion.

Artifacts:

- `outputs/halp/lut/t4_fp16_backbone.{json,csv}`
- `outputs/halp/lut/environment.json`
- `outputs/halp/lut/latency_steps.{json,csv}`

This is only a measured latency LUT. Taylor accumulation, DepGraph-aware
latency grouping, augmented knapsack, iterative structural pruning, fine-tuning,
and test evaluation remain TODO.
