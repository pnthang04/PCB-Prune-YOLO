# HALP Stage 3 TensorRT gate

## Classification of decisions

**PAPER.** Section 3.4 defines iterative pruning from a trained network. Every
minibatch performs normal weight-gradient training and computes Taylor
importance. Importance is averaged over multiple minibatches. At each of `k`
exponentially decreasing latency milestones, remaining widths and latency
contributions are recomputed, neurons are regrouped, and augmented knapsack is
solved. Final recovery fine-tuning happens only after the last milestone.

**OFFICIAL CODE.** `train/training.py` calls `loss.backward()`,
`pruner.update_metric(global_step)`, and `optimizer.step()` on the normal
training path. `pruner.prune_step()` is called after the batch; every
`prune_interval` it averages accumulated importance, recalculates adaptive
group latency, applies the next target, and resets importance. Between pruning
events `mask_weights()` maintains the current sparse structure. The reviewed
ResNet config uses interval 40 and 30 milestones. These constants are evidence,
not defaults copied to YOLO.

**ADAPTATION.** YOLO uses physical DepGraph channel removal instead of official
zero masks, explicit pruning-safe C2f branches, a static TensorRT FP16 batch-1
T4 LUT, and protected Detect/DFL outputs. The first 5% target is a safety gate,
not a claim that the official 30-step schedule has been reproduced.

## Fair TensorRT result

Both engines were built on Tesla T4 with TensorRT 10.16.1.11, CUDA 12.8,
Ultralytics 8.4.115, FP16, static `[1,3,640,640]`, batch 1, and workspace 4 GiB.
Each benchmark used 50 warm-ups and 200 synchronized iterations; build/load
time was excluded and the execution context and buffers were reused.

| Model | Forward mean/median/p95 (ms) | Forward FPS | E2E mean/median/p95 (ms) | E2E FPS | Engine MiB | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.780 / 1.580 / 3.090 | 561.89 | 4.959 / 4.938 / 5.395 | 201.66 | 7.479 | 0.78572 |
| HALP M05 | 1.838 / 1.605 / 3.098 | 543.93 | 4.608 / 4.533 / 5.142 | 217.01 | 7.011 | 0.65601 |

Forward-only speedup is `0.968x`: M05 is 3.30% slower. The independent
`trtexec` per-layer run also shows no gain: 1.6741 ms baseline versus 1.6768 ms
M05 (`0.998x`). Therefore the LUT's 6.32% eligible-root reduction does not
predict full-engine latency.

E2E is 1.076x faster, but that measurement includes NMS and M05 loses 0.12971
mAP50-95 and 0.07235 recall. The lower candidate/detection workload can make
NMS cheaper. It is content-dependent and is not accepted as TensorRT model
acceleration while forward-only compute is slower.

## Bottleneck evidence

The per-layer profile contains 159 baseline layers and 166 M05 layers. M05 has
34 reformat nodes versus 28 and 30 pointwise nodes versus 14. Pointwise time
increases from 0.1087 to 0.2405 ms. Convolution time falls only from 1.0049 to
0.9976 ms, which is too small to offset graph overhead. Explicit C2f branches
remove stock Split operations but introduce extra activation/reformat/fusion
boundaries. Kernel-launch and small-tensor overhead dominate the modest 3.81%
MAC reduction.

All eight targeted operator LUT measurements selected a different TensorRT
tactic from their native-width record. These are isolated-operator tactics, not
proof of the tactic used inside the serialized full engine. The Ultralytics
engines were built without detailed tactic metadata, so full-engine tactic IDs
cannot be recovered honestly after serialization; `trtexec` layer timing and
fusion names are retained instead.

## Decision

Do not continue to another pruning milestone and do not perform final
fine-tuning yet. First fix the C2f structural representation/export fusion and
extend the cost/grouping objective to account for full-engine reformat,
pointwise, concat/copy, and launch overhead. Rebuild a fresh M05 candidate and
require forward-only TensorRT improvement before resuming the paper's
multi-milestone weight-update/Taylor-accumulation loop. Test was not used.
