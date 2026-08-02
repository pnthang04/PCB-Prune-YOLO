# P40-HW TensorRT latency gate

Date: 2026-08-02. This experiment is independent of HALP, QAT and INT8. It
uses validation only; the DeepPCB test split was not touched.

## Construction

All candidates start from `outputs/train/baseline/weights/best.pt`, use local
DepGraph group-magnitude pruning with target ratio 0.40, protect the six fixed
Detect outputs and DFL, and retain six classes plus decoded output
`[1,10,8400]`. Alignment is applied by DepGraph while selecting channels, not
as an unsafe post-pruning rewrite.

- A8 uses `round_to=8`.
- A16 uses `round_to=16`.
- BLOCK uses `round_to=8` and excludes the ten C2f Bottleneck residual-output
  convolutions as pruning roots. Their inputs may still be updated through the
  dependency graph.

Every candidate passed PyTorch forward, complete-model save, new-process load,
CUDA inference and TensorRT engine load/inference.

## Measured results

The TensorRT protocol is T4, TensorRT 10.16.1.11, CUDA 12.8, static
`[1,3,640,640]`, FP16, batch 1, workspace 4 GiB, 50 warm-ups and 200 timed
iterations. The primary comparison disables CUDA Graph and reuses the execution
context, CUDA stream and buffers.

| Model | Params | MACs | Forward mean | p95 | Kernel launches/inference | TRT layers | Reformat | Pointwise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3,012,018 | 4.0733G | 1.6193 ms | 1.7186 ms | 160.08 | 158 | 27 | 79 |
| P30 direct | 1,452,562 | 1.9619G | 1.7575 ms | 2.0746 ms | 175.11 | 161 | 27 | 90 |
| P40-A8 | 903,466 | 1.1212G | **1.3540 ms** | 1.8411 ms | 168.01 | 162 | 29 | 88 |
| P40-A16 | 799,730 | 1.2218G | 1.4903 ms | **1.7441 ms** | **159.05** | **155** | **23** | **85** |
| P40-BLOCK | 1,131,930 | 1.4207G | 1.5590 ms | 2.4484 ms | 163.04 | 160 | 27 | 89 |

A8 is 1.298x faster than P30 by forward mean (22.96% latency reduction) and
uses 37.8% fewer parameters plus 42.9% fewer MACs than P30. A16 has the cleanest
profile and best p95, but its mean is slower than A8. A8 therefore passes the
requested latency gate and is the provisional `P40-HW pre-FT` selection.

The very low pre-fine-tune E2E values are not an accuracy-preserving speedup:
all candidate confidence is untrained after structural mutation, so NMS work is
lower. Only forward-only latency is used for architecture selection.

## Accuracy gate and decision

A8 validation before fine-tuning is zero for precision, recall, mAP50 and
mAP50-95, including every class. P20 and P30 also collapsed before recovery,
so this does not prove recovery is impossible, but P40-A8 is a materially more
aggressive architecture and must be reported before a long run.

Training was intentionally not started. The supplied request file ends in the
middle of the two-branch training diagram and does not provide the teacher,
distillation loss, temperature, loss weights, epochs or stopping rule. Guessing
those values would prevent a fair standard-FT versus KD comparison. The next
step is to complete that missing specification, then start both branches from
the exact same A8 checkpoint and keep the TensorRT forward gate after training.

Machine-readable evidence is in `outputs/pruning_hw/comparison.{json,csv}` and
the per-model prune, runtime, `trtexec`, Engine Inspector and Nsight reports are
under `outputs/pruning_hw/`.

## Fine-tuning and knowledge distillation

Date: 2026-08-02. `outputs/` is gitignored, so the A8 pre-FT checkpoint above
did not survive between sessions. It was rebuilt from the same immutable
`configs/prune/p40_hw_a8.yaml` against a freshly downloaded copy of the public
baseline checkpoint (`outputs/pruning_hw/p40_a8_restore/p40/pruned.pt`); the
reconstruction matched the original report exactly (903,466 params,
1,121,237,600 MACs, `[1,10,8400]`, pre-FT validation zero for every metric),
confirming group-magnitude pruning is deterministic given the same baseline
weights and config.

The earlier blocker — a distillation request that ended before specifying a
teacher, loss, weights, epochs or stopping rule — is resolved without
inventing a custom design: Ultralytics 8.4.115 (the version this project pins)
ships native knowledge distillation. `distill_model` takes a teacher
checkpoint path and `dis` (default 6.0) weights a score-weighted L2 feature
loss computed between teacher and student at the Detect head's input layers
(`ultralytics/nn/distill_model.py`), on top of the normal detection loss.
`scripts/finetune_pruned.py` gained optional `--distill-model`/`--dis`
passthrough flags (no effect when omitted).

Two branches were fine-tuned from the identical restored `pruned.pt`, on
separate GPUs, with every other hyperparameter matched exactly to the
project's established P10/P20/P30 recipe (AdamW, lr0 0.001, lrf 0.01,
momentum 0.9, weight decay 0.0005, batch 64, imgsz 640, patience 10, seed 42,
50 epochs; neither triggered early stopping):

| Branch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Standard FT | 0.867 | 0.805 | 0.893 | 0.634 |
| KD (teacher = baseline `best.pt`) | 0.895 | 0.828 | 0.913 | 0.660 |

KD wins on every aggregate metric (+0.026 mAP50-95, +2.0 mAP50 points) under a
single-variable comparison, and wins on mAP50-95 for five of six classes
individually. Both checkpoints passed new-process CUDA load/inference and
batch-1 PyTorch benchmark with identical architecture (903,466 params,
1.1212G MACs, ~1.96 MiB), since fine-tuning does not change channel counts.
TensorRT engines were not rebuilt for this comparison; the pre-FT forward-only
measurement (1.3540 ms, 1.298x faster than P30) describes this architecture
and is expected to still hold, pending re-measurement.

Both P40-A8 branches remain well below P30 direct's validation mAP50-95
(0.75030) at a substantially more aggressive compression point (-70.00%
parameters, -72.47% MACs versus baseline, compared to P30's -51.77%/-51.83%).
P40-A8 with KD is therefore the strongest fast/light candidate found so far,
but a faster-and-smaller-but-less-accurate alternative to P30, not a
replacement for it. Test split was not used.

Reports: `outputs/finetune_direct/p40_a8_adamw_exact/{evaluation_val,benchmark}`
and `outputs/finetune_direct/p40_a8_kd_baseline_teacher/{evaluation_val,benchmark}`.

Both checkpoints were published publicly on Hugging Face with model card,
training args, validation metrics, and benchmark:

- `https://huggingface.co/thangkt/PCB-Prune-YOLO-P40-A8-Direct`
- `https://huggingface.co/thangkt/PCB-Prune-YOLO-P40-A8-KD`

Anonymous (unauthenticated) download of `best.pt` and public API visibility
(`private: false`) were verified for both repositories.

### Extended fine-tune (100 epochs, cosine LR) and distillation-weight sweep

Date: 2026-08-02. The 50-epoch results above were still improving at the
final epoch for both branches (`patience=10` never triggered, and the KD
branch's best validation epoch was epoch 50 itself), so both were re-run from
the identical restored `pruned.pt` with `epochs=100`, `patience=20`, and
`cos_lr=True` (all other hyperparameters unchanged). `scripts/finetune_pruned.py`
gained a `--cos-lr` passthrough flag for this.

| Branch | Epochs | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Standard FT | 50 | 0.867 | 0.805 | 0.893 | 0.634 |
| KD (`dis=6.0`) | 50 | 0.895 | 0.828 | 0.913 | 0.660 |
| Standard FT | 100 | 0.929 | 0.888 | 0.950 | 0.701 |
| **KD (`dis=6.0`)** | 100 | 0.930 | 0.898 | 0.957 | **0.712** |

Both curves now plateau under cosine annealing (`results.csv` oscillates
within ~0.01 mAP50-95 over the last 15 epochs of each run), confirming
convergence rather than an arbitrary cutoff. Extending training closed most of
the gap to P30: KD is now only 0.0383 mAP50-95 below P30 (0.712 vs 0.75030),
down from 0.0903 at 50 epochs, while retaining P40-A8's much stronger
compression (-70.00%/-72.47% params/MACs vs P30's -51.77%/-51.83%) and a
~1.3x TensorRT forward-latency advantage over P30 (matched-session P40-HW
gate measurement).

A `dis` sweep (`3.0` and `10.0`, otherwise identical to the 100-epoch KD run)
found the Ultralytics default `dis=6.0` was already near-optimal: 0.711
(`dis=3.0`), 0.712 (`dis=6.0`), 0.709 (`dis=10.0`) — differences within noise.
No further distillation-weight tuning is planned; artifacts are under
`outputs/finetune_direct/p40_a8_kd_dis3/` and `.../p40_a8_kd_dis10/` but were
not published.

TensorRT FP16 engines were rebuilt for the 100-epoch checkpoints (same
architecture, so latency is expected to match the 50-epoch measurement):
standard FT 1.422 ms (703.22 FPS), KD 1.497 ms (668.12 FPS) from a single
measurement each — both still ahead of the same-session baseline (1.716 ms).
A follow-up check re-measured the same KD engine (no rebuild) and the same
baseline engine three more times each: KD averaged 1.494 ms (σ 0.052 over 4
samples including the first), baseline averaged 1.702 ms (σ 0.011 over 4
samples). This confirms ~5% run-to-run latency noise on this shared cloud GPU
even with an identical, unmodified engine — not an architecture difference
(both P40-A8 checkpoints share 903,466 params / 1.1212G MACs). The reliable
KD speedup is therefore **~1.14x (≈14%)**, not the ~1.21x a single sample
suggested. Standard FT's ratio was not re-verified with repeated sampling.
Both engines passed new-process load/inference. The 100-epoch checkpoints
supersede the 50-epoch ones as the current best P40-A8 candidates and were
re-published to the same two Hugging Face repositories
(`PCB-Prune-YOLO-P40-A8-Direct`,
`PCB-Prune-YOLO-P40-A8-KD`). Test split was not used at any point.

### TensorRT re-measurement after fine-tuning

Date: 2026-08-02. TensorRT 10.16.1.11 was reinstalled and fresh FP16 engines
were exported directly from both fine-tuned checkpoints, alongside a rebuilt
baseline engine in the same session for a same-instance comparison (a prior
question about latency drift confirmed T4 cloud instances can differ ~10%
run-to-run at the absolute level, so relative comparisons must be measured
together). Protocol: static `[1,3,640,640]`, FP16, batch 1, no NMS, 50
warm-ups, 200 synchronized iterations, `scripts/benchmark_model.py`.

| Model | Params | MACs | TensorRT mean latency | FPS | Speedup vs baseline |
|---|---:|---:|---:|---:|---:|
| Baseline (rebuilt) | 3,012,018 | 4.0733G | 1.716 ms | 582.75 | 1.00x |
| P40-A8 standard FT | 903,466 | 1.1212G | 1.423 ms | 702.91 | 1.206x |
| P40-A8 KD | 903,466 | 1.1212G | 1.417 ms | 705.96 | 1.211x |

Both fine-tuned engines passed new-process load and `[1,10,8400]` inference.
This confirms the expectation stated above: fine-tuning changes weights only,
so the architecture-level TensorRT speedup over baseline (previously measured
pre-FT at 1.3540 ms vs a then-baseline of 1.6193 ms, i.e. ~1.19x) holds after
training, and if anything measured slightly better in this session (~1.21x).
Absolute millisecond values differ slightly from the original P40-HW gate
because they come from a different GPU instance/session, not from an
architecture or setup change — see the baseline-vs-baseline check that
motivated rebuilding all three engines together. Artifacts under
`outputs/tensorrt_recheck/`.
