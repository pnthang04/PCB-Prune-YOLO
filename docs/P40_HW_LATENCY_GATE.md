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
