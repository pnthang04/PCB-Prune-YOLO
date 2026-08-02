# Workflows

Run all commands from `/kaggle/PCB-Prune-YOLO`.

## Integrity checks

```bash
python -m compileall -q src scripts tests
python -m pytest -q
python scripts/validate_dataset.py
python scripts/check_environment.py --require-gpus 2
```

## Baseline

Full training uses DDP on `device=0,1`:

```bash
python scripts/train_baseline.py
```

Short DDP smoke test:

```bash
python scripts/train_baseline.py --smoke --batch 128 --fraction 0.2
```

Evaluate validation while selecting configurations. Evaluate test only for the final report:

```bash
python scripts/evaluate_model.py --checkpoint MODEL.pt --split val --device 0
python scripts/evaluate_model.py --checkpoint MODEL.pt --split test --device 0
```

Benchmark batch 1:

```bash
python scripts/benchmark_model.py --model MODEL.pt --device cuda:0 --output OUTPUT_DIR
```

The benchmark includes warm-up, CUDA synchronization, params, MACs, estimated FLOPs, checkpoint size, latency distribution, FPS, peak memory, GPU, Python, PyTorch, CUDA, and Ultralytics versions.

## DepGraph

Sparse-train from the unpruned baseline before the main P10 experiment:

```bash
python scripts/train_sparse.py --config configs/prune/depgraph_sparse.yaml
```

The completed stronger-regularization experiment uses a separate immutable
config and output name:

```bash
python scripts/train_sparse.py --config configs/prune/depgraph_sparse_reg5e4.yaml
```

For a short hook check, add `--smoke --epochs 2 --fraction 0.1 --batch 32`.
The sparse trainer is currently single-GPU because its in-memory DepGraph object
is not reconstructed by Ultralytics DDP. It logs group norms and regularizer
gradient evidence to `sparse_metrics.json` and `results.csv`.

Ultralytics optimizer-stripped checkpoints may load with every parameter frozen.
The project pruning wrapper must restore `requires_grad=True` on its in-memory
model before tracing DepGraph. Do not remove this compatibility step.

Dry-run a ratio without deleting channels:

```bash
python scripts/prune_model.py \
  --checkpoint outputs/train/baseline/weights/best.pt \
  --pruning-ratio 0.10 \
  --dry-run
```

For the paper-path P10, point `--checkpoint` to the validation-selected sparse
checkpoint, use `--output outputs/pruning_sparse`, and set `--round-to 0`.

Prune with channels rounded for CUDA alignment:

```bash
python scripts/prune_model.py \
  --checkpoint outputs/train/baseline/weights/best.pt \
  --pruning-ratio 0.10 \
  --round-to 8 \
  --no-dry-run
```

Disable channel rounding with `--round-to 0`. Use validation, never test, to decide between rounding policies.

After every pruning run, require all of the following:

- parameter count decreases;
- MAC count is recorded or an explicit error/TODO is stored;
- forward output remains `[1, 4 + nc, 8400]` at 640 px;
- six class names remain correct;
- new-process load and inference succeed;
- validation JSON/CSV is generated before fine-tuning;
- batch-1 benchmark is generated.

## Fine-tune a changed architecture

Use the dedicated entry point, which injects the already-pruned model into the Ultralytics trainer instead of rebuilding the original YAML architecture:

```bash
python scripts/finetune_pruned.py \
  --model outputs/pruning_no_round/p10/pruned.pt \
  --epochs 50 \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.01 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --batch 64 \
  --device 0 \
  --name p10
```

Current safe implementation supports one GPU per fine-tune process. Do not pass `0,1`: default Ultralytics DDP subprocess reconstruction can lose the changed in-memory structure. Add and verify explicit pruned-model DDP support before using two GPUs.

After fine-tuning, evaluate the best checkpoint on validation and benchmark it. Do not evaluate its test split until the final pruning configuration has been selected.

The sparse `reg=5e-4` P10 fine-tune command was:

```bash
python scripts/finetune_pruned.py \
  --model outputs/pruning_sparse_reg5e4/p10/pruned.pt \
  --epochs 50 --optimizer AdamW --lr0 0.001 --lrf 0.01 \
  --momentum 0.9 --weight-decay 0.0005 --batch 64 --device 0 --patience 10 \
  --project outputs/finetune_sparse_reg5e4 --name p10
```

## P20 and P30

Use the same rounding policy, seed, image size, dataset split, and importance method as the selected P10 experiment:

```bash
python scripts/prune_model.py --checkpoint outputs/train/baseline/weights/best.pt --pruning-ratio 0.20 --round-to 0 --no-dry-run
python scripts/prune_model.py --checkpoint outputs/train/baseline/weights/best.pt --pruning-ratio 0.30 --round-to 0 --no-dry-run
```

Do not run the whole matrix automatically. Execute, inspect, and record one stage at a time.

## Save/load constraints

Structured pruning changes module dimensions. Never save only a state dict and expect the original YAML model to load it. Save the complete architecture, use `weights_only=False` where direct `torch.load` requires it, and test through `YOLO(checkpoint)` in a separate process.

## Publishing

- Commit lightweight experiment evidence under `outputs/`: JSON/CSV reports, used YAML configs, learning curves, confusion matrices, metric curves, and validation previews.
- Do not commit datasets, checkpoints, training-batch images, caches, tokens, or generated `egg-info` changes.
- Use short commit messages.
- Update README metrics only from committed JSON/CSV artifacts.
- Publish large checkpoints to Hugging Face, keep repositories public only when explicitly requested, and verify anonymous download.

Current public P10 repository:
`https://huggingface.co/thangkt/PCB-Prune-YOLO-P10-DepGraph`. It contains the
complete changed model object and requires this project installation for the
serialized `PrunableC2f` class.

## TensorRT FP16 export and benchmark

Build each engine separately on the target Tesla T4. The export command refuses
to overwrite an existing engine/report:

```bash
python scripts/export_tensorrt.py \
  --checkpoint outputs/train/baseline/weights/best.pt \
  --name baseline --output outputs/tensorrt_fp16 \
  --device cuda:0 --imgsz 640 --workspace 4

python scripts/verify_tensorrt_engine.py \
  --engine outputs/tensorrt_fp16/baseline/model.engine \
  --device cuda:0 --imgsz 640

python scripts/evaluate_model.py \
  --checkpoint outputs/tensorrt_fp16/baseline/model.engine \
  --data configs/data/deeppcb.yaml --split val --device 0 \
  --output outputs/tensorrt_fp16/baseline/validation_pipeline

python scripts/benchmark_model.py \
  --model outputs/tensorrt_fp16/baseline/model.engine \
  --source-model outputs/train/baseline/weights/best.pt \
  --device cuda:0 --imgsz 640 --warmup-iterations 50 \
  --benchmark-iterations 200 \
  --output outputs/tensorrt_fp16/baseline/benchmark
```

Repeat with the direct P10/P20/P30 fine-tuned checkpoints and unique names.
The benchmark measures pure engine forward and excludes preprocessing/NMS;
validation reports record those pipeline stages separately. Do not use test or
INT8 for this experiment. Build and run engines only against the same TensorRT,
CUDA, GPU, and Ultralytics versions.

Absolute PyTorch/TensorRT latency can still drift roughly 10-20% between
different cloud T4 instances/sessions even with identical declared GPU name,
memory, and pinned software versions (observed when re-measuring baseline
PyTorch latency across sessions: 8.289 ms historically vs ~7.3-7.4 ms in a
later session, stable to <1% within each session). Never compare a new
measurement against a historical absolute number from a different
session/instance; always rebuild/rebenchmark the reference model
(baseline, or whichever model you are comparing against) in the same session
before quoting a speedup ratio.

## HALP Stage 1 LUT

Read `docs/HALP_ADAPTATION_PLAN.md` before changing the sampling or cliff
definition. Generate the full static T4 FP16 backbone LUT with:

```bash
python scripts/profile_halp_lut.py \
  --checkpoint outputs/train/baseline/weights/best.pt \
  --output outputs/halp/lut --device cuda:0 --imgsz 640 \
  --warmup 50 --iterations 200 --workspace-gib 1
```

The script refuses to overwrite existing outputs. Use `--resume` after an
interruption; completed `(layer,Cin,Cout)` records are skipped. Use
`--max-signatures 1 --warmup 5 --iterations 10` only for a smoke test, never as
the reported LUT. The dense-input output sweep uses increments of eight solely
as a candidate grid adaptation. Proposed group sizes come from measured cliff
spacing and can differ by layer.

Stage 1 must not invoke knapsack, pruning, fine-tuning, or test evaluation.
Stage 2 must use exact LUT pairs or explicitly profile/refine missing pairs; it
must not silently collapse the surface to output-channel-only latency.

For a clean server recovery, follow `docs/RESUME_HALP.md`. It restores the
baseline from the public Hugging Face repository, installs pinned TensorRT via
`requirements-tensorrt.txt`, optionally checks out official HALP at the reviewed
commit, validates the committed LUT, and lists the context files in reading
order.

## HALP Stage 3 TensorRT gate

Use unique output names and the same export settings for baseline and a HALP
candidate. Forward-only uses `benchmark_model.py`; E2E without disk I/O uses
preloaded validation images:

```bash
python scripts/benchmark_tensorrt_e2e.py \
  --engine MODEL.engine --output UNIQUE_OUTPUT \
  --warmup 50 --iterations 200
```

The E2E result includes preprocessing, H2D, engine execution, NMS, and result
construction. It is content-dependent: never treat an E2E gain as architecture
speedup when forward-only does not improve or accuracy/recall changes strongly.

Ultralytics prefixes `.engine` files with a length-prefixed JSON metadata
header. `trtexec --loadEngine` needs the raw plan after that header; retain the
original engine and extract a separate diagnostic plan. Use TensorRT 10.16.1.11
`--dumpProfile --separateProfileRun --profilingVerbosity=detailed` with 200
iterations. The current serialized engines do not retain detailed tactic IDs;
do not invent them from layer timing. Isolated operator tactics remain recorded
in the HALP LUT and must be labeled as operator-level evidence only.

## Direct P30 deployment optimization

This workflow is independent of HALP and must remain validation-only. Benchmark
a metadata-wrapped Ultralytics engine with reused context, buffers and stream:

```bash
python scripts/benchmark_tensorrt_runtime.py \
  --engine outputs/tensorrt_fp16/p30_direct/model.engine \
  --output outputs/deployment_optimization/runtime/p30_direct/UNIQUE_NAME \
  --warmup 50 --iterations 200 [--cuda-graph]
```

Export a new P30 PTQ engine without touching the checkpoint or FP16 engine:

```bash
python scripts/export_tensorrt_int8.py \
  --checkpoint outputs/finetune_direct/p30_adamw_exact/weights/best.pt \
  --name UNIQUE_NAME --data configs/data/deeppcb.yaml \
  --calibration-count 500 --calibration-seed 42
```

The export script selects only `images/train`, records the exact manifest and
refuses overwrites. Always audit TensorRT layer formats and validate on `val`
before latency claims. The current implicit PTQ result fails the accuracy gate;
do not reuse its cache for a changed graph and do not run distillation before a
plain explicit-Q/DQ QAT control.

The completed QAT smoke uses `configs/qat/p30_int8_qat.yaml` and
`scripts/train_qat.py`. Its checkpoint must be restored with
`modelopt.torch.opt.restore(base_model, checkpoint)`; ordinary `YOLO(checkpoint)`
is invalid. Export with `scripts/export_qat_tensorrt.py`, which emits a strongly
typed explicit-Q/DQ engine with detailed inspector data. Current decision is
`FIX_GRAPH_FIRST`; do not extend epochs until a modified graph beats P30 FP16
forward latency under the same 50/200 runtime protocol.

## P40-HW latency-first pruning

This workflow is independent of HALP, QAT and INT8. Reproduce candidates with
the immutable configs, one at a time:

```bash
python scripts/prune_model.py --config configs/prune/p40_hw_a8.yaml
python scripts/prune_model.py --config configs/prune/p40_hw_a16.yaml
python scripts/prune_model.py --config configs/prune/p40_hw_block.yaml
```

`hardware_policy=block` protects only the ten C2f Bottleneck residual-output
convolutions as pruning roots; DepGraph still propagates safe input dependency
changes. Use `scripts/export_tensorrt.py`, `benchmark_tensorrt_runtime.py` and
`profile_tensorrt_engine.py` with unique output directories. The gate is
forward-only graph-off latency; do not use pre-FT E2E because collapsed
confidence changes NMS work. A8 passed the latency gate but has zero pre-FT
validation.

`outputs/` is gitignored, so a rebuilt server/session must reconstruct the
pruned checkpoint before fine-tuning it; this is deterministic since
group-magnitude pruning depends only on baseline weights:

```bash
mkdir -p outputs/train/baseline/weights
curl -L -o outputs/train/baseline/weights/best.pt \
  https://huggingface.co/thangkt/PCB-Prune-YOLO-Baseline/resolve/main/best.pt
curl -L -o deeppcb_processed.zip \
  https://huggingface.co/datasets/thangkt/PCB-Prune-YOLO-DeepPCB/resolve/main/deeppcb_processed.zip
mkdir -p data/processed && unzip -q deeppcb_processed.zip -d data/processed

python scripts/prune_model.py --config configs/prune/p40_hw_a8.yaml \
  --output outputs/pruning_hw/p40_a8_restore --no-dry-run
```

A8's fair FT/KD branches are complete. Ultralytics 8.4.115 ships native
knowledge distillation (`distill_model` teacher path, `dis` loss weight,
default 6.0) implemented in `ultralytics/nn/distill_model.py` as a
score-weighted feature-L2 loss at the Detect head's input layers — this
supplies the previously missing teacher/loss/weight specification without a
custom design. `scripts/finetune_pruned.py` now accepts optional
`--distill-model`/`--dis` passthrough (no effect when omitted). Run both
branches from the identical restored `pruned.pt`, with every other
hyperparameter matched, one GPU each:

```bash
python scripts/finetune_pruned.py \
  --model outputs/pruning_hw/p40_a8_restore/p40/pruned.pt \
  --epochs 50 --optimizer AdamW --lr0 0.001 --lrf 0.01 \
  --momentum 0.9 --weight-decay 0.0005 --batch 64 --device 0 --patience 10 \
  --project outputs/finetune_direct --name p40_a8_adamw_exact

python scripts/finetune_pruned.py \
  --model outputs/pruning_hw/p40_a8_restore/p40/pruned.pt \
  --distill-model outputs/train/baseline/weights/best.pt \
  --epochs 50 --optimizer AdamW --lr0 0.001 --lrf 0.01 \
  --momentum 0.9 --weight-decay 0.0005 --batch 64 --device 1 --patience 10 \
  --project outputs/finetune_direct --name p40_a8_kd_baseline_teacher
```

KD beat standard FT by 2.7 mAP50-95 percentage points at seed 42 (0.660 vs
0.634); see `references/project-state.md` for the full comparison. Fine-tuning
does not change channel counts, so the pre-FT TensorRT forward measurement
(1.3540 ms) still describes this architecture; TensorRT was not rebuilt for
this comparison. Test split was not used.
