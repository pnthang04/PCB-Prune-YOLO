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

Dry-run a ratio without deleting channels:

```bash
python scripts/prune_model.py \
  --checkpoint outputs/train/baseline/weights/best.pt \
  --pruning-ratio 0.10 \
  --dry-run
```

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
  --lr0 0.001 \
  --batch 64 \
  --device 0 \
  --name p10
```

Current safe implementation supports one GPU per fine-tune process. Do not pass `0,1`: default Ultralytics DDP subprocess reconstruction can lose the changed in-memory structure. Add and verify explicit pruned-model DDP support before using two GPUs.

After fine-tuning, evaluate the best checkpoint on validation and benchmark it. Do not evaluate its test split until the final pruning configuration has been selected.

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
