# DepGraph sparse-learning implementation

## Source mapping

- **PAPER:** Section 3.3, Equation (4) regularizes every coupled parameter in a
  dependency group. Equation (5) sets channel shrinkage to
  `gamma_k = 2 ** (alpha * normalized_inverse_importance)` and fixes `alpha=4`.
- **OFFICIAL CODE:** Torch-Pruning v1.6.0 `GroupNormPruner` uses
  `update_regularizer()` once per epoch and `regularize(model)` after backward and
  before the optimizer step. Its live signature is `regularize(model, alpha=16,
  bias=False)`; this `alpha` is the exponential base, so this project passes
  `2 ** 4` explicitly.
- **OFFICIAL CODE:** The v1.6.0 reproduction uses `GroupMagnitudeImportance(p=2)`,
  global pruning, and `reg=1e-4` with 30 sparse epochs for ImageNet ResNet-50.
- **ADAPTATION:** YOLOv8/DeepPCB is not evaluated in the paper. The project starts
  from the trained YOLO checkpoint, retains its detection loss and split, uses
  `lr0=0.001`, one pruning step, P10, no channel rounding, and protects the six
  fixed-width Detect output convolutions plus DFL.
- **ADAPTATION:** Torch-Pruning 1.6.0 divides by `max(importance)-min(importance)`.
  The compatibility hook skips and logs any group with a zero/non-finite range so
  a degenerate group cannot introduce a NaN gradient. No group was skipped in the
  accepted smoke run.

## Exact hook location

`SparseDetectionTrainerMixin.optimizer_step()` is called by Ultralytics after
`scaler.scale(loss).backward()`. It performs:

1. `scaler.unscale_(optimizer)`;
2. `GroupNormPruner.regularize(model, alpha=2**4)`;
3. gradient clipping;
4. `scaler.step(optimizer)` and `scaler.update()`.

This preserves the native YOLO detection loss and avoids changing Ultralytics
source. `update_regularizer()` is called at `on_train_epoch_start`.

## Commands

Smoke (limited to at most three epochs):

```bash
python scripts/train_sparse.py --smoke --epochs 2 --fraction 0.1 --batch 32
```

Full sparse training with the reviewed adaptation config:

```bash
python scripts/train_sparse.py --config configs/prune/depgraph_sparse.yaml
```

After selecting the sparse checkpoint on validation, prune P10 without rounding:

```bash
python scripts/prune_model.py \
  --checkpoint outputs/sparse/depgraph_sparse_p10/weights/best.pt \
  --pruning-ratio 0.10 --round-to 0 \
  --output outputs/pruning_sparse --no-dry-run
```

Do not use the test set to choose sparse-training or pruning settings.
