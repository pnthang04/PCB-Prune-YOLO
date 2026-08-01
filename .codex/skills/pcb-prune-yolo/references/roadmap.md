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

## Not completed

- Run the proposed 30-epoch group-level sparse training on the full train split.
- Prune P10 from the validation-selected sparse checkpoint and compare it with
  the preserved direct-P10 ablation.

- Fine-tune P10 for 30–50 epochs.
- Confirm P10 accuracy recovery and benchmark the best fine-tuned checkpoint.
- Run P20 pre-fine-tune validation, fine-tune, and benchmark.
- Run P30 pre-fine-tune validation, fine-tune, and benchmark.
- Decide whether `round_to=8` or no rounding is the standard policy. Current validation favors no rounding before fine-tuning, but both are extremely weak.
- Implement safe multi-GPU DDP fine-tuning for changed model objects.
- Produce the final comparison table covering baseline and P10/P20/P30 before/after fine-tuning.
- Select the best pruned model using validation mAP50-95.
- Evaluate only that selected pruned model on test.
- Publish final pruned checkpoint(s) and update README/model cards.

## Known issues and decisions

- Directly tracing stock C2f produced one unusable dependency group. Replace C2f with explicit two-branch blocks first.
- Ignoring the entire Detect module expands to all submodules in vendored Torch-Pruning 1.6.0 and can suppress all safe groups. Ignore only the fixed-width terminal output convolutions and DFL convolution.
- `round_to=8` makes nominal P10 substantially stronger in small layers: observed parameter reduction is 24%, not 10%.
- Both tested P10 variants lose nearly all validation accuracy before fine-tuning. Do not present them as successful compressed detectors yet.
- Lower MACs did not improve T4 latency for P10. Always measure deployment latency.
- Pruned FP32 checkpoint size can exceed the stripped FP16 baseline checkpoint even with fewer parameters.
- Fine-tuning must preserve the in-memory changed architecture; ordinary Ultralytics rebuilding from model YAML is unsafe.

## Next decision gate

Fine-tune the no-round P10 candidate for up to 50 epochs with `lr0=0.001`, batch 64, image size 640, seed 42, AMP, and validation-based early stopping. Continue to P20/P30 only after confirming that the P10 training path preserves the pruned dimensions and recovers meaningful validation mAP50-95.
