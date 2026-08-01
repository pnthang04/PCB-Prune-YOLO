---
name: pcb-prune-yolo
description: Continue, diagnose, evaluate, prune, fine-tune, benchmark, document, or publish the PCB-Prune-YOLO project. Use whenever working in this repository on DeepPCB data, the YOLOv8n baseline, DepGraph/Torch-Pruning structured pruning, experiment comparison, checkpoints, metrics, README results, Hugging Face artifacts, or the long-term project roadmap.
---

# PCB-Prune-YOLO

Treat `/kaggle/PCB-Prune-YOLO` as the repository root and run project commands there.

## Load context

1. Read [references/project-state.md](references/project-state.md) before making decisions or reporting results.
2. Read [references/workflows.md](references/workflows.md) before running training, evaluation, pruning, benchmark, save/load verification, or publishing.
3. Read [references/roadmap.md](references/roadmap.md) when planning new work, choosing the next experiment, or reporting what remains.
4. Inspect the current filesystem and Git state. Treat references as a maintained snapshot, not a substitute for verification.

## Operating rules

- Preserve the official DeepPCB test split. Select pruning settings and checkpoints only with validation metrics.
- Never invent metrics. Mark an experiment incomplete when its artifact or report is absent.
- Reuse existing scripts and modules; do not create duplicate entry points.
- Do not modify Ultralytics source. Put compatibility code under `src/pcb_prune_yolo/`.
- Use the repository's vendored `torch_pruning` and inspect its live API before assuming behavior from another release.
- Build DepGraph with autograd enabled and input `[1, 3, 640, 640]`.
- Protect fixed-width detection outputs. Verify class count, names, and decoded output shape after pruning.
- For structurally pruned models, save the complete changed architecture, then load it in a new process and run inference.
- Compare parameters, MACs, accuracy, latency, FPS, memory, and checkpoint size. Do not infer latency improvement from MAC reduction.
- Do not automatically launch long training or the complete P10/P20/P30 matrix unless explicitly requested.
- Keep secrets out of source, logs, commits, model cards, and skill files.
- Keep lightweight reports and evaluation plots trackable in Git; continue ignoring datasets, checkpoints, caches, and training-batch images.
- When committing, use short messages such as `add pruning workflow` or `update results`.

## Maintain this memory

After a material experiment or implementation change:

1. Update `references/project-state.md` with exact artifact paths and measured results.
2. Update `references/roadmap.md` by moving completed work and recording new blockers.
3. Update `references/workflows.md` if commands or compatibility constraints changed.
4. Keep `docs/PROJECT_MEMORY.md` aligned as the short human-readable repository summary.
5. Validate this skill with the skill-creator `quick_validate.py`.
