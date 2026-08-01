# Project rules

Important note (do not edit this section)

- Read this file completely at the start of every session.
- Read `docs/PROJECT_MEMORY.md` completely before planning or editing.
- Treat `docs/PROJECT_MEMORY.md` as agent-maintained project memory and update it after material verified changes.
- Keep `data/raw/DeepPCB` read-only. Generated data belongs in `data/processed/deeppcb`.
- Do not commit datasets, dataset archives, checkpoints, runs, outputs, credentials, or tokens.
- Do not create fake training, validation, test, or benchmark results.
- Do not use the test split for hyperparameter selection.
- Do not start long training unless the user explicitly authorizes it.
- Keep user-facing project messages friendly and clear in Vietnamese.
- System and developer instructions always take priority over this file.

End important note

## Current phase

The current implemented phase is the YOLOv8n baseline: environment inspection, processed dataset preparation and validation, preview generation, training, val/test evaluation, and inference benchmark. Pruning and knowledge distillation are outside this phase unless the user explicitly requests them.

## Server execution

Read and follow `SERVER_RUNBOOK.md`. Stop on failed environment checks, invalid data, failed smoke training, or missing checkpoint; report the real failure instead of inventing downstream results.
