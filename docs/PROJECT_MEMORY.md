# PCB-Prune-YOLO project memory

Last updated: 2026-08-02

## Purpose

Train and evaluate a YOLOv8n baseline for six-class PCB defect detection. The current phase excludes pruning and knowledge distillation.

## Repository structure

- `configs/`: dataset, training, benchmark, and future pruning configuration.
- `scripts/`: environment check, data preparation/validation/preview, training, evaluation, and benchmark entry points.
- `src/pcb_prune_yolo/`: project implementation.
- `src/torch_pruning/`: internal module retained for a later phase; not used by the baseline pipeline.
- `tests/`: CPU-only unit tests.
- `SERVER_RUNBOOK.md`: authoritative server execution order.

## Dataset

- Public processed archive: `https://huggingface.co/datasets/thangkt/PCB-Prune-YOLO-DeepPCB`.
- Expected processed root: `data/processed/deeppcb`.
- Only tested PCB images are used; template images are excluded.
- Official test split is preserved.
- Original train split is divided 80/20 with seed 42.
- Verified counts: train 800 images/5,485 boxes; val 200/1,388; test 500/3,140.
- Verified class mapping: `0 open`, `1 short`, `2 mousebite`, `3 spur`, `4 copper`, `5 pin-hole`.
- Content-hash validation found no duplicate images across train, val, and test.

## Baseline configuration

- Model: `yolov8n.pt` pretrained.
- Two-GPU device setting: `0,1`.
- Image size: 640.
- Global batch: 32; reduce to 16 if two T4 GPUs run out of memory.
- Maximum epochs: 100.
- Early-stopping patience: 20.
- Seed: 42; AMP and deterministic mode enabled.
- Smoke mode: 5 epochs, normally invoked with global batch 8.

## Outputs

- Smoke run: `outputs/train/smoke`.
- Baseline checkpoint: `outputs/train/baseline/weights/best.pt`.
- Evaluation: `outputs/evaluation/metrics_{val,test}.{json,csv}`.
- Benchmark: `outputs/benchmark/benchmark.{json,csv}`.
- Dataset preview: `outputs/dataset_preview`.

## Verified state

- Processed archive contains 3,000 image/label files and extracts under `deeppcb/`.
- Dataset YAML resolves train, val, and test correctly from the repository root.
- Training config parses as device `0,1`, batch 32, epochs 100, patience 20.
- Syntax check passes.
- CPU unit tests: 10 passed.
- Dataset validation passes and 20 preview images were generated.
- Training, GPU evaluation, and GPU benchmark have not yet been run or verified in this local environment.

## Environment policy

Do not replace PyTorch until the server CUDA/driver is known. Prefer the GPU-enabled PyTorch already provided by the server image. Install project dependencies from `requirements.txt`, then install the local package with `python -m pip install -e . --no-deps`.

## Next execution

Clone the repository on the two-T4 server and follow `SERVER_RUNBOOK.md` from environment check through benchmark. Report actual checkpoint path, best epoch, stopping reason, val/test metrics, latency, FPS, and peak GPU memory.
