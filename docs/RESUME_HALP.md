# Resume HALP work from a clean server

The authoritative branch is `server/depgraph-p10`. HALP Stage 1 is complete at
commit `c54e543`; use the latest commit on that branch if it has advanced.

## 1. Clone and install

```bash
git clone --branch server/depgraph-p10 \
  https://github.com/pnthang04/PCB-Prune-YOLO.git
cd PCB-Prune-YOLO

# Install the CUDA-compatible PyTorch build for the new server first.
pip install -r requirements-tensorrt.txt
pip install -e .
```

The measured environment was Python 3.12.12, PyTorch 2.10.0+cu128, CUDA 12.8,
TensorRT 10.16.1.11, Ultralytics 8.4.115, and Tesla T4. Existing TensorRT
engines and LUT measurements should only be compared directly on the same stack.

## 2. Restore the baseline checkpoint

```bash
mkdir -p outputs/train/baseline/weights
curl -L \
  -o outputs/train/baseline/weights/best.pt \
  https://huggingface.co/thangkt/PCB-Prune-YOLO-Baseline/resolve/main/best.pt
```

Other public checkpoints and TensorRT engines are linked from `README.md`.
Stage 2 starts from the unpruned baseline above, not P10/P20/P30.

## 3. Restore the official research reference if needed

The official HALP checkout is deliberately not committed because its NVIDIA
license is limited to non-commercial research/evaluation.

```bash
git clone https://github.com/NVlabs/HALP.git references/HALP
git -C references/HALP checkout dfee297d55d1638b968359e7ffff878be846ec02
git -C references/HALP rev-parse HEAD
```

Project-native code does not import this checkout. It is needed only to audit
paper/code behavior. See `references/HALP_PROVENANCE.md`.

## 4. Verify persisted Stage 1 state

The measured LUT is committed to Git and does not need to be regenerated:

```bash
python - <<'PY'
from pathlib import Path
from pcb_prune_yolo.halp.lut import load_lut, validate_lut

payload = load_lut(Path("outputs/halp/lut/t4_fp16_backbone.json"))
validate_lut(payload)
print(len(payload["profiled_layers"]), len(payload["records"]))
PY

python -m pytest -q
```

Expected LUT counts are 27 profiled layer names and 598 successful records.

## 5. Context and next implementation

Read in this order:

1. `.codex/skills/pcb-prune-yolo/SKILL.md`
2. `.codex/skills/pcb-prune-yolo/references/project-state.md`
3. `.codex/skills/pcb-prune-yolo/references/workflows.md`
4. `.codex/skills/pcb-prune-yolo/references/roadmap.md`
5. `docs/PROJECT_MEMORY.md`
6. `docs/HALP_ADAPTATION_PLAN.md`

Next work is HALP Stage 2 only:

```text
YOLO detection-loss Taylor saliency
→ DepGraph dependency groups
→ exact 2D LUT costs / targeted LUT refinement
→ latency-aware channel grouping
→ augmented-knapsack dry-run
```

Do not prune, fine-tune, run the test split, or claim a complete HALP model
until the Stage 2 dry-run invariants pass. The suggested first eventual latency
target is 5%, selected using validation only.
