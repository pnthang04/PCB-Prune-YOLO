# Multi-depth feature distillation for Hard-Concrete gated pruning

Status: proposed, not implemented. No code has changed and no experiment has
run. This plan only extends the already-implemented design in
`docs/DIARIZEN_GATED_PRUNING_DESIGN.md`; read that file first, it is the
source of truth for the gate mechanics, sparsity constraint, and schedule that
this plan builds on top of, unchanged.

## Motivation

`train_gated` (`src/pcb_prune_yolo/training/trainer.py:111`) already wires
Ultralytics 8.4.115's native `distill_model`/`dis` KD into gated training
through the `overrides` dict passed to `GatedDetectionTrainer`. That native
path is a score-weighted feature L2 loss at exactly the three feature tensors
entering the Detect head (`ultralytics/nn/distill_model.py`) — one supervision
point, near the end of the network.

The official BUTSpeechFIT/DiariZen pruning recipe
(`recipes/diar_ssl_pruning/conf/s80_base.toml`, reviewed at commit
`844f5555b0a98acd0931511fc641a8c5b8ba92c7`, confirmed unchanged as of
2026-08-20) does the analogous thing differently: `distill_layers = "0,4,8,12"`
picks four transformer layers spread across the *entire* depth of a 12-layer
encoder, not just the last one, and computes `L1 + (1 - cosine)` between
teacher and student hidden states at each of them
(`diarizen/models/pruning/model_distill_prune.py`,
`diarizen/models/pruning/utils.DistillLoss`). The stated intent (and the
reason this project is revisiting it) is to keep the compute cost of
distillation low — comparing 4 tensors, not every layer — while still
supervising representation quality at multiple depths, so a channel gate
learns to compensate for pruning noise at each stage rather than only being
corrected once at the output.

Open question this plan tries to answer experimentally: does adding this kind
of depth-spread feature supervision to gated YOLOv8n pruning recover more
validation mAP50-95 (for a fixed target sparsity) than the current
Detect-only native KD alone? No claim is made about the answer yet.

## Why YOLOv8n makes this cheaper than DiariZen's version

DiariZen's teacher and student can have different hidden widths mid-training
once heads/FFN units are masked, so DPHuBERT-style code needs care about what
"the same tensor" means across two differently-shaped models. This project's
Hard-Concrete gate (`src/pcb_prune_yolo/pruning/hard_concrete.py`,
`ChannelGateHook`) only *multiplies* a Conv output by a soft mask — it never
changes a tensor's shape until `GatedGroupRegistry.materialize()` physically
prunes channels after training ends. During the entire gated-training phase,
student and teacher activations at any chosen layer are therefore guaranteed
identical shape, so no projector or channel-alignment layer is needed, unlike
DiariZen's own `Conv-ReLU-Conv` projector. This makes the multi-depth version
strictly simpler to implement here than in the original recipe.

## Proposed anchors

YOLOv8n backbone module names are already used elsewhere in this repo's HALP
LUT work (`outputs/halp/lut/t4_fp16_backbone.json`) and are stable:

| Anchor | Module | Role |
|---|---|---|
| early | `model.2` | first C2f block output |
| P3 | `model.4` | existing FPN input, 1/8 scale |
| P4 | `model.6` | existing FPN input, 1/16 scale |
| P5 | `model.9` | SPPF output, existing FPN input, 1/32 scale |

Three of the four (`model.4`, `model.6`, `model.9`) are tensors Ultralytics
already computes and routes into the neck — capturing them for KD adds a
forward hook, not a new forward pass, so there is no extra teacher/student
compute beyond the teacher's forward pass the native KD path already pays for.
This mirrors DiariZen's even depth-spacing (0/4/8/12 of 12) at four points
spread through backbone depth instead of one point at the very end.

## Design decision: gate order

Capture the student activation **after** the Hard-Concrete gate hook has
multiplied it (i.e., the tensor the rest of the network actually sees during
that step), not the pre-gate raw Conv output. The point of combining pruning
with distillation is to teach the model to compensate for the pruning noise
it is currently experiencing, not to distill against a hypothetical unpruned
version of itself. `ChannelGateHook` (`hard_concrete.py:63`) already runs as a
`register_forward_hook` on the gate-owning module, so registering the KD
capture hook on the same module executes after it by hook-registration order;
this ordering must be asserted in a test, not assumed.

## Minimal implementation shape

New file `src/pcb_prune_yolo/pruning/feature_distillation.py`:

- `FeatureDistillationRegistry`: given a frozen teacher `nn.Module` and the
  student `nn.Module` plus a list of module names, registers a paired forward
  hook on each named module in both models that stores the output tensor in a
  small buffer keyed by name. Raises immediately if a configured name is
  absent from `named_modules()` — do not silently skip a missing anchor.
- `feature_kd_loss(student_acts, teacher_acts, l1_weight, cos_weight)`:
  reproduces DiariZen's `DistillLoss` formula per anchor
  (`l1_weight * L1(s, t.detach()) + cos_weight * (1 - mean(cosine(s, t.detach(), dim=channel)))`),
  averaged across anchors with equal weight by default.

Extend `GatedDetectionTrainerMixin` (`training/trainer.py:360`):

- `_setup_train`: build a frozen teacher copy the same way `distill_model`
  already does (or reuse Ultralytics' existing teacher object if the native
  path is left enabled — avoid loading the baseline checkpoint twice), then
  construct a `FeatureDistillationRegistry` over the configured anchors.
- `optimizer_step`: after the existing `self._gate_loss` computation and
  before calling `super().optimizer_step()`, compute
  `self._feature_kd_loss = feature_kd_weight * feature_kd_loss(...)` from the
  buffers populated during the forward pass that already happened in this
  step, and back it through the same `self.scaler.scale(...).backward()`
  pattern already used for `_gate_loss`. This keeps AMP scaling consistent
  with the existing sparsity-loss injection point and requires no change to
  Ultralytics' own training loop.
- `save_metrics`: log `gated/feature_kd_loss` per anchor and combined, next to
  the existing `gated/*` sparsity metrics, so a smoke run's log is enough to
  confirm the loss is finite and nonzero without a separate script.

Config addition to `configs/prune/gated_p10.yaml` (new block, native
`distill_model`/`dis` left untouched so the two KD signals can be toggled
independently for a clean ablation):

```yaml
feature_kd:
  enabled: false        # default off; existing behavior is unchanged until this is set true
  layers: ["model.2", "model.4", "model.6", "model.9"]
  l1_weight: 1.0
  cos_weight: 1.0
  weight: 1.0            # scales the combined per-anchor loss before backward
```

## Verification plan, in order

Follow the project's existing gated-pruning acceptance gate
(`DIARIZEN_GATED_PRUNING_DESIGN.md`, "First controlled experiment") plus these
additions; do not skip a step on the way to a full run:

1. Unit test on a synthetic tiny module graph: teacher/student shapes match
   at every configured anchor, loss is finite, gradient reaches only student
   parameters (teacher stays `requires_grad=False`), and a deliberately wrong
   anchor name raises instead of being silently ignored.
2. Unit test asserting hook execution order: gate hook fires before the
   feature-KD capture hook on a shared owner module (see "Design decision:
   gate order" above) — assert on a module with both hooks attached that the
   captured tensor differs from a pre-gate reference by exactly the gate
   factor.
3. One-batch forward/backward smoke with `feature_kd.enabled: true` on the
   real YOLOv8n graph (synthetic input, no dataset): confirm finite loss and
   that `gated/expected_sparsity` still moves as it does today with the flag
   `false`, i.e. the new loss does not silently break the existing sparsity
   path.
4. One-epoch dataset smoke
   (`python scripts/train_gated_pruning.py --epochs 1 --name p10_smoke_kd`)
   with `feature_kd.enabled: true`, compared against the existing
   `feature_kd.enabled: false` smoke. Both must show finite gate gradients
   and a logged `gated/feature_kd_loss` history; this is a mechanism check,
   not a result.
5. Only after 1–4 pass, and only with explicit authorization for a longer
   run: two full matched gated-training runs at the same target sparsity,
   seed, and epoch budget — one with `feature_kd.enabled: false` (today's
   behavior) and one `true` — then materialize and fine-tune both identically,
   and compare **validation** mAP50-95 only. Test split is not touched at any
   point in this plan.

## Non-goals

- No change to the augmented-Lagrangian sparsity math, the schedule, or
  `GatedGroupRegistry.materialize()`.
- No MAC- or latency-aware term in any loss here, matching DiariZen's own
  restraint (`get_num_macs()` in the reference repo is only used for
  post-prune reporting in `apply_pruning.py`, never inside a training loss).
- No change to the existing native `distill_model`/`dis` path; it stays the
  default single-point KD unless a future decision retires it in favor of
  the multi-depth version.
- Not a replacement for the still-pending dataset-backed one-epoch gated
  smoke that `docs/DIARIZEN_GATED_PRUNING_DESIGN.md` already calls the next
  required gate before any 30-epoch run; this plan's own step 4 folds into
  that same requirement rather than adding a second one.
