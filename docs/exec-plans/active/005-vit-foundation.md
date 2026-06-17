# 005 — Sprint 1: ViT Foundation

- **Sprint:** 1 (Foundation) · maps to `docs/vit/project_plan.md §3` (S1) and `§4` (Stage 0)
- **Branch:** `sprint1/vit-foundation` (off `main`)
- **Status:** active · **Created:** 2026-06-16
- **Compute:** 0 A100 units (frozen path only — VLD-11)
- Follow `AGENTS.md` ("Two projects" addendum — ViT work). Source of truth:
  `docs/vit/project_plan.md`; locked decisions: `docs/vit/decisions.md` (VLD-*).

---

## 1. Goal

Stand up the **ViT feature-extraction pipeline end-to-end in the frozen regime**:
a timm-based backbone factory for **ViT-B/16, Swin-T, BEiT-B/16**, wired into the
existing fusion model, with a ViT-aware frozen feature cache and **one non-NaN
single-ViT frozen smoke run**. This proves the plumbing (data → features → fusion
→ MLP → loss → macro-F1) works on transformer backbones. No fine-tuning, no fusion
ablation, no 5-fold — just a working foundation.

## 2. Inputs

- `docs/vit/project_plan.md` §3 (S1), §4 (Stage 0), §8 (frozen-relevant recipe only)
- `docs/vit/project_structure.md` §2.1 (verified timm strings + 768-d), §6 (module contracts)
- `docs/vit/decisions.md`: VLD-01/02/03/04/05/06/07/09/10/11/14
- `references/methodology_backbones/{vit,swin,beit}/paper.md` (feature-extraction method)
- Shared code reused: `src/models/full_model.py`, `projections.py`, `fusion/*`,
  `classifiers.py`, `src/data/feature_cache.py`, `datasets.py`, `splits.py`,
  `src/evaluation/metrics.py`
- `data/splits/official/5_fold_split.csv` (fold materialization logic)

## 3. Scope (Sprint 1)

- `src/models/vit_backbones.py` — `ViTFeatureExtractor` (timm `num_classes=0`, (B,768)).
- Verify `model.blocks` (ViT/BEiT = 12) / `model.layers` (Swin depths {2,2,6,2})
  counts so the `blocks[9:]` last-3 slice (VLD-08) is correct. *(Stored only; not
  exercised — no fine-tune in S1.)*
- Extractor **dispatch** in `full_model.py` (CNN vs ViT by name), CNN path unchanged.
- **configs/vit full method skeleton** (single + pairs + triple × concat/weighted)
  + `vit_frozen.yaml` + `experiment_matrix.yaml` *(decision: write all now; only
  single ViT-B is executed this sprint)*.
- `feature_cache.py` made **extractor-aware + bicubic** (decision: edit the shared
  file; CNN behavior byte-identical), outputs under `results/vit/feature_cache/`.
- Smoke tests in `tests/`: (2,768) shape per backbone; fusion output shapes.
- **One end-to-end run:** single **ViT-B frozen**, fold 0, **off-A100** → non-NaN
  macro-F1 + `metrics.json`.
- `uv run pytest tests/` green (CNN + new ViT tests).

## 4. Out of scope (do NOT do in Sprint 1)

- Fine-tuning / ViT LLRD / block unfreezing → **Sprint 3**.
- Concat/weighted ablation across all single/pair/triple configs → **Sprint 2**.
- 5-fold CV, bootstrap CI, seed/TTA → **Sprint 4**.
- Attention rollout, UMAP, LaTeX report → **Sprint 5**.
- **GMU and any extra fusion arm → deferred stretch, NOT deleted** (VLD-07;
  `fusion/gmu.py` + reference are kept). Added only after the MVP is stable.
- Any A100 usage (frozen only here). Any edit to the frozen `src/models/backbones.py`.

## 5. Files expected to be modified later (during S1 implementation)

| File | Action |
|---|---|
| `src/models/vit_backbones.py` | **NEW** — `ViTFeatureExtractor` |
| `src/models/full_model.py` | EDIT — CNN/ViT extractor dispatch by name |
| `src/data/feature_cache.py` | EDIT — extractor-aware + bicubic (CNN byte-identical) |
| `configs/vit/method/*.yaml` | NEW — single/pair/triple × concat/weighted |
| `configs/vit/training/vit_frozen.yaml` | NEW — frozen recipe |
| `configs/vit/experiment_matrix.yaml` | NEW — S1/S2 config list |
| `tests/test_vit_backbones.py` (+ fusion shape test) | NEW |
| `docs/vit/experiment_log.md`, `docs/vit/results_progress.md` | EDIT — log results |
| `src/models/backbones.py`, `projections.py`, `fusion/*`, `classifiers.py` | **NOT edited** (reused as-is / frozen) |

## 6. Step-by-step implementation plan

1. **Backbone factory.** Implement `ViTFeatureExtractor(name, pretrained=True,
   unfreeze_blocks=0)`: aliases `vit_b`/`swin_t`/`beit_b` → verified timm strings;
   build with `num_classes=0`; `forward(x) -> (B,768)` via `model(x)`; `feature_dim`
   property = 768; store `unfreeze_blocks` (default 3, unused in frozen S1).
2. **Verify counts.** Assert `len(model.blocks)==12` (ViT/BEiT) and Swin stage
   depths, confirming `blocks[9:]` = last 3 (record in `known_issues.md` if surprising).
3. **Dispatch.** In `full_model.py`, choose `ViTFeatureExtractor` when the name is a
   ViT alias, else `BackboneFeatureExtractor`; projection uses `in_dim=feature_dim`
   (generic). Verify CNN `full_model` still constructs and forwards.
4. **Feature cache.** Make `cache_frozen_features` pick the right extractor by name
   and use a **bicubic** transform (`crop_pct≈0.9`, VLD-09) for the ViT path; write to
   `results/vit/feature_cache/`. Keep the CNN branch unchanged.
5. **Configs.** Write the full `configs/vit/` method skeleton + `vit_frozen.yaml`
   (frozen, `ce_label_smooth`, no CutMix/MixUp, no unfreeze) + `experiment_matrix.yaml`.
6. **Tests.** Shape asserts (2,768) for V/S/B; fusion output shapes (concat=N×512,
   weighted=512); a tiny integration test that the single-ViT frozen forward yields
   a finite loss.
7. **Smoke run.** Extract ViT-B frozen features (fold 0, off-A100) → train
   projection+MLP a few epochs → confirm **non-NaN macro-F1**, save `metrics.json`
   under `results/vit/runs/`.
8. **Green + log.** `uv run pytest tests/` passes; append results to the docs.

### Implementation slices (execution order)

The 8 steps above are implemented as **3 sequential slices**, each ending with its
own `uv run pytest tests/` + approval gate (AGENTS task workflow). Dependency order
is strict: **1 → 2 → 3**.

| Slice | Name | Steps | Rationale | Done when |
|---|---|---|---|---|
| **1** | Model | 1, 2, 3 + shape tests of 6 | Pure model code, no data/training — fastest to test, foundational; factory + count-verify + dispatch are one unit | `(2,768)` asserts pass; CNN `full_model` still works; pytest green |
| **2** | Inputs | 4, 5 | Prepares the run's inputs: cached ViT features + configs describing the experiments | ViT `.pt` aligned (count==dataset); configs parse; CNN cache byte-identical |
| **3** | Run | 7, 8 | End-to-end validation; depends on Slices 1–2 | non-NaN macro-F1 + `metrics.json`; all pytest green; docs logged |

- Slice 1 may optionally be split into **1a** (factory + count + tests) and **1b**
  (dispatch) for smaller approval gates; combined is recommended (both are small).
- In Slice 2, only the **single ViT-B** config + cache are on the critical path; the
  remaining pair/triple configs are written but not executed until Sprint 2.

## 7. Risks

- **timm path/count mismatch** → Step 2 verification (paths already confirmed present 2026-06-16).
- **Shared `feature_cache.py` edit breaks CNN** → keep CNN branch byte-identical; run CNN tests.
- **Bilinear vs bicubic** silently lowers fidelity → enforce VLD-09 bicubic for ViT.
- **CLS logic misapplied to Swin** → `num_classes=0` + `model(x)` only (VLD-04).
- **Manifests absent on fresh checkout** → regenerate via `scripts/make_splits.py`.

## 8. Acceptance criteria

- `ViTFeatureExtractor(name)(x).shape == (2,768)` for all three (test passes).
- `model.blocks`/`layers` counts verified; `blocks[9:]` = exactly 3 blocks.
- A single-ViT `full_model` forwards to logits without error; **CNN `full_model` still works**.
- ViT feature cache `.pt` written under `results/vit/feature_cache/`, count == dataset size.
- **One single ViT-B frozen run (fold 0): non-NaN macro-F1 + `metrics.json` saved.**
- `uv run pytest tests/` all green (CNN + new ViT tests).
- 0 A100 units used; `backbones.py` untouched; no large artifacts committed.

## 9. Commands to run

```bash
uv sync
# (if fold manifests absent) regenerate:
uv run python scripts/make_splits.py --config configs/dataset/hyperkvasir_23class_official.yaml
# verify backbones load + shapes:
uv run pytest tests/test_vit_backbones.py
# cache ViT-B frozen features (off-A100):
uv run python scripts/extract_features.py --config configs/dataset/hyperkvasir_23class_official.yaml --backbone vit_b
# single ViT-B frozen smoke run, fold 0:
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 01_single_vit_b_frozen_official
# full suite green:
uv run pytest tests/
```
*(Exact script flags to be confirmed against `scripts/` during implementation.)*

## 10. Documentation updates required

- `docs/vit/experiment_log.md` — append the S1 completion entry (run id, pytest result).
- `docs/vit/results_progress.md` — tick Sprint 1 milestones #2–5 + first smoke result.
- `docs/vit/known_issues.md` — log any timm count/path surprise (KI-VIT-NNN).
- On completion: move this plan `active/ → completed/`.
- Housekeeping (separate, optional): move the finished CNN
  `docs/exec-plans/active/004-week-4-analysis-report.md` → `completed/`.
