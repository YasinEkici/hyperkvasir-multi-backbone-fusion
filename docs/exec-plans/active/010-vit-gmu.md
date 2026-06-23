# 010 - ViT Sprint 4.5: GMU Fusion Ablation (optional stretch)

## 1. Goal

Evaluate **GMU (Gated Multimodal Unit, Arevalo et al. 2017)** as a third fusion
method alongside concat + weighted, to enrich the report's fusion discussion
(§5.5) — an optional stretch (VLD-07), run only now that the concat/weighted MVP
is complete (Sprint 4). First **fix the GMU implementation to match the paper**
(faithful element-wise gate), then a cheap **fold-0 screen** of GMU on the
multi-backbone sets, and **promote only a competitive GMU config to 5-fold CV**
for an apples-to-apples comparison vs the Sprint 4 best (`11_triple_weighted`,
macro-F1 0.6094 / pooled 0.6119 [0.593, 0.630]).

Source of truth: `docs/vit/project_plan.md` §2 (stretch) / §4 Stage 4; locked
decisions VLD-07, VLD-10, VLD-11, VLD-12, VLD-13, VLD-15, VLD-17, VLD-18; and the
GMU paper `references/methodology_fusion/gmu_arevalo_2017_iclr_workshop/paper.md`
§3.1.

Branch: `sprint4.5/vit-gmu` (off a `main` that already includes Sprint 4).
**Time-box (deadline Fri 23:59): keep this minimal — it must NOT delay the
mandatory Sprint 5 report + YouTube video.**

## 2. Inputs

- GMU module `src/models/fusion/gmu.py`; dispatch in `src/models/full_model.py`
  (`MultiCNNFusionClassifier`, `fusion_type="gmu"`) and `scripts/train.py`
  (`FrozenHeadModel`).
- GMU paper §3.1 equations (the fidelity reference).
- Tuned fine-tune config `configs/vit/training/vit_finetune.yaml` (VLD-17).
- Method configs `configs/vit/method/*.yaml`; matrix
  `configs/vit/experiment_matrix.yaml`.
- Fold manifests `data/splits/hyperkvasir_official_5fold/fold_{0..4}.csv`.
- Aggregation: `scripts/summarize_vit_cv.py`, `scripts/eval_tta_ensemble.py`.
- Sprint 4 tuned-config fold-0 baselines (for the screen): `11_triple_weighted_cv`
  (0.6102), `09_pair_swin_t_beit_b_weighted_cv`, `05_pair_vit_b_beit_b_concat_cv`.
- Colab runner `colab/vit_finetune_runner.ipynb`.

## 3. Scope

- **GMU = fusion → MULTI-backbone only.** Configs: pair V+S, pair V+B, pair S+B,
  triple V+S+B. **No singles** (no fusion).
- **Faithful GMU gate** (Slice 0): the paper's gate is **element-wise** (per
  feature), `z = σ(W_z·[x_1..x_N])` with `z` the same dim as the branch
  activations; `h = z ⊙ h_v + (1−z) ⊙ h_t` (bimodal). The current `gmu.py` uses a
  **per-branch scalar softmax** gate (z ∈ ℝ^N) — a simplification that loses the
  element-wise property. Add a config-driven `gate_mode` so ViT GMU uses the
  faithful element-wise gate; keep the scalar mode as the default for CNN
  back-compat (the CNN project's `15_triple_gmu` run used the scalar gate).
- 4 GMU method configs + 4 `*_gmu_cv` matrix rows pointing at `vit_finetune.yaml`,
  `fold-0`-screenable and CV-promotable without renaming.
- **Stage 1 (fold-0 screen, A100, 4 runs):** compare GMU vs the **tuned-config**
  fold-0 baselines (headline = **triple-GMU vs triple-weighted**; pair GMU vs
  their tuned counterparts where available — V+S has none, mark caveated).
- **Stage 2 (conditional, +5 runs):** ONLY if a GMU config is competitive with /
  beats the best weighted at fold-0, promote that single config to 5-fold CV and
  aggregate (mean±std + bootstrap 95% CI, leakage-free OOF) vs `11_triple_weighted`.
  Else STOP after Stage 1 and document "GMU evaluated, no gain".
- Reuse the tuned config (VLD-17), the A100 gate (VLD-11), archive staging
  (KI-VIT-001), fast provenance, and the runner. Total: 4 runs (screen) or 9
  (screen + 1 CV); ~10 A100 units.

## 4. Out of Scope

- GMU on single backbones; re-running concat/weighted; changing the Sprint 4 CV
  results, the tuned config, splits, class defs, eval rules, or any locked VLD.
- Sprint 5: attention rollout / Grad-CAM / UMAP / LaTeX report / YouTube video.
- Changing the CNN scalar-gate GMU behavior (kept as the default; CNN frozen
  artifacts untouched).
- Editing the frozen CNN files or `src/models/backbones.py`; committing large
  artifacts (`*.pt` / predictions / run dirs stay gitignored).
- Any 4th backbone or fusion method beyond GMU.

## 5. Files Expected to Be Modified Later

- `src/models/fusion/gmu.py` - add `gate_mode` ("scalar" default / "elementwise"
  faithful); element-wise gate reduces to the paper's bimodal form for N=2.
- `src/models/full_model.py` (+ `scripts/train.py` `FrozenHeadModel` if trivial) -
  pass GMU `fusion_kwargs` (e.g. `gate_mode`) from the method config.
- `configs/vit/method/{pair_vit_b_swin_t,pair_vit_b_beit_b,pair_swin_t_beit_b,
  triple_vit_swin_beit}_gmu.yaml` - new (fusion_type: gmu, gate_mode: elementwise).
- `configs/vit/experiment_matrix.yaml` - 4 `*_gmu_cv` rows.
- `tests/test_vit_configs.py` - GMU-rows test; `tests/test_*` - GMU gate tests.
- `scripts/summarize_vit_cv.py` - add the promoted GMU CV id (Stage 2 only).
- `colab/vit_finetune_runner.ipynb` - control-panel preset for the GMU screen/CV.
- `docs/vit/decisions.md` - VLD note: GMU gate fidelity (element-wise) + the GMU
  result; `docs/vit/results_progress.md`, `docs/vit/experiment_log.md`.
- `docs/vit/project_plan.md §3` - insert Sprint 4.5 (`010-vit-gmu.md`); report
  becomes `011-vit-report.md`.

Must not be modified: `src/models/backbones.py`; locked VLD recipe values; the
Sprint 4 `_cv` results.

## 6. Step-by-Step Implementation Plan

### Slice 0 - GMU correctness fix (faithful element-wise gate)
1. Read GMU paper §3.1; confirm the gate is **element-wise** ("importance to
   various features simultaneously").
2. Add `gate_mode` to `gmu.py`:
   - `"scalar"` (default): current per-branch softmax (CNN back-compat).
   - `"elementwise"`: `W_z: (N·D)→(N·D)`, reshape `(B, N, D)`, softmax over the
     branch axis per dimension; `h = Σ_i z_i ⊙ h_i`. For N=2 this reduces exactly
     to the paper's tied `z ⊙ h_v + (1−z) ⊙ h_t`.
3. Wire `gate_mode` config-driven (method `fusion_kwargs`) through
   `full_model.py` (default scalar so CNN/existing configs are unchanged).
4. Update `gmu.py` docstring; add a VLD note in `decisions.md` (faithful gate;
   CNN GMU used the scalar gate — documented, frozen artifacts unchanged).
5. Tests: element-wise gate shape `(B,N,D)` and per-dimension sum-to-1; bimodal
   reduces to the paper form; scalar mode byte-unchanged. `uv run pytest tests/`.

### Slice 1 - GMU method configs + matrix rows
1. Add the 4 GMU method configs (512-d, MLP, `fusion_type: gmu`,
   `fusion_kwargs: {gate_mode: elementwise}`).
2. Add 4 `*_gmu_cv` rows pointing at `vit_finetune.yaml`, fold 0. Config test.

### Slice 2 - Colab runner preset
1. Control-panel preset for the fold-0 screen (4 GMU ids, FOLDS=[0], SEEDS=[42]).

### Slice 3 - Stage 1 fold-0 screen (A100, 4 runs)
1. Run the 4 GMU configs on fold 0; verify 4 artifacts + finite metrics.
2. Compare to the tuned-config fold-0 baselines (triple-GMU vs `11_triple_weighted_cv`
   fold-0 0.6102 is the headline; pairs vs their tuned counterparts, V+S caveated).

### Slice 4 - Conditional CV promotion (A100, +5)
1. If a GMU config is competitive with / beats triple-weighted at fold-0, run it
   on folds 1-4 (5-fold total). Else stop and record "no gain".

### Slice 5 - Aggregate + document + validate
1. If promoted: aggregate via `scripts/summarize_vit_cv.py` (mean±std + 95% CI);
   compare vs `11_triple_weighted` (0.6094) leakage-free.
2. Update `results_progress.md`, `experiment_log.md`, `decisions.md` (GMU result),
   `project_plan.md §3`. `uv run pytest tests/`; hygiene check.

## 7. Risks

- **Marginal gain:** GMU may not beat weighted; the **faithful element-wise gate
  makes the comparison meaningful** (the old scalar gate was near-identical to
  `weighted`, likely why a gain would be small). Document the result either way.
- **Time/scope:** keep minimal; do not block the mandatory Sprint 5 report+video.
- **CNN back-compat:** `gate_mode` default scalar keeps the CNN GMU + all existing
  configs unchanged; only ViT GMU opts into element-wise.
- **Fold-0 comparison consistency:** compare against the **tuned** `_cv` fold-0
  baselines (not the pre-3.5 Sprint 3 `_finetune_official`); V+S has no tuned
  counterpart → caveated.
- **A100 / Windows:** A100 gate (VLD-11); KI-VIT-002 is Windows-only (Colab safe).
- **Large artifacts / Sprint 5 scope creep:** run dirs gitignored; no Sprint 5.

## 8. Acceptance Criteria

- `gmu.py` element-wise gate matches the paper (bimodal reduces to §3.1);
  config-driven; scalar default keeps CNN/existing behavior; tests pass.
- 4 GMU fold-0 runs complete (4 artifacts each, finite metrics), compared to the
  tuned `_cv` fold-0 baselines; triple-GMU vs triple-weighted is the headline.
- If competitive: promoted GMU 5-fold CV aggregated (mean±std + 95% CI) vs
  `11_triple_weighted`, leakage-free, traceable to `metrics.json`. Else a clear
  "GMU evaluated, no gain" record.
- GMU fidelity + result documented (`decisions.md`, `results_progress.md`); a
  legitimate "GMU (Arevalo 2017)" citation (no overclaim).
- `uv run pytest tests/` passes; `backbones.py` untouched; no large artifacts;
  Sprint 5 not started.

## 9. Commands to Run

```powershell
git switch -c sprint4.5/vit-gmu   # off a main that includes Sprint 4

# Stage 1 screen (A100, fold 0):
uv run --no-sync python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment <gmu_cv_id> --fold 0 --device cuda
# Stage 2 (if promoted): folds 1-4 for the competitive id
uv run --no-sync python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment <gmu_cv_id> --fold <k> --device cuda

uv run python scripts/summarize_vit_cv.py   # if a GMU id is added to CV_CONFIGS
uv run pytest tests/
git diff -- src/models/backbones.py
```

## 10. Documentation Updates Required

- `docs/vit/decisions.md`: VLD note — GMU gate fidelity (element-wise, paper-faithful;
  CNN used scalar) + the GMU ablation result.
- `docs/vit/results_progress.md`: GMU fold-0 screen table (+ 5-fold row if promoted)
  vs concat/weighted; conclusion.
- `docs/vit/experiment_log.md`: commands, A100/provenance, GMU metrics, pytest.
- `docs/vit/project_plan.md §3`: insert Sprint 4.5; renumber report to
  `011-vit-report.md`.
- `docs/exec-plans/active/010-vit-gmu.md`: keep active until Sprint 4.5 completion
  is explicitly approved; move to `completed/` only then.
