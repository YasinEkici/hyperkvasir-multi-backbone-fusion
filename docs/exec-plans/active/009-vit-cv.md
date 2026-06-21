# 009 - ViT Sprint 4: 5-fold CV + Confidence Intervals

## 1. Goal

Run official 5-fold cross-validation for the Sprint 3 top-4 ViT fusion
configurations using the Sprint 3.5 tuned fine-tune config, on A100, and report
each configuration's per-fold and **mean ± std** Accuracy / macro-F1 / macro
precision / macro recall plus a **bootstrap 95% CI on the headline macro-F1**
(VLD-10). Identify the overall best ViT fusion configuration for the report.

As leakage-free extras (VLD-13), add **TTA** (inference-only) on the CV
checkpoints and a **top-1 seed ensemble** (best config only) so the headline
number carries an ensemble + CI. No new architecture, no new candidate configs.

Source of truth: `docs/vit/project_plan.md` §3 Sprint 4 / §4 Stage 3; locked
decisions VLD-10, VLD-11, VLD-12, VLD-13, VLD-15, VLD-17.

Branch: `sprint4/vit-cv` (off `main`, **after Sprint 3.5 / `008-vit-perf.md` is
merged** — Sprint 4 depends on the tuned `vit_finetune.yaml` and VLD-17).

## 2. Inputs

- Dataset config: `configs/dataset/hyperkvasir_23class_official.yaml`.
- Fold manifests: `data/splits/hyperkvasir_official_5fold/fold_{0..4}.csv`
  (all five verified present).
- Tuned fine-tune config: `configs/vit/training/vit_finetune.yaml` (VLD-17).
- Method configs (reused, unchanged):
  `single_swin_t.yaml`, `pair_swin_t_beit_b_weighted.yaml`,
  `triple_vit_swin_beit_weighted.yaml`, `pair_vit_b_beit_b_concat.yaml`.
- Sprint 3 top-4 (by fold-0 test macro-F1):
  1. `02_single_swin_t` (S, none) — 0.5918837684.
  2. `09_pair_swin_t_beit_b_weighted` (S+B, weighted) — 0.5909205632.
  3. `11_triple_weighted` (V+S+B, weighted) — 0.5850707743.
  4. `05_pair_vit_b_beit_b_concat` (V+B, concat) — 0.5783726429.
  Dropped: `04_pair_vit_b_swin_t_concat` — NOT included.
- Engine: `scripts/train.py` (`--fold`/`--seed` run-dir naming; A100 gate),
  `scripts/summarize_vit_finetune.py`, `colab/vit_finetune_runner.ipynb`.
- Provenance/runtime tooling: archive dataset staging (KI-VIT-001), fast
  provenance check.

## 3. Scope

- **CV run namespace (decision, approved):** add **four new CV experiment ids**
  so the tuned-config fold-0 runs do NOT overwrite the Sprint 3 fold-0 artifacts:
  - `02_single_swin_t_cv` -> `single_swin_t.yaml`
  - `09_pair_swin_t_beit_b_weighted_cv` -> `pair_swin_t_beit_b_weighted.yaml`
  - `11_triple_weighted_cv` -> `triple_vit_swin_beit_weighted.yaml`
  - `05_pair_vit_b_beit_b_concat_cv` -> `pair_vit_b_beit_b_concat.yaml`
  Each points at `vit_finetune.yaml`; all 5 folds use the tuned config.
- Run **4 configs × folds 0-4 = 20 fine-tune runs** on A100. Run dirs follow
  `scripts/train.py` naming: fold 0 -> `{id}`, fold k>=1 -> `{id}_fold_{k}`.
- Each completed run must contain `metrics.json`, `config.yaml`,
  `predictions.npz`, `best.pt`.
- **Aggregate** per config (every number from `results/vit/runs/{id}/metrics.json`):
  per-fold + mean ± std for accuracy, macro-F1, macro precision, macro recall;
  **bootstrap 95% CI on macro-F1** computed leakage-free over the concatenated
  out-of-fold test predictions (each sample scored by the fold that did not train
  on it). Preserve per-class metrics + confusion matrix per fold.
- Produce the **final 5-fold ranking table (mean ± 95% CI macro-F1)** and name the
  overall best ViT fusion config.
- **Leakage-free extras (option B):**
  - Persist **test softmax probabilities** (needed for averaging) — small, contained
    output addition; `predictions.npz` currently stores argmax preds only.
  - **TTA** (inference-only) on the 20 CV checkpoints: average softmax over a small
    set of test-time augmentations **within each fold's test set**, then macro-F1
    + CI. No retraining.
  - **Top-1 seed ensemble:** for the single best config only, train **folds 0-4
    with 2 additional seeds** (e.g. 123, 2024; seed 42 already in the CV) =
    **+10 runs**; average softmax across seeds **within each fold's test set**,
    then macro-F1 + CI. Never average fold models (VLD-13 / anti-halluc §5.1.7).
- A100 / provenance: keep the VLD-11 A100 gate before every fine-tune; reuse
  archive staging + fast provenance. Colab/Linux makes the tuned `num_workers`
  safe (KI-VIT-002 is Windows-only).

Total: **~30 A100 runs** (20 CV + 10 seed-ensemble), well within the 500-unit
Pro+ budget.

## 4. Out of Scope

- Sprint 5: attention rollout / Grad-CAM / UMAP / LaTeX report / YouTube video.
- Any config beyond the top-4; GMU; re-running the dropped `04` config.
- Full seed ensemble across all 4 configs (top-1 only — the CNN project found seed
  ensembling gave no macro-F1 gain on this dataset).
- Changing the tuned config, dataset splits, class definitions, evaluation rules,
  or any locked VLD decision.
- Editing the frozen CNN files or `src/models/backbones.py`.
- Committing large artifacts (`*.pt`, `predictions.npz`, run dirs stay gitignored).

## 5. Files Expected to Be Modified Later

- `configs/vit/experiment_matrix.yaml` - add the four `_cv` rows (and, for the
  seed-ensemble stage, none extra — seeds are a CLI `--seed` override).
- `scripts/summarize_vit_cv.py` (new, small) OR extend
  `scripts/summarize_vit_finetune.py` - per-fold + mean ± std + bootstrap 95% CI
  aggregation and the final ranking table.
- `scripts/train.py` / `src/training/trainer.py` - minimal: persist test softmax
  probabilities in `predictions.npz` (default-safe; needed for TTA/ensemble).
- A TTA + seed-ensemble inference helper (new small script under `scripts/`) -
  leakage-free within-fold softmax averaging.
- `colab/vit_finetune_runner.ipynb` - top-4 × folds 0-4 (and the top-1 seed loop)
  via `train.py --experiment {cv_id} --fold k [--seed S]`, overnight-safe.
- `docs/vit/experiment_log.md`, `docs/vit/results_progress.md` - CV table with
  mean ± 95% CI, TTA / seed-ensemble numbers, best config.
- `docs/vit/decisions.md` - a VLD note only if the final-model selection or the
  CV/CI methodology is locked as a decision.
- `docs/vit/known_issues.md` - only for newly observed issues.

Must not be modified: `src/models/backbones.py`; locked VLD decisions; the tuned
`vit_finetune.yaml` recipe values.

## 6. Step-by-Step Implementation Plan

### Slice 1 - Preflight + CV matrix rows
1. Confirm branch `sprint4/vit-cv` off a `main` that includes Sprint 3.5.
2. Confirm the five fold manifests and the tuned `vit_finetune.yaml` (VLD-17).
3. Add the four `_cv` rows to `experiment_matrix.yaml` (reuse method configs,
   point at `vit_finetune.yaml`, fold 0 default). Add a focused config test.

### Slice 2 - Colab multi-fold runner
1. Extend the runner to iterate the four `_cv` ids × folds 0-4 via
   `train.py --fold`, overnight-safe (restore-from-Drive / skip-if-metrics /
   per-run backup), reusing archive staging + fast provenance + the A100 gate.

### Slice 3 - Run the 20 CV fine-tune runs (A100)
1. Run all four `_cv` configs × folds 0-4 with the tuned config.
2. Verify each run dir has the four required artifacts and finite test metrics.

### Slice 4 - Aggregate + CI + ranking
1. Verify the five fold test sets are disjoint and cover the dataset once
   (required for pooled out-of-fold bootstrap; else fall back to per-fold CIs).
2. Compute per-fold + mean ± std (Acc / macro-F1 / macro-P / macro-R) and the
   bootstrap 95% CI on macro-F1 over concatenated OOF test predictions.
3. Produce the final ranking table; identify the overall best config.

### Slice 5 - Leakage-free extras (TTA + top-1 seed ensemble)
1. Persist test softmax probabilities (default-safe code change).
2. TTA on the 20 CV checkpoints (within-fold softmax averaging) -> macro-F1 + CI.
3. Seed-ensemble the best config: run folds 0-4 with seeds 123 and 2024 (+10),
   average softmax across {42,123,2024} within each fold -> macro-F1 + CI.

### Slice 6 - Documentation + validation
1. Update `results_progress.md` (CV table + extras + best config) and
   `experiment_log.md` (commands, A100/provenance, per-fold metrics, pytest).
2. Add a VLD note only if the final model / CI methodology is locked.
3. `uv run pytest tests/`; confirm hygiene (no large artifacts, `backbones.py`
   untouched).

## 7. Risks

- **A100 budget:** ~30 runs; with VLD-17 (~5-6x) and 500 units this is comfortable,
  but the first fold should calibrate the real per-run cost.
- **Fold-0 overwrite:** mitigated by the new `_cv` namespace (Sprint 3 artifacts
  untouched).
- **CI leakage:** bootstrap must be within / across held-out test predictions only;
  never average fold models (VLD-13). Verify fold test-set disjointness.
- **Missing softmax outputs:** TTA + seed ensemble need probabilities; the prob
  output addition must stay default-safe (frozen/CNN unaffected).
- **Run-id collisions:** `_cv` ids + `--fold`/`--seed` suffixes must be distinct.
- **KI-VIT-002:** Windows local only; the A100/Linux target is unaffected.
- **Large artifacts / scope creep:** run dirs/checkpoints stay gitignored; no
  Sprint 5 work.

## 8. Acceptance Criteria

- Four `_cv` configs × folds 0-4 = 20 runs complete, each with `metrics.json`,
  `config.yaml`, `predictions.npz`, `best.pt`, fold-tagged correctly, finite
  metrics.
- Per-config per-fold + mean ± std table and a bootstrap 95% CI on macro-F1;
  every number traces to a `metrics.json`.
- Final 5-fold ranking (mean ± 95% CI macro-F1) produced; overall best config
  named.
- TTA (within-fold) and top-1 seed ensemble (+10 runs) reported leakage-free with
  CI.
- Sprint 3 fold-0 artifacts untouched; `backbones.py` untouched; no large
  artifacts committed.
- `uv run pytest tests/` passes; Sprint 5 not started.

## 9. Commands to Run

### Branch
```powershell
git switch -c sprint4/vit-cv   # off a main that includes Sprint 3.5
```

### CV runs (top-4 × folds 0-4, A100, tuned config)
```powershell
# for each cv id in {02_single_swin_t_cv, 09_pair_swin_t_beit_b_weighted_cv,
#                     11_triple_weighted_cv, 05_pair_vit_b_beit_b_concat_cv}
# and each fold k in 0..4:
uv run --no-sync python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment <cv_id> --fold <k> --device cuda
```

### Top-1 seed ensemble (best config only, +2 seeds × 5 folds)
```powershell
# best_cv_id determined after Slice 4; seeds 123 and 2024, folds 0..4:
uv run --no-sync python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment <best_cv_id> --fold <k> --seed <S> --device cuda
```

### Aggregate + CI + TTA
```powershell
uv run python scripts/summarize_vit_cv.py --runs-dir results/vit/runs --output-dir results/vit/tables
# TTA + seed-ensemble inference helper (within-fold softmax averaging)
```

### Tests + hygiene
```powershell
uv run pytest tests/
git status --short
git diff -- src/models/backbones.py
```

## 10. Documentation Updates Required

- `docs/vit/experiment_log.md`: commands, A100/provenance status, per-fold + mean
  ± std metrics, CI, TTA + seed-ensemble numbers, pytest result, best config.
- `docs/vit/results_progress.md`: Sprint 4 5-fold ranking table (mean ± 95% CI
  macro-F1), extras, selected best ViT fusion config.
- `docs/vit/decisions.md`: VLD note only if the final model / CI methodology is
  locked.
- `docs/vit/known_issues.md`: only for newly observed issues.
- `docs/exec-plans/active/009-vit-cv.md`: keep active until Sprint 4 completion is
  explicitly approved; move to `completed/` only then.
