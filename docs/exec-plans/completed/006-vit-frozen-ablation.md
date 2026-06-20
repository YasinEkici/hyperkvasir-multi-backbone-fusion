# 006 - ViT Frozen Ablation

## 1. Goal

Run the Sprint 2 frozen-path ablation for the ViT fusion project on official
fold 0, off-A100, using cached features. The sprint ranks all frozen single,
pair, and triple ViT configs by test macro-F1 and identifies the top candidates
to carry into Sprint 3 fine-tuning.

Source of truth:
- `docs/vit/project_plan.md` Section 3 Sprint 2 and Section 4 Stage 1.
- `docs/vit/project_structure.md` Sections 2.1, 2.3, and 6.
- `docs/vit/decisions.md` VLD-03 through VLD-14.

Branch: `sprint2/vit-frozen-ablation`, off `main` or the merged Sprint 1
foundation branch.

Compute budget: 0 A100 units. Frozen caching and frozen-head training must run
on local/non-A100 GPU or CPU.

## 2. Inputs

- Dataset config: `configs/dataset/hyperkvasir_23class_official.yaml`.
- Fold manifest: `data/splits/hyperkvasir_official_5fold/fold_0.csv`.
- Experiment matrix: `configs/vit/experiment_matrix.yaml`.
- Frozen training config: `configs/vit/training/vit_frozen.yaml`.
- Method configs under `configs/vit/method/`.
- Existing Sprint 1 cache/run:
  - `results/vit/feature_cache/fold_0_vit_b_features.pt`.
  - `results/vit/runs/01_single_vit_b_frozen_official/metrics.json`.
- Current implementation:
  - `src/models/vit_backbones.py`.
  - `src/models/full_model.py`.
  - `src/data/feature_cache.py`.
  - `scripts/train.py`.
  - `scripts/extract_features.py`.

The frozen matrix is expected to contain exactly these 11 fold-0 rows:

| id | backbones | fusion |
|---|---|---|
| `01_single_vit_b_frozen_official` | V | none |
| `02_single_swin_t_frozen_official` | S | none |
| `03_single_beit_b_frozen_official` | B | none |
| `04_pair_vit_b_swin_t_concat_frozen_official` | V+S | concat |
| `05_pair_vit_b_beit_b_concat_frozen_official` | V+B | concat |
| `06_pair_swin_t_beit_b_concat_frozen_official` | S+B | concat |
| `07_pair_vit_b_swin_t_weighted_frozen_official` | V+S | weighted |
| `08_pair_vit_b_beit_b_weighted_frozen_official` | V+B | weighted |
| `09_pair_swin_t_beit_b_weighted_frozen_official` | S+B | weighted |
| `10_triple_concat_frozen_official` | V+S+B | concat |
| `11_triple_weighted_frozen_official` | V+S+B | weighted |

## 3. Scope

- Confirm fold-0 manifests exist, or regenerate them with the existing split
  script.
- Confirm the matrix contains the intended 11 frozen ablation rows and no Sprint
  3 fine-tune rows.
- Ensure fold-0 feature caches exist for all three ViT aliases:
  - `results/vit/feature_cache/fold_0_vit_b_features.pt`.
  - `results/vit/feature_cache/fold_0_swin_t_features.pt`.
  - `results/vit/feature_cache/fold_0_beit_b_features.pt`.
- Validate each cache has one feature row per fold manifest row and feature
  dimension 768.
- Run all 11 frozen fold-0 configs from `configs/vit/experiment_matrix.yaml`.
- Verify each run writes `metrics.json`, `config.yaml`, `predictions.npz`, and
  `best.pt` under `results/vit/runs/{experiment_id}/`.
- Produce a ranked frozen ablation table ordered by test macro-F1.
- Update ViT progress docs with metrics, command provenance, cache status,
  device, pytest result, and Sprint 3 candidate recommendations.

## 4. Out of Scope

- Fine-tuning, LLRD, EMA fine-tune recipe, or unfreezing transformer blocks.
- A100-gated training. Never spend Colab A100 units on Sprint 2 frozen work.
- 5-fold CV, bootstrap CI, seed ensembling, TTA, or cross-fold aggregation.
- Attention rollout, Grad-CAM, UMAP, LaTeX/report writing, or video work.
- Adding GMU or any extra fusion arm unless explicitly approved after the MVP
  frozen ablation is complete.
- Changing dataset splits, class definitions, evaluation rules, or locked VLD
  decisions.
- Editing frozen CNN root files or `src/models/backbones.py`.
- Committing large artifacts such as `*.pt`, checkpoints, predictions, or run
  directories.

## 5. Files Expected to Be Modified Later

Required during Sprint 2 execution:
- `docs/vit/experiment_log.md` - append run commands, device, cache status,
  pytest result, and top-ranked frozen configs.
- `docs/vit/results_progress.md` - add the ranked frozen ablation table with
  metric sources.
- `docs/exec-plans/active/006-vit-frozen-ablation.md` - mark progress as slices
  complete, if using slice tracking.

Conditional:
- `docs/vit/known_issues.md` - only if a real issue occurs, such as cache
  mismatch, missing raw images, timm load drift, failed run, or unstable metric.
- A small metrics aggregation helper may be added only if existing utilities are
  insufficient for a macro-F1-ranked ViT table. Prefer reusing
  `scripts/generate_report_tables.py --runs-dir results/vit/runs` first.
- `scripts/extract_features.py` may be minimally extended only if a cache-only
  fold-manifest CLI is needed. Its current CLI looks for separate
  `train.csv`/`val.csv`/`test.csv` manifests, while official fold-0 caching in
  `scripts/train.py` uses `data/splits/{protocol}/fold_0.csv`.

Must not be modified:
- `src/models/backbones.py`.
- Frozen CNN root project docs/configs unless a separate approved task requires
  them.
- `docs/vit/decisions.md`, unless a locked decision genuinely changes. Sprint 2
  should not change any VLD decision.

## 6. Step-by-Step Implementation Plan

### Slice 1 - Preflight and Matrix Audit

1. Check git status and confirm the branch is `sprint2/vit-frozen-ablation`.
2. Confirm the official fold-0 manifest exists. If absent, regenerate fold
   manifests from `configs/dataset/hyperkvasir_23class_official.yaml`.
3. Inspect `configs/vit/experiment_matrix.yaml` and confirm it contains exactly
   the 11 expected Sprint 2 rows listed above.
4. Confirm every row uses `configs/vit/training/vit_frozen.yaml`, fold 0, and a
   method config under `configs/vit/method/`.
5. Confirm `vit_frozen.yaml` has `unfreeze_blocks: 0`, `ema.enabled: false`, and
   no fine-tune-only optimizer settings.

### Slice 2 - Cache Presence and Alignment

1. Check whether the three fold-0 cache files exist under
   `results/vit/feature_cache/`.
2. If Swin-T or BEiT-B cache is missing, generate it off-A100. The current
   no-refactor path is to run the relevant frozen experiment through
   `scripts/train.py`; it will build missing cache files before head training.
3. Do not recompute `fold_0_vit_b_features.pt` if it already exists and passes
   validation.
4. Validate every cache:
   - `features.shape[0] == len(fold_0.csv)`.
   - `features.shape[1] == 768`.
   - `labels`, `paths`, and `indices` lengths match `features.shape[0]`.
   - cache `backbone` matches the expected alias.
5. If a cache mismatch occurs, stop and log it in `docs/vit/known_issues.md`
   before rebuilding.

### Slice 3 - Run Frozen Ablation Rows

1. Run all 11 experiments from `configs/vit/experiment_matrix.yaml` on fold 0,
   using `--device cuda` when a non-A100 CUDA device is available, otherwise
   `--device cpu`.
2. Keep `unfreeze_blocks: 0`; do not override the frozen training config with a
   fine-tune config.
3. After each run, verify these files exist:
   - `results/vit/runs/{experiment_id}/metrics.json`.
   - `results/vit/runs/{experiment_id}/config.yaml`.
   - `results/vit/runs/{experiment_id}/predictions.npz`.
   - `results/vit/runs/{experiment_id}/best.pt`.
4. Inspect each `metrics.json` for non-NaN `test.macro_f1`, `test.accuracy`,
   `test.macro_precision`, and `test.macro_recall`.
5. If any run fails after one attempted fix, stop and ask before changing
   methodology or config semantics.

### Slice 4 - Aggregate and Rank

1. Generate or collect a table from completed `results/vit/runs/*/metrics.json`
   files.
2. Rank rows by `test.macro_f1` descending.
3. Include at minimum:
   - config id.
   - backbones.
   - fusion.
   - transfer mode (`frozen`).
   - fold.
   - accuracy.
   - macro-F1.
   - macro precision.
   - macro recall.
   - metrics source path.
4. Identify the top candidates for Sprint 3 fine-tune funnel. Prefer a diverse
   top set that includes the strongest single, strongest pair, and strongest
   triple if performance is close; otherwise rank strictly by macro-F1 and note
   the rationale.
5. Do not run any Sprint 3 fine-tune command.

### Slice 5 - Documentation and Validation

1. Update `docs/vit/results_progress.md` with the ranked frozen ablation table.
2. Update `docs/vit/experiment_log.md` with:
   - commands run.
   - device and confirmation that no A100 units were used.
   - cache status and cache shapes.
   - pytest result.
   - top-ranked frozen configs.
3. Update `docs/vit/known_issues.md` only if a real issue occurred.
4. Run `uv run pytest tests/`.
5. Check `git diff` and confirm no large artifacts are staged/committed and
   `src/models/backbones.py` remains untouched.

## 7. Risks

- **Wrong pooling path:** using `forward_features(x)[:,0]` would break Swin and
  may be wrong for BEiT. Use `ViTFeatureExtractor` with timm
  `num_classes=0` and `model(x)` only (VLD-03/VLD-04).
- **Wrong preprocessing:** using bilinear or CNN resize behavior for ViT caches
  can silently change feature quality. ViT cache path must keep bicubic resize
  and `crop_pct` about 0.9 (VLD-09).
- **Cache misalignment:** features must remain in manifest order. DataLoaders
  must not shuffle during cache extraction, and feature counts must match
  `fold_0.csv`.
- **Stale partial run directories:** a failed run may leave incomplete artifacts.
  Verify required files before reporting a metric.
- **Metrics table drift:** generated tables must read `results/vit/runs`, not
  root CNN `results/runs`, and must rank by test macro-F1.
- **Scope creep:** GMU, fine-tuning, 5-fold CV, TTA, and report figures are
  Sprint 3+ or stretch work.
- **A100 waste:** frozen work is prohibited on A100. If running in Colab, use a
  non-A100 runtime for Sprint 2 or run locally.
- **Large artifacts in git:** feature caches and run outputs are outputs only.
  Confirm they remain ignored.

## 8. Acceptance Criteria

- `data/splits/hyperkvasir_official_5fold/fold_0.csv` exists or is regenerated
  from the official split config.
- `configs/vit/experiment_matrix.yaml` contains the intended 11 frozen fold-0
  rows and no unexpected fine-tune rows.
- All three ViT fold-0 feature caches exist, are aligned, and have shape
  `(10662, 768)`, or `(manifest_row_count, 768)` if the manifest count differs.
- Every planned frozen ablation row runs successfully off-A100 and saves
  `metrics.json`.
- Every run directory contains `metrics.json`, `config.yaml`, `predictions.npz`,
  and `best.pt`.
- Every reported metric traces to
  `results/vit/runs/{experiment_id}/metrics.json`.
- Frozen ablation table is ranked by test macro-F1.
- Top candidates for Sprint 3 are identified, but not fine-tuned.
- `uv run pytest tests/` passes after any code/config/doc updates.
- No large artifacts are committed, and `src/models/backbones.py` remains
  untouched.

## 9. Commands to Run

### Setup and manifest check

```powershell
uv sync
Test-Path data/splits/hyperkvasir_official_5fold/fold_0.csv
uv run python scripts/make_splits.py --config configs/dataset/hyperkvasir_23class_official.yaml
```

Only run `make_splits.py` if `fold_0.csv` is missing.

### Cache shape inspection

```powershell
uv run python -c "from pathlib import Path; import csv, torch; manifest=Path('data/splits/hyperkvasir_official_5fold/fold_0.csv'); n=sum(1 for _ in csv.DictReader(manifest.open(newline=''))); cache_dir=Path('results/vit/feature_cache'); names=['vit_b','swin_t','beit_b']; print('manifest_rows', n); [print(name, torch.load(cache_dir / f'fold_0_{name}_features.pt', map_location='cpu', weights_only=True)['features'].shape) for name in names if (cache_dir / f'fold_0_{name}_features.pt').exists()]"
```

### Generate missing caches through the frozen train path

The current `scripts/train.py` frozen path auto-builds missing fold-0 caches from
`data/splits/{split_protocol}/fold_0.csv`. Use this path unless a cache-only
fold-manifest CLI is added later.

```powershell
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 02_single_swin_t_frozen_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 03_single_beit_b_frozen_official --device cuda
```

If CUDA is unavailable, replace `--device cuda` with `--device cpu`.

### Run all frozen ablation rows

```powershell
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 01_single_vit_b_frozen_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 02_single_swin_t_frozen_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 03_single_beit_b_frozen_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 04_pair_vit_b_swin_t_concat_frozen_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 05_pair_vit_b_beit_b_concat_frozen_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 06_pair_swin_t_beit_b_concat_frozen_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 07_pair_vit_b_swin_t_weighted_frozen_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 08_pair_vit_b_beit_b_weighted_frozen_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 09_pair_swin_t_beit_b_weighted_frozen_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 10_triple_concat_frozen_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 11_triple_weighted_frozen_official --device cuda
```

Skip rerunning a completed Sprint 1 row only if its `metrics.json`,
`config.yaml`, `predictions.npz`, and `best.pt` already exist and the metrics are
non-NaN.

### Inspect required run artifacts

```powershell
uv run python -c "from pathlib import Path; ids=['01_single_vit_b_frozen_official','02_single_swin_t_frozen_official','03_single_beit_b_frozen_official','04_pair_vit_b_swin_t_concat_frozen_official','05_pair_vit_b_beit_b_concat_frozen_official','06_pair_swin_t_beit_b_concat_frozen_official','07_pair_vit_b_swin_t_weighted_frozen_official','08_pair_vit_b_beit_b_weighted_frozen_official','09_pair_swin_t_beit_b_weighted_frozen_official','10_triple_concat_frozen_official','11_triple_weighted_frozen_official']; required=['metrics.json','config.yaml','predictions.npz','best.pt']; root=Path('results/vit/runs'); [print(i, {r:(root/i/r).exists() for r in required}) for i in ids]"
```

### Generate and inspect tables

```powershell
uv run python scripts/generate_report_tables.py --runs-dir results/vit/runs --output-dir results/vit/tables
uv run python -c "from pathlib import Path; import json; rows=[]; root=Path('results/vit/runs'); [rows.append((json.load((d/'metrics.json').open())['test']['macro_f1'], d.name, json.load((d/'metrics.json').open())['test'].get('accuracy'))) for d in root.iterdir() if d.is_dir() and (d/'metrics.json').exists()]; [print(f'{f1:.6f} {acc:.6f} {name}') for f1,name,acc in sorted(rows, reverse=True)]"
```

The generated table may need manual reordering or a minimal helper because the
current report-table script sorts by experiment id, not macro-F1.

### Tests and git hygiene

```powershell
uv run pytest tests/
git status --short
git diff -- src/models/backbones.py
```

## 10. Documentation Updates Required

- `docs/vit/experiment_log.md`:
  - commands run.
  - device name and off-A100 confirmation.
  - cache status and cache shapes.
  - result source paths.
  - `uv run pytest tests/` result.
  - top-ranked frozen configs for Sprint 3.
- `docs/vit/results_progress.md`:
  - ranked frozen ablation table with config, backbones, fusion, transfer, fold,
    accuracy, macro-F1, macro precision, macro recall, and metrics source.
- `docs/vit/known_issues.md`:
  - update only for observed issues, not anticipated risks.
- `docs/exec-plans/active/006-vit-frozen-ablation.md`:
  - keep active until Sprint 2 completion is explicitly approved.
- After Sprint 2 completion:
  - move this plan from `active/` to `completed/` only when explicitly approved.

