# Experiment Log — ViT Fusion

Record successful runs and environment notes here (chronological, append-only).
Each reportable run must also have a `metrics.json` under `results/vit/runs/{id}/`
and trace to a resolved config (provenance gate, CNN D-09 reused).

## 2026-06-16 — Project scaffolding (Sprint 0 / setup)

- `docs/vit/` created: `final_assignment.md`, `project_structure.md`,
  `project_plan.md`, `decisions.md` (VLD-01…16), `results_progress.md`,
  `known_issues.md`, `environment.md`.
- References: 10 ViT folders added (`INDEX.md` Section H, #32–41); `dropblock`
  and `swa` removed. timm strings verified loadable (768-d) on 2026-06-16.
- `AGENTS.md` ViT addendum added (timm/LayerNorm/path overrides).
- No code or runs yet — implementation starts in Sprint 1.

## 2026-06-17 - Sprint 1 Slice 2 inputs

- Implemented ViT-aware frozen feature caching: ViT aliases route to
  `ViTFeatureExtractor`, use bicubic resize plus center crop, and the extraction
  script defaults ViT caches to `results/vit/feature_cache/`.
- Added frozen `configs/vit/` method/training/matrix skeleton for single, pair,
  and triple ViT configs with concat/weighted fusion where applicable.
- Validation: `uv run pytest tests/` passed (212 passed). No experiment metrics
  were generated; the single ViT-B frozen smoke run remains pending.

## 2026-06-17 - Sprint 1 Slice 3 smoke run

- Command: `uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 01_single_vit_b_frozen_official --device cuda`.
- Device: NVIDIA GeForce RTX 5080 Laptop GPU (off-A100; 0 A100 units).
- Cache: `results/vit/feature_cache/fold_0_vit_b_features.pt`, shape
  `10662 x 768`, labels/paths/indices all length `10662`.
- Run: `results/vit/runs/01_single_vit_b_frozen_official/metrics.json`.
- Result: test accuracy `0.8416588124`, test macro-F1 `0.5575947093`,
  best validation macro-F1 `0.5683442025`, early stopped at epoch 11.
- Validation: `uv run pytest tests/` passed (`217 passed`).

## 2026-06-19 - Sprint 2 frozen ablation completion

- Scope: completed the fold-0 frozen ablation matrix from
  `configs/vit/experiment_matrix.yaml` using `configs/vit/training/vit_frozen.yaml`
  (`unfreeze_blocks: 0`, `ema.enabled: false`).
- Commands run:
  - Cache/run path: `uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment <experiment_id> --device cuda`
    for all 11 Sprint 2 frozen rows:
    `01_single_vit_b_frozen_official`,
    `02_single_swin_t_frozen_official`,
    `03_single_beit_b_frozen_official`,
    `04_pair_vit_b_swin_t_concat_frozen_official`,
    `05_pair_vit_b_beit_b_concat_frozen_official`,
    `06_pair_swin_t_beit_b_concat_frozen_official`,
    `07_pair_vit_b_swin_t_weighted_frozen_official`,
    `08_pair_vit_b_beit_b_weighted_frozen_official`,
    `09_pair_swin_t_beit_b_weighted_frozen_official`,
    `10_triple_concat_frozen_official`,
    `11_triple_weighted_frozen_official`.
  - Table helper: `uv run python scripts/generate_report_tables.py --runs-dir results/vit/runs --output-dir results/vit/tables`.
  - Ranked table generation: local metrics aggregation over
    `results/vit/runs/*/metrics.json`, sorted by `test.macro_f1` descending.
  - Slice 5 validation: `uv run python -` validation script checked manifest
    counts, cache shapes, required run artifacts, fold IDs, device fields, and
    local CUDA device name.
- Device: NVIDIA GeForce RTX 5080 Laptop GPU (`cuda`, off-A100; 0 A100 units).
- Fold manifest: `data/splits/hyperkvasir_official_5fold/fold_0.csv` has
  `10662` rows (`6418` train / `2122` val / `2122` test).
- Cache status:
  - `results/vit/feature_cache/fold_0_vit_b_features.pt`: `(10662, 768)`,
    labels/paths/indices all length `10662`, backbone `vit_b`.
  - `results/vit/feature_cache/fold_0_swin_t_features.pt`: `(10662, 768)`,
    labels/paths/indices all length `10662`, backbone `swin_t`.
  - `results/vit/feature_cache/fold_0_beit_b_features.pt`: `(10662, 768)`,
    labels/paths/indices all length `10662`, backbone `beit_b`.
- Run artifact status: all 11 run directories under `results/vit/runs/` contain
  `metrics.json`, `config.yaml`, `predictions.npz`, and `best.pt`; every run is
  fold 0 and reports finite test macro-F1.
- Ranked table artifacts:
  - `results/vit/tables/frozen_ablation_ranked.csv`.
  - `results/vit/tables/frozen_ablation_ranked.md`.
- Top Sprint 3 candidates by fold-0 frozen test macro-F1:
  1. `02_single_swin_t_frozen_official` - S, none, macro-F1 `0.5759686248`.
  2. `11_triple_weighted_frozen_official` - V+S+B, weighted, macro-F1 `0.5648876413`.
  3. `09_pair_swin_t_beit_b_weighted_frozen_official` - S+B, weighted, macro-F1 `0.5647640280`.
  4. `04_pair_vit_b_swin_t_concat_frozen_official` - V+S, concat, macro-F1 `0.5618023184`.
  5. `05_pair_vit_b_beit_b_concat_frozen_official` - V+B, concat, macro-F1 `0.5613159913`.
- No Sprint 3 fine-tuning was run.
- Validation: `uv run pytest tests/` passed on 2026-06-19 (`217 passed`).
