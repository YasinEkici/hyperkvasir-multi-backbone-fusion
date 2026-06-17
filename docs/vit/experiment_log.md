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
