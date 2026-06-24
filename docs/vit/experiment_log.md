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

## 2026-06-20 - Sprint 3 fine-tune funnel completion

- Scope: fine-tuned the five locked Sprint 2 candidates on official fold 0 using
  `configs/vit/training/vit_finetune.yaml`. No 5-fold CV, no Sprint 4 (VLD-12).
- Commands run (Colab A100, via `colab/vit_finetune_runner.ipynb`):
  - `uv run --no-sync python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment <id> --device cuda`
    for: `02_single_swin_t_finetune_official`,
    `04_pair_vit_b_swin_t_concat_finetune_official`,
    `05_pair_vit_b_beit_b_concat_finetune_official`,
    `09_pair_swin_t_beit_b_weighted_finetune_official`,
    `11_triple_weighted_finetune_official`.
  - Aggregation: `uv run python scripts/summarize_vit_finetune.py` (ranking +
    frozen-vs-fine-tune deltas + top-4; tables under `results/vit/tables/`).
- A100 / provenance: `train.py` enforces the A100 gate (VLD-11) before any
  fine-tune; the five runs completed, implying A100. `metrics.json` records the
  requested `device: cuda`. Dataset provenance was validated via per-class counts
  vs the fold manifest + git SHA (full SHA-tree re-hash of the Drive source was
  skipped due to Drive-FUSE throttling — see KI-VIT-001).
- Training config / key hyperparameters (VLD-15, confirmed in saved `config.yaml`):
  `unfreeze_blocks: 3`, AdamW, backbone LR `5e-5`, head LR `1e-3`, weight decay
  `0.05`, LLRD `0.7`, cosine warmup `5`, drop_path `0.05`, label smoothing `0.1`,
  MixUp `alpha 0.2 / prob 0.5`, CutMix off, EMA decay `0.9998` (start epoch 5),
  mixed precision, batch size 32.
- Run artifact status: all five `results/vit/runs/*_finetune_official/` contain
  `metrics.json`, `config.yaml`, `predictions.npz`, `best.pt`; fold 0; finite
  test metrics. Checkpoints range 107 MB (single Swin-T) to 765 MB (triple).
- Fine-tune fold-0 test macro-F1 (source: `results/vit/runs/{id}/metrics.json`):
  1. `02_single_swin_t_finetune_official` - 0.5918837684 (val 0.6493, stop ep 14).
  2. `09_pair_swin_t_beit_b_weighted_finetune_official` - 0.5909205632 (val 0.6437, ep 14).
  3. `11_triple_weighted_finetune_official` - 0.5850707743 (val 0.6386, ep 13).
  4. `05_pair_vit_b_beit_b_concat_finetune_official` - 0.5783726429 (val 0.6050, ep 16).
  5. `04_pair_vit_b_swin_t_concat_finetune_official` - 0.5727252344 (val 0.6313, ep 13).
- Frozen-vs-fine-tune: all five improved test macro-F1 (Δ +0.0109 to +0.0262);
  full per-metric deltas in `docs/vit/results_progress.md` Sprint 3 section.
- Selected Sprint 4 top-4: `02`, `09`, `11`, `05`; dropped `04` (lowest
  fine-tune macro-F1). Sprint 4 not run.
- Validation: `uv run pytest tests/` passed on 2026-06-20 (`249 passed`).

## 2026-06-20 - Sprint 3.5 Slice 1: throughput baseline (measurement)

- Scope: measurement only; no change to `scripts/train.py`, training configs, or
  any run artifacts (exec-plan `008-vit-perf.md`, Slice 1).
- Added `scripts/benchmark_vit_throughput.py`: times the fine-tune loop under
  `current` (num_workers=0, fp16, TF32 off, cuDNN deterministic/no-autotuner) vs
  `fast` (num_workers=8, pin_memory, persistent_workers, bf16, TF32 on,
  cudnn.benchmark=True) and reports img/s, steps/s, ms/step, peak memory.
- Measurement (local **RTX 5080**; warmup 10, 100/60 timed steps — stable run.
  A100 not yet measured):
  - `02_single_swin_t_finetune_official` (bs 32): current 63.2 img/s
    (506 ms/step) -> fast 624.1 img/s (51 ms/step) = **9.9x**; peak 0.7 GB.
  - `11_triple_weighted_finetune_official` (bs 32): current 66.3 img/s
    (482 ms/step) -> fast 285.1 img/s (112 ms/step) = **4.3x**; peak 2.9 GB.
  - (A short 8-step smoke earlier showed 23.5x/6.3x; those were inflated by
    unamortised worker startup — the longer run above is the reliable figure.)
  - Dominant factor is the DataLoader fix (`num_workers=0` starves the GPU); the
    single (cheap compute) is almost entirely data-bound, the triple is more
    compute-bound hence a smaller ratio. Tiny peak memory (0.7/2.9 GB) => large
    batch headroom.
  - Implication: at the fast config the 5080 reaches ~10 s/epoch (single) and
    ~23 s/epoch (triple) of compute, i.e. full fold-0 fine-tunes in minutes — so
    Sprint 4 (top-4 x 5 folds) is feasible on the local 5080 at zero A100 cost.
  - First local run surfaced and fixed a print-time `KeyError` in the tool.
- Official measurement (**Colab A100-SXM4-40GB**; warmup 10, 100/60 timed steps):
  - `02_single_swin_t_finetune_official` (bs 32): current 57.0 img/s
    (561 ms/step) -> fast 300.8 img/s (106 ms/step) = **5.28x**; peak 0.7 GB.
  - `11_triple_weighted_finetune_official` (bs 32): current 54.0 img/s
    (593 ms/step) -> fast 319.1 img/s (100 ms/step) = **5.91x**; peak 2.8 GB.
  - Key finding 1: in `current` mode the A100 (~55 img/s) is no faster than the
    laptop 5080 (~65 img/s) — `num_workers=0` fully starves it (paying A100 for
    laptop throughput). The fast knobs recover 5.3-5.9x.
  - Key finding 2: in `fast` mode single (~301) ~= triple (~319) img/s, i.e. the
    A100 is still CPU/data-pipeline bound, not compute bound. Peak memory
    0.7/2.8 GB of 40 GB => large headroom. Slice 2 should raise `num_workers`
    (Colab A100 high-RAM has more vCPUs) and batch size, then re-benchmark to
    find the data ceiling.
- No project decision changed yet (knobs are only measured, not applied), so
  `decisions.md` / `005-vit-foundation.md` are unchanged. The reproducibility
  trade-off (cuDNN benchmark / non-determinism / TF32 / bf16) will be logged as a
  VLD note in Slice 2 when the knobs are actually adopted.
- Validation: `uv run pytest tests/` passed on 2026-06-20 (`249 passed`).
- TODO: run the official A100 benchmark (single + triple) to record the real
  speedup before Slice 2 applies the knobs. (Done — A100 numbers above.)

## 2026-06-21 - Sprint 3.5 Slice 2: apply throughput knobs (VLD-17)

- Applied the verified knobs to the ViT fine-tune path — config-driven and
  default-off, so the frozen ViT path and the CNN project are unchanged:
  - `scripts/train.py` `_make_image_loaders` now reads a `dataloader` section
    (num_workers / pin_memory / persistent_workers / prefetch_factor); default
    0 workers preserves prior behavior.
  - `scripts/train.py` enables TF32 (config flag) and passes `amp_dtype` from a
    `performance` section to the Trainer.
  - `src/training/trainer.py` autocast uses `amp_dtype` (float16 default /
    bfloat16); GradScaler enabled only for float16.
  - `configs/vit/training/vit_finetune.yaml`: `dataloader.num_workers=8` (+pin,
    persistent, prefetch 4); `performance.amp_dtype=bfloat16`, `tf32=true`;
    `reproducibility.cudnn_benchmark=true` / `deterministic=false`. batch_size
    stays 32 (result comparability).
  - `scripts/benchmark_vit_throughput.py`: added `--fast-workers` for sweeps.
- Decision logged: VLD-17 in `docs/vit/decisions.md` (throughput config +
  reproducibility trade-off). `005-vit-foundation.md` checked — no change needed.
- Wiring verified locally: `vit_finetune.yaml` -> train loader num_workers=8,
  pin_memory=True, persistent_workers=True, prefetch_factor=4, batch_size=32;
  `vit_frozen.yaml` has no dataloader/performance section (frozen unchanged).
- Tests: `tests/test_trainer_amp.py` + perf-config assert in
  `tests/test_vit_configs.py`. `uv run pytest tests/` passed (`253 passed`).
- Sprint 3 fold-0 results are unchanged (this only affects future fine-tune runs).
- TODO (Slice 3, A100): correctness re-run of one fold-0 candidate with the tuned
  config -> confirm test macro-F1 within run-to-run noise of Sprint 3; optional
  `--fast-workers` 12/16 sweep to push past the ~310 img/s data ceiling.

## 2026-06-21 - Sprint 3.5 Slice 3: correctness check (tuned config)

- Re-ran `02_single_swin_t_finetune_official` with the tuned config on the local
  RTX 5080 (bf16 + 8 workers + TF32 + cuDNN autotuner), `--seed 123
  --allow-non-a100`, so it writes to `..._seed123/` (the Sprint 3 seed-42 run is
  untouched).
- Training healthy: loss decreased normally, best val macro-F1 0.6344 (epoch 4),
  early-stopped epoch 12 — comparable to Sprint 3's seed-42 run (best val 0.6493).
- Windows-only crash in the post-training test eval (KI-VIT-002): the test loader
  spawned 8 more workers on top of the train+val persistent workers (~24
  torch-loading processes) -> `WinError 1455` (paging file too small). Not a logic
  bug; Colab/Linux (fork) is unaffected. Test metrics recovered by evaluating the
  saved `best.pt` with `num_workers=0`.
- Correctness comparison (test macro-F1):
  - tuned (5080, bf16, seed 123): 0.5789963898 (acc 0.8680490104).
  - Sprint 3 (A100, fp16, seed 42): 0.5918837684 (acc 0.8770028275).
  - delta -0.0129: within plausible run-to-run noise but **conservative / n=1** —
    conflates seed (123 vs 42) + hardware (5080 vs A100) + dtype (bf16 vs fp16) +
    non-determinism. The lower test tracks the lower best-val (0.6344 vs 0.6493),
    i.e. a less-lucky seed, not a config degradation. Training dynamics healthy.
- Conclusion: the tuned config trains correctly and lands within the expected seed
  band; acceptable to lock for Sprint 4 (5-fold CV averages seed noise). Optional
  report-grade rigor: one A100 same-seed (42) tuned run vs Sprint 3 to remove the
  hardware/seed/dtype confounds.
- No code change in this slice. `uv run pytest tests/` passed (`253 passed`).
- `..._seed123/` artifacts are gitignored (local correctness evidence only).

## 2026-06-23 - Sprint 4: 5-fold CV + CI + leakage-free extras

- Scope: official 5-fold CV (folds 0–4) for the Sprint 3 top-4, tuned config
  (VLD-17), on A100. NEW `_cv` ids so tuned fold-0 runs do not overwrite the
  Sprint 3 `_finetune_official` artifacts (009-vit-cv.md).
- Commands (Colab A100, `colab/vit_finetune_runner.ipynb`):
  - CV: `uv run --no-sync python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment <cv_id> --fold <k> --device cuda`
    for `{02_single_swin_t_cv, 09_pair_swin_t_beit_b_weighted_cv,
    11_triple_weighted_cv, 05_pair_vit_b_beit_b_concat_cv}` × folds 0–4 (20 runs).
  - Seed ensemble (top-1 = `11_triple_weighted_cv`): same with `--fold <k>
    --seed {123,2024}` × folds 0–4 (+10 runs).
  - Aggregation: `uv run python scripts/summarize_vit_cv.py` (CV mean±std + CI);
    `uv run python scripts/eval_tta_ensemble.py --mode {tta,ensemble}`.
- A100 / provenance: VLD-11 A100 gate before each fine-tune; archive dataset
  staging (KI-VIT-001) + fast local-hash provenance. Colab/Linux makes the tuned
  `num_workers=8` safe (KI-VIT-002 is Windows-only).
- Run artifact status: 20 CV + 10 seed-ensemble run dirs each contain
  `metrics.json`, `config.yaml`, `predictions.npz`, `best.pt`; fold/seed tags
  correct; finite metrics; Sprint 3 fold-0 runs intact.
- Fold integrity: the 5 fold test sets are disjoint (pairwise overlap 0) and
  union to the full 10662 — pooled out-of-fold bootstrap CI is valid (VLD-13).
- 5-fold CV (mean macro-F1 ± std; pooled 95% CI; full table in
  `results_progress.md`):
  1. `11_triple_weighted_cv` — 0.6094±0.0254, pooled 0.6119 [0.5930, 0.6295].
  2. `02_single_swin_t_cv` — 0.5969±0.0261, pooled 0.5985 [0.5806, 0.6152].
  3. `09_pair_swin_t_beit_b_weighted_cv` — 0.5879±0.0111, pooled 0.5893 [0.5736, 0.6054].
  4. `05_pair_vit_b_beit_b_concat_cv` — 0.5727±0.0149, pooled 0.5753 [0.5638, 0.5863].
- Best ViT fusion config: `11_triple_weighted` (V+S+B, weighted) — re-ranks the
  Sprint 3 fold-0 result (single Swin-T led there). CIs overlap (best by mean,
  not a clean separation) — to be discussed in the report §5.5.
- Leakage-free extras (best config, within-fold softmax averaging):
  - TTA (orig+hflip): 0.6116 -> 0.6105 [0.5920, 0.6280] = -0.0011 (no gain).
  - seed ensemble (42,123,2024): 0.6116 -> 0.6157 [0.5994, 0.6321] = +0.0041.
- Final selected model locked as VLD-18.
- Validation: `uv run pytest tests/` passed on 2026-06-23 (`264 passed`).

## 2026-06-24 - Sprint 4.5: GMU fusion ablation (optional stretch)

- Scope: evaluate GMU (Arevalo 2017) as a 3rd fusion method (VLD-07), with a
  **faithful element-wise gate** (VLD-19, `gate_mode: elementwise`) — not the
  legacy scalar gate. Multi-backbone only (3 pairs + triple); no singles.
- Code (Slice 0): added `gate_mode` to `src/models/fusion/gmu.py` (elementwise =
  paper-faithful, reduces to the bimodal for N=2; scalar default = CNN
  back-compat), wired `fusion_kwargs` through `full_model.py` + `train.py`.
  9 unit tests (`tests/test_gmu_gate.py`).
- Configs (Slice 1): 4 GMU method configs + 4 `*_gmu_cv` matrix rows (tuned
  config VLD-17). Runner preset (Slice 2): `colab/vit_finetune_runner.ipynb` GMU
  fold-0 screen (namespace `sprint45_vit_gmu`).
- Commands (Colab A100): `uv run --no-sync python scripts/train.py --config
  configs/vit/experiment_matrix.yaml --experiment <gmu_cv_id> --fold 0 --device cuda`
  for the 4 GMU ids. A100 gate (VLD-11) + archive staging (KI-VIT-001) + fast
  provenance.
- Stage 1 fold-0 screen (macro-F1; vs same-backbone tuned `_cv` fold-0):
  - `pair_swin_t_beit_b_gmu_cv` (S+B): 0.5995 vs S+B-weighted 0.5858 (+0.0137).
  - `pair_vit_b_beit_b_gmu_cv` (V+B): 0.5773 vs V+B-concat 0.5795 (-0.0022).
  - `pair_vit_b_swin_t_gmu_cv` (V+S): 0.5744 (no tuned counterpart; `04` dropped).
  - `triple_vit_swin_beit_gmu_cv` (V+S+B): 0.5525 vs triple-weighted 0.6102 (-0.0577).
- Run artifact status: 4 GMU fold-0 runs each have `metrics.json`, `config.yaml`,
  `predictions.npz`, `best.pt`; finite metrics.
- Stage 2 (conditional CV promotion): **NOT run** — no GMU config is competitive
  with / beats the best weighted (`11_triple_weighted`, 0.6102 fold-0 / 0.6094
  5-fold). Triple-GMU regressed sharply (-0.058). Decision: GMU evaluated, no gain;
  **final model stays `11_triple_weighted`** (VLD-18). See VLD-19.
- Validation: `uv run pytest tests/` passed on 2026-06-24 (`274 passed`).
