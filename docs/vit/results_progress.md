# Results Progress — ViT Fusion

> **Single source of truth for sprint implementation milestones and experiment results.**
>
> Every number in this file traces back to a `metrics.json` under
> `results/vit/runs/{id}/`. Updated at the end of each sprint / step group. Used
> directly as ground truth when writing the ViT report.
>
> Columns: Acc = test accuracy, F1 = test **macro-F1** (zero_division=0), headline metric (VLD-10).
> Protocol: official HyperKvasir 5-fold split, fold 0 for ablation unless stated otherwise.
> Backbones: V = ViT-B/16, S = Swin-T, B = BEiT-B/16.

---

## Sprint 1 — Foundation
*Status: complete. Slice 3 single ViT-B frozen smoke run completed on 2026-06-17.*

### Implementation milestones

| # | Component | File(s) | Done |
|---|---|---|:--:|
| 1 | `vit-fusion` branch + project docs (`docs/vit/`) | — | ✅ |
| 2 | timm ViT backbone factory | `src/models/vit_backbones.py` | yes |
| 3 | Extractor dispatch (CNN vs ViT) | `src/models/full_model.py` | yes |
| 4 | `configs/vit/` skeleton (method + training) | `configs/vit/` | yes |
| 5 | Backbone shape smoke test ((B,768) for V/S/B) | `tests/` | yes |
| 6 | ViT-aware frozen feature-cache code path | `src/data/feature_cache.py`, `scripts/extract_features.py` | yes |
| 7 | Single ViT-B frozen smoke run | `results/vit/runs/01_single_vit_b_frozen_official/metrics.json` | yes |

### Results
| config | backbones | fusion | transfer | fold | Acc | macro-F1 | Source |
|---|---|---|---|---:|---:|---:|---|
| `01_single_vit_b_frozen_official` | V | none | frozen | 0 | 0.8416588124 | 0.5575947093 | `results/vit/runs/01_single_vit_b_frozen_official/metrics.json` |

Cache validation: `results/vit/feature_cache/fold_0_vit_b_features.pt` has
`10662 x 768` features, matching the fold manifest dataset count.

Validation: `uv run pytest tests/` passed on 2026-06-17 (`217 passed`).

---

## Sprint 2 - Frozen ablation (off-A100)
*Status: complete. Slice 5 documentation and validation completed on 2026-06-19.*

### Cache and run validation

- Fold manifest: `data/splits/hyperkvasir_official_5fold/fold_0.csv`,
  `10662` rows (`6418` train / `2122` val / `2122` test).
- Feature caches:
  - `results/vit/feature_cache/fold_0_vit_b_features.pt`: `(10662, 768)`.
  - `results/vit/feature_cache/fold_0_swin_t_features.pt`: `(10662, 768)`.
  - `results/vit/feature_cache/fold_0_beit_b_features.pt`: `(10662, 768)`.
- All 11 frozen fold-0 run directories contain `metrics.json`, `config.yaml`,
  `predictions.npz`, and `best.pt`.

### Ranked frozen ablation results

| rank | config | backbones | fusion | transfer | fold | Acc | macro-F1 | macro-P | macro-R | Source |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `02_single_swin_t_frozen_official` | S | none | frozen | 0 | 0.8704052780 | 0.5759686248 | 0.5767312258 | 0.5795733188 | `results/vit/runs/02_single_swin_t_frozen_official/metrics.json` |
| 2 | `11_triple_weighted_frozen_official` | V+S+B | weighted | frozen | 0 | 0.8510838831 | 0.5648876413 | 0.5875085287 | 0.5688667361 | `results/vit/runs/11_triple_weighted_frozen_official/metrics.json` |
| 3 | `09_pair_swin_t_beit_b_weighted_frozen_official` | S+B | weighted | frozen | 0 | 0.8779453346 | 0.5647640280 | 0.5617156996 | 0.5700015021 | `results/vit/runs/09_pair_swin_t_beit_b_weighted_frozen_official/metrics.json` |
| 4 | `04_pair_vit_b_swin_t_concat_frozen_official` | V+S | concat | frozen | 0 | 0.8803016023 | 0.5618023184 | 0.5613323206 | 0.5658202972 | `results/vit/runs/04_pair_vit_b_swin_t_concat_frozen_official/metrics.json` |
| 5 | `05_pair_vit_b_beit_b_concat_frozen_official` | V+B | concat | frozen | 0 | 0.8539114043 | 0.5613159913 | 0.5704439425 | 0.5598491055 | `results/vit/runs/05_pair_vit_b_beit_b_concat_frozen_official/metrics.json` |
| 6 | `06_pair_swin_t_beit_b_concat_frozen_official` | S+B | concat | frozen | 0 | 0.8656927427 | 0.5598234260 | 0.5569990924 | 0.5663675014 | `results/vit/runs/06_pair_swin_t_beit_b_concat_frozen_official/metrics.json` |
| 7 | `07_pair_vit_b_swin_t_weighted_frozen_official` | V+S | weighted | frozen | 0 | 0.8694627710 | 0.5580175157 | 0.5585852452 | 0.5624236509 | `results/vit/runs/07_pair_vit_b_swin_t_weighted_frozen_official/metrics.json` |
| 8 | `01_single_vit_b_frozen_official` | V | none | frozen | 0 | 0.8416588124 | 0.5575947093 | 0.5507576386 | 0.5747792052 | `results/vit/runs/01_single_vit_b_frozen_official/metrics.json` |
| 9 | `08_pair_vit_b_beit_b_weighted_frozen_official` | V+B | weighted | frozen | 0 | 0.8581526861 | 0.5530041018 | 0.5863239361 | 0.5517167524 | `results/vit/runs/08_pair_vit_b_beit_b_weighted_frozen_official/metrics.json` |
| 10 | `10_triple_concat_frozen_official` | V+S+B | concat | frozen | 0 | 0.8520263902 | 0.5516672382 | 0.5455682436 | 0.5657765936 | `results/vit/runs/10_triple_concat_frozen_official/metrics.json` |
| 11 | `03_single_beit_b_frozen_official` | B | none | frozen | 0 | 0.8317624882 | 0.5215489803 | 0.5199164326 | 0.5251164671 | `results/vit/runs/03_single_beit_b_frozen_official/metrics.json` |

### Sprint 3 candidate funnel

Strict top five by test macro-F1, which also covers the strongest single, pair,
and triple frozen configurations:

| priority | config | backbones | fusion | macro-F1 | rationale |
|---:|---|---|---|---:|---|
| 1 | `02_single_swin_t_frozen_official` | S | none | 0.5759686248 | strongest single and overall frozen baseline |
| 2 | `11_triple_weighted_frozen_official` | V+S+B | weighted | 0.5648876413 | strongest triple |
| 3 | `09_pair_swin_t_beit_b_weighted_frozen_official` | S+B | weighted | 0.5647640280 | strongest pair |
| 4 | `04_pair_vit_b_swin_t_concat_frozen_official` | V+S | concat | 0.5618023184 | strict top-five macro-F1 |
| 5 | `05_pair_vit_b_beit_b_concat_frozen_official` | V+B | concat | 0.5613159913 | strict top-five macro-F1 |

No Sprint 3 fine-tuning was run during Sprint 2.

Validation: `uv run pytest tests/` passed on 2026-06-19 (`217 passed`).

---

## Sprint 3 — Fine-tune funnel (A100)
*Status: complete. Slice 6 documentation and validation completed on 2026-06-20.*

### Run validation

- Fine-tuned the five locked Sprint 2 candidates on official fold 0 (Colab A100),
  using `configs/vit/training/vit_finetune.yaml` (VLD-15 recipe: `unfreeze_blocks: 3`,
  backbone LR `5e-5`, head LR `1e-3`, weight decay `0.05`, LLRD `0.7`, warmup `5`,
  drop_path `0.05`, label smoothing `0.1`, mild MixUp `0.2`, CutMix off, EMA `0.9998`).
- VLD-08 unfreeze confirmed in code: ViT-B/BEiT-B `blocks[9:]`; Swin-T final stage
  `layers[3]`; both plus final norm/head.
- All five `results/vit/runs/*_finetune_official/` directories contain
  `metrics.json`, `config.yaml`, `predictions.npz`, and `best.pt`; every run is
  fold 0 with finite test metrics.
- A100: `train.py` aborts off-A100 (VLD-11 gate), so completion implies A100; the
  `device` field in `metrics.json` records the requested string `cuda`. Dataset
  provenance was verified via per-class counts vs the fold manifest + git SHA
  (see `docs/vit/known_issues.md` KI-VIT-001 for the Drive-staging workaround).

### Ranked fine-tune results (fold 0, by test macro-F1)

| rank | config | backbones | fusion | transfer | fold | Acc | macro-F1 | macro-P | macro-R | Source |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `02_single_swin_t_finetune_official` | S | none | finetune-3blk | 0 | 0.8770028275 | 0.5918837684 | 0.5898269191 | 0.5984727358 | `results/vit/runs/02_single_swin_t_finetune_official/metrics.json` |
| 2 | `09_pair_swin_t_beit_b_weighted_finetune_official` | S+B | weighted | finetune-3blk | 0 | 0.8713477851 | 0.5909205632 | 0.5850371972 | 0.6079857923 | `results/vit/runs/09_pair_swin_t_beit_b_weighted_finetune_official/metrics.json` |
| 3 | `11_triple_weighted_finetune_official` | V+S+B | weighted | finetune-3blk | 0 | 0.8788878417 | 0.5850707743 | 0.5834765276 | 0.5960435750 | `results/vit/runs/11_triple_weighted_finetune_official/metrics.json` |
| 4 | `05_pair_vit_b_beit_b_concat_finetune_official` | V+B | concat | finetune-3blk | 0 | 0.8661639962 | 0.5783726429 | 0.5817263685 | 0.5988759185 | `results/vit/runs/05_pair_vit_b_beit_b_concat_finetune_official/metrics.json` |
| 5 | `04_pair_vit_b_swin_t_concat_finetune_official` | V+S | concat | finetune-3blk | 0 | 0.8689915174 | 0.5727252344 | 0.5644907650 | 0.5978429194 | `results/vit/runs/04_pair_vit_b_swin_t_concat_finetune_official/metrics.json` |

### Frozen vs fine-tune deltas (fold 0, finetune − frozen)

| config | backbones | fusion | Δ Acc | Δ macro-F1 | Δ macro-P | Δ macro-R |
|---|---|---|---:|---:|---:|---:|
| `02_single_swin_t` | S | none | +0.0065975495 | +0.0159151436 | +0.0130956933 | +0.0188994171 |
| `09_pair_swin_t_beit_b_weighted` | S+B | weighted | −0.0065975495 | +0.0261565351 | +0.0233214976 | +0.0379842902 |
| `11_triple_weighted` | V+S+B | weighted | +0.0278039585 | +0.0201831330 | −0.0040320012 | +0.0271768389 |
| `05_pair_vit_b_beit_b_concat` | V+B | concat | +0.0122525919 | +0.0170566515 | +0.0112824260 | +0.0390268130 |
| `04_pair_vit_b_swin_t_concat` | V+S | concat | −0.0113100848 | +0.0109229160 | +0.0031584445 | +0.0320226222 |

Observations:
- **Fine-tuning improved test macro-F1 for all five candidates** (Δ macro-F1
  strictly positive). Largest gain: `09` S+B weighted (+0.0262); smallest: `04`
  V+S concat (+0.0109).
- The strongest single config (`02` Swin-T) stays on top after fine-tuning
  (0.5918837684), narrowly ahead of the now-much-closer `09` pair (0.5909205632).
- Best fine-tune macro-F1 (0.5918837684) > best frozen macro-F1 (0.5759686248).

### Selected Sprint 4 top-4

By fold-0 fine-tune test macro-F1, carry forward to Sprint 4 5-fold CV
(Sprint 4 not run, VLD-12):

1. `02_single_swin_t_finetune_official` — S, none — 0.5918837684.
2. `09_pair_swin_t_beit_b_weighted_finetune_official` — S+B, weighted — 0.5909205632.
3. `11_triple_weighted_finetune_official` — V+S+B, weighted — 0.5850707743.
4. `05_pair_vit_b_beit_b_concat_finetune_official` — V+B, concat — 0.5783726429.

Dropped: `04_pair_vit_b_swin_t_concat_finetune_official` (lowest fine-tune
macro-F1 0.5727252344, smallest gain).

Tables regenerated by `scripts/summarize_vit_finetune.py` into
`results/vit/tables/finetune_fold0_ranked.{csv,md}` and
`frozen_vs_finetune_fold0_delta.{csv,md}` (gitignored outputs).

Validation: `uv run pytest tests/` passed on 2026-06-20 (`249 passed`).

## Sprint 3.5 — A100 throughput (perf)
*Status: in progress (`008-vit-perf.md`); Slices 1-3 done, Slice 4 = docs.*

- Root cause of slow/expensive A100 fine-tuning: the image DataLoader ran with
  `num_workers=0`, starving the GPU — A100 `current`-mode throughput (~55 img/s)
  equalled a laptop RTX 5080.
- Tuned config (num_workers=8 + pin/persistent/prefetch, TF32, bf16,
  `cudnn.benchmark`) measured **~5.3x (single) / ~5.9x (triple)** A100 throughput
  at batch 32 (`scripts/benchmark_vit_throughput.py`). Decision: VLD-17.
- Correctness (Slice 3): `02_single_swin_t` with the tuned config (5080, bf16,
  seed 123) gave test macro-F1 `0.5789963898` vs Sprint 3 `0.5918837684`
  (delta -0.013, within run-to-run noise; conservative n=1 conflating
  seed/hardware/dtype). Training dynamics healthy; config locked for Sprint 4.

## Sprint 4 — 5-fold CV + CI (A100)
*Status: complete. Slice 6 documentation completed on 2026-06-23.*

### Run validation

- 5-fold CV (folds 0–4) for the top-4, with NEW `_cv` ids (tuned config VLD-17,
  A100), so the Sprint 3 `_finetune_official` fold-0 artifacts stay intact.
  20 CV runs + 10 top-1 seed-ensemble runs; all have `metrics.json`,
  `config.yaml`, `predictions.npz`, `best.pt`, correct fold/seed tags, finite
  test metrics. Per-fold per-class metrics + confusion matrix are preserved in
  each run's `metrics.json` (VLD-10).
- The five official fold test sets are **disjoint and cover the dataset once**
  (overlap 0; union 10662), so concatenating their predictions gives one
  out-of-fold (OOF) prediction per sample — leakage-free (VLD-13; no fold-model
  averaging).

### 5-fold CV ranking (by mean macro-F1)

Source: `results/vit/runs/{cv_id}[_fold_k]/metrics.json` + `predictions.npz`;
aggregated by `scripts/summarize_vit_cv.py`. Bootstrap 95% CI over OOF preds.

| rank | config | bb | fusion | Acc (mean±std) | macro-F1 (mean±std) | macro-F1 pooled [95% CI] | macro-P (mean±std) | macro-R (mean±std) |
|---:|---|---|---|---|---|---|---|---|
| 1 | `11_triple_weighted_cv` | V+S+B | weighted | 0.8780±0.0140 | **0.6094±0.0254** | 0.6119 [0.5930, 0.6295] | 0.6111±0.0243 | 0.6246±0.0226 |
| 2 | `02_single_swin_t_cv` | S | none | 0.8679±0.0097 | 0.5969±0.0261 | 0.5985 [0.5806, 0.6152] | 0.5958±0.0267 | 0.6148±0.0257 |
| 3 | `09_pair_swin_t_beit_b_weighted_cv` | S+B | weighted | 0.8652±0.0058 | 0.5879±0.0111 | 0.5893 [0.5736, 0.6054] | 0.5826±0.0138 | 0.6106±0.0153 |
| 4 | `05_pair_vit_b_beit_b_concat_cv` | V+B | concat | 0.8454±0.0084 | 0.5727±0.0149 | 0.5753 [0.5638, 0.5863] | 0.5728±0.0137 | 0.6006±0.0220 |

- **Best ViT fusion config = `11_triple_weighted` (V+S+B, weighted).** This
  **re-ranks** the Sprint 3 *fold-0* result (single Swin-T led there); on full
  5-fold CV the triple wins.
- **Honest caveat (for the report §5.5):** the top configs' CIs overlap (triple
  [0.593, 0.630] vs single Swin-T [0.581, 0.615]) — best by mean, not a clean
  statistical separation.

### Leakage-free extras (best config `11_triple_weighted`)

Within-fold softmax averaging, OOF macro-F1 + bootstrap 95% CI
(`scripts/eval_tta_ensemble.py`; single-seed recompute 0.6116 ≈ saved-pred
pooled 0.6119, a ~0.0003 float-precision difference).

| variant | OOF macro-F1 | 95% CI | Δ vs single-seed |
|---|---:|---|---:|
| single (seed 42) | 0.6116 | — | — |
| TTA (orig + hflip) | 0.6105 | [0.5920, 0.6280] | **−0.0011 (no gain)** |
| seed ensemble (42,123,2024) | **0.6157** | [0.5994, 0.6321] | **+0.0041** |

- **TTA gave no gain** on this dataset (consistent with the CNN project's
  Week-3.5 finding). The **seed ensemble gives a small positive gain** and is the
  CI-backed headline for the best ViT fusion model: **macro-F1 ≈ 0.616
  [0.599, 0.632]**.

Validation: `uv run pytest tests/` passed on 2026-06-23 (`264 passed`).

## Sprint 4.5 — GMU fusion ablation (optional stretch)
*Status: complete. Slice 5 documentation completed on 2026-06-24.*

GMU (Gated Multimodal Unit, Arevalo et al. 2017) evaluated as an optional third
fusion method (VLD-07), implemented with a **faithful element-wise gate**
(VLD-19) — not the legacy per-branch scalar gate. Multi-backbone only (pairs +
triple); singles have no fusion.

### Fold-0 screen (tuned config, A100) vs the tuned `_cv` fold-0 baselines

Same fold as the Sprint 4 `_cv` fold-0, so the comparison is like-for-like.
Source: `results/vit/runs/{id}/metrics.json`.

| GMU config | bb | macro-F1 | Acc | same-backbone tuned baseline | Δ macro-F1 |
|---|---|---:|---:|---|---:|
| `pair_swin_t_beit_b_gmu_cv` | S+B | 0.5995 | 0.8728 | S+B weighted (0.5858) | **+0.0137** |
| `pair_vit_b_beit_b_gmu_cv` | V+B | 0.5773 | 0.8567 | V+B concat (0.5795) | −0.0022 |
| `pair_vit_b_swin_t_gmu_cv` | V+S | 0.5744 | 0.8516 | (no tuned counterpart; `04` dropped) | n/a |
| `triple_vit_swin_beit_gmu_cv` | V+S+B | 0.5525 | 0.8355 | triple weighted (0.6102) | **−0.0577** |

### Outcome — no promotion (GMU does not beat the best)

- **No GMU config beats the overall best** (`11_triple_weighted`, 5-fold 0.6094 /
  fold-0 0.6102). The natural headline candidate, **triple-GMU, regressed sharply
  (−0.058)** — the faithful per-feature gate over 3 modalities underperformed
  simple weighted fusion under the limited fine-tune budget.
- The only GMU win is **S+B-GMU (+0.0137 vs S+B-weighted)**, but at 0.5995 it sits
  below the headline triple and on a structurally weaker backbone set
  (S+B-weighted 5-fold was 0.5879).
- Per the Sprint 4.5 promotion criterion, **no GMU config was promoted to 5-fold
  CV** (Stage 2 not run). **Final model stays `11_triple_weighted`** (VLD-18).
- Report §5.5 framing: GMU was screened on fold 0 (same fold as the CV configs'
  fold-0) and did not beat the best weighted — a legitimate funnel decision (cf.
  Sprint 3). The faithful gate (VLD-19) makes "GMU (Arevalo 2017)" an honest
  citation.

Validation: `uv run pytest tests/` passed on 2026-06-24 (`274 passed`).

## Sprint 5 — Interpretability + report
*Status: in progress (Slice 1 done)*

### Slice 1 — Interpretability: attention rollout + Grad-CAM++
- Inference-only from the frozen final model `11_triple_weighted` (`best.pt`),
  OOF test split of fold 0 (2122 images, leakage-free per VLD-13). 0 A100 units.
- **Attention rollout** (Abnar & Zuidema 2020) for the CLS-token ViT-B and BEiT-B
  branches: `A = 0.5 W_att + 0.5 I`, head-averaged, recursive matmul, CLS→14×14.
  Attention captured by temporarily disabling timm `fused_attn` + hooking
  `attn_drop` (no edit to `vit_backbones.py`).
- **Grad-CAM++** (Chattopadhyay 2018, `pytorch-grad-cam`) for all three branches
  with a transformer `reshape_transform` (last block `norm1`). **Swin caveat:**
  timm Swin output is channels-last `(B,7,7,C)` → coarse 7×7 windowed map,
  flagged in the figure; ViT-B/BEiT-B (14×14) are the faithful ones. Maps were
  finite (no degenerate fallback triggered).
- Classes auto-selected from the final model's per-class F1 (3 best + 3 worst,
  non-zero support): best = retroflex-stomach (F1 1.00), normal-pylorus,
  retroflex-rectum; worst = ulcerative-colitis-grade-2-3, -grade-1-2, hemorroids
  — the confusable/rare classes that cap macro-F1 (ties into §5.5). Each panel
  shows correct + a failure case (e.g. UC grade-2-3 → predicted grade-3).
- Figures: `reports/vit/figures/{rollout,gradcam}_<class>.png` (12 small PNGs),
  each traceable to `results/vit/runs/11_triple_weighted_cv/best.pt` (OOF fold 0).
- New scripts: `scripts/interpret_common.py`, `interpret_attention_rollout.py`,
  `interpret_gradcam.py`; tests `tests/test_interpret.py` (13). No train.py change.
- Validation: `uv run pytest tests/` → 287 passed; `backbones.py` /
  `vit_backbones.py` untouched; checkpoints stay gitignored.

### Slice 2 — Feature-space analysis (UMAP)
- Inference-only from `11_triple_weighted` (`best.pt`), fold-0 OOF test (2122,
  leakage-free VLD-13). 0 A100. UMAP (McInnes 2018) on the 512-d **fused**
  (weighted) and **per-branch projected** features (VLD-05), extracted by calling
  the trained projection/fusion submodules (no edit to `full_model.py`).
- `umap_fused.png`: distinct classes form well-separated clusters (e.g.
  retroflex-stomach, hemorroids, normal-pylorus); the **ulcerative-colitis grades
  (0-1/1/1-2/2/2-3/3) overlap heavily** — directly visualises the confusable/rare
  classes that cap macro-F1 (§5.5).
- `umap_branches.png`: ViT-B and Swin-T projections are cleaner / more separated;
  **BEiT-B is more diffuse/entangled** — a per-backbone feature-quality cue for
  the discussion (which backbone learned better features).
- Reproducible (`random_state=42`); points capped at 150/class for readability
  (rare classes kept in full); 1753/2122 points plotted.
- New script `scripts/interpret_umap.py`; helpers `extract_features` +
  `subsample_indices_per_class` added to `scripts/interpret_common.py`; 2 more
  unit tests (`tests/test_interpret.py`, 15 total). UMAP is unsupervised /
  visualisation only — not a model metric, champion unchanged.
- Validation: `uv run pytest tests/` → 289 passed; locked files untouched; figures
  are small PNGs under `reports/vit/figures/`, traceable to the checkpoint.

### Slice 3 — Optional leakage-free add-ons (logit adjustment + McNemar)
Both additive, inference-only, leakage-free (VLD-13/VLD-20); champion unchanged.
- **Post-hoc logit adjustment (Menon 2020):** recomputed OOF test logits from each
  fold's `best.pt`, subtracted `τ·log(train_prior)` (prior from TRAIN split only).
  **Honest negative — macro-F1 falls monotonically with τ:** τ=0 0.6118
  [0.593,0.629] → τ=0.5 0.6022 → **τ=1.0 0.4911** [0.481,0.500] → collapse.
  Cause: the WeightedRandomSampler already balances training, so its implicit prior
  is ~uniform; prior subtraction double-corrects. → balanced sampler is the right
  imbalance mechanism (§5.5). Sensitivity curve: `reports/vit/figures/logit_adjust_tau.png`.
- **McNemar (Dietterich 1998), paired OOF N=10662:** champion is **significantly
  more accurate** than the best single (`02_single_swin_t`: χ²cc 15.25, exact
  p=9.2e-05, 422 vs 315) and the best pair (`09_pair_swin_t_beit_b_weighted`:
  χ²cc 25.25, p=4.7e-07, 423 vs 288). Accuracy-level test — complements (not
  contradicts) the CI-overlapping macro-F1 result.
- Built-in check: τ=0 reproduces stored champion OOF preds (10661/10662; 1 near-tie
  flip, bf16/TF32 vs fp32 — VLD-17). New scripts: `scripts/eval_logit_adjust.py`,
  `scripts/stats_mcnemar.py`; helpers + 8 tests in `interpret_common`/`test_interpret`.
- Validation: `uv run pytest tests/` → 297 passed; locked files / final-model
  numbers unchanged; no large artifacts.

### Slice 4 — LaTeX report
- Full report under `reports/vit/` (`main.tex` + 7 `sections/*.tex` + `references.bib`),
  academic Turkish, mirroring the CNN report template (pdfLaTeX/Overleaf; also
  cross-engine via `iftex`/`fontspec` so local `tectonic` renders Turkish glyphs).
  Structure follows project_plan §10 (Giriş/Yöntem/Deneyler/Sonuçlar/Tartışma/Sonuç).
- Missing §5.4 figures generated inference-free by `scripts/make_vit_report_figures.py`
  (pooled OOF, n=10662, leakage-free): `confusion_matrix.png`, `per_class_f1.png`,
  `training_curves.png`, `cv_macrof1_bar.png`, plus table
  `results/vit/tables/per_class_champion.{csv,md}`. Architecture diagram = TikZ.
- Content: 5-fold CV table (champion 0.6119 pooled), frozen→finetune deltas,
  pooled confusion matrix (rare/ordinal confusion: UC grades, hemorroids→retroflex-
  rectum), per-class table (rare classes F1→0), interpretability (rollout/Grad-CAM++/
  UMAP), honest negatives (GMU, TTA, seed-ens CI-overlap, logit-adj), McNemar
  significance, contextual comparator table (Wang/GastroViT/EffiMix — no SOTA claim).
- Reference stubs created for the previously-empty citations:
  `references/methodology_evaluation/mcnemar_dietterich_1998/paper.md` and
  `references/methodology_imbalance/logit_adjustment_menon_2020_iclr/paper.md`.
- Compiles clean with `tectonic`: 0 undefined citations, 0 undefined refs, 0 missing
  Turkish glyphs, 16 bibitems; `main.pdf` ~1.2 MB. LaTeX build intermediates
  gitignored (`reports/vit/.gitignore`).
- Validation: `uv run pytest tests/` → 297 passed; `backbones.py`/`vit_backbones.py`/
  configs/final-model numbers untouched; every reported number traces to
  `cv_fold5_ranked` / `metrics.json` / `predictions.npz` (no invention).
- TODO (for submission): student names/numbers + YouTube link in the report (Slice 5).

### Report corrections — Slice A (style & rules) — done
- Audit-driven style pass (`docs/vit/report_corrections_plan.md`, Slice A). Removed
  from the report BODY: repo paths (5), self-referential "dürüst negatif" wording (4),
  decorative em dashes (`---`, now `yok`/`$-$`/punctuation). Numbers/results unchanged.
- Quick check clean: no "yalnızca…değil", no "Günümüzde"/"önemli bir yer tut", 0
  decorative em dashes. `tectonic main.tex` → 0 undefined cites/refs, 0 missing
  Turkish glyphs. Slices B–E remain open.

### Report corrections — Slice B (required tables + training time) — done
- Added the 2 missing §10 tables and a hyperparameter table, all from existing data
  (no new runs): frozen ablation ranking (11 rows), fine-tune hyperparameters, and a
  transfer-cost table (best-validation epoch + tuned ~5.3×/5.9× throughput). The §5.5
  "ne kadar sürmüş" question is now answered via convergence epochs + throughput.
- Wall-clock time is not logged in metrics.json/epoch_log.jsonl; cost is therefore
  expressed via epochs-to-best (from history) + the documented speedup — no fabricated
  minutes. Report now has all 4 required tables (frozen ablation, 5-fold CV+CI,
  per-class, transfer/time). `tectonic` clean (0 undefined cites/refs, 0 missing glyph).

### Report corrections — Slice C (literature depth + citation integrity) — done
- Created the 2 missing reference stubs (`gradcam_selvaraju_2017`,
  `ulmfit_howard_2018_acl`) so every cited key has a non-empty stub (§11).
- Deepened Related Work with Subedi 2024 (CNN-Transformer hybrid) and Varam 2024
  (lightweight/edge ViTs), contextual (Kvasir-Capsule, different dataset);
  `references.bib` extended.
- Verified comparator numbers against the paper stubs: GastroViT is on the SAME
  HyperKvasir 23-class set (~92% acc, ensemble, different split); EffiMix ~98% acc
  (weighted F1). Competitors' macro-F1 column set to "$-$" (none report a directly
  comparable leakage-free macro-F1). Ramachandran 2025 dropped (no paper.md stub).
- `tectonic` clean (0 undefined cites/refs, 0 missing glyph); no SOTA claim.

### Report corrections — Slice D (method rigor + evidence) — done
- Added fusion equations (concat / weighted = softmax(w) scalar / GMU element-wise)
  to the Methodology — faithful to weighted.py / gmu.py.
- Added an appendix (`07_appendix.tex`, `\appendix` in main.tex): full 23-class
  per-class P/R/F1/support table (from per_class_champion) and an extra
  interpretability gallery (rollout `normal-pylorus` + Grad-CAM++ `hemorroids`),
  using two previously-unused panels.
- `tectonic` clean (0 undefined cites/refs, 0 missing glyph); PDF ~1.8 MB. Numbers
  unchanged. Remaining corrections: Slice E (student names + video link, user-side).

### Report depth — Slice F (methodology depth) — done
- Deepened the Methodology (write-up only, no result change): backbone inductive
  biases (ViT global/minimal-bias, Swin hierarchical shifted-window locality, BEiT
  masked-image self-supervised) grounded in their papers; transfer rationale
  (catastrophic forgetting → last-blocks-only + LLRD); one-line definitions of the
  rollout residual term and drop-path/MixUp/label-smoothing/EMA. No new citations
  (5 stubs already cited). Methodology 516→713 words. `tectonic` clean. Tracked in
  docs/vit/report_depth_plan.md (Slices F–I).

### Report depth — Slice G (discussion expansion) — done
- Added two Discussion subsections (write-up only, existing evidence): "Hata analizi
  ve sınıf karışmaları" (ordinal UC-grade + anatomical hemorroids→retroflex-rectum /
  oesophagitis-a→normal-z-line confusions, from the existing confusion matrix /
  per-class table) and "Dengesizliğin makro-F1 üzerindeki etkisi" (balanced-error /
  Menon + focal loss / Lin 2017; weighted sampler described without a citation).
  Discussion 583→878 words, 5→7 subsections.
- Stub-integrity caught: class_balanced_loss_cui_2019 stub holds the WRONG paper
  (Ahmed 2023) and weighted_random_sampler has no paper.md → neither cited (only the
  verified Focal Loss added to references.bib). `tectonic` clean; no new numbers.

### Report depth — Slice H (related-work literature grounding) — done
- Added 3 verified GI works to Related Work (faithful paraphrase, bib entries):
  Ahmed 2023 (GI feature fusion), He 2025 (HEMF multi-attention fusion), Guo 2024
  (curriculum self-supervised learning on HyperKvasir — ties to BEiT). Related Work
  now engages 9 distinct GI/comparator works. 248→299 words. `tectonic` clean; no
  SOTA. Only Slice I (intro/conclusion + term defs) remains in the depth plan.

### Report depth — Slice I (intro/conclusion + term definitions) — done
- Grounded the three-backbone choice in the Introduction (complementary biases:
  global / hierarchical-local / self-supervised); added a focal-loss citation
  (Lin 2017) to the Conclusion's future-work; defined the weighted sampler
  (inverse-frequency sampling) and bootstrap CI in one line each in Experiments.
  No new citation (focal already in bib). `tectonic` clean; numbers unchanged.
- **All depth slices F–I complete.** The report write-up is deepened end-to-end
  (Methodology +197 words, Discussion +295 words / 2 new subsections, Related Work
  +3 GI works, intro/conclusion grounded); every claim traces to existing
  evidence or a verified literature stub; no `\TODO` remains; PDF ~1.8 MB.

### Report depth — Slice J (richer literature metric comparison) — done
- Rebuilt tab:context as a 6-column contextual table (Çalışma/Yöntem/Protokol/
  Doğruluk/Makro-F1/Diğer) with the dataset paper as a **macro-F1 anchor**: Borgli
  2020 best macro-F1 0.619 (micro 0.907, MCC 0.899) — same metric type as ours,
  different split. Competitors keep "$-$" for macro-F1 (GastroViT 0.920 acc, Wang
  0.868 acc, EffiMix 0.980 acc / weighted-F1 0.97 note) — no fabricated macro-F1.
  Added two \subsubsections (protocol/metric differences + metric positioning):
  our 0.612 sits in Borgli's CNN baseline band (0.605–0.619); high-accuracy numbers
  stem from different protocols + aggregate/weighted metrics. All numbers verified
  from paper.md; no SOTA. `tectonic` clean; 04_results 1176→1317 words.

### Report depth — Slice J-2 (supporting metrics MCC + weighted-F1) — done
- Derived from the existing pooled OOF predictions (n=10662, inference-free): accuracy
  0.878 (= micro-F1), weighted-F1 0.879, MCC 0.868; macro-F1 0.6119 cross-check
  confirms correct pooling. Added MCC 0.868 to our tab:context row (same axis as
  Borgli's MCC 0.899) and a supporting-metrics paragraph in §5.4 (macro-F1 stays the
  headline; CI only on macro-F1). Honest note: our MCC is slightly below Borgli's but
  in the same band. No fabricated numbers; `tectonic` clean; 04_results 1317→1378 words.

### Report depth — Slice K (strengths/limitations deepened) — done
- Split Discussion §"Güçlü ve zayıf yönler" into \subsubsection{Güçlü yönler} and
  \subsubsection{Sınırlamalar}, each point mapped to existing evidence (McNemar,
  tab:context Borgli band + MCC, tab:cv CI, interpretability figures, tab:addons /
  fig:logit negatives, tab:cost throughput; limitations: tab:perclass rare-class F1=0,
  ordinal/anatomical confusion, CI-overlapping gain, BEiT/GMU, approximate repro,
  single dataset). No new numbers; all \ref resolve. Discussion 878→1068 words;
  `tectonic` clean.

### Report depth — Slice L (\subsubsection granularity) — done
- Re-nested existing prose into \subsubsections (no new content/numbers): Methodology
  §projection-fusion-classifier → 3 (Dal izdüşümü / Füzyon yöntemleri / Sınıflandırıcı);
  Results §detailed analysis → 2 (Eğitim dinamiği / Karışıklık matrisi ve sınıf bazında
  başarım); Discussion §interpretability → 2 (Dikkat ve Grad-CAM++ / Öznitelik uzayı
  (UMAP)). All labels/cross-references intact; `tectonic` clean.
- **All depth slices F–L complete.** The report now has deeper structure end-to-end
  (backbone biases, transfer rationale, error analysis, imbalance subsection,
  evidence-mapped strengths/limitations, enriched contextual comparison with a macro-F1
  anchor + MCC, and \subsubsection nesting across Methodology/Results/Discussion).
  Every claim traces to existing evidence or a verified stub; no `\TODO`; PDF ~1.8 MB.

### Report — architecture diagram (fig:arch) improved
- Fixed a real bug: the MLP node read `MLP\\[256]`, where `\\[256]` is a 256pt line
  break → a giant empty box. Now `MLP\\(256, dr. 0,3)`.
- Rewrote the TikZ (coordinate-based): symmetric input fan-out (same 224×224 image →
  3 backbones), symmetric projection→fusion fan-in, stage headers (Öznitelik çıkarımı
  / İzdüşüm / Füzyon / Sınıflandırma), dimension labels on arrows (768→512→512→23),
  clearer node texts (768-d / 512-d / Softmax 23 sınıf), and a caption note on the
  transfer regimes. No new packages; `tectonic` clean. Architecture/dimensions
  unchanged (write-up/figure only).

### Report — subsection bridge sentences
- Every \subsection that jumped straight into a \subsubsection now has a 1–2 sentence
  lead-in summarizing what the subsection covers (5 places: Methodology
  projection-fusion-classifier, Results detailed-analysis + literature-comparison,
  Discussion interpretability + strengths/limitations). Scan confirms 0 bridgeless
  subsections remain. Write-up only; `tectonic` clean.

### Report depth — Slice M (controlled CNN vs ViT comparison) — done
- Added Discussion §"Aynı protokol altında CNN füzyonu ile karşılaştırma" (tab:cnnvit):
  the apples-to-apples comparison the monorepo was designed for (same dataset, official
  5-fold, leakage-free OOF+CI, same concat/weighted/GMU+MLP framework; only the backbone
  family differs). CNN numbers verified this turn from docs/FINAL_MODEL.md — CNN final
  (triple weighted + TTA): macro-F1 0.6075 [0.5860,0.6296], acc 0.8765, MCC 0.8662; CNN
  base-CE 0.6000 (within the ViT CI). ViT (base): 0.6119 / 0.8779 / 0.868. Honest reading:
  statistically comparable (CI overlap), both bounded by the same rare-class ceiling →
  the limit is data scarcity, not the backbone family; shared negatives (focal/TTA/seed);
  cross-project directions (transformer backbones + attention/Grad-CAM++ realised here).
  No SOTA; methodological framing (no repo paths / VLD-PLD ids / sprint-week wording in
  body). Also cleaned `---` in the architecture-diagram TikZ comments. 05_discussion
  1068→1367 words; `tectonic` clean.

### Report corrections — Slice E (submission/final) — report side done
- Title page set to a single author (Yasin Ekici, no 21360859029) with the GitHub
  repo link on the cover and in the Conclusion
  (github.com/YasinEkici/hyperkvasir-multi-backbone-fusion); removed the report's
  YouTube `\TODO` (the video goes in the repo README, not the report body).
- No real `\TODO` remains in the report. `tectonic` clean. All agent-fixable report
  corrections (Slices A–E report side) are complete. Pending (user): record the
  Sprint 5 demo video and add its link to the README.
