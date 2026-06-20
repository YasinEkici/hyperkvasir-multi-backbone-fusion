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

## Sprint 4 — 5-fold CV + CI (A100)
*Status: not started*

## Sprint 5 — Interpretability + report
*Status: not started*
