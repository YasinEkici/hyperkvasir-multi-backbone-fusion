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

## Sprint 2 — Frozen ablation (off-A100)
*Status: not started*

| config | backbones | fusion | transfer | fold | Acc | macro-F1 |
|---|---|---|---|---|---|---|
| _to be filled_ | | | | | | |

---

## Sprint 3 — Fine-tune funnel (A100)
*Status: not started*

## Sprint 4 — 5-fold CV + CI (A100)
*Status: not started*

## Sprint 5 — Interpretability + report
*Status: not started*
