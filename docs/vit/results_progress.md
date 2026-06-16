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
*Status: not started*

### Implementation milestones

| # | Component | File(s) | Done |
|---|---|---|:--:|
| 1 | `vit-fusion` branch + project docs (`docs/vit/`) | — | ✅ |
| 2 | timm ViT backbone factory | `src/models/vit_backbones.py` | ☐ |
| 3 | Extractor dispatch (CNN vs ViT) | `src/models/full_model.py` | ☐ |
| 4 | `configs/vit/` skeleton (method + training) | `configs/vit/` | ☐ |
| 5 | Backbone shape smoke test ((B,768) for V/S/B) | `tests/` | ☐ |

### Results
*(none yet — Sprint 2 frozen ablation produces the first numbers)*

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
