# Known Issues — ViT Fusion

Track reproducibility, dataset, and runtime issues here as they arise (KI-VIT-NNN).
No issues are pre-invented; anticipated risks live in `docs/vit/project_plan.md §9`.

## Anticipated watch-list (not yet observed)

These are risks to log here **if/when** they actually occur:

- **timm attribute paths** — `model.blocks` (ViT/BEiT) vs `model.layers` (Swin)
  confirmed present (2026-06-16); verify exact block/stage counts in the S1 smoke
  test before the fine-tune slice (`blocks[9:]`, VLD-08) relies on them.
- **A100 availability** — Colab Pro may allocate a non-A100 GPU; frozen stages
  degrade gracefully, fine-tune stages must abort (D-09 gate).
- **Interpolation** — timm ViTs expect bicubic + `crop_pct≈0.9` (VLD-09); using
  torchvision's default bilinear would silently lower transfer fidelity.
- **Catastrophic forgetting** — low backbone LR (2e-5–5e-5), last-3-block unfreeze,
  LLRD, EMA are the mitigations; watch fold-to-fold variance.
