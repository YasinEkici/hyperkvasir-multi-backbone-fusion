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

## Observed issues

### KI-VIT-001 — Google Drive FUSE throttles many-small-file staging (Sprint 3)

- **Observed (2026-06-20):** staging the 10,662-image dataset to Colab by copying
  loose files off the mounted Drive (FUSE) ran at ~1.5 files/s and stalled for
  20–30 min — Drive throttles sustained small-file reads. The provenance gate hit
  the same wall because it re-hashed the Drive source tree (`sha256_tree`).
- **Workaround (implemented in `colab/vit_finetune_runner.ipynb`):** stage from a
  single **archive** on Drive (`labeled-images.zip`/`.tar` with `labeled-images/`
  at its root) — one sequential copy + local extract (minutes). The provenance
  cell now hashes only the **local** staged tree and verifies per-class counts vs
  the fold manifest + git SHA, recording the Drive archive as `approved_source`
  (the Drive-vs-staged byte re-hash is redundant: the archive was verified to hold
  10,662 jpg at build time and cell 5 re-asserts the staged count).
- **Also observed:** the first artifact copy-back was partial (files split across
  runs); re-fetching the complete run dirs from the Drive per-model backup
  resolved it. All five runs verified complete afterward.
- **Status:** resolved; no methodology impact. Dataset integrity and A100 gating
  were preserved.

### KI-VIT-002 — Windows: high DataLoader num_workers crash at eval (Sprint 3.5)

- **Observed (2026-06-21):** running the tuned fine-tune config
  (`dataloader.num_workers=8`, `persistent_workers=true`) on the local RTX 5080
  (Windows, multiprocessing `spawn`) trained fine for all epochs, then crashed in
  the post-training **test** evaluation with `OSError [WinError 1455] paging file
  too small` while loading `cublas64_*.dll` in a freshly spawned worker. Cause:
  train + val persistent workers (8 each) plus the test loader's 8 new workers =>
  ~24 processes each re-importing torch/CUDA under Windows `spawn` -> commit-memory
  limit.
- **Impact:** Windows local runs only. The Sprint 4 target (Colab A100, Linux
  `fork`) is unaffected. Training itself completed; only the final test eval +
  `metrics.json` write were skipped (recoverable by evaluating `best.pt`).
- **Mitigations (Windows local):** lower `dataloader.num_workers` (e.g. 2) and/or
  set `persistent_workers: false`, or raise the Windows paging-file size. No code
  change needed for the A100/Linux path.
- **Status:** documented; not a code-logic bug. Optional future hardening: cap
  num_workers when `os.name == 'nt'`.
