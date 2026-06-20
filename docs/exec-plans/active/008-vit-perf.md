# 008 - ViT Sprint 3.5: A100 Throughput & Runner Efficiency

## 1. Goal

Make ViT fine-tuning use the A100 efficiently before Sprint 4 5-fold CV, and stop
burning Colab units on idle/setup time. Measure first, then apply only the changes
that show a real, verified speedup, and lock the tuned settings as the Sprint 4
default. No change to model architecture, evaluation, or the locked candidate set.

Motivation: Sprint 3 fine-tuning was slow and units were wasted on (a) Drive-FUSE
small-file staging and provenance re-hashing (already mitigated in the Colab
runner), and (b) likely GPU starvation during training — the fine-tune image
DataLoader uses `num_workers=0`, TF32 is off, and cuDNN runs in deterministic /
no-autotuner mode (`scripts/train.py`, `src/utils/reproducibility.py`).

Numbering note: this is an inserted Sprint 3.5; Sprint 4 (5-fold CV) becomes
`009-vit-cv.md`. Update `docs/vit/project_plan.md §3` if/when this is confirmed.

## 2. Inputs

- Throughput tool: `scripts/benchmark_vit_throughput.py` (measurement only).
- Fine-tune path: `scripts/train.py` (`_make_image_loaders`, Trainer wiring).
- `src/utils/reproducibility.py` (`seed_all` cuDNN flags).
- `src/training/trainer.py` (AMP autocast dtype).
- `configs/vit/training/vit_finetune.yaml`.
- Sprint 3 runs for a correctness baseline:
  `results/vit/runs/02_single_swin_t_finetune_official/metrics.json` and the
  triple `11_triple_weighted_finetune_official/metrics.json`.
- Best-practice references (see Slice 4 doc notes): PyTorch DataLoader / Lightning
  speed guide, PyTorch reproducibility notes, NVIDIA TF32 blog, PyTorch mixed
  precision blog.

## 3. Scope

- Add a measurement tool that times the fine-tune loop under `current` vs `fast`
  settings and reports img/s, steps/s, ms/step, peak GPU memory (done:
  `scripts/benchmark_vit_throughput.py`).
- Candidate tuned knobs (apply only if measured to help, lowest risk first):
  - DataLoader: `num_workers>0`, `pin_memory=True`, `persistent_workers=True`,
    `prefetch_factor` — plumb through `_make_image_loaders` (default preserves
    current behavior).
  - cuDNN: `benchmark=True` + `deterministic=False` for the fine-tune path
    (seeds kept for approximate reproducibility).
  - TF32: enable `matmul.allow_tf32` / `cudnn.allow_tf32` on the fine-tune path.
  - AMP: bf16 autocast on A100 (range-safe for ViT) instead of fp16.
  - (Optional) larger batch size for single/pair configs with LR re-check.
- Keep changes opt-in/config-driven; the frozen cached-feature path and CNN
  behavior stay unchanged.
- Document the speedup, the reproducibility trade-off, and log it as a VLD note.

## 4. Out of Scope

- Sprint 4 5-fold CV / bootstrap CI runs (that is `009-vit-cv.md`).
- Changing backbones, fusion, projection dim, classifier, metrics, or the locked
  Sprint 3 candidate/top-4 set.
- Re-running or invalidating Sprint 3 fold-0 results.
- Editing the frozen CNN files or `src/models/backbones.py`.
- GPU-side augmentation libraries (DALI/Kornia) unless a later, separate task.

## 5. Files Expected to Be Modified

- `scripts/benchmark_vit_throughput.py` - measurement tool (added).
- `scripts/train.py` - DataLoader knobs + AMP dtype wiring (config-driven).
- `src/training/trainer.py` - autocast dtype (bf16 option), if adopted.
- `src/utils/reproducibility.py` / `seed_all` call - cuDNN/TF32 flags for the
  fine-tune path (must not change frozen/CNN behavior).
- `configs/vit/training/vit_finetune.yaml` - tuned defaults (num_workers, amp
  dtype, cudnn benchmark, tf32, batch size).
- `docs/vit/decisions.md` - VLD note on the throughput/reproducibility trade-off.
- `docs/vit/experiment_log.md`, `docs/vit/results_progress.md` - benchmark
  numbers and the verified-equivalent smoke check.
- Tests for any new config plumbing.

## 6. Step-by-Step Implementation Plan

### Slice 1 - Measure baseline (no training change)
1. Stage the dataset (archive path) and confirm A100.
2. Run `benchmark_vit_throughput.py --modes both` for a cheap single
   (`02_single_swin_t_finetune_official`) and the heavy triple
   (`11_triple_weighted_finetune_official`).
3. Record img/s, steps/s, ms/step, peak memory for `current` vs `fast`, and the
   speedup factor.

### Slice 2 - Apply verified knobs
1. Plumb DataLoader knobs through `_make_image_loaders` (defaults preserve
   current behavior).
2. Add cuDNN benchmark + TF32 + bf16 for the fine-tune path only.
3. Make all of it config-driven in `vit_finetune.yaml`.

### Slice 3 - Correctness check
1. Re-run one fold-0 candidate end-to-end with the tuned config.
2. Confirm test macro-F1 is within run-to-run noise of the Sprint 3 value
   (non-determinism is expected; large drift is a bug).

### Slice 4 - Document & lock for Sprint 4
1. Record benchmark + correctness numbers in the ViT docs.
2. Add the VLD note (throughput config + reproducibility trade-off).
3. Run `uv run pytest tests/`; confirm hygiene (no large artifacts, frozen files
   untouched).

## 7. Risks

- **Reproducibility:** `cudnn.benchmark=True` / non-deterministic / TF32 drop
  bitwise reproducibility. Mitigation: keep seeds, treat as approximate; document.
- **bf16 vs fp16:** numerical differences; bf16 is range-safer for ViT but verify
  loss is stable.
- **num_workers on Colab:** too many workers can exhaust RAM; tune to the runtime
  (~8 on A100 high-RAM).
- **Scope creep:** must not alter the Sprint 3 results or start Sprint 4 CV.
- **Frozen/CNN regressions:** knobs must be gated to the fine-tune path; default
  behavior unchanged.

## 8. Acceptance Criteria

- Benchmark reports `current` vs `fast` img/s and a speedup factor on the A100.
- Applied knobs are config-driven and default-safe (frozen/CNN unchanged).
- A tuned fold-0 re-run yields test macro-F1 within run-to-run noise of Sprint 3.
- Throughput + reproducibility trade-off documented; VLD note added.
- `uv run pytest tests/` passes; no large artifacts committed; frozen files and
  `src/models/backbones.py` untouched.

## 9. Commands

```bash
# Slice 1 - measure (A100, after staging)
uv run --no-sync python scripts/benchmark_vit_throughput.py --experiment 02_single_swin_t_finetune_official --modes both --max-steps 100
uv run --no-sync python scripts/benchmark_vit_throughput.py --experiment 11_triple_weighted_finetune_official --modes both --max-steps 60

# Slice 3 - correctness re-run (example)
uv run --no-sync python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 02_single_swin_t_finetune_official --device cuda

# Slice 4
uv run pytest tests/
```

## 10. Documentation Updates Required

- `docs/vit/experiment_log.md`: benchmark numbers, applied knobs, correctness
  check, pytest result.
- `docs/vit/results_progress.md`: short Sprint 3.5 note (throughput before/after).
- `docs/vit/decisions.md`: VLD note (throughput config + reproducibility).
- `docs/vit/project_plan.md §3`: insert Sprint 3.5; renumber CV to `009-vit-cv.md`.
- Move this plan to `completed/` only when explicitly approved.
