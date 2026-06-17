# 004 — Week 4: Analysis, Report, and Submission

> **Status legend:** ✅ Done · 🔄 In progress · ⬜ Pending
>
> All steps begin at ⬜ Pending. Do not auto-advance — each step requires
> explicit user approval (see memory: `feedback_step_advancement.md`).
>
> **Insert position:** after Week 3.5 close (2026-06-08, exec plan 003.5 completed).
> **Hard deadline: 2026-06-08, 23:59 Turkey time (TODAY).** Submission requires
> BOTH an ekampüs `.rar` (report PDF + code + README) AND a YouTube video link.
> **No video = the project is not graded** (assignment §6).

---

## 1. Goal

Turn the frozen final model and the completed experiments into the graded
deliverables: a complete report PDF, a YouTube presentation video, and a packaged
`.rar` submission — all built on the **frozen model (exp 11 triple weighted CE +
TTA, macro-F1 0.6075, CI [0.5860, 0.6296]; see `docs/FINAL_MODEL.md`)** with **no
new training**. Every reported number must trace to `metrics.json` /
`results/tables/` / `docs/FINAL_MODEL.md`, and the honest finding — *no Week 3.5
technique beat the CE champion at the 95% CI level* — must be stated plainly, not
hidden.

This plan is **deadline-driven**: the mandatory path (figures → report → video →
packaging) is protected; interpretability extras (Grad-CAM++, UMAP) are
explicitly optional and droppable.

---

## 2. Constraints (read before editing)

- **No new training / experiments / config changes.** All analysis is computed
  from saved predictions and checkpoints of the frozen model.
- **Build on `docs/FINAL_MODEL.md`** (exp 11 + TTA, seed 42). Week 4 figures must
  reflect this model, not a stale single fold.
- **Honest reporting (non-negotiable):** state that focal lost, TTA is the best
  point estimate (+0.0075, **within CI**), and the seed ensemble gave no macro-F1
  gain — none is statistically significant; the architecture is at its ceiling on
  the official 5-fold protocol. Do not overclaim.
- **No PLD-\* change.** MLP-only classifier (PLD-06); macro-F1 headline (PLD-11).
- **No classifier-wise comparison.** The assignment's "3-classifier" phrasing is an
  instructor-confirmed error (`project_plan.md` §scope, PLD-06). Comparisons are
  CNN-wise, fusion-wise, transfer-learning-wise only.
- **Cross-protocol honesty (PLD-17):** Borgli's official-split macro-F1 (~0.62) is
  the only protocol-aligned external anchor. Effimix (0.9799 acc) and GastroViT
  (0.92 acc / 0.64 F1) use augmented/non-official or single-split protocols →
  **contextual discussion only, never in the head-to-head table.** Do not mix
  GastroViT's 16-class numbers with our 23-class.
- **Every external number cited from `references/INDEX.md`** with its comparability
  caveat and table/section evidence location.
- **No `.pt`, feature cache, or raw data committed to git** (checkpoints stay
  referenced by path, per `docs/FINAL_MODEL.md`).
- **Deadline beats completeness:** if time runs short, ship report + video with the
  core tables/figures; drop optional extras with a one-line "future work" note.

---

## 3. Scope

### Track A — Final figures & tables from the frozen model — MANDATORY (assistant, CPU)
Computed from saved predictions/logs; no training.
- **A1 Confusion matrix (frozen model):** from pooled TTA predictions
  (`results/runs/11_*/predictions_tta.npz`) — full 23×23, plus optionally a
  representative fold. Labelled with the 23 class names.
- **A2 Table 3 — per-class precision/recall/F1/support** for the frozen model
  (TTA), via `classification_report` on pooled TTA preds; highlight the rare-class
  failures (support ≤ 10) that drive macro-F1 down.
- **A3 Table 4 — training time / epochs-to-early-stop** across configs, parsed
  from `results/runs/*/epoch_log.jsonl` (+ recorded per-fold A100/RTX timings).
- **A4 Refresh comparison bar chart + per-class F1 chart** so they represent the
  frozen model (exp 11 + TTA), not a stale single fold.

### Track B — Report PDF — MANDATORY (assistant drafts; user finalizes)
Map to assignment §5, building on the existing LaTeX skeleton (`reports/final/`).
- Intro, Method, Experiments, Results, Discussion (§5.5 — most critical, 25%),
  Conclusion. Three comparison axes (CNN/fusion/transfer). 5-fold CV + bootstrap CI
  methodology. Honest Week 3.5 findings. Cross-protocol context with caveats.
  Headline macro-F1; accuracy as support. One-sentence note on the corrected
  "3-classifier" requirement.

### Track C — YouTube video — MANDATORY (user records/uploads; assistant scripts)
- Script/outline → record → upload (unlisted is fine) → capture link. Cover
  problem, dataset, method, results, honest discussion, conclusion.

### Track D — Submission packaging — MANDATORY (user; assistant preps)
- `.rar` per assignment §6 naming, containing report PDF + code + README + YouTube
  link; upload to ekampüs before 23:59.

### Track E — Interpretability extras — OPTIONAL / DROPPABLE (assistant, time permitting)
- **E1 Grad-CAM++** on a single representative fold/seed checkpoint (fold 0,
  seed 42). **Open design question:** the model fuses 3 backbones — decide which
  backbone(s) to visualize (e.g. per-branch CAMs) and document the choice. Uses the
  `grad-cam` package (already in `env/requirements-colab.txt`).
- **E2 UMAP** of the fused 512-d features (uses `umap-learn`).
- Both are stretch; drop cleanly with a "future work" note if the deadline nears.

---

## 4. Out of Scope

- Any new training, fine-tuning, seed, or config (frozen model only).
- Any PLD-\* change; classifier-wise comparison; recipe-v2 (Step 6, skipped).
- Re-running Week 3.5 experiments or re-opening the frozen-model decision.
- New external baselines beyond those already in `references/INDEX.md`.
- DeepWeeds / cross-protocol reproduction / McNemar (all previously deferred).

---

## 5. Files Expected to Be Modified / Created

| File | Change |
|---|---|
| `results/runs/11_*/confusion_matrix_tta.png` (or `results/figures/`) | New — frozen-model confusion matrix (A1) |
| `results/tables/per_class_frozen_tta.csv` / `.md` | New — Table 3 (A2) |
| `results/tables/training_time.csv` / `.md` | New — Table 4 (A3) |
| `results/figures/comparison_bar_chart.png`, `per_class_f1_best.png` | Refresh for frozen model (A4) |
| `scripts/` (small analysis helper if needed) | New — per-class/CM/time from saved preds (no training) |
| `reports/final/sections/*.tex`, `reports/final/tables/`, `reports/final/figures/` | Report content (B) — gitignored working dir |
| `reports/final/notes/video_script.md` | New — video outline (C) |
| `README.md` | Ensure run/setup + results summary current for submission (D) |
| `docs/results_progress.md`, `docs/experiment_log.md` | Week 4 entries |
| `docs/exec-plans/active/004-...md` → `completed/` | On close-out |

> Report working files live under `reports/` (gitignored by user choice) — they do
> not get committed; the final PDF is submitted via ekampüs, not git.

---

## 6. Step-by-Step Implementation Plan

> One step at a time. After each: verify, update docs, **stop and wait for approval**.
> Deadline-prioritized: Steps 1–4 are mandatory; Step 5 (extras) only if time remains.

### ⬜ Step 1 — Final figures & tables (Track A) — MANDATORY
- A1 confusion matrix, A2 Table 3 (per-class P/R/F1/support), A3 Table 4 (time/
  epochs), A4 refreshed charts — all from frozen-model TTA predictions / logs.
- **Who:** assistant (CPU). **Acceptance:** every figure/table generated from saved
  artifacts; numbers match `ci_11_*_tta.json` / `extra_metrics_11_*_tta.json`;
  rare-class failures visible in Table 3; no training run.

### ⬜ Step 2 — Report draft (Track B) — MANDATORY
- Fill the §5 sections on the LaTeX skeleton; insert Tables 1–4 + figures; write
  the §5.5 discussion answering all required questions; add cross-protocol context
  with caveats; state the honest within-CI verdict.
- **Who:** assistant drafts text/tables; user finalizes LaTeX + compiles PDF.
- **Acceptance:** all §5 sections present; every number traces to a source file;
  comparisons are CNN/fusion/transfer (no classifier axis); macro-F1 headline;
  honest finding stated; PDF compiles.

### ⬜ Step 3 — Video script + recording (Track C) — MANDATORY
- Assistant writes `video_script.md` (problem → dataset → method → results →
  discussion → conclusion, ~timed). User records and uploads to YouTube; captures
  link.
- **Who:** assistant (script); USER (record + upload). **Acceptance:** video
  uploaded, link captured, content matches the honest results.

### ⬜ Step 4 — Packaging + submission (Track D) — MANDATORY
- Assemble `.rar` (report PDF + code + README + YouTube link) per §6 naming; upload
  to ekampüs before 23:59.
- **Who:** USER (assistant preps README + file checklist). **Acceptance:** `.rar`
  uploaded before deadline with all required contents and the video link.

### ⬜ Step 5 — Interpretability extras (Track E) — OPTIONAL
- E1 Grad-CAM++ (document backbone choice), E2 UMAP. Only if Steps 1–4 are done
  with time to spare.
- **Who:** assistant. **Acceptance:** figure(s) produced from the frozen
  checkpoint + added to the report; OR explicitly dropped as documented future work.

### ⬜ Step 6 — Close-out
- Update `results_progress.md` / `experiment_log.md` with Week 4 completion; move
  this plan to `completed/`.

---

## 7. Risks / Open Questions

| Risk / Question | Likelihood | Mitigation |
|---|---|---|
| **Deadline missed (dominant risk)** | High | Protect Steps 1–4; drop Step 5; video + report are the must-haves |
| **No video uploaded → project ungraded** | Med | Record video as soon as report results are stable; do not leave to last hour |
| Report number not traceable to a source | Med | Every table/figure cites its `results/` source; no hand-typed numbers |
| Over-claiming TTA/ensemble gains | Med | Encode "within CI, not significant" in Results + Discussion |
| Cross-protocol numbers misread as head-to-head | Med | Separate table + caveat; Borgli official-split is the only aligned anchor |
| LaTeX compile/figure-path failures near deadline | Med | Compile early and often; keep figures in `reports/final/figures/` |
| **Open question:** Grad-CAM++ on a 3-backbone fusion — which backbone(s)? | — | Decide in Step 5 (per-branch CAMs on fold-0/seed-42); drop if no time |

---

## 8. Acceptance Criteria

- **Report PDF** complete with all assignment §5 sections; every number traceable
  to `metrics.json` / `results/tables/` / `docs/FINAL_MODEL.md`; macro-F1 headline;
  comparisons CNN/fusion/transfer (no classifier axis); honest within-CI verdict
  stated; cross-protocol numbers contextual-only with caveats.
- **Confusion matrix + Table 3 (per-class) + Table 4 (time)** generated from the
  frozen model's saved predictions/logs.
- **YouTube video uploaded** and the link captured (mandatory — no video = ungraded).
- **`.rar` submitted to ekampüs before 23:59** with report PDF + code + README +
  video link, per §6 naming.
- No PLD-\* change; no `.pt`/cache/raw data committed; frozen model unchanged.
- Grad-CAM++/UMAP either included or explicitly documented as dropped future work.

---

## 9. Commands

```bash
# ── Track A (assistant, CPU — from saved predictions/logs, NO training) ──────────
# A1/A2 confusion matrix + per-class table for the frozen model (TTA):
uv run python scripts/plot_results.py --confusion-matrix --experiments 11_triple_weighted_finetune_wide_official
# (A2/A3/A4 may need a small saved-prediction analysis helper — added in Step 1,
#  reads predictions_tta.npz / epoch_log.jsonl; no model training.)
uv run python scripts/generate_report_tables.py

# ── Verify nothing regressed ────────────────────────────────────────────────────
uv run pytest tests/ -v

# ── Track E (optional) ──────────────────────────────────────────────────────────
# Grad-CAM++ / UMAP from the frozen fold-0/seed-42 checkpoint (inference only).

# ── Close-out ───────────────────────────────────────────────────────────────────
git mv docs/exec-plans/active/004-week-4-analysis-report.md docs/exec-plans/completed/
```

---

## 10. Documentation Updates Required

- **During Step 1:** note final figure/table sources in `results_progress.md`.
- **After Step 2:** report draft status in `experiment_log.md`.
- **After Steps 3–4:** record YouTube link + submission in `experiment_log.md`.
- **After Step 6:** Week 4 close-out in `results_progress.md` / `experiment_log.md`;
  update `known_issues.md` if any Grad-CAM/UMAP/LaTeX issue surfaced; move this plan
  to `completed/`.
