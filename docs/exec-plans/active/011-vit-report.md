# 011 - ViT Sprint 5: Interpretability + Report + Video

## 1. Goal

Close the **ViT fusion** project with the **mandatory grade-driving deliverables**:
(a) **interpretability** evidence for the frozen final model — transformer-native
**attention rollout** (Abnar & Zuidema 2020) + **Grad-CAM++** (Chattopadhyay et al.
2018) panels and a **UMAP** (McInnes et al. 2018) feature-space map; (b) a complete
**LaTeX report** under `reports/vit/` that tells the full, honest experimental
story; and (c) the **YouTube video** (without it the project is NOT graded). All of
this is **inference-only** from the existing `best.pt` checkpoints — no retraining,
**0 A100 units** (VLD-11). An optional, time-boxed slice adds two cheap,
leakage-free analyses (post-hoc logit adjustment + McNemar) to enrich the
discussion.

Final model is **FROZEN** (VLD-18/VLD-19): `11_triple_weighted` (ViT-B + Swin-T +
BEiT-B, weighted fusion, MLP head) — 5-fold CV macro-F1 **0.6094 ± 0.0254**, pooled
OOF **0.6119 [0.593, 0.630]**, top-seed ensemble **0.6157 [0.599, 0.632]**,
accuracy ≈ 0.88. Sprint 5 **explains and reports** this model; it does not change
it.

Source of truth: `docs/vit/project_plan.md` §3 (S5 row), §10 (report deliverables →
assignment §5), §11 (submission checklist); `docs/vit/final_assignment.md`; locked
decisions VLD-10, VLD-11, VLD-13, VLD-14, VLD-18, VLD-19; and the interpretability
references `references/methodology_evaluation/attention_rollout_abnar_2020_acl/`,
`.../gradcam_pp_chattopadhyay_2018_wacv/`, `.../umap_mcinnes_2018_joss/`,
`.../mcnemar_dietterich_1998/`, plus `references/methodology_imbalance/` (logit
adjustment / Menon 2020 context).

Branch: `sprint5/vit-report` (off a `main` that already includes Sprint 4 + 4.5).
**Time-box (deadline Fri 2026-06-26 23:59, ~2 days): the report + video are
mandatory and must ship; interpretability is cheap/local; keep the optional
add-ons (Slice 3) from blocking Slices 4-5.**

## 2. Inputs

- Frozen checkpoints + predictions for the final model and CV folds under
  `results/vit/runs/` (e.g. `11_triple_weighted[_fold_k]/best.pt`, `metrics.json`,
  OOF prediction files). Authoritative numbers in `docs/FINAL_MODEL.md`.
- The trained model graph: `src/models/full_model.py` (`MultiCNNFusionClassifier`),
  `src/models/fusion/{concat,weighted,gmu}.py`, `src/models/backbones.py`
  (READ-ONLY), per-branch 512-d projections (VLD-05).
- Eval/aggregation tooling to reuse without retraining:
  `scripts/summarize_vit_cv.py` (mean±std + leakage-free bootstrap 95% CI),
  `scripts/eval_tta_ensemble.py` (recompute softmax from `best.pt`),
  `scripts/benchmark_vit_throughput.py` (training-time table).
- Fold manifests `data/splits/hyperkvasir_official_5fold/fold_{0..4}.csv`
  (OOF test images for figures, leakage-free per VLD-13).
- Interpretability references (Section F & H.3) and comparator references
  (Section A & H.4: Wang 2023 HSW-ViT ≈86.81% acc; GastroViT 2025 ~0.64 macro-F1 /
  91.98% acc; Effimix 2022; Borgli 2020 dataset paper).
- Result tables already produced: `results/vit/tables/cv_fold5_ranked.{csv,md}`,
  TTA/ensemble outputs, frozen-ablation tables, throughput table.

## 3. Scope

- **Interpretability is INFERENCE-ONLY** from existing `best.pt` — no retraining, no
  new training runs, **0 A100 units**. Runs on local CPU/GPU.
- **Attention rollout** for the ViT-B and BEiT-B branches (CLS-token, isotropic
  blocks — rollout `A = 0.5·W_att + 0.5·I` across layers). **Grad-CAM++** via a
  transformer `reshape_transform` for token→grid; **document the Swin windowed-layout
  caveat** (hierarchical/shifted-window features need care — if Grad-CAM++ on Swin is
  unreliable, fall back to rollout/attention on the ViT-B + BEiT-B branches and say so
  honestly rather than ship a misleading map).
- **Per-class qualitative panels** drawn from **OOF test images only** (VLD-13):
  correct predictions + a few representative **failure cases**, spanning common and
  rare classes.
- **UMAP** of the fused and/or per-branch 512-d features on the OOF test set, colored
  by class — to visualize class separability and the rare-class crowding that caps
  macro-F1.
- **Optional Slice 3 (time-boxed, leakage-free):**
  (i) **post-hoc logit adjustment / class-prior correction** (Menon 2020) as ONE more
  inference-time arm — subtract `log(train prior)` from logits (priors from training
  folds only), recompute macro-F1 over OOF the same way as TTA/ensemble;
  (ii) **McNemar's test** (Dietterich 1998) for champion-vs-a-baseline significance on
  paired OOF predictions. Both are documented as honest results (expected small/zero;
  set expectations) and must NOT alter the frozen final model.
- **LaTeX report** under `reports/vit/` covering the full story (frozen vs fine-tune,
  concat/weighted/GMU, 5-fold CV + CI, seed-ensemble, TTA, throughput), the comparator
  positioning table, the interpretability figures, and an honest
  limitations/imbalance discussion — mapped to assignment §5 (`project_plan.md §10`)
  and the §11 submission checklist.
- **YouTube video**: a script/outline + recording checklist + what to demo; a link
  placeholder wired into the report and README.

## 4. Out of Scope

- Any retraining, new backbone, new fusion method, A100 usage, or change to the
  splits, class defs, eval rules, tuned recipe, or any locked VLD.
- Changing the **frozen final model** (`11_triple_weighted`) or any Sprint 4 `_cv` /
  Sprint 4.5 GMU number — Slice 3's optional arms are *additional reported analyses*,
  never a re-selection of the champion.
- Editing `src/models/backbones.py` or the frozen CNN project files; committing large
  artifacts (`*.pt`, predictions, run dirs stay gitignored; only small PNG/PDF figures
  under `reports/vit/figures/`).
- The final `.rar` packaging / ekampüs upload (that is the human submission step,
  tracked in `project_plan.md §11`, not this plan).

## 5. Files Expected to Be Modified Later

- `scripts/` (new, inference-only, no train.py change):
  `scripts/interpret_attention_rollout.py`, `scripts/interpret_gradcam.py`,
  `scripts/interpret_umap.py`; optional `scripts/eval_logit_adjust.py`,
  `scripts/stats_mcnemar.py` (or extend `scripts/eval_tta_ensemble.py` /
  `scripts/summarize_vit_cv.py` with leakage-free helpers).
- `tests/` - small pure-function tests for the new helpers (rollout matrix
  normalization, `reshape_transform` shape, logit-adjustment shift, McNemar
  contingency-table math). `uv run pytest tests/`.
- `reports/vit/` - LaTeX sources (`main.tex`, `sections/*.tex`, `refs.bib`) +
  `reports/vit/figures/` (small PNG/PDF: rollout panels, Grad-CAM++ panels, UMAP,
  confusion matrix, training curves, architecture diagram).
- `results/vit/tables/` - any new derived table (e.g. logit-adjust arm, McNemar
  p-value) if Slice 3 runs.
- `README.md` - video link placeholder + report pointer.
- `docs/vit/results_progress.md`, `docs/vit/experiment_log.md`,
  `docs/vit/decisions.md` (VLD note only if Slice 3 yields one),
  `docs/vit/project_plan.md §3` (mark S5 done at close-out).

Must not be modified: `src/models/backbones.py`; locked VLD recipe values; the
Sprint 4 `_cv` and final-model numbers; frozen CNN artifacts.

## 6. Step-by-Step Implementation Plan

### Slice 1 - Interpretability: attention rollout + Grad-CAM++
1. Add `scripts/interpret_attention_rollout.py`: load `11_triple_weighted/best.pt`,
   capture per-layer attention from the ViT-B and BEiT-B branches, compute rollout
   (`A = 0.5·W_att + 0.5·I`, multiply across layers, normalize), map the CLS-row to
   the 14×14 patch grid, overlay on the input.
2. Add `scripts/interpret_gradcam.py`: Grad-CAM++ with a transformer
   `reshape_transform` (token sequence → grid); target the last block of each branch.
   Validate on ViT-B/BEiT-B; **document the Swin windowed-layout caveat** and fall
   back gracefully if Swin maps are unreliable.
3. Select OOF test images (correct + failure cases) across common + rare classes;
   render per-class panels to `reports/vit/figures/`.
4. Tests: rollout normalization (rows sum→1, identity-mix), `reshape_transform`
   output shape. `uv run pytest tests/`.
- **Acceptance:** rollout + Grad-CAM++ panels generated from OOF images only, each
  traceable to `best.pt` and a fold manifest; Swin caveat documented; tests pass.

### Slice 2 - Feature-space analysis (UMAP)
1. Add `scripts/interpret_umap.py`: extract fused (and/or per-branch) 512-d features
   for the OOF test set from `best.pt`, fit UMAP, scatter colored by class.
2. Save figure + a short note on separability / rare-class crowding for the
   discussion.
- **Acceptance:** one UMAP figure over the OOF test set, colored by 23 classes,
  saved under `reports/vit/figures/`, traceable to the checkpoint + manifests.

### Slice 3 - Optional leakage-free add-ons (time-boxed)
1. **Logit adjustment:** compute training-fold class priors, subtract `log(prior)`
   from OOF logits, recompute macro-F1 / Acc the same leakage-free way as
   `eval_tta_ensemble.py`; report vs the unadjusted champion (honest, expect
   small/zero).
2. **McNemar:** paired OOF predictions (champion vs a chosen baseline, e.g. best
   single backbone or concat), build the 2×2 discordant table, report the statistic
   + p-value (Dietterich 1998).
3. Tests for the logit-shift and the McNemar contingency math.
- **Acceptance:** both analyses leakage-free and documented either way; the frozen
  final model is unchanged; tests pass. **Skippable without blocking Slices 4-5.**

### Slice 4 - LaTeX report
1. Scaffold `reports/vit/main.tex` + `sections/*.tex` + `refs.bib` following
   `project_plan.md §10` (Giriş / Yöntem / Deneyler / Sonuçlar / Tartışma / Sonuç).
2. Tables: (1) frozen ablation, (2) selected 5-fold mean±std + bootstrap CI,
   (3) per-class P/R/F1/support for the final model, (4) training-time / transfer
   comparison; **comparator positioning table** (Wang 2023 / GastroViT / Effimix).
3. Figures: architecture diagram, confusion matrix, training curves, rollout +
   Grad-CAM++ panels, UMAP.
4. Discussion (§5.5, most weighted): which backbone learned better features & why;
   did fusion help (concat vs weighted vs **GMU = honest negative**); seed-ensemble /
   TTA = CI-overlapping; frozen vs fine-tune trade-off + throughput; **honest
   imbalance / rare-class limitation** (macro-F1 ceiling); future work.
5. Every headline number traces to `metrics.json` / a predictions file / a paper
   reference (no overclaim); `refs.bib` cites the exact reference folders.
- **Acceptance:** report compiles to PDF, satisfies §10 structure + the §11 checklist
  items it owns, includes the negative results, and every number is traceable.

### Slice 5 - YouTube video
1. Write a script/outline (problem → method → experiments → results → interpretability
   → honest limitations) + a recording checklist; note what to demo on screen.
2. Add the video link placeholder to the report + README.
- **Acceptance:** a complete script/outline + checklist committed; link placeholder
  wired in. (Recording/upload is the human step.)

### Slice 6 - Close-out
1. Docs sync: `results_progress.md`, `experiment_log.md`, `decisions.md` (only if
   Slice 3 produced a VLD note), `project_plan.md §3` (mark S5 done).
2. `uv run pytest tests/`; hygiene check (no large artifacts, `backbones.py`
   untouched, final-model numbers unchanged).
3. Move `011-vit-report.md` to `completed/` **only on explicit sprint-close approval**.
- **Acceptance:** docs consistent; pytest green; hygiene clean; plan archived on
  approval.

## 7. Risks

- **Deadline (total impact):** report + video are mandatory; ~2 days. Mitigation —
  do Slices 1-2 + 4-5 first; Slice 3 is explicitly skippable; freeze experiments
  (already done end-S4/S4.5).
- **Grad-CAM++ on Swin:** hierarchical/shifted-window layout breaks the naive
  `reshape_transform`. Mitigation — validate on ViT-B/BEiT-B, document the caveat,
  fall back rather than ship a misleading map.
- **Leakage in figures/Slice 3:** any per-image or per-prior analysis must use OOF
  test images and training-fold priors only (VLD-13). Mitigation — reuse the
  `eval_tta_ensemble.py` / fold-manifest discipline; tests on the math.
- **Scope creep into modeling:** the temptation to "just retrain with logit adj /
  focal." Mitigation — Slice 3 is inference-only and additive; the champion is frozen
  (VLD-18/VLD-19).
- **Large artifacts:** checkpoints/predictions stay gitignored; only small PNG/PDF
  figures committed under `reports/vit/figures/`.
- **Number drift:** all report numbers must match `docs/FINAL_MODEL.md` /
  `metrics.json`; no re-derivation by hand.

## 8. Acceptance Criteria

- Attention rollout + Grad-CAM++ panels and a UMAP map generated **inference-only**
  from `11_triple_weighted/best.pt` over **OOF test images** (VLD-13), saved under
  `reports/vit/figures/`, each traceable to a checkpoint + fold manifest; Swin caveat
  documented.
- (If run) Slice 3's logit-adjustment arm and McNemar test are leakage-free,
  documented honestly either way, and leave the frozen final model unchanged.
- LaTeX report under `reports/vit/` compiles to PDF, follows `project_plan.md §10`,
  includes the comparator table + the honest negative results + the imbalance
  limitation, and every headline number traces to `metrics.json` / predictions / a
  paper reference.
- YouTube video script/outline + recording checklist committed; link placeholder in
  report + README.
- `uv run pytest tests/` passes; `src/models/backbones.py` untouched; locked VLD
  values, splits, class defs, and final-model numbers unchanged; **0 A100 units**; no
  large artifacts committed.
- Sprint closed only on explicit approval; `011-vit-report.md` then moved to
  `completed/`.

## 9. Commands to Run

```powershell
git switch -c sprint5/vit-report   # off a main that includes Sprint 4 + 4.5

# Interpretability (inference-only, local; no A100):
uv run python scripts/interpret_attention_rollout.py --run 11_triple_weighted --split oof
uv run python scripts/interpret_gradcam.py          --run 11_triple_weighted --split oof
uv run python scripts/interpret_umap.py             --run 11_triple_weighted --split oof

# Optional Slice 3 (leakage-free, no retraining):
uv run python scripts/eval_logit_adjust.py --run 11_triple_weighted
uv run python scripts/stats_mcnemar.py     --champion 11_triple_weighted --baseline <baseline_id>

# Report:
#   cd reports/vit && latexmk -pdf main.tex   (or the project's LaTeX toolchain)

uv run pytest tests/
git diff -- src/models/backbones.py    # must be empty
```

## 10. Documentation Updates Required

- `docs/vit/results_progress.md`: Sprint 5 section — interpretability figures
  summary, (optional) logit-adjust / McNemar result, report/video status.
- `docs/vit/experiment_log.md`: commands run, checkpoints used, figure provenance,
  pytest result.
- `docs/vit/decisions.md`: a VLD note **only if** Slice 3 yields a decision (e.g.
  "logit adjustment evaluated, no gain — final model unchanged").
- `docs/vit/project_plan.md §3`: mark S5 done (End-S5 checkpoint: report draft v1 +
  video outline).
- `README.md`: report pointer + YouTube link placeholder.
- `docs/exec-plans/active/011-vit-report.md`: keep active until Sprint 5 completion
  is explicitly approved; move to `completed/` only then.
