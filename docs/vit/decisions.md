# Decisions — Multi-ViT Feature Fusion Project

Append-only log of locked decisions for the ViT project. Mirrors the CNN
project's `docs/decisions.md`. Locked decisions (VLD-*) cannot be changed
silently; the canonical summary table is in `docs/vit/project_plan.md §6`.

This project lives in the same repository as the completed CNN project
(monorepo, additive). It reuses shared infrastructure and references; CNN
decisions D-07/D-08/D-09 from `docs/decisions.md` are reused where noted.

---

## 2026-06-15 — VLD-01: Dataset (HyperKvasir 23-class, reused)

- **Dataset = HyperKvasir 23-class labeled subset** (10,662 images, severe class
  imbalance), reused unchanged from the CNN project. Satisfies assignment §3.0
  (public, multi-class, used in 2022–2025 GI-endoscopy studies).
- The official **5-fold** split (`data/splits/official/5_fold_split.csv`) and its
  fold-materialization logic are the primary protocol (shared with the CNN project).

## 2026-06-15 — VLD-14: Repository strategy (monorepo, additive)

- The ViT project is **added to the existing repo**, not a new repo and not a
  branch that deletes CNN. Rationale: shared code (`src/`, `scripts/`, `data/`,
  `references/`) is reused without duplication, and the eventual CNN-vs-ViT-vs-
  literature comparison becomes native instead of requiring a later reunification.
- **Additive layout now; symmetric `cnn/`–`vit/` split deferred to post-assignment.**
  CNN files at the repo root stay untouched (frozen). New ViT artifacts are
  namespaced under `vit/`: `configs/vit/`, `results/vit/`, `reports/vit/`,
  `docs/vit/`. Shared engine code is **not** split by project.
- `src/` additions for ViT are new files (`vit_backbones.py`,
  `attention_rollout.py`) + small edits to `full_model.py` and
  `training/optimizers.py`. The CNN `backbones.py` is **not edited**.
- Safety: tag the CNN submission commit `cnn-submission` before any reorg so the
  exact frozen state is always recoverable. Work happens on branch `vit-fusion`,
  merged to `main` at submission.
- The GitHub repo may be renamed to `hyperkvasir-multi-backbone-fusion`; the
  local folder name is left unchanged (keeps tooling/memory paths stable).

## 2026-06-15 — VLD-11, VLD-12: Compute strategy (Colab Pro A100)

- Primary compute is **Google Colab Pro — A100 (40 GB), ~100 units/month**. This
  budget (not VRAM) is the binding constraint. The RTX 5080 is no longer used for
  this project.
- **Never spend A100 units on frozen work.** Feature caching and the entire
  frozen-path ablation run on any non-A100 session at zero A100 cost.
- **Funnel (VLD-12):** ablation on fold-0 for all configs; promote only the top
  ~5 to fold-0 fine-tune; 5-fold CV only for the top-4. Target ≈65 A100 units,
  leaving margin for reruns.
- Reuse CNN **D-09** provenance gate (asserts A100, aborts T4) and Drive staging
  for ephemeral sessions; **D-08** A100 exploration config conventions carry over.

## 2026-06-15 — VLD-02, VLD-03: Backbone selection (N=3, transformer-pure, timm)

- **N=3 backbones, locked:** ViT-B/16 + Swin-T + **BEiT-B/16**. The mandatory pair
  (Vanilla ViT + Swin, §3.1) plus a 3rd model satisfies §2's "üç farklı ViT".
- Three distinct design axes: isotropic+supervised (ViT-B), hierarchical+windowed
  (Swin-T), isotropic+self-supervised/MIM (BEiT-B).
- **3rd model = BEiT-B/16** chosen over alternatives: CaiT-S12 and original
  PVT-Small are **not in the public timm registry** (verified); ConvNeXt-Tiny is a
  CNN and was rejected to keep scope true to a "ViT" project despite strong
  complementarity in the literature.
- **No 4th model** on the Pro budget. If upgraded to Pro+, the transformer-pure
  4th would be **PVTv2-B2** (`pvt_v2_b2.in1k`), not EfficientFormer-L1.
- **VLD-03: timm with `pretrained=True` is the required backbone source.** This
  **reverses** the CNN project's PLD-15 ("do not use timm"). Verified strings
  (timm 1.x, 2026-06-16): `vit_base_patch16_224.orig_in21k_ft_in1k`,
  `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`,
  `beit_base_patch16_224.in22k_ft_in22k_in1k` — all load, 768-d pooled output.

## 2026-06-15 — VLD-04, VLD-09: Feature extraction & preprocessing

- **VLD-04:** extract features via `timm.create_model(name, pretrained=True,
  num_classes=0)` then `model(x)`, returning each backbone's **native pooled
  pre-logits** (768-d for all three). This is architecture-correct for both CLS
  (ViT/BEiT) and CLS-less (Swin) models. Do **not** hard-code
  `forward_features(x)[:,0]`.
- **VLD-09:** 224×224 input, ImageNet normalization, **bicubic** interpolation
  with `crop_pct≈0.9` (timm ViT training fidelity), not torchvision's bilinear.

## 2026-06-15 — VLD-05, VLD-06, VLD-07, VLD-08: Projection / classifier / fusion / transfer

- **VLD-05:** per-branch projection `Linear→LayerNorm→GELU` to **512-d**, reused
  unchanged from the CNN project (parametric in input dim).
- **VLD-06:** classifier is **MLP only** (hidden=256, dropout=0.3). The assignment
  §5.1 "3 sınıflandırıcı" phrase is the **same instructor wording error** as in the
  CNN project (§3.4 specifies only MLP). Comparisons are ViT-wise / fusion-wise /
  transfer-wise, never classifier-wise. Report will note this explicitly.
- **VLD-07:** mandatory fusion = **concatenation + weighted**. **GMU is an
  optional/stretch** extra (reuses `gmu_arevalo_2017_iclr_workshop`,
  backbone-agnostic on 512-d features); add only after the MVP is stable.
- **VLD-08:** two transfer regimes — fully **frozen** feature extraction and
  **fine-tune the last transformer blocks**. Per assignment §4.2 ("son 2 veya 3
  ... katman"), we unfreeze the **last 3 blocks** for ViT/BEiT (`blocks[9:]`) +
  final norm + head, and the **final stage** for Swin-T (+ final norm + head).
  3 chosen over 2 for more task-adaptation capacity while staying within the
  assignment's 2–3 range. **No BatchNorm logic** (LayerNorm models): a frozen
  branch is `.eval()` to disable drop_path; fine-tune uses a modest `drop_path_rate`.

## 2026-06-15 — VLD-10, VLD-13: Evaluation & ensembling

- **VLD-10:** headline metric **macro-F1** (HyperKvasir is imbalanced); also report
  Accuracy / macro Precision / Recall / F1 (assignment §4.2) with bootstrap 95% CI.
- **VLD-13:** only **leakage-free** ensembling — seed ensemble and TTA averaging
  softmax within each fold's held-out test set. **No cross-fold model averaging.**
  Direct reuse of CNN **D-07** policy.

## 2026-06-15 — VLD-15: ViT training recipe (deltas vs CNN project)

- **backbone LR 2e-5–5e-5** (ViT) / 3e-5–8e-5 (Swin) for fine-tune — NOT the CNN
  project's 1e-4 (too high for pretrained ViTs; risks catastrophic forgetting).
- **LLRD 0.65–0.75 per encoder layer** — the CNN-stage `_BACKBONE_BLOCKS` spec in
  `src/training/optimizers.py` must be re-mapped to ViT layers (12 for ViT/BEiT;
  Swin stages).
- **Reduce CutMix:** α=1.0/p=0.5 can erase tiny mucosal lesions in endoscopy →
  use **mild MixUp (0.1–0.2), CutMix off/small**. (Deviation from the CNN config.)
- weight_decay **0.05**; drop_path **0.05** (replaces BN handling); label smoothing
  0.05→0.1; EMA decay **0.9998**; warmup **5** (extend to 8–10 only on instability);
  **probe (head-only) → unfreeze top blocks** schedule.

## 2026-06-16 — VLD-16: References (ViT additions / removals)

- Added (auto-extracted `paper.md` + `metadata.yaml` + `original.pdf`), indexed in
  `references/INDEX.md` **Section H** (#32–41):
  - Backbones: `vit_dosovitskiy_2021_iclr`, `swin_liu_2021_iccv`, `beit_bao_2022_iclr`.
  - Training: `stochastic_depth_huang_2016_eccv`, `mixup_zhang_2018_iclr`,
    `mae_he_2022_cvpr` (optional).
  - Interpretability: `attention_rollout_abnar_2020_acl`.
  - Baselines: `09_wang_2023_hsw_vit` (highest-priority direct HyperKvasir ViT
    comparator), `10_subedi_2024_cnn_swin_fusion` (fusion context), `11_varam_2024_edge_vits_capsule` (contextual).
- Removed: `dropblock_ghiasi_2018_neurips` (superseded by stochastic depth) and
  `swa_izmailov_2018_uai` (EMA used, not SWA). Marked REMOVED in INDEX.
- CNN backbone refs (ResNet/MobileNet/EfficientNet) retained as comparison/prior-work.
- Citation traceability rule unchanged: report numbers must trace to a `paper.md`
  Table/Section or a `metrics.json` (`references/INDEX.md §0`, `AGENTS.md`).

## 2026-06-21 — VLD-17: ViT fine-tune throughput config (Sprint 3.5)

- **Problem:** the pre-3.5 fine-tune pipeline ran the A100 at laptop-5080
  throughput. Root cause measured with `scripts/benchmark_vit_throughput.py`:
  the image DataLoader used `num_workers=0` (GPU starved), TF32 off, fp16
  autocast, and cuDNN in deterministic / no-autotuner mode.
- **Decision:** the ViT **fine-tune** path (`configs/vit/training/vit_finetune.yaml`)
  adopts a throughput config — DataLoader `num_workers=8` + `pin_memory` +
  `persistent_workers` + `prefetch_factor=4`; `cudnn.benchmark=true` +
  `deterministic=false`; TF32 on; **bfloat16** autocast (range-safe for ViT on
  A100). All knobs are config-driven and default-off, so the frozen ViT path and
  the CNN project are unchanged. `batch_size` stays 32 (a batch increase changes
  optimization and is out of scope for this perf work).
- **Measured speedup (Colab A100, bs 32):** single Swin-T 5.3x, triple 5.9x
  (img/s). In `fast` mode single throughput ~= triple => still CPU/data bound;
  higher `num_workers`/batch can be swept via `--fast-workers` in the benchmark.
- **Reproducibility trade-off:** the autotuner / non-determinism / TF32 / bf16
  drop bitwise reproducibility. Seeds are still set, so runs are **approximately**
  reproducible. Sprint 3 fold-0 results were produced with the deterministic fp16
  config and are NOT invalidated; Sprint 4 onward uses this throughput config.
  A Slice-3 correctness re-run confirms test macro-F1 stays within run-to-run
  noise of Sprint 3.
- Applies to the ViT fine-tune path only; does not change the VLD-08/09/15
  modelling decisions.

## 2026-06-23 — VLD-18: Final ViT model selection + CV/CI methodology (Sprint 4)

- **Final selected ViT fusion model = `11_triple_weighted`** — ViT-B/16 + Swin-T +
  BEiT-B/16, **weighted** fusion of the 512-d branch projections, MLP head,
  fine-tuned last blocks with the tuned recipe (VLD-15 + VLD-17). Chosen as best
  by 5-fold mean macro-F1.
- **CV/CI methodology (locked):** official 5-fold (folds 0–4); per-fold + mean ±
  std for Acc / macro-F1 / macro-P / macro-R; **bootstrap 95% CI on macro-F1**
  over the concatenated out-of-fold test predictions (folds verified disjoint and
  covering the dataset once → leakage-free, VLD-13; never average fold models).
  Headline metric = macro-F1 (VLD-10).
- **Result:** 5-fold macro-F1 **0.6094 ± 0.0254** (pooled OOF 0.6119
  [0.5930, 0.6295]); with the 3-seed ensemble **0.6157 [0.5994, 0.6321]**.
- **Leakage-free extras (VLD-13):** TTA (orig+hflip, within-fold) gave **no gain**
  (−0.0011); the top-1 **seed ensemble** (seeds 42/123/2024, within-fold softmax
  average) gave **+0.0041**. Both reported with CI.
- **Honest caveats (for the report §5.5):** (a) the top configs' CIs overlap
  (triple [0.593, 0.630] vs single Swin-T [0.581, 0.615]) — best by mean, not a
  clean statistical separation; (b) the CV used the throughput config (VLD-17,
  approximate reproducibility), so the numbers are not bitwise-reproducible.
- Supersedes the Sprint 3 fold-0 ranking for model selection; does not change any
  earlier VLD modelling decision.

## 2026-06-23 — VLD-19: GMU gate fidelity (element-wise, Sprint 4.5)

- **Problem:** `src/models/fusion/gmu.py` used a **per-branch scalar softmax gate**
  (z ∈ ℝ^N), but Arevalo et al. 2017 §3.1 specify an **element-wise** gate
  ("multiplicative gates that assign importance to various features
  simultaneously"): z has the dimensionality of the branch activations,
  `h = z ⊙ h_v + (1−z) ⊙ h_t`. The scalar version loses the per-feature gating
  (it behaves like an input-gated weighted sum, close to our `weighted` fusion).
- **Decision:** add a config-driven `gate_mode`:
  - `"elementwise"` (faithful): `z = softmax_over_branches(reshape(W_z·concat(x),
    (N,D)))` → (B,N,D); `h = Σ_i z_i ⊙ h_i`. For N=2 this reduces **exactly** to
    the paper's tied bimodal element-wise gate.
  - `"scalar"` (default): the legacy per-branch gate, kept so the CNN project's
    `15_triple_gmu` run and all existing configs are byte-for-byte unchanged.
- **ViT GMU (Sprint 4.5) uses `gate_mode: elementwise`** (method `fusion_kwargs`,
  wired through `full_model.py` / `FrozenHeadModel`), so the "GMU (Arevalo et al.
  2017)" citation is faithful. The CNN GMU number was produced with the scalar
  gate — documented here; the frozen CNN artifacts are not changed (PLD-/tag
  `cnn-submission`).
- Does not change any other VLD; concat/weighted fusion untouched.
- **Ablation result (Sprint 4.5 fold-0 screen, faithful gate):** GMU did NOT beat
  the best weighted. Triple-GMU 0.5525 vs triple-weighted 0.6102 (−0.058); only
  S+B-GMU edged its weighted counterpart (+0.0137) but stayed below the headline.
  No GMU config was promoted to 5-fold CV; the **final model remains
  `11_triple_weighted`** (VLD-18). GMU is reported as an honest negative result.

## 2026-06-24 — VLD-20: Sprint 5 leakage-free add-ons (logit adjustment + McNemar)

- **Scope:** two ADDITIVE, inference-only, leakage-free analyses for the report —
  they never re-select the champion (`11_triple_weighted` stays final, VLD-18).
- **Post-hoc logit adjustment (Menon et al. 2020):** subtract `τ·log(train_prior)`
  from recomputed OOF test logits (prior from each fold's TRAIN split only,
  VLD-13). Result is a **clean honest negative**: OOF macro-F1 falls monotonically
  with τ — τ=0 (baseline) 0.6118 [0.593, 0.629] → τ=0.5 0.6022 → **τ=1.0 (headline)
  0.4911** [0.481, 0.500] → collapses beyond. **Why:** the model trains with a
  WeightedRandomSampler (balanced sampling), so its implicit prior is already
  ~uniform; subtracting the train prior **double-corrects** and over-shifts toward
  rare classes. Takeaway for §5.5: the balanced sampler is the correct imbalance
  mechanism here; post-hoc prior correction does not help (degrades macro-F1).
- **McNemar's test (Dietterich 1998), paired OOF (N=10662):** the champion is
  **significantly more accurate** than the best single backbone
  (`02_single_swin_t`: 422 vs 315 discordant, χ²cc 15.25, exact p = 9.2e-05) and
  than the best pair (`09_pair_swin_t_beit_b_weighted`: 423 vs 288, χ²cc 25.25,
  p = 4.7e-07). Note: this is an **accuracy-level** paired-correctness test, not a
  macro-F1 test — it complements (does not contradict) the CI-overlapping macro-F1
  story (fusion adds significantly correct predictions overall, while the macro-F1
  gain is bounded by the rare classes).
- Built-in check: τ=0 reproduces the stored champion OOF predictions
  (10661/10662; 1 near-tie argmax flip from the tuned bf16/TF32 path vs the fp32
  recompute — VLD-17 "reproducibility approximate"). Does not change any VLD; final
  model, splits, recipe, and all Sprint 4/4.5 numbers unchanged.

## Open instructor-ambiguity flags (to address in the report)

1. **§2 "üç farklı ViT" vs §3.1 (two mandatory).** Resolved by N=3 (VLD-02).
2. **§5.1 "3 sınıflandırıcı" vs §3.4 (MLP only).** Instructor wording error; only
   MLP implemented (VLD-06).
