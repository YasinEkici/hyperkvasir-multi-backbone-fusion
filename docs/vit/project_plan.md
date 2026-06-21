# Project Plan — Multi-ViT Feature Fusion for HyperKvasir

> **HOW TO USE THIS DOCUMENT**
>
> Pairs with `docs/vit/project_structure.md` (what to build) and
> `docs/vit/final_assignment.md` (instructor's verbatim requirements). This file
> defines what to do, when, and how to verify the work.
>
> **Scope note:** the classifier is **MLP only**. The assignment §5.1 mentions a
> "3 sınıflandırıcı bazlı" comparison, but §3.4 specifies only MLP — this is the
> same instructor wording error as in the prior CNN project. All comparisons are
> **ViT-model-wise, fusion-wise, and transfer-learning-wise**, never classifier-wise.

---

## 1. Project Context

**Course:** Deep Learning, Spring 2026
**Assignment:** "Çoklu ViT Tabanlı Özellik Füzyonu ile Sınıflandırma"
**Deadline:** **TBD** — instructor left the date blank in `final_assignment.md §6`
(fill once announced; a YouTube video upload is **mandatory** for grading).
**Primary hardware:** Google Colab **Pro** — A100 (40 GB), **100 units/mo**.

**Reference paper supplied by instructor:**
Bilal, H., et al. (2025). *M-ViT-RF: A hybrid deep learning model for accurate
pediatric skeletal age estimation using hand bone radiographs.* Alexandria
Engineering Journal, 129, 289-301.

**Dataset:** HyperKvasir 23-class labeled subset (reused from the CNN project;
public, multi-class, used in 2022-2025 GI-endoscopy studies — satisfies §3.0).

### Assignment requirement → our design mapping

| Requirement (final_assignment.md) | Our design |
|---|---|
| §3.1 ViT models (Vanilla ViT + Swin mandatory); §2 "üç farklı ViT" | **N=3:** ViT-B/16 + Swin-T + **BEiT-B/16** (satisfies both the pair floor and the "three" wording) |
| §3.2 remove head, take feature vector | `timm num_classes=0` → native pooled 768-d per branch |
| §3.3 concat + weighted fusion | both mandatory; GMU optional extra |
| §3.4 MLP classifier | MLP only (hidden=256, dropout=0.3) |
| §4.1 single / two(/three) ablation | full single/pair/triple matrix |
| §4.2 frozen feature-extraction + fine-tune last transformer blocks | both transfer regimes |
| §4.2 metrics: Acc, Precision, Recall, F1 | reported; **macro-F1 headline**, + bootstrap 95% CI |

---

## 2. Scope by Priority

**MVP / must finish:**
1. Correct data pipeline reusing official 5-fold manifests.
2. Three timm backbones: ViT-B/16, Swin-T, BEiT-B/16 (frozen feature extractors).
3. Mandatory fusion: concatenation and weighted.
4. MLP classifier only.
5. Single / two / three backbone ablations.
6. Frozen feature extraction **and** fine-tune-last-blocks variants.
7. Accuracy, macro precision/recall/F1, per-class metrics, confusion matrix.

**Strong version / target if MVP stable:**
1. 5-fold CV on the selected top-4 configurations.
2. Bootstrap 95% CI for headline macro-F1.
3. Attention-rollout + Grad-CAM interpretability for the best model.
4. Probe-then-finetune schedule; LLRD; EMA.

**Stretch / only if time and units remain:**
1. GMU as an extra advanced fusion method.
2. Seed ensemble + TTA (leakage-free, per D-07).
3. UMAP feature-space visualization.
4. 4th backbone (PVTv2-B2) — only if upgraded to Colab Pro+.

---

## 3. Timeline (Sprints) + Compute Funnel

Continues the repo's existing `docs/exec-plans/` sprint machinery (CNN used
`001..004`; ViT continues at `005`). Each sprint = one exec-plan, approval-gated
(no auto-advance).

| Sprint | Exec-plan | Focus | A100 units |
|---|---|---|---:|
| **S1 — Foundation** | `005-vit-foundation.md` | `vit-fusion` branch; timm backbone factory (verify attribute paths); `configs/vit/` skeleton; feature-extraction path; smoke test (shape asserts) | 0 |
| **S2 — Frozen ablation** | `006-vit-frozen-ablation.md` | cache 3 ViT features (all folds, off-A100); run **all** frozen single/pair/triple configs; rank by fold-0 macro-F1 | **0** |
| **S3 — Fine-tune funnel** | `007-vit-finetune.md` | fine-tune **top ~5** configs on fold-0 (A100); pick top-4 | ~11 |
| **S3.5 — A100 throughput** | `008-vit-perf.md` | fix DataLoader starvation (num_workers) + TF32 + bf16 + cuDNN autotuner; benchmark + correctness; lock tuned fine-tune config (VLD-17) | ~few (benchmark) |
| **S4 — 5-fold CV + CI** | `009-vit-cv.md` | top-4 × 5 folds; bootstrap CI; (optional) seed draw + TTA | ~8–11 (tuned; was ~43–54) |
| **S5 — Interpretability + report** | `010-vit-report.md` | attention rollout / Grad-CAM / UMAP; LaTeX report under `reports/vit/`; YouTube video | 0 |

**Funnel rule (protects the ~100-unit budget):** frozen-rank everything off-A100
→ fine-tune only the pruned top set on A100 → 5-fold only the top-4. Never
fine-tune all configs. After the Sprint 3.5 throughput tuning (VLD-17), the A100
fine-tune cost dropped ~5–6×, so the realistic total is well under budget.

### Hard checkpoints
- **End S1:** smoke test asserts the three backbones return `(B, 768)`; timm
  `model.blocks` / `model.layers` paths confirmed.
- **End S2:** frozen ablation table ranks all configs; top set chosen.
- **End S3:** fold-0 fine-tune done; top-4 locked.
- **End S3.5:** ViT fine-tune DataLoader/AMP/TF32 tuned (~5–6× A100 throughput);
  correctness within run-to-run noise; tuned config locked for S4 (VLD-17).
- **End S4:** 5-fold results with CIs are report-ready.
- **End S5:** report draft v1 + video outline.

---

## 4. Experiment Matrix

Backbones: `V`=ViT-B/16, `S`=Swin-T, `B`=BEiT-B/16. Strategy = fold-0 for all
ablation rows, 5-fold only for the selected top-4 (mirrors the CNN funnel).

### Stage 0 — smoke (fold-0, tiny)
Single V/S/B frozen (shape + dim sanity); triple concat frozen (fusion/MLP sanity).

### Stage 1 — frozen ablation (fold-0, **off-A100**, rank all)
| Tier | Configs |
|---|---|
| Single | V, S, B (frozen) |
| Pair | V+S, V+B, S+B × {concat, weighted} (frozen) |
| Triple | V+S+B × {concat, weighted} (frozen) |

### Stage 2 — fine-tune funnel (fold-0, A100)
Fine-tune only the **top ~5** configs from Stage-1 ranking (singles + best pairs/triple).

### Stage 3 — selected 5-fold CV (A100)
Top-4 overall × 5 folds; mean ± std; bootstrap 95% CI on macro-F1.

### Stage 4 — stretch
GMU triple; seed ensemble + TTA; UMAP. Drop if budget/time runs out.

---

## 5. Anti-Hallucination Protocol (ViT-specific)

### 5.1 Common hallucinations here
1. Wrong timm attribute paths (`model.blocks` vs `model.stages` vs `model.layers`).
2. Applying CLS-token logic to Swin/BEiT (no/added tokens).
3. Inventing a timm model string that isn't in the registry (CaiT-S12, PVT-Small).
4. Carrying over BatchNorm-freeze logic to LayerNorm models.
5. Reusing the CNN backbone-LR (1e-4) — too high for pretrained ViTs.
6. Strong CutMix erasing small endoscopic lesions.
7. Averaging fold models before macro-F1 (leakage).

### 5.2 Verify before declaring a module done
1. Match the signature in `project_structure.md §6`.
2. Add/update the matching test.
3. `uv run pytest`.
4. Backbone shape check:
```python
x = torch.randn(2, 3, 224, 224)
for n in ["vit_base_patch16_224", "swin_tiny_patch4_window7_224", "beit_base_patch16_224"]:
    assert ViTFeatureExtractor(n)(x).shape == (2, 768)
```
5. Fusion output-shape check (concat → N×512, weighted → 512).
6. Note each successful run in `docs/vit/experiment_log.md`.

### 5.3 Stop-and-ask triggers
Ambiguous paper equation; live timm behavior differs from assumed; a test fails
after a fix; new dependency needed; wants to simplify scope; wants a non-MLP
classifier; wants to change class count; A100 unavailable for an A100-only stage.

---

## 6. Locked Design Decisions (VLD-*)

| ID | Decision | Reason |
|---|---|---|
| VLD-01 | Dataset: HyperKvasir 23-class labeled subset | Reuse; public, multi-class, recent literature |
| VLD-02 | Backbones N=3: ViT-B/16 + Swin-T + BEiT-B/16 | Mandatory pair + 3rd (SSL axis); satisfies §2 "three" |
| VLD-03 | Backbone source: **timm, pretrained=True** | Required for ViT family (reverses CNN PLD-15) |
| VLD-04 | Feature extraction: `num_classes=0` → native pooled 768-d | Architecture-correct for CLS and CLS-less models |
| VLD-05 | Per-branch projection dim: 512, Linear+LayerNorm+GELU | Reuse; handles any input dim |
| VLD-06 | Classifier: MLP only (hidden=256, dropout=0.3) | Assignment; §5.1 "3 classifiers" is a wording error |
| VLD-07 | Mandatory fusion: concat + weighted; GMU optional | Assignment §3.3 |
| VLD-08 | Transfer: frozen + fine-tune last transformer blocks | Assignment §4.2; no BN/conv logic |
| VLD-09 | Image size 224, ImageNet norm, **bicubic** interpolation | timm ViT training fidelity |
| VLD-10 | Headline metric: macro-F1 (+ Acc/P/R/F1, bootstrap CI) | Imbalance; assignment metrics |
| VLD-11 | Compute: Colab Pro A100, 100 units/mo; never bill frozen work | Budget is the binding constraint |
| VLD-12 | Ablation on fold-0; 5-fold only for top-4 | Funnel; fits unit budget |
| VLD-13 | Leakage-free ensembling only (seed/TTA), no cross-fold averaging | Reuse CNN D-07 policy |
| VLD-14 | Monorepo additive; ViT artifacts under `vit/`; src/ shared | Serves later CNN-vs-ViT comparison |

---

## 7. Methods to Implement

| Method | Read first | Difficulty |
|---|---|---:|
| timm ViT feature extractor | `references/methodology_backbones/{vit,swin,beit}/paper.md` | medium |
| Concatenation fusion | (reuse) | low |
| Weighted fusion | (reuse) | low |
| MLP classifier | (reuse) | low |
| Frozen feature cache | `project_structure.md §6` | low |
| Fine-tune + ViT LLRD | `references/methodology_training/llrd_*`, `mae_he_2022` | medium |
| Attention rollout | `references/methodology_evaluation/attention_rollout_abnar_2020_acl/paper.md` | medium |
| GMU (optional) | `references/methodology_fusion/gmu_arevalo_2017_iclr_workshop/paper.md` | low-medium |

> **Reference status:** only `gmu_*` and `llrd_*` folders already exist (gmu has a
> stub; llrd is an empty placeholder). The `vit/swin/beit`, `attention_rollout`,
> and `mae_he_2022` stubs are **to be created** (see the references plan) before
> these are cited in the report.

---

## 8. Training Recipe (ViT deltas vs CNN project)

| Hyperparameter | Value | Note |
|---|---|---|
| head_lr | 1e-3 | fresh head |
| backbone_lr (fine-tune) | **2e-5 – 5e-5** (ViT), 3e-5 – 8e-5 (Swin) | **NOT 1e-4** — avoid forgetting |
| LLRD decay | **0.65 – 0.75 per encoder layer** | re-mapped from CNN stages |
| weight_decay | **0.05** | higher than CNN's 1e-4 |
| warmup | **5 epochs** | extend to 8–10 only if loss spikes |
| drop_path | **0.05** | replaces BN logic; up to 0.1 if overfitting |
| label smoothing | 0.05 → 0.1 | |
| MixUp / CutMix | **mild MixUp 0.1–0.2, CutMix off/small** | CutMix erases small lesions |
| EMA | on, decay **0.9998** | |
| schedule | probe (head-only) → unfreeze top blocks | small-data best practice |
| optimizer | AdamW | |

---

## 9. Risk Register

| Risk | Prob | Impact | Mitigation |
|---|---:|---:|---|
| A100 unavailable on Pro | Medium | High | Frozen stages run on any GPU; reserve A100 for fine-tune only |
| 100-unit budget exhausted | Medium | High | Funnel; frozen off-A100; prune before fine-tuning |
| timm attribute paths differ | Medium | Medium | S1 smoke test verifies before relying |
| Catastrophic forgetting on small data | Medium | High | Low backbone LR, partial unfreeze, LLRD, EMA, probe-first |
| Class-collapse under imbalance | Medium | Medium | macro-F1 selection, weighted sampler |
| Over-augmentation erases lesions | Medium | Medium | mild MixUp, CutMix off |
| Ephemeral Colab sessions lose work | Medium | Medium | per-fold checkpoint to Drive; resumable runs |
| Video/report deadline | Medium | Total | freeze experiments by end S4 |

---

## 10. Report Deliverables (maps to assignment §5)

1. **Giriş (§5.1):** problem, approach overview, comparison axes (fusion-wise,
   ViT-model-wise; note MLP-only and why the "3 classifier" line is not applicable).
2. **Yöntem (§5.2):** ViT-B/Swin-T/BEiT-B; feature extraction; projection; concat
   & weighted fusion; MLP; frozen vs fine-tune.
3. **Deneyler (§5.3):** dataset, split protocol, hyperparameters, metrics.
4. **Sonuçlar (§5.4):** ablation table, 5-fold mean±std, per-class table,
   confusion matrix, training-time comparison.
5. **Tartışma (§5.5 — most critical):** which backbone learned better features &
   why; did fusion help & why; best combination; strengths/weaknesses; which
   transfer approach won and how long it took.
6. **Sonuç:** summary, limitations, future work.

**Tables:** (1) frozen ablation, (2) selected 5-fold mean±std + CI, (3) per-class
P/R/F1/support for best model, (4) training-time / transfer comparison.
**Figures:** architecture diagram, confusion matrix, training curves, attention
rollout / Grad-CAM, optional UMAP. No classifier-wise table (MLP fixed).

---

## 11. Submission Checklist

- [ ] Code on `vit-fusion`, merged to main before deadline.
- [ ] `pyproject.toml`, `uv.lock`, `.python-version`, `env/requirements-colab.txt` present.
- [ ] `README.md` covers ViT setup + Colab.
- [ ] Report PDF complete (`reports/vit/`).
- [ ] All cited papers have reference stubs; all headline numbers trace to `metrics.json` or paper evidence.
- [ ] **YouTube video uploaded** (mandatory for grading).
- [ ] `.rar` named `birinci_ad_no_ikinci_ad_no.rar` with report + code + video link.
- [ ] Uploaded to ekampüs before deadline.
