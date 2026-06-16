# References Index — Multi-Backbone Feature Fusion (CNN + ViT)

> **Purpose:** This file lists every paper to be referenced in the project,
> organized by role and priority. It also defines the metadata, evidence, and
> caveat structure required before a paper can be cited in the report or used by
> a coding agent.
>
> **Repo scope (2026-06):** this repository now hosts **two projects** — the
> completed CNN fusion project (Sections A–G below) and the new **ViT fusion
> project**. ViT-specific additions are consolidated in **Section H**; the CNN
> backbone references (ResNet/MobileNet/EfficientNet) remain valid as
> comparison/prior-work. See `docs/vit/project_plan.md` for the ViT scope.
>
> **Classifier scope:** both projects use **MLP as the only classifier**. Papers
> using SVM, RandomForest, XGBoost, or other classifiers are cited only as
> background or methodological context.
>
> **Folder convention note:** earlier "Target file" paths below were written as
> flat `*.md`; the actual layout is one folder per paper containing
> `paper.md` + `metadata.yaml` + `original.pdf` + `assets/`.

---

## 0. Metadata and Evidence Rules

Every markdown file under `references/` must include this block near the top:

```yaml
metadata:
  title:
  authors:
  venue:
  year:
  doi:
  arxiv:
  url:
  pdf_url:
  code_url:
  dataset_used:
  class_count:
  directly_comparable: exact | approximate | contextual | not_comparable
  paper_type: peer_reviewed | preprint | documentation | repository
  download_date:
  pdf_sha256:
  zotero_key:
```

Every result or claim we rely on must be written as:

```markdown
## Claims we rely on

| Claim | Evidence location | Caveat | Used in our project? |
|---|---|---|---|
| The paper reports X accuracy on HyperKvasir 23-class. | Table 2 / Section 4.3 | Exact split is not fully specified. | Cross-protocol discussion only. |
```

Do not let an LLM summary become a citation source by itself. The LLM can draft
summaries, but the final markdown must be checked against DOI/arXiv/PDF/source
metadata.

---

## 1. Recommended Metadata Extraction Workflow

Use this order for each paper:

1. **Zotero first** for bibliographic metadata.
   - Add by DOI/arXiv/URL when possible.
   - Export BibTeX or Better BibTeX key.
2. **Crossref / OpenAlex / Semantic Scholar** for DOI-level metadata checks.
   - Use these to fill title, authors, venue, year, DOI, references, abstract
     when available.
3. **GROBID** for PDF header and reference extraction.
   - Use especially when the PDF has a long reference list or unclear metadata.
4. **LLM last** for conversion into the project markdown template.
   - Give the LLM the PDF text or GROBID output.
   - Ask it to fill the template, but require it to quote exact table/section
     locations for all results.
5. **Manual verification** for every number used in the report.
   - Especially accuracy, F1, class count, train/test split, and dataset subset.

---

## Section A — HyperKvasir Comparison Papers (PRIMARY)

These papers are used for the report's comparison and discussion. Only papers
with matching dataset/class protocol are considered direct comparisons; others
are contextual.

### 1. Ramamurthy et al. 2022 — Effimix [HIGHEST PRIORITY]

**Why it matters:** Most direct methodological precedent for feature fusion on
HyperKvasir. It uses EfficientNet-B0 + a custom CNN feature-fusion design on the
23-class HyperKvasir setting and reports very high performance.

- **Citation:** Ramamurthy, K., George, T. T., Shah, Y., & Sasidhar, P. (2022).
  A Novel Multi-Feature Fusion Method for Classification of Gastrointestinal
  Diseases Using Endoscopy Images. *Diagnostics, 12*(10), 2316.
- **DOI:** 10.3390/diagnostics12102316
- **Open Access:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9600128/
- **Target file:** `references/primary_baselines/01_ramamurthy_2022_effimix.md`
- **Comparability:** approximate direct comparison if exact split is not fully reproducible.
- **Verification needed:** confirm reported accuracy/F1, class count, and split protocol from the paper tables.

### 2. GastroViT 2025 [HIGHEST PRIORITY]

**Why it matters:** Modern transformer/ensemble comparison point on HyperKvasir.
Useful to position CNN fusion against ViT-style methods.

- **Citation:** *GastroViT: A Vision Transformer Based Ensemble Learning
  Approach for Gastrointestinal Disease Classification with Grad CAM & SHAP
  Visualization.* (2025).
- **arXiv:** 2509.26502
- **Open Access:** https://arxiv.org/abs/2509.26502
- **Target file:** `references/primary_baselines/02_gastrovit_2025.md`
- **Comparability:** direct only for the 23-class result; 16-class result must not be mixed with our 23-class table.
- **Verification needed:** confirm 23-class accuracy, macro-F1, split, and whether the result is peer-reviewed or preprint.

### 3. Borgli et al. 2020 — HyperKvasir Dataset Paper

**Why it matters:** Canonical dataset citation. Use this for dataset description,
license, class structure, and original baseline context.

- **Citation:** Borgli, H., et al. (2020). HyperKvasir, a comprehensive
  multi-class image and video dataset for gastrointestinal endoscopy.
  *Scientific Data, 7*, 283.
- **DOI:** 10.1038/s41597-020-00622-y
- **Open Access:** https://www.nature.com/articles/s41597-020-00622-y
- **Target file:** `references/primary_baselines/03_borgli_2020_hyperkvasir_dataset.md`
- **Comparability:** dataset source and historical baseline, not a primary method competitor.

### 4. Ahmed et al. 2023 — Hybrid 3-CNN Fusion (Kvasir-ROI)

**Why it matters:** GI endoscopy feature-fusion precedent using multiple CNN
features. Not a direct HyperKvasir 23-class comparison.

- **Citation:** Ahmed, I. A., Senan, E. M., & Shatnawi, H. S. A. (2023).
  Hybrid Models for Endoscopy Image Analysis for Early Detection of
  Gastrointestinal Diseases Based on Fused Features. *Diagnostics, 13*(10), 1758.
- **DOI:** 10.3390/diagnostics13101758
- **Open Access:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10217222/
- **Target file:** `references/primary_baselines/04_ahmed_2023_hybrid_fusion.md`
- **Comparability:** contextual; uses Kvasir-ROI and XGBoost, while our classifier is MLP only.

### 5. He et al. 2025 — HEMF

**Why it matters:** Recent multi-attention feature-fusion framework for medical
image classification. Use only after full-text verification.

- **Citation:** He, J., Shi, Q., Ma, J., Shi, D., & Min, T. (2025). HEMF: an
  adaptive hierarchical enhanced multi-attention feature fusion framework for
  cross-scale medical image classification. *PeerJ Computer Science, 11*, e3181.
- **DOI:** 10.7717/peerj-cs.3181
- **Open Access / Code:** https://github.com/Esgjgd/HEMF
- **Target file:** `references/primary_baselines/05_he_2025_hemf.md`
- **Caveat:** do not cite HyperKvasir numbers unless full text confirms them.

### 6. Ramachandran et al. 2025 — Interpretable Deep Learning (medRxiv preprint)

**Why it matters:** Useful for interpretability and single-CNN discussion.

- **Citation:** Ramachandran et al. (2025). Interpretable Deep Learning
  Approaches for Reliable GI Image Classification: A Study with the HyperKvasir
  Dataset. *medRxiv* preprint.
- **Open Access:** https://www.medrxiv.org/content/10.1101/2025.07.22.25332009v1
- **Target file:** `references/primary_baselines/06_ramachandran_2025_interpretable.md`
- **Comparability:** contextual; preprint and subset-based.

### 7. Curriculum Self-Supervised Learning 2024

**Why it matters:** Current HyperKvasir training strategy baseline, useful as a
non-fusion comparison point.

- **Open Access:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10937990/
- **Target file:** `references/primary_baselines/07_curriculum_ssl_2024.md`
- **Comparability:** contextual unless the exact class count and protocol match.

### 8. Sutton et al. — DenseNet121 Single-CNN Baseline

**Why it matters:** Historical single-CNN context if exact source can be confirmed.

- **Citation:** to be confirmed from Ramachandran 2025 reference list.
- **Target file:** `references/primary_baselines/08_sutton_densenet121.md`
- **Caveat:** do not cite until original source is identified.

---

## Section B — Instructor's Reference Paper

### 9. Bilal et al. 2025 — M-CNN-RF

**Why it matters:** Instructor-provided example of a multi-CNN hybrid method.
Use for motivation and methodological framing, not as an implementation target.

- **Citation:** Bilal, H., Tian, Y., Ullah, I., Garg, S., Choi, B. J., & Hassan,
  M. M. (2025). M-CNN-RF: A hybrid deep learning model for accurate pediatric
  skeletal age estimation using hand bone radiographs. *Alexandria Engineering
  Journal, 129*, 289-301.
- **Target file:** `references/instructor_reference/bilal_2025_mcnn_rf.md`
- **Comparability:** contextual; different task and likely non-MLP classifier.

---

## Section C — Methodology: Fusion Techniques

### 10. GMU — Gated Multimodal Unit (PRIMARY ADVANCED METHOD)

- **Workshop version:** Arevalo, J., Solorio, T., Montes-y-Gómez, M., &
  González, F. A. (2017). Gated Multimodal Units for Information Fusion.
  *ICLR Workshop*. arXiv:1702.01992.
- **Extended journal version:** Arevalo, J., et al. (2020). Gated multimodal
  networks. *Neural Computing and Applications, 32*, 10209-10228.
- **Target files:**
  - `references/methodology_fusion/gmu_arevalo_2017_iclr_workshop.md`
  - `references/methodology_fusion/gmu_arevalo_2020_ncaa.md`
- **Use:** implement one beyond-spec fusion arm after concat and weighted fusion.
- **Priority:** core advanced method.
- **Shared with ViT project (2026-06):** the `gmu_arevalo_2017_iclr_workshop/`
  folder is reused by the ViT fusion project as an **optional/stretch** fusion
  arm (mandatory ViT fusions are concat + weighted only; see Section H and
  `docs/vit/project_plan.md` VLD-07). Backbone-agnostic: operates on the 512-d
  projected features. The 2020 NCAA folder is not present; the 2017 workshop
  stub is sufficient to cite.

### 11. AFF — Attentional Feature Fusion (STRETCH)

- **Citation:** Dai, Y., Gieseke, F., Oehmcke, S., Wu, Y., & Barnard, K. (2021).
  Attentional Feature Fusion. *IEEE WACV*, 3560-3569.
- **DOI:** 10.1109/WACV48630.2021.00360
- **Code:** https://github.com/YimianDai/open-aff
- **Target file:** `references/methodology_fusion/aff_dai_2021_wacv.md`
- **Use:** stretch advanced fusion if GMU and mandatory experiments are complete.

### 12. LMF — Low-rank Multimodal Fusion (STRETCH)

- **Citation:** Liu, Z., Shen, Y., Lakshminarasimhan, V. B., Liang, P. P.,
  Zadeh, A., & Morency, L.-P. (2018). Efficient Low-rank Multimodal Fusion
  with Modality-Specific Factors. *ACL*, 2247-2256.
- **arXiv:** 1806.00064
- **Target file:** `references/methodology_fusion/lmf_liu_2018_acl.md`
- **Use:** stretch tensor-fusion comparison if time remains.

### 13. SE Block (Hu et al. 2018)

- **Citation:** Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation
  Networks. *CVPR*, 7132-7141.
- **Target file:** `references/methodology_fusion/se_block_hu_2018_cvpr.md`
- **Use:** background for channel weighting/gating.

### 14. CBAM (Woo et al. 2018)

- **Citation:** Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM:
  Convolutional Block Attention Module. *ECCV*, 3-19.
- **Target file:** `references/methodology_fusion/cbam_woo_2018_eccv.md`
- **Use:** background for channel/spatial attention.

### 15. MFuRe-CNN / MF-CNN (Montalbo 2024)

- **Citation:** Montalbo, F. J. P. (2024). Development of a multi-fusion
  convolutional neural network for enhanced gastrointestinal disease diagnosis
  in endoscopy image analysis. *PeerJ Computer Science, 10*, e1950.
- **Target file:** `references/methodology_fusion/mfure_cnn_montalbo_2024.md`
- **Use:** 3-backbone GI fusion precedent.

---

## Section D — Methodology: Training Recipe

### 16. CutMix (Yun et al. 2019)

- **Citation:** Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y.
  (2019). CutMix: Regularization Strategy to Train Strong Classifiers with
  Localizable Features. *ICCV*, 6023-6032.
- **Target file:** `references/methodology_training/cutmix_yun_2019_iccv.md`
- **Use:** training augmentation; cite ImageNet gain only as background, not as a promised project gain.

### 17. Label Smoothing (Müller et al. 2019)

- **Citation:** Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label
  Smoothing Help? *NeurIPS*, 4694-4703.
- **Target file:** `references/methodology_training/label_smoothing_muller_2019_neurips.md`

### 18. ULMFiT / LLRD (Howard & Ruder 2018)

- **Citation:** Howard, J., & Ruder, S. (2018). Universal Language Model
  Fine-tuning for Text Classification. *ACL*, 328-339.
- **arXiv:** 1801.06146
- **Target file:** `references/methodology_training/llrd_howard_ruder_2018_acl.md`
- **Caveat:** NLP source; use as general fine-tuning rationale, not as direct CNN evidence.

### 19. AdamW (Loshchilov & Hutter 2019)

- **Citation:** Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay
  Regularization. *ICLR*. arXiv:1711.05101.
- **Target file:** `references/methodology_training/adamw_loshchilov_2019_iclr.md`

### 20. Cosine Annealing (Loshchilov & Hutter 2017)

- **Citation:** Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient
  Descent with Warm Restarts. *ICLR*. arXiv:1608.03983.
- **Target file:** `references/methodology_training/cosine_annealing_loshchilov_2017.md`

### 21. SWA / EMA (Izmailov et al. 2018) — **REMOVED (2026-06)**

- **Status:** folder deleted. The ViT project uses EMA, not SWA, and no canonical
  SWA citation is needed. Not cited.

### 22. RandAugment (Cubuk et al. 2020)

- **Citation:** Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. (2020).
  RandAugment. *NeurIPS Workshops*.
- **Target file:** `references/methodology_training/randaugment_cubuk_2020.md`

### 23. DropBlock (Ghiasi et al. 2018) — **REMOVED (2026-06)**

- **Status:** folder deleted. DropBlock is conv-feature-map specific; the ViT
  project uses **stochastic depth / drop_path** instead (see Section H, #35). Not cited.

---

## Section E — Methodology: Class Imbalance Handling

### 24. Focal Loss (Lin et al. 2017)

- **Citation:** Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
  Focal Loss for Dense Object Detection. *ICCV*, 2980-2988.
- **Target file:** `references/methodology_imbalance/focal_loss_lin_2017_iccv.md`

### 25. Class-Balanced Loss (Cui et al. 2019)

- **Citation:** Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019).
  Class-Balanced Loss Based on Effective Number of Samples. *CVPR*, 9268-9277.
- **Target file:** `references/methodology_imbalance/class_balanced_loss_cui_2019_cvpr.md`

### 26. Weighted Random Sampler (PyTorch documentation)

- **Source:** PyTorch official documentation for `torch.utils.data.WeightedRandomSampler`.
- **Target file:** `references/methodology_imbalance/weighted_random_sampler_torch.md`
- **Use:** compute weights from the training fold only.

---

## Section F — Evaluation, Statistics, Interpretability

### 27. McNemar's Test (Dietterich 1998)

- **Citation:** Dietterich, T. G. (1998). Approximate Statistical Tests for
  Comparing Supervised Classification Learning Algorithms. *Neural Computation,
  10*(7), 1895-1923.
- **Target file:** `references/methodology_evaluation/mcnemar_dietterich_1998.md`

### 28. Grad-CAM (Selvaraju et al. 2017/2020)

- **Citation:** Selvaraju, R. R., et al. Grad-CAM: Visual Explanations from Deep
  Networks via Gradient-based Localization. *IJCV, 128*, 336-359.
- **Target file:** `references/methodology_evaluation/gradcam_selvaraju_2017.md`

### 29. Grad-CAM++ (Chattopadhyay et al. 2018)

- **Citation:** Chattopadhyay, A., Sarkar, A., Howlader, P., & Balasubramanian,
  V. N. (2018). Grad-CAM++: Improved Visual Explanations for Deep
  Convolutional Networks. *IEEE WACV*, 839-847.
- **Target file:** `references/methodology_evaluation/gradcam_pp_chattopadhyay_2018_wacv.md`

### 30. UMAP (McInnes et al. 2018)

- **Citation:** McInnes, L., Healy, J., & Melville, J. (2018). UMAP. *Journal of
  Open Source Software, 3*(29), 861.
- **Target file:** `references/methodology_evaluation/umap_mcinnes_2018_joss.md`

---

## Section G — Secondary Dataset

### 31. DeepWeeds (Olsen et al. 2019)

- **Citation:** Olsen, A., Konovalov, D. A., Philippa, B., Ridd, P., Wood, J. C.,
  Johns, J., et al. (2019). DeepWeeds: A multiclass weed species image dataset
  for deep learning. *Scientific Reports, 9*, 2058.
- **DOI:** 10.1038/s41598-018-38343-3
- **Code/Data:** https://github.com/AlexOlsen/DeepWeeds
- **Target file:** `references/secondary_dataset/deepweeds_olsen_2019_scirep.md`
- **Use:** stretch domain-generality test.

---

## Section H — ViT Project References (added 2026-06)

All folders below contain auto-extracted `paper.md` + `metadata.yaml` +
`original.pdf` + `assets/`. Backbone model strings were verified loadable in
timm 1.x with 768-d pooled output (see `docs/vit/project_structure.md §2.1`).

### H.1 ViT Backbones (mandatory + 3rd model)

#### 32. Vanilla ViT (Dosovitskiy et al. 2021)

- **Citation:** Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words:
  Transformers for Image Recognition at Scale. *ICLR*. arXiv:2010.11929.
- **Folder:** `references/methodology_backbones/vit_dosovitskiy_2021_iclr/`
- **timm:** `vit_base_patch16_224.orig_in21k_ft_in1k` (768-d, CLS token).
- **Use:** mandatory backbone #1 — isotropic, supervised. ViT-Base = 12 layers,
  hidden 768, 86M (paper Table 1).

#### 33. Swin Transformer (Liu et al. 2021)

- **Citation:** Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., &
  Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted
  Windows. *ICCV*. arXiv:2103.14030.
- **Folder:** `references/methodology_backbones/swin_liu_2021_iccv/`
- **timm:** `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` (768-d, global avg pool, no CLS).
- **Use:** mandatory backbone #2 — hierarchical, shifted windows. Swin-T C=96,
  depths {2,2,6,2}, ~29M (paper §3.3 / Table 1).

#### 34. BEiT (Bao et al. 2022)

- **Citation:** Bao, H., Dong, L., Piao, S., & Wei, F. (2022). BEiT: BERT
  Pre-Training of Image Transformers. *ICLR*. arXiv:2106.08254.
- **Folder:** `references/methodology_backbones/beit_bao_2022_iclr/`
- **timm:** `beit_base_patch16_224.in22k_ft_in22k_in1k` (768-d).
- **Use:** backbone #3 — same ViT-B scale but **self-supervised masked image
  modeling** pretraining (distinct design axis vs ViT/Swin).

### H.2 ViT Training Methodology

#### 35. Stochastic Depth / drop_path (Huang et al. 2016)

- **Citation:** Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q.
  (2016). Deep Networks with Stochastic Depth. *ECCV*. arXiv:1603.09382.
- **Folder:** `references/methodology_training/stochastic_depth_huang_2016_eccv/`
- **Use:** ViT regularization (replaces DropBlock); `drop_path_rate≈0.05` for fine-tuning.

#### 36. MixUp (Zhang et al. 2018)

- **Citation:** Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).
  mixup: Beyond Empirical Risk Minimization. *ICLR*. arXiv:1710.09412.
- **Folder:** `references/methodology_training/mixup_zhang_2018_iclr/`
- **Use:** mild MixUp (0.1–0.2) for ViT fine-tuning; CutMix is reduced for endoscopy.

#### 37. MAE (He et al. 2022) — optional

- **Citation:** He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R.
  (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR*. arXiv:2111.06377.
- **Folder:** `references/methodology_training/mae_he_2022_cvpr/`
- **Use:** source of the ViT fine-tuning recipe (LLRD, warmup, drop_path) and SSL framing.

### H.3 ViT Interpretability

#### 38. Attention Rollout (Abnar & Zuidema 2020)

- **Citation:** Abnar, S., & Zuidema, W. (2020). Quantifying Attention Flow in
  Transformers. *ACL*. arXiv:2005.00928.
- **Folder:** `references/methodology_evaluation/attention_rollout_abnar_2020_acl/`
- **Use:** ViT-native interpretability (complements Grad-CAM); `A = 0.5·W_att + 0.5·I` rollout.

### H.4 ViT / Fusion Baselines (GI endoscopy)

#### 39. Wang et al. 2023 — HSW-ViT [HIGHEST PRIORITY ViT comparator]

- **Citation:** Wang, W., Yan, X., & Tan, J. (2023). Vision Transformer With
  Hybrid Shifted Windows for Gastrointestinal Endoscopy Image Classification.
  *IEEE TCSVT*.
- **Folder:** `references/primary_baselines/09_wang_2023_hsw_vit/`
- **Comparability:** **direct** — reports HyperKvasir (≈86.81% acc) and Kvasir v2
  (≈95.42%). Verify exact split/class count from paper tables before citing numbers.

#### 40. Subedi et al. 2024 — CNN-Transformer Fusion

- **Citation:** Subedi, A., Regmi, S., Regmi, N., Bhusal, B., Bagci, U., & Jha, D.
  (2024). Classification of Endoscopy and Video Capsule Images using
  CNN-Transformer Model. arXiv:2408.10733.
- **Folder:** `references/primary_baselines/10_subedi_2024_cnn_swin_fusion/`
- **Comparability:** contextual — DenseNet201 + Swin **image-branch fusion** on
  GastroVision (22-class) and Kvasir-Capsule (14), **not** HyperKvasir 23-class.
  Preprint. Use as a fusion-methodology precedent in the discussion.

#### 41. Varam et al. 2024 — Edge ViTs (Kvasir-Capsule)

- **Citation:** Varam, D., Khalil, L., & Shanableh, T. (2024). On-Edge Deployment
  of Vision Transformers for Medical Diagnostics Using the Kvasir-Capsule Dataset.
  *Applied Sciences, 14*(18), 8115. DOI: 10.3390/app14188115.
- **Folder:** `references/primary_baselines/11_varam_2024_edge_vits_capsule/`
- **Comparability:** contextual — lightweight ViT zoo (EfficientFormerV2, MobileViT,
  RepViT) on Kvasir-Capsule; useful for lightweight-backbone discussion, not a direct comparator.

---

## Markdown Conversion Template

````markdown
# {Paper Title}

```yaml
metadata:
  title: ""
  authors: []
  venue: ""
  year:
  doi: ""
  arxiv: ""
  url: ""
  pdf_url: ""
  code_url: ""
  dataset_used: ""
  class_count:
  directly_comparable: exact | approximate | contextual | not_comparable
  paper_type: peer_reviewed | preprint | documentation | repository
  download_date: "YYYY-MM-DD"
  pdf_sha256: ""
  zotero_key: ""
```

## Why we cite this paper

## Method core

## Hyperparameters reported

## Reported results on relevant benchmarks

## Claims we rely on

| Claim | Evidence location | Caveat | Used in our project? |
|---|---|---|---|

## Known caveats and limitations

## What we use from this paper in our project

## Direct quote of any equation we implement
````

---

## Quick Order of Conversion

**Tier 1 — must have before implementation/report core:**
1. Borgli 2020 — HyperKvasir dataset
2. Ramamurthy 2022 — Effimix
3. GastroViT 2025
4. GMU — Arevalo 2020
5. CutMix — Yun 2019
6. Focal Loss — Lin 2017
7. AdamW — Loshchilov & Hutter 2019
8. Grad-CAM++ — Chattopadhyay 2018

**Tier 2 — needed for report strengthening:**
9. Bilal 2025 — instructor reference
10. Ahmed 2023 — 3-CNN fusion context
11. McNemar — Dietterich 1998
12. Class-Balanced Loss — Cui 2019
13. UMAP — McInnes 2018
14. Curriculum SSL 2024

**Tier 3 — stretch:**
15. AFF
16. LMF
17. DeepWeeds
18. HEMF, only after HyperKvasir claims are verified
19. Remaining training-method papers


