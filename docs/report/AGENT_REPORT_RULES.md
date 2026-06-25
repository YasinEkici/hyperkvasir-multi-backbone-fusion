# Agent Report Writing Rules

This document defines the writing rules for the final report of the HyperKvasir **multi-ViT feature fusion** project (`reports/vit/`). The report must be in Turkish (academic register).

The purpose is to make the report academically honest, project-specific, traceable to repository evidence, and clearly based on the actual engineering process followed in this project (Sprints 1–5, exec-plans `005`–`011`, decisions `VLD-01..VLD-20`).

## 1. Core Writing Principle

The report must describe this specific project, not a generic deep learning study.

Avoid generic textbook-style openings such as:

> Görüntü dönüştürücüleri son yıllarda tıbbi görüntü analizini kökten değiştirmiştir.

Prefer project-specific writing such as:

> Bu projede asıl zorluk yüksek doğruluk elde etmek değil, nadir sınıfların toplu ölçütlerce gizlenebildiği dengesiz 23-sınıf protokolünde model davranışını makro-F1 ve güven aralıklarıyla doğru biçimde değerlendirmekti.

The report should sound like it was written from the actual experiment history, project decisions (VLD-*), limitations, and documented results — not from a template.

## 2. AI Patterns to Avoid

Patterns that sound noticeably AI-generated, even in formal writing:

* **Empty framing:** "Günümüzde", "Gelişen teknolojiyle birlikte", "Literatürde önemli bir yer tutmaktadır". Address the problem or finding directly.
* **Forced contrast:** "yalnızca X değil, aynı zamanda Y". Use a direct, affirmative sentence.
* **Artificial three-part lists:** Decorative adjective sequences such as "hızlı, güvenilir ve etkili".
* **Repetitive template transitions:** Avoid beginning every paragraph with "Ayrıca / Bununla birlikte / Dahası / Sonuç olarak".
* **Overuse of `-maktadır/-mektedir`:** Use a cleaner verb form whenever possible.
* **Unsupported general claims:** Every claim should be supported by a figure, a result, or a citation.
* **Decorative em dashes (`—`)** and empty buzzwords such as "değer katmak" and "fark yaratmak".

**Quick check:** After finishing the text, search for "yalnızca... değil", "Günümüzde", and "önemli bir yer tut".

## 3. Evidence-Based Writing

Every major claim must be traceable to one of the following:

* documented project decisions (`docs/vit/decisions.md`, `VLD-*`),
* experiment logs (`docs/vit/experiment_log.md`),
* progress notes (`docs/vit/results_progress.md`),
* result tables (`results/vit/tables/`, e.g. `cv_fold5_ranked`, `frozen_ablation_ranked`, `per_class_champion`),
* result figures (`reports/vit/figures/`),
* per-run metrics (`results/vit/runs/{id}/metrics.json`, `predictions.npz`),
* documented configuration files (`configs/vit/`),
* literature stubs already included in the repository (`references/`).

Do not invent:

* numerical results,
* completed experiments,
* model improvements,
* comparison values,
* citations,
* runtime values,
* hardware details,
* conclusions that are not supported by results.

If a result is missing, write `TODO` (or `\TODO{...}` in LaTeX) instead of guessing.

## 4. Project-Specific Details to Include

The report should include concrete details from the actual implementation when relevant:

* HyperKvasir 23-class classification task ($n = 10{,}662$),
* official 5-fold evaluation protocol with leakage-free, pooled out-of-fold (OOF) test predictions,
* the three ViT backbones: **ViT-B/16, Swin-T, BEiT-B/16** (loaded via `timm`, `num_classes=0`, native pooled 768-d),
* per-branch projection to 512-d (Linear + LayerNorm + GELU),
* the three fusion methods: **concatenation, weighted, and GMU** (Gated Multimodal Unit, faithful element-wise gate),
* MLP-only classifier decision (hidden 256, dropout 0.3),
* the two transfer regimes: **frozen feature extraction** vs **fine-tuning the last transformer blocks** (ViT/BEiT last blocks, Swin final stage),
* macro-F1 as the headline metric; accuracy / macro precision / macro recall as supporting,
* bootstrap 95% confidence intervals over pooled OOF predictions,
* per-class analysis and confusion-matrix interpretation (ordinal/anatomical confusions: UC grades, `hemorroids`→`retroflex-rectum`),
* rare-class limitations (very low-support classes drive macro-F1 down),
* the final model `11_triple_weighted` (V+S+B weighted, fine-tuned; pooled macro-F1 0.6119, acc ≈ 0.88),
* interpretability: attention rollout (ViT-B/BEiT-B), Grad-CAM++ (with the Swin windowed-layout caveat), UMAP of the fused/per-branch features,
* the McNemar significance test (champion vs best single / best pair),
* experiment IDs when useful,
* Colab Pro A100 and the throughput tuning (TF32, bfloat16, dataloader parallelism) only where documented and relevant.

Do not overload the report with implementation trivia. Use details only when they support the methodology, reproducibility, or discussion.

## 5. Explain Decisions, Not Only Actions

The report should not only say what was done. It should explain why it was done.

Important questions to address:

* Why is macro-F1 the headline metric, and why is accuracy insufficient on this imbalanced dataset?
* Why three different transformer families (ViT-B, Swin-T, BEiT-B) rather than one?
* Why is the classifier fixed as MLP (so all differences are attributable to backbone/fusion/transfer)?
* Why is the official 5-fold protocol used, and why are fold models never averaged before computing macro-F1?
* Why is weighted fusion preferred over concatenation, and why did the more complex GMU fusion not help?
* Why did post-hoc logit adjustment degrade macro-F1 (the balanced sampler already addresses imbalance)?
* Why are some literature results only contextual and not direct comparisons?
* Why should negative or neutral ablation results (GMU, TTA, seed ensemble) still be reported?

These explanations make the report defensible and connected to the actual engineering process.

## 6. Handling Negative or Neutral Results

Do not hide weak, negative, or neutral results. This project has several honest negatives that strengthen the report:

* GMU fusion did not beat weighted fusion (VLD-19),
* TTA gave no macro-F1 gain and the seed ensemble's gain is within overlapping confidence intervals,
* post-hoc logit adjustment monotonically reduced macro-F1 (VLD-20).

State them plainly. Example:

> Ağırlıklı füzyon en iyi sonucu verse de, birleştirme füzyonuna kıyasla farkın güven aralıkları yakın olduğundan kazanç dikkatle yorumlanmalıdır; buna karşılık eşli McNemar testi, üçlü füzyonun en iyi tekli omurgadan anlamlı biçimde daha doğru olduğunu göstermiştir.

This kind of careful discussion is preferred over exaggerating small differences.

## 7. Literature Use and Citation Rules

Do not copy wording from papers.

When using literature:

* paraphrase in our own words,
* cite the source properly (entries must exist in `references.bib` and have a stub under `references/`),
* separate what the paper reports from what our project concludes,
* avoid direct head-to-head claims if the protocols differ.

External results (e.g. Wang 2023 HSW-ViT, GastroViT 2025, EffiMix, Borgli 2020) must be described as **contextual** when:

* the split protocol differs,
* augmentation strategy differs,
* class balancing differs,
* dataset version or class count differs,
* the training/evaluation setup is not directly comparable.

Note specifically that "≈97%" accuracy figures in some HyperKvasir papers stem from imbalance-inflated accuracy and different splits; macro-F1 bands are far closer.

## 8. Claims That Must Not Be Made

Do not claim:

* state-of-the-art performance,
* direct superiority over papers with different protocols,
* that GMU, TTA, the seed ensemble, or logit adjustment improved the result (they did not),
* that accuracy alone proves success,
* that the model is clinically usable,
* that rare classes are solved (per-class F1 shows several at or near zero),
* that protocol-mismatched literature values are direct baselines.

Prefer careful wording:

* "seçilen resmi protokol altında rekabetçi",
* "kontrollü karşılaştırma",
* "yöntemsel olarak tutarlı değerlendirme",
* "literatürle bağlamsal karşılaştırma",
* "doğrudan kıyaslanabilirlik sınırlı",
* "kazanç dikkatle yorumlanmalıdır".

## 9. Tone and Style

The writing should be: academic, clear, specific, honest, technically precise, and connected to repository evidence. Use "seed" (not "tohum").

Avoid: generic filler, marketing-style claims, exaggerated language, polished-but-empty sentences, repeated phrases, unnecessary buzzwords, and the AI patterns listed in Section 2. Do not include repository file paths, internal decision IDs (VLD-*), or self-referential "honesty"/process wording in the report body itself; those belong in the docs, not the report prose.

Do not write "make it human-like." Instead, write in a project-specific and evidence-based way.

## 10. Manual Student Contribution

Some parts should be reviewed and manually refined by the student:

* final paragraph of the Introduction,
* explanation of why the protocol was selected,
* interpretation of the most important results,
* limitations,
* future work,
* final conclusion.

The final report should reflect the student's understanding of the project, not only automatically generated text.

## 11. Section-Specific Rules

### Introduction
Explain the task, why the dataset is challenging (imbalance), why multi-ViT feature fusion is investigated, and the comparison axes (backbone-wise, fusion-wise, transfer-wise). Avoid broad generic claims unless directly relevant.

### Related Work
Briefly cover the transformer backbones (ViT, Swin, BEiT), fusion methods (concat, weighted, GMU), HyperKvasir classification literature (contextual), and the interpretability/statistics methods used. Paraphrase; cite stubs.

### Methodology
Explain the backbones, feature extraction (`num_classes=0`, 768-d), projection (512-d), the three fusion methods, the MLP classifier, and the frozen vs fine-tune-last-blocks setup. Do not introduce models or methods that were not implemented. The architecture diagram should match the implemented pipeline.

### Experimental Setup
Explain the dataset setup, the official 5-fold protocol, leakage-free OOF evaluation, metrics, hyperparameters (where documented in `configs/vit/`), and reproducibility caveats (approximate under tuned bf16/TF32, seeds fixed). Do not fabricate runtime or hardware details.

### Results
Use only documented values. Present: frozen ablation, fine-tune vs frozen deltas, the 5-fold CV table with confidence intervals, the pooled confusion matrix, per-class results, the additive-technique table (TTA / seed ensemble / logit adjustment / GMU), and the McNemar significance result.

### Discussion (most weighted)
Explain what the results mean: which backbone learned better features and why; whether fusion helped and why the gain is bounded by rare classes; why some classes are difficult (ordinal/anatomical confusion); why literature comparison is limited; what the project did well; what remains weak. Support claims with the interpretability figures (rollout / Grad-CAM++ / UMAP).

### Conclusion
Summarize the actual findings without exaggeration. Do not introduce new results. Mark the YouTube demo link and student identity as `\TODO` until provided.

## 12. Final Integrity Checklist

Before finalizing the report, check:

* Are all numerical results traceable to repository files (`results/vit/`, `docs/vit/`)?
* Are all major claims supported by a figure, table, or citation?
* Are incomplete items clearly marked (`\TODO`) or omitted?
* Are literature comparisons described as contextual where protocols differ?
* Is macro-F1 treated as the headline metric, and accuracy not over-emphasized?
* Are the negative/neutral results (GMU, TTA, seed ensemble, logit adjustment) reported honestly?
* Is the McNemar result framed as an accuracy-level test that complements (not contradicts) the macro-F1 CIs?
* Are limitations (rare-class ceiling) included?
* Did the AI-pattern quick check pass (no "yalnızca... değil", "Günümüzde", "önemli bir yer tut"; no decorative em dashes; limited `-maktadır`)?
* Are there any generic paragraphs that could belong to any deep learning paper?
* Does the report reflect this specific project's process?

If any answer is problematic, revise the report before final submission.
