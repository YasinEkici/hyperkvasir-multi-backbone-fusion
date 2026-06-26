# HyperKvasir Multi-Backbone Feature Fusion

This repository contains two related Deep Learning course projects on the
HyperKvasir 23-class gastrointestinal endoscopy image-classification benchmark:

- **CNN fusion project:** completed and frozen at the `cnn-submission` state.
- **ViT fusion project:** final report work under `docs/vit/`, `configs/vit/`,
  `results/vit/`, and `reports/vit/`.

Both projects share the same core engine (`src/`, `scripts/`, `data/`,
`references/`, evaluation utilities), but their decisions, reports, and result
artifacts are separated.

Youtube video explaining the CNN backbones: https://www.youtube.com/watch?v=odPTc5wz2Cc&t=347s (in Turkish)

Youtube video explaining the Vision Transformer (ViT) backbones and comparing it with CNNs: https://www.youtube.com/watch?v=r9erkNX8rO0 (in Turkish)

## Repository Status

| Track | Status | Main docs | Main outputs |
|---|---|---|---|
| CNN fusion | Completed / frozen | `project_structure.md`, `project_plan.md`, `docs/decisions.md`, `docs/FINAL_MODEL.md` | `results/`, `reports/final/` |
| ViT fusion | Final report/project work | `docs/vit/project_structure.md`, `docs/vit/project_plan.md`, `docs/vit/decisions.md`, `docs/vit/results_progress.md` | `results/vit/`, `reports/vit/` |

The root CNN files remain valid for the CNN submission. ViT-specific work is
additive and namespaced under `vit/` where possible.

## Project Overview

The shared task is multi-class classification of the **HyperKvasir 23-class
labeled subset**: 10,662 gastrointestinal endoscopy images with severe class
imbalance. Some classes contain hundreds or more than one thousand images, while
rare classes have only a few samples.

Both projects use the same high-level idea:

1. Extract one feature vector from each pretrained backbone.
2. Project each branch to a common 512-dimensional space.
3. Fuse the projected features.
4. Classify the fused representation with an MLP.

Macro-F1 is the main metric because accuracy can hide failures on rare classes.
Accuracy is reported as a supporting metric.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/` | Shared model, data, training, evaluation, and utility code |
| `scripts/` | Shared command-line entry points for preparation, training, evaluation, analysis, and figures |
| `configs/` | CNN/root configs and shared dataset configs |
| `configs/vit/` | ViT method and training configs |
| `results/` | CNN run outputs, tables, figures, and local artifacts |
| `results/vit/` | ViT run outputs, tables, figures, and feature caches |
| `reports/final/` | CNN report sources/artifacts |
| `reports/vit/` | ViT LaTeX report, PDF, bibliography, and figures |
| `docs/` | CNN decisions, final-model record, and shared planning docs |
| `docs/vit/` | ViT decisions, project plan, assignment text, result log, and report plans |
| `references/` | Local paper stubs and metadata used for report citations |
| `tests/` | Unit and smoke tests for shared and project-specific code |

Raw datasets, feature caches, checkpoints, and large run artifacts are not meant
to be included in git or a small course-submission archive.

## CNN Fusion Project

The CNN project is the completed first track. It uses three torchvision
backbones:

| Backbone | Feature dim |
|---|---:|
| ResNet50 | 2048 |
| MobileNetV2 | 1280 |
| EfficientNetB0 | 1280 |

CNN branches are projected with `Linear + LayerNorm + GELU` to 512 dimensions.
The official classifier is always an MLP. Concatenation and weighted fusion are
the main fusion methods; GMU was also evaluated as an ablation. The CNN track
uses torchvision for these backbones; `timm` is not used for the CNN backbones.
BatchNorm handling applies to the CNN fine-tuning path.

Final CNN model, from `docs/FINAL_MODEL.md`:

| Field | Value |
|---|---|
| Experiment | `11_triple_weighted_finetune_wide_official` + deterministic TTA |
| Backbones | ResNet50 + MobileNetV2 + EfficientNetB0 |
| Fusion | Weighted fusion |
| Protocol | Official 5-fold, pooled OOF predictions, n=10,662 |
| Macro-F1 | 0.6075 [0.5860, 0.6296] |
| Accuracy | 0.8765 |
| MCC | 0.8662 |

The CNN project should be treated as frozen. See `docs/FINAL_MODEL.md`,
`docs/results_progress.md`, and `results/tables/` for audit details.

## ViT Fusion Project

The ViT project implements "Coklu ViT Tabanli Ozellik Fuzyonu ile
Siniflandirma" on the same HyperKvasir 23-class task.

### Task And Dataset

- Dataset: HyperKvasir 23-class labeled subset.
- Images: 10,662.
- Task: gastrointestinal endoscopy image classification.
- Challenge: severe class imbalance and visually adjacent classes.
- Evaluation: official 5-fold leakage-free OOF protocol.

### Backbones

| Backbone | Source | Feature dim | Role |
|---|---|---:|---|
| ViT-B/16 | `timm` | 768 | global patch-token representation |
| Swin-T | `timm` | 768 | hierarchical shifted-window representation |
| BEiT-B/16 | `timm` | 768 | masked-image-modeling pretraining |

All ViT backbones are loaded with:

```python
timm.create_model(..., pretrained=True, num_classes=0)
```

The model forward pass returns native pooled features. Swin is not treated as a
CLS-token model.

### Feature Extraction And Projection

Each backbone produces a 768-dimensional pooled feature vector. Each branch then
uses a 512-dimensional projection:

```text
768-d feature -> Linear -> LayerNorm -> GELU -> 512-d feature
```

ViT models use LayerNorm and drop-path behavior. CNN BatchNorm-freezing logic is
not part of the ViT method.

### Fusion Strategies

The core fusion methods are:

- **Concatenation:** concatenate projected branch vectors.
- **Weighted fusion:** learn scalar branch weights and combine 512-d vectors.
- **GMU:** evaluated as an additional gated-fusion ablation; it did not beat the
  final weighted fusion model.

The classifier remains MLP-only across all ViT experiments.

### Transfer Learning Modes

The ViT project evaluates:

- **Frozen feature extraction:** backbones fixed, projection/fusion/MLP trained.
- **Fine-tuning:** last 3 blocks for ViT-B/16 and BEiT-B/16, final stage for
  Swin-T.

Preprocessing uses 224x224 images, ImageNet mean/std, and bicubic interpolation.

### Final ViT Model

Final ViT model:

```text
11_triple_weighted_cv
ViT-B/16 + Swin-T + BEiT-B/16 -> 512-d projections -> weighted fusion -> MLP
```

Main result from `results/vit/tables/cv_fold5_ranked.md`:

| Metric | Value |
|---|---:|
| 5-fold mean macro-F1 | 0.6094 +/- 0.0254 |
| Pooled OOF macro-F1 | 0.6119 [0.5930, 0.6295] |
| Accuracy mean | 0.8780 +/- 0.0140 |
| Macro precision mean | 0.6111 +/- 0.0243 |
| Macro recall mean | 0.6246 +/- 0.0226 |

## ViT Results Summary

| Variant | Macro-F1 | Accuracy | Interpretation |
|---|---:|---:|---|
| Final triple weighted, `11_triple_weighted_cv` | 0.6119 [0.5930, 0.6295] pooled; 0.6094 +/- 0.0254 mean | 0.8780 +/- 0.0140 | Selected final ViT model |
| Best single, `02_single_swin_t_cv` | 0.5985 [0.5806, 0.6152] pooled; 0.5969 +/- 0.0261 mean | 0.8679 +/- 0.0097 | Strongest single backbone |
| Best pair, `09_pair_swin_t_beit_b_weighted_cv` | 0.5893 [0.5736, 0.6054] pooled; 0.5879 +/- 0.0111 mean | 0.8652 +/- 0.0058 | Strongest pair baseline |
| Seed ensemble | 0.6157 [0.5994, 0.6321] | not the selection metric | Small gain; CI overlaps; final model unchanged |
| TTA | 0.6105 [0.5920, 0.6280] | not promoted | No macro-F1 gain |
| Triple GMU fold-0 | 0.5525 | 0.8355 | Negative ablation; below weighted fusion |
| Logit adjustment, tau=1.0 | about 0.4911 | not promoted | Negative result; over-correction |

McNemar's test shows the final ViT model is significantly better at the
accuracy level than the best single and best pair models. This is not a
macro-F1 significance test.

## CNN vs ViT Comparison

This comparison summarizes two separate project tracks in the same repository.
It should not be read as a general claim that one backbone family is universally
better.

| Aspect | CNN fusion | ViT fusion |
|---|---|---|
| Project status | Completed/frozen | Final report/project work |
| Backbones | ResNet50, MobileNetV2, EfficientNetB0 | ViT-B/16, Swin-T, BEiT-B/16 |
| Backbone library | torchvision | timm |
| Native feature dims | 2048 / 1280 / 1280 | 768 / 768 / 768 |
| Projection | 512-d `Linear + LayerNorm + GELU` | 512-d `Linear + LayerNorm + GELU` |
| Fusion/classifier | concat, weighted, GMU + MLP | concat, weighted, GMU + MLP |
| Transfer detail | CNN block fine-tuning with BatchNorm considerations | last transformer blocks/final Swin stage; no BatchNorm logic |
| Main protocol | official 5-fold OOF | official 5-fold OOF |

| Final model | Macro-F1 | Accuracy | MCC |
|---|---:|---:|---:|
| CNN triple weighted + TTA | 0.6075 [0.5860, 0.6296] | 0.8765 | 0.8662 |
| ViT triple weighted | 0.6119 [0.5930, 0.6295] | about 0.878 | 0.868 |

The confidence intervals overlap, and both tracks are bounded by the same
rare-class limitation. The careful conclusion is that the two families are very
close under this protocol; the ViT track adds transformer-native analysis such
as attention rollout, while the CNN track remains the frozen submitted baseline.
Literature comparisons in the ViT report are contextual because splits, class
counts, metrics, and protocols differ.

## Reproducibility And Running

The project uses Python 3.11 and `uv`.

Install dependencies:

```bash
uv sync
```

Check the environment:

```bash
uv run python -c "import torch, torchvision, timm; print(torch.__version__, torchvision.__version__, timm.__version__, torch.cuda.is_available())"
```

Run tests:

```bash
uv run pytest tests/ -q
```

Prepare HyperKvasir manifests:

```bash
uv run python scripts/prepare_data.py --dataset hyperkvasir --config configs/dataset/hyperkvasir_23class_official.yaml
uv run python scripts/make_splits.py --config configs/dataset/hyperkvasir_23class_official.yaml
```

Example CNN run:

```bash
uv run python scripts/train.py --config configs/experiment_matrix.yaml --experiment 11_triple_weighted_finetune_wide_official --fold 0 --device cuda
```

Example ViT run:

```bash
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 11_triple_weighted_finetune_official --fold 0 --device cuda
```

The commands assume the dataset is available locally and `DATA_ROOT` is set.
Large checkpoints and raw data are not included.

## Reports And Submission Artifacts

ViT report:

- PDF: `reports/vit/main.pdf`
- LaTeX source: `reports/vit/main.tex`
- Sections: `reports/vit/sections/`
- Figures: `reports/vit/figures/`
- Result tables: `results/vit/tables/`
- Bibliography: `reports/vit/references.bib`

CNN report/artifacts:

- Report directory: `reports/final/`
- Final model record: `docs/FINAL_MODEL.md`
- Result tables: `results/tables/`
- Figures: `results/figures/`

For a small submission archive, include source code, configs, docs, report PDF,
report figures/tables, and the README. Do not include raw data, feature caches,
or model checkpoints unless explicitly allowed by the submission size limit.

## References

Primary dataset:

- Borgli et al. 2020, "HyperKvasir, a comprehensive multi-class image and video
  dataset for gastrointestinal endoscopy", Scientific Data.

The ViT report bibliography is in `reports/vit/references.bib`. Local paper
stubs and metadata are under `references/`.
