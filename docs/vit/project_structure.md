# Project Structure — Multi-ViT Feature Fusion for HyperKvasir Classification

> **HOW TO USE THIS DOCUMENT (READ FIRST)**
>
> This file is the **single source of truth** for the ViT project's file layout,
> module contracts, architectural facts, and naming conventions. It pairs with
> `docs/vit/project_plan.md` (what to do, when) and `docs/vit/final_assignment.md`
> (the instructor's verbatim requirements).
>
> **Monorepo context:** this ViT project lives in the same repository as the
> completed CNN project. **Shared engine code** (`src/`, `scripts/`, `data/`,
> `references/`) is reused, not duplicated. **Project-specific artifacts** are
> namespaced under `vit/` (`configs/vit/`, `results/vit/`, `reports/vit/`,
> `docs/vit/`). The frozen CNN files at the repo root are legacy — do not edit
> them; the exact CNN submission is preserved at the `cnn-submission` git tag.
>
> **Rules for the agent:**
> 1. Facts in §2 marked **VERIFIED** were confirmed against the deep-research
>    pass or are standard architecture facts. Facts marked **TO-VERIFY** must be
>    confirmed against live `timm` in the Sprint-1 smoke test before relying on
>    exact attribute paths. Do not silently "correct" verified values.
> 2. Module signatures in §6 are binding. Do not invent new names; ask if a
>    signature feels wrong.
> 3. Every entry in "Forbidden Patterns" has a reason. Do not work around them.
> 4. Prefer an option already locked here over introducing a new alternative.

---

## 1. Project Goal (one paragraph)

Build a **multi-ViT feature-fusion** classifier for the **HyperKvasir 23-class
labeled subset** (gastrointestinal endoscopy). **Three transformer backbones —
Vanilla ViT-B/16, Swin-T, and BEiT-B/16** — are used as frozen / fine-tuned
feature extractors via `timm`; their native pooled features (768-d each) are
projected to a shared 512-d space, fused (**concatenation** and **weighted**
fusion are mandatory; GMU is an optional extra), and classified by an **MLP**.
Mandatory ablations cover **single / two / three** backbone combinations and
**frozen feature extraction vs. fine-tuning the last transformer blocks**.
Headline metric is **macro-F1**; accuracy, precision, recall, and F1 are also
reported per the assignment.

The three backbones span three deliberately distinct design axes: **isotropic +
supervised** (ViT-B/16), **hierarchical + shifted-window** (Swin-T), and
**isotropic + self-supervised / masked-image-modeling** (BEiT-B/16).

---

## 2. Verified / To-Verify Facts

### 2.1 Backbone architecture (timm) — **VERIFIED against installed timm 1.x, 2026-06-16**

The three rows below (timm string + pretrained tag, native pooled dim, container
attribute) were confirmed by `timm.list_pretrained()` + a `num_classes=0` forward
pass on this machine. Only the exact unfreeze slice indices in §2.2 remain to be
confirmed in the S1 smoke test.

| Property | ViT-B/16 | Swin-T | BEiT-B/16 |
|---|---|---|---|
| timm string | `vit_base_patch16_224.orig_in21k_ft_in1k` ✓ | `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` ✓ | `beit_base_patch16_224.in22k_ft_in22k_in1k` ✓ |
| Params (≈) | 86M | 28M | 86M |
| Native pooled feature dim | **768** ✓ | **768** ✓ | **768** ✓ |
| Pooling (timm `num_classes=0`) | CLS token (`global_pool='token'` → `x[:,0]`) | global avg pool of final stage (**no CLS**) | CLS / mean pool (timm default) |
| Design axis | isotropic, supervised | hierarchical, windowed | isotropic, SSL (MIM) |
| Block container | `model.blocks` ✓ (12 blocks) | `model.layers` ✓ (4 stages, depths `[2,2,6,2]`) | `model.blocks` ✓ (12 blocks) |

**Feature extraction rule (all branches):**
```python
m = timm.create_model(name, pretrained=True, num_classes=0)
feat = m(x)            # native pooled pre-logits, (B, 768)
```
Do **not** hard-code `forward_features(x)[:,0]` — that is wrong for Swin/BEiT
(no/added tokens). `num_classes=0` + `m(x)` is architecture-correct everywhere.

### 2.2 Fine-tuning unfreeze targets ("last transformer blocks") — locked to last 3 (VLD-08)

Container paths (`model.blocks`, `model.layers`) are verified (§2.1); confirm exact
block/stage counts in the S1 smoke test. Locked choice (assignment §4.2 "son 2
veya 3 katman"): unfreeze the **last 3 blocks** for ViT/BEiT, the **final stage**
for Swin — 3 over 2 for more task-adaptation capacity, still within the 2–3 range.

| Backbone | Locked unfreeze (last 3) | Fallback if forgetting |
|---|---|---|
| ViT-B/16 | last **3** blocks (`blocks[9:]`) + final `norm` + head | reduce to last **2** (`blocks[10:]`) |
| Swin-T | final **stage** (`layers[3]`, 2 blocks) + final `norm` + head | head + norm only |
| BEiT-B/16 | last **3** blocks (`blocks[9:]`) + final `norm` + head | reduce to last **2** |

**LayerNorm, not BatchNorm.** ViTs have no running statistics, so the CNN
project's "freeze BatchNorm running stats" rule is **irrelevant here**. The
equivalent constraint: a **frozen** branch is put in `.eval()` to disable
dropout / stochastic-depth (drop_path); a **fine-tuned** branch uses a modest
`drop_path_rate` (~0.05). There are no BN buffers to freeze.

### 2.3 Image preprocessing

- Input resolution: **224×224** for all three backbones (controlled comparison).
- Normalization: ImageNet stats — `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`.
- Resize-shortest to 256 then center/random crop to 224.
- **Interpolation: bicubic** (timm ViT/Swin/BEiT weights were trained with
  bicubic + `crop_pct≈0.9`; do not use torchvision's default bilinear for the
  ViT path).

### 2.4 HyperKvasir labeled subset (unchanged from CNN project)

- **23 classes**, **10,662 labeled images**, JPEG.
- Source: `https://datasets.simula.no/hyperkvasir/` · License: CC BY 4.0
- Severe class imbalance (e.g., majority ≈1028 images, rarest ≈6 images).
- Official **5-fold** split is the primary protocol; reuse the existing
  `data/splits/official/5_fold_split.csv` and fold materialization logic.

---

## 3. Hardware & Environment

- **Primary:** Google Colab **Pro** — **A100 (40 GB)**, **100 compute units/mo**.
  This is the binding constraint (not VRAM). See `docs/vit/project_plan.md §3`
  for the compute funnel that fits N=3 in ~65 A100 units.
- **Rule:** never spend A100 units on frozen work — feature caching and the
  entire frozen-path ablation run on any non-A100 session at zero A100 cost.
- A100 is not guaranteed even on Pro; frozen stages degrade gracefully on any
  GPU. Reuse the existing D-09 provenance gate (asserts A100, aborts T4) and
  Drive staging for ephemeral sessions.
- **Python:** 3.11 · **Stack:** torch≥2.12, torchvision≥0.27, **timm≥1.0**,
  numpy, pandas, scikit-learn, matplotlib, pyyaml, tqdm, grad-cam, umap-learn.
- Environment managed by `uv`. See `AGENTS.md`.

---

## 4. File Hierarchy (monorepo, additive)

```
hyperkvasir-multi-backbone-fusion/
├── AGENTS.md                          # shared, READ-FIRST (needs ViT addendum, see §9)
├── README.md                          # shared (update: two projects, one engine)
├── project_structure.md               # CNN (legacy, frozen)
├── project_plan.md                    # CNN (legacy, frozen)
│
├── docs/
│   ├── decisions.md                   # CNN (legacy)
│   ├── environment.md                 # shared
│   ├── exec-plans/                    # SHARED machinery
│   │   ├── TEMPLATE.md
│   │   ├── active/   005-vit-foundation.md ...   # ViT sprints
│   │   └── completed/ 001..004 (CNN)
│   └── vit/                           # ← ViT project docs
│       ├── final_assignment.md        # instructor requirements (verbatim)
│       ├── project_structure.md        # THIS FILE
│       ├── project_plan.md
│       ├── decisions.md               # ViT locked decisions log (VLD-*)
│       └── experiment_log.md
│
├── references/                        # SHARED (CNN + ViT citations together)
│   ├── INDEX.md
│   ├── methodology_backbones/
│   │   ├── resnet_he_2016_cvpr/ ...   # CNN (now comparison/prior-work)
│   │   ├── vit_dosovitskiy_2021_iclr/ # ← new
│   │   ├── swin_liu_2021_iccv/        # ← new
│   │   └── beit_bao_2022_iclr/        # ← new
│   ├── methodology_training/  + stochastic_depth_huang_2016_eccv/, mixup_zhang_2018_iclr/  (− dropblock, − swa)
│   ├── methodology_evaluation/ + attention_rollout_abnar_2020_acl/
│   └── primary_baselines/     + 09_wang_2023_hsw_vit/, 10_subedi_2024_cnn_swin_fusion/
│
├── configs/
│   ├── dataset/                       # SHARED
│   ├── method/ , training/ , experiment_matrix.yaml   # CNN (legacy)
│   └── vit/                           # ← ViT configs
│       ├── method/    (vit_single_*.yaml, vit_pair_*.yaml, vit_triple_*.yaml)
│       ├── training/  (vit_frozen.yaml, vit_finetune.yaml)
│       └── experiment_matrix.yaml
│
├── data/                              # SHARED (gitignored raw; splits/official tracked)
├── results/
│   ├── runs/ ...                      # CNN (legacy)
│   └── vit/  runs/ tables/ figures/ feature_cache/   # ← ViT outputs
├── reports/
│   ├── final/                         # CNN report (legacy)
│   └── vit/                           # ← ViT report (LaTeX)
│
├── src/                               # SHARED ENGINE (not split by project)
│   ├── data/                          # reused as-is (+ bicubic option)
│   ├── models/
│   │   ├── backbones.py               # CNN extractor — DO NOT EDIT (frozen)
│   │   ├── vit_backbones.py           # ← NEW: timm ViT extractor
│   │   ├── projections.py             # reused as-is (parametric in_dim)
│   │   ├── fusion/                    # reused as-is (concat/weighted/gmu)
│   │   ├── classifiers.py             # reused as-is (MLP)
│   │   └── full_model.py              # minor edit: dispatch CNN vs ViT extractor
│   ├── training/
│   │   ├── optimizers.py              # extend _BACKBONE_BLOCKS with ViT LLRD
│   │   └── ... (trainer, ema, schedulers, losses) reused
│   ├── evaluation/
│   │   ├── interpretability.py        # CNN Grad-CAM (reused)
│   │   └── attention_rollout.py       # ← NEW: ViT-native interpretability
│   └── utils/
│
├── scripts/                           # SHARED (generic; pass configs/vit/...)
│   └── make_vit_figures.py            # ← NEW only if ViT-specific figures needed
├── tests/                             # SHARED (+ ViT backbone shape tests)
└── env/
```

---

## 5. References Folder (shared)

Convention unchanged (see root `project_structure.md §5` and `references/INDEX.md`):
`{NN}_{author}_{year}_{slug}/` for `primary_baselines/`, no `NN` elsewhere. Each
cited paper needs the INDEX.md metadata block (+ a "Claims we rely on" table for
baselines whose numbers appear in our comparison). ViT additions and the two new
baselines (`09_wang_2023_hsw_vit`, `10_subedi_2024_cnn_swin_fusion`) follow the
same rule. CNN backbone stubs remain as comparison/prior-work references.

---

## 6. Module Contracts

**New (ViT):**
- `src/models/vit_backbones.py::ViTFeatureExtractor(name, pretrained=True, unfreeze_blocks=0)`
  - `.forward(x) -> (B, feature_dim)` native pooled pre-logits via `num_classes=0`.
  - `.feature_dim -> int` (768 for all three).
  - `.trainable_param_groups(head_lr, backbone_lr, llrd_decay)` for fine-tune.
- `src/evaluation/attention_rollout.py::attention_rollout(model, x, ...) -> heatmap`

**Reused as-is (backbone-agnostic — DO NOT duplicate):**
- `src/models/projections.py::BranchProjection(in_dim, out_dim=512)`
- `src/models/fusion/{concat,weighted,gmu}.py::FusionModule(num_branches, feature_dim, **kwargs)`
- `src/models/classifiers.py::MLPClassifier(input_dim, num_classes, hidden_dims, dropout)`
- `src/data/feature_cache.py::cache_frozen_features(backbones, dataset_config, split_manifest, output_dir, ...)` (swap in ViT extractor; bicubic)
- `src/evaluation/statistical.py::bootstrap_ci(...)` — implemented, reuse as-is.
- `src/evaluation/statistical.py::mcnemar_test(...)` — **signature only, raises
  `NotImplementedError` (scaffold)**; implement if a McNemar comparison is needed
  (not in the MVP scope).

**Edited:**
- `src/models/full_model.py` — choose CNN vs ViT extractor by backbone name; the
  rest of the projection→fusion→MLP wiring is already generic.
- `src/training/optimizers.py::build_adamw_with_llrd(...)` — add ViT per-encoder-layer
  block spec to `_BACKBONE_BLOCKS` (12 layers for ViT/BEiT; Swin stages).

---

## 7. Config Schema

Dataset configs are **shared** (`configs/dataset/`, reused unchanged). ViT method
and training configs live under `configs/vit/method/` and `configs/vit/training/`,
with master `configs/vit/experiment_matrix.yaml`. Method configs name `backbone_names`
(timm strings or short aliases `vit_b`/`swin_t`/`beit_b`), `fusion_type`
(`concat`/`weighted`/`none`), `projection_dim: 512`, `classifier: mlp`. Training
configs carry the ViT recipe (see `project_plan.md §8`).

---

## 8. Naming Conventions

- **Experiment IDs:** `{NN}_{method}_{transfer}_official` — e.g.
  `11_triple_weighted_finetune_official`, `vit_` prefix optional since outputs
  live under `results/vit/`.
- **Backbone aliases:** `vit_b`, `swin_t`, `beit_b`.
- **Checkpoints:** `best.pt`, `last.pt`, `ema.pt`.
- **Exec plan files:** `005-vit-foundation.md`, `006-vit-frozen-ablation.md`, …
- **Git branches:** work happens on `vit-fusion`; per-sprint `feat/vit-s{N}-{slug}` optional.
- **Variables:** snake_case functions/vars, PascalCase classes.

---

## 9. Forbidden Patterns (ViT-updated)

1. **Splitting `src/` or `scripts/` into a parallel `vit/` tree** — they are the
   shared engine; add ViT-specific *files* in place, never a mirror tree.
2. **Editing the frozen CNN files** (root `*.md`, `src/models/backbones.py`,
   `configs/method/*`, `results/runs/*`). Preserved at `cnn-submission` tag.
3. Hard-coding `forward_features(x)[:,0]`; use `num_classes=0` + `m(x)`.
4. Freezing BatchNorm stats / any BN logic for ViT (there is none — use drop_path).
5. Adding non-MLP classifiers as official baselines (assignment is MLP-only;
   the §5.1 "3 classifiers" phrase is the same instructor error as the CNN
   project — only MLP exists).
6. Spending A100 compute units on frozen feature caching or frozen ablation.
7. Averaging across folds before computing macro-F1 (leakage; see D-07 policy).
8. Saving raw datasets, feature caches, or checkpoints to git.
9. Citing a report number without a `paper.md`/Table reference or a `metrics.json`.
10. Using a model string not confirmed loadable in timm (CaiT-S12 and original
    PVT-Small are **not** in the public registry — do not use).

> **Note vs CNN project:** the CNN `project_structure.md §9` forbade `timm` for
> backbones. **That rule is REVERSED here** — timm with `pretrained=True` is the
> required backbone source for all three ViTs.

---

## 10. Working Style for Agents

1. Read `AGENTS.md`, then this file, then `docs/vit/project_plan.md`, then the
   active exec plan, then the relevant `paper.md` files.
2. Implement one module at a time; write/update the matching test first.
3. For each method (weighted fusion, attention rollout, …), copy the paper
   equation verbatim into the module docstring before coding.
4. Sprint-1 smoke test must confirm the **TO-VERIFY** timm attribute paths
   (`model.blocks`, `model.layers`, pooled dims) before fine-tuning code relies
   on them.
5. Save run artifacts: resolved config, metrics, predictions, split manifest.
6. Ask before deviating from scope or locked decisions (VLD-*); never auto-advance
   a sprint without explicit approval.
