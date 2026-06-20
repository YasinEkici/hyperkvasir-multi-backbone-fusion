# 007 - ViT Fine-tune Funnel

## 1. Goal

Fine-tune only the top Sprint 2 frozen candidates for the ViT fusion project on
official fold 0, using the A100-gated path. The sprint compares each fine-tuned
candidate against its frozen counterpart, ranks the fine-tuned results by test
macro-F1, and selects the top 4 configurations to carry into Sprint 4 5-fold CV.

Source of truth:
- `docs/vit/project_plan.md` Section 3 Sprint 3, Section 4 Stage 2, and
  Section 8.
- `docs/vit/project_structure.md` Sections 2.1, 2.2, 2.3, and 6.
- `docs/vit/decisions.md` VLD-03 through VLD-15.
- Sprint 2 outputs in `docs/vit/results_progress.md`,
  `docs/vit/experiment_log.md`, and `results/vit/tables/`.

Branch: `sprint3/vit-finetune-funnel`, off the merged Sprint 2 frozen ablation
branch.

Compute budget: A100 units are allowed only for fine-tuning. Do not spend A100
units on frozen feature-cache recomputation or frozen-head reruns.

## 2. Inputs

- Dataset config: `configs/dataset/hyperkvasir_23class_official.yaml`.
- Fold manifest: `data/splits/hyperkvasir_official_5fold/fold_0.csv`.
- Sprint 2 matrix: `configs/vit/experiment_matrix.yaml`.
- Frozen training config: `configs/vit/training/vit_frozen.yaml`.
- Fine-tune training config to create if missing:
  `configs/vit/training/vit_finetune.yaml`.
- Method configs under `configs/vit/method/`.
- Sprint 2 ranked frozen table:
  `results/vit/tables/frozen_ablation_ranked.csv`.
- Sprint 2 fold-0 caches:
  - `results/vit/feature_cache/fold_0_vit_b_features.pt`.
  - `results/vit/feature_cache/fold_0_swin_t_features.pt`.
  - `results/vit/feature_cache/fold_0_beit_b_features.pt`.
- Current implementation:
  - `src/models/vit_backbones.py`.
  - `src/models/full_model.py`.
  - `src/models/projections.py`.
  - `src/models/fusion/`.
  - `src/models/classifiers.py`.
  - `src/training/optimizers.py`.
  - `src/training/trainer.py`.
  - `src/training/ema.py`.
  - `src/training/schedulers.py`.
  - `src/training/losses.py`.
  - `scripts/train.py`.

Sprint 3 candidate funnel from Sprint 2:

| frozen id | fine-tune id | backbones | fusion | frozen macro-F1 |
|---|---|---|---|---|
| `02_single_swin_t_frozen_official` | `02_single_swin_t_finetune_official` | S | none | 0.5759686248 |
| `11_triple_weighted_frozen_official` | `11_triple_weighted_finetune_official` | V+S+B | weighted | 0.5648876413 |
| `09_pair_swin_t_beit_b_weighted_frozen_official` | `09_pair_swin_t_beit_b_weighted_finetune_official` | S+B | weighted | 0.5647640280 |
| `04_pair_vit_b_swin_t_concat_frozen_official` | `04_pair_vit_b_swin_t_concat_finetune_official` | V+S | concat | 0.5618023184 |
| `05_pair_vit_b_beit_b_concat_frozen_official` | `05_pair_vit_b_beit_b_concat_finetune_official` | V+B | concat | 0.5613159913 |

## 3. Scope

- Create or confirm `configs/vit/training/vit_finetune.yaml` with the locked
  VLD-15 recipe:
  - AdamW.
  - ViT/BEiT backbone LR about 2e-5 to 5e-5.
  - Swin backbone LR about 3e-5 to 8e-5.
  - head LR about 1e-3.
  - weight decay 0.05.
  - LLRD decay about 0.65 to 0.75.
  - warmup 5 epochs.
  - drop path about 0.05.
  - label smoothing 0.05 to 0.1.
  - mild MixUp 0.1 to 0.2.
  - CutMix off or very small.
  - EMA enabled with decay about 0.9998.
- Add fine-tune experiment rows only for the five Sprint 3 candidates, with
  distinct fine-tune run ids.
- Confirm `scripts/train.py` uses image loaders, not cached features, whenever
  `unfreeze_blocks > 0`.
- Audit and minimally fix, if needed:
  - ViT/BEiT last-3-block unfreezing and Swin final-stage unfreezing.
  - ViT/BEiT/Swin LLRD parameter groups.
  - drop path config plumbing into timm model creation.
  - MixUp support for fine-tune training.
  - bicubic ViT image preprocessing for fine-tune loaders.
  - A100/provenance gating before fine-tune runs.
- Run the five fold-0 fine-tune candidates on A100.
- Verify each run writes `metrics.json`, `config.yaml`, `predictions.npz`, and
  `best.pt` under `results/vit/runs/{experiment_id}/`.
- Produce a fine-tune ranking table ordered by test macro-F1.
- Compare fine-tuned metrics against the matching Sprint 2 frozen metrics.
- Select the top 4 configurations for Sprint 4 5-fold CV without running
  Sprint 4.
- Update ViT progress docs with command provenance, A100 status,
  hyperparameters, metrics, frozen-vs-fine-tune deltas, pytest result, and top-4
  selection.

## 4. Out of Scope

- Fine-tuning all 11 Sprint 2 frozen ablation configs.
- 5-fold CV, bootstrap CI, seed ensembling, TTA, or cross-fold aggregation.
- Attention rollout, Grad-CAM, UMAP, LaTeX/report writing, or video work.
- Adding GMU or any extra fusion arm unless explicitly approved after the
  Sprint 3 MVP.
- Changing dataset splits, class definitions, evaluation rules, or locked VLD
  decisions.
- Recomputing frozen feature caches on A100.
- Frozen-head reruns unless needed to verify an existing result artifact.
- Editing frozen CNN root files or `src/models/backbones.py`.
- Committing large artifacts such as feature caches, checkpoints, predictions,
  or run directories.

## 5. Files Expected to Be Modified Later

Required during Sprint 3 execution:
- `configs/vit/training/vit_finetune.yaml` - fine-tune recipe following VLD-15.
- `configs/vit/experiment_matrix.yaml` - add only the five fine-tune rows, or
  otherwise point commands to an approved ViT fine-tune matrix if one is created.
- `src/models/vit_backbones.py` - only if needed for drop path plumbing and
  correct ViT/BEiT/Swin unfreeze behavior.
- `src/training/optimizers.py` - only if needed for ViT/BEiT/Swin LLRD parameter
  groups.
- `src/training/trainer.py` - only if needed for mild MixUp support in the
  fine-tune path.
- `scripts/train.py` - only if needed for A100 gating, bicubic fine-tune
  preprocessing, drop path propagation, or fine-tune config wiring.
- `docs/vit/experiment_log.md` - append commands, A100/provenance status,
  hyperparameter config, pytest result, metrics, and top-4 selection.
- `docs/vit/results_progress.md` - add fine-tune ranking and
  frozen-vs-fine-tune delta table.
- `docs/exec-plans/active/007-vit-finetune.md` - mark slice progress if using
  slice tracking.

Conditional:
- `docs/vit/known_issues.md` - only if a real issue occurs, such as A100
  unavailable, OOM, unstable loss, cache/run mismatch, timm drift, failed run,
  or a preprocessing/config incompatibility.
- A small metrics aggregation helper may be added only if existing utilities are
  insufficient for the fine-tune ranking and frozen-vs-fine-tune delta table.
  Prefer `scripts/generate_report_tables.py` and small existing utilities first.

Must not be modified:
- `src/models/backbones.py`.
- Frozen CNN root project docs/configs unless a separate approved task requires
  them.
- `docs/vit/decisions.md`, unless a locked decision genuinely changes. Sprint 3
  should not change any VLD decision.

## 6. Step-by-Step Implementation Plan

### Slice 1 - Preflight and Fine-tune Config Audit

1. Check git status and confirm the branch is `sprint3/vit-finetune-funnel`.
2. Confirm Sprint 2 output files exist:
   - `results/vit/tables/frozen_ablation_ranked.csv`.
   - `docs/vit/results_progress.md`.
   - `docs/vit/experiment_log.md`.
3. Confirm the official fold-0 manifest exists.
4. Confirm the five Sprint 3 candidate frozen runs have `metrics.json`,
   `config.yaml`, `predictions.npz`, and `best.pt`.
5. Confirm the three fold-0 feature caches exist only as provenance inputs; do
   not recompute them on A100.
6. Create `configs/vit/training/vit_finetune.yaml` if missing.
7. Ensure the fine-tune config sets `unfreeze_blocks > 0`, enables EMA, uses the
   VLD-15 LR ranges, and does not inherit the frozen cache-only semantics.

### Slice 2 - Fine-tune Path Implementation Audit

1. Confirm `scripts/train.py` enters the image-loader fine-tune branch when
   `unfreeze_blocks > 0`.
2. Audit `src/models/vit_backbones.py`:
   - ViT-B and BEiT-B must unfreeze only `blocks[9:]` plus final norm/head where
     applicable.
   - Swin-T must unfreeze only `layers[3]` plus final norm/head where
     applicable.
   - Do not add BatchNorm logic; ViTs use LayerNorm.
   - Preserve timm `pretrained=True, num_classes=0` and native pooled
     `model(x)`.
3. Audit drop path support:
   - If `drop_path_rate` is absent from model creation, add minimal config
     plumbing so fine-tune timm backbones are created with about 0.05.
   - Frozen runs must remain unaffected.
4. Audit `src/training/optimizers.py`:
   - Add ViT/BEiT/Swin LLRD groups if the current optimizer path is CNN-only or
     flat for ViT.
   - Keep projection, fusion, and classifier groups at head LR.
   - Keep weight decay at 0.05 unless explicitly overridden in the fine-tune
     config.
5. Audit `scripts/train.py` preprocessing:
   - Fine-tune ViT transforms must use bicubic resize and ImageNet
     normalization.
   - Avoid changing CNN frozen behavior.
6. Audit augmentation:
   - Add mild MixUp support if absent.
   - Keep CutMix off or very small per VLD-15.
7. Audit A100 gating:
   - Fine-tune commands must fail early unless CUDA is available and the selected
     device name includes `A100`, or an equivalent explicit provenance check is
     logged.
8. Add or update focused tests for any code/config behavior changed in this
   slice.

### Slice 3 - Fine-tune Matrix Rows

1. Add exactly five fine-tune rows for the Sprint 3 candidates.
2. Reuse existing method configs for the selected backbones/fusion methods.
3. Point each row at `configs/vit/training/vit_finetune.yaml`.
4. Use fold 0 only.
5. Use distinct run ids:
   - `02_single_swin_t_finetune_official`.
   - `11_triple_weighted_finetune_official`.
   - `09_pair_swin_t_beit_b_weighted_finetune_official`.
   - `04_pair_vit_b_swin_t_concat_finetune_official`.
   - `05_pair_vit_b_beit_b_concat_finetune_official`.
6. Confirm no new GMU, classifier-wise baseline, 5-fold CV, or Sprint 4 row is
   added.

### Slice 4 - A100 Fine-tune Runs

1. Run an A100/provenance check before any fine-tune command.
2. Run each of the five fine-tune rows on official fold 0.
3. Do not run cache extraction or frozen experiments on the A100 runtime.
4. After each run, verify:
   - `results/vit/runs/{experiment_id}/metrics.json`.
   - `results/vit/runs/{experiment_id}/config.yaml`.
   - `results/vit/runs/{experiment_id}/predictions.npz`.
   - `results/vit/runs/{experiment_id}/best.pt`.
5. Inspect every `metrics.json` for non-NaN `test.macro_f1`,
   `test.accuracy`, `test.macro_precision`, and `test.macro_recall`.
6. If an OOM or unstable-loss issue occurs, stop after one conservative
   adjustment attempt and log the issue before changing methodology.

### Slice 5 - Aggregate, Compare, and Select Top 4

1. Generate or collect fine-tune metrics from
   `results/vit/runs/*_finetune_official/metrics.json`.
2. Rank fine-tuned rows by `test.macro_f1` descending.
3. Compare each fine-tuned row with its frozen counterpart:
   - test accuracy delta.
   - test macro-F1 delta.
   - macro precision delta.
   - macro recall delta.
4. Preserve metric source paths for every reported number.
5. Select the top 4 configurations for Sprint 4 5-fold CV.
6. Do not run any Sprint 4 command.

### Slice 6 - Documentation and Validation

1. Update `docs/vit/results_progress.md` with:
   - fine-tune fold-0 ranking.
   - frozen-vs-fine-tune delta table.
   - selected Sprint 4 top 4.
   - metric source paths.
2. Update `docs/vit/experiment_log.md` with:
   - commands run.
   - A100/provenance status.
   - training config and key hyperparameters.
   - run artifact status.
   - pytest result.
   - top-4 selection.
3. Update `docs/vit/known_issues.md` only for observed issues.
4. Run `uv run pytest tests/`.
5. Check `git diff` and confirm no large artifacts are staged/committed and
   `src/models/backbones.py` remains untouched.

## 7. Risks

- **Wrong unfreeze scope:** current code must be checked carefully because Swin
  uses stages, not `blocks[9:]`. VLD-08 requires ViT/BEiT last 3 transformer
  blocks and Swin final stage only.
- **Flat ViT optimizer groups:** a CNN-only LLRD implementation can silently use
  one flat backbone LR for ViT. Sprint 3 must verify real ViT/BEiT/Swin LLRD
  groups before A100 runs.
- **Wrong preprocessing:** fine-tune image loaders must use bicubic resize for
  ViT. Default torchvision resize behavior may not be acceptable unless
  explicitly set.
- **Missing drop path plumbing:** VLD-15 requires modest drop path for
  fine-tuning. Passing no `drop_path_rate` to timm can silently disable this
  regularizer.
- **Missing MixUp support:** the frozen trainer path has CutMix support, but
  Sprint 3 must verify or add mild MixUp without altering frozen CNN behavior.
- **A100 misuse:** A100 units are allowed for fine-tuning only. Cache extraction
  and frozen reruns must not be launched on A100.
- **Run id collision:** fine-tune runs must not overwrite Sprint 2 frozen run
  directories. Use distinct `_finetune_official` ids.
- **Metrics traceability drift:** every reported fine-tune and delta number must
  trace to a `metrics.json` file under `results/vit/runs/`.
- **Large artifacts in git:** `best.pt`, `predictions.npz`, run directories, and
  feature caches are outputs only and must remain ignored.
- **Scope creep:** 5-fold CV, attention maps, report writing, GMU, and
  additional candidates are outside Sprint 3 MVP.

## 8. Acceptance Criteria

- `configs/vit/training/vit_finetune.yaml` exists and follows the VLD-15
  hyperparameter ranges.
- Only the top five Sprint 2 candidates are planned and run for fold-0
  fine-tuning.
- A100/provenance gating is included before any fine-tune run.
- Fine-tune image loaders use ViT preprocessing with bicubic resize and ImageNet
  normalization.
- ViT/BEiT unfreeze only `blocks[9:]` plus final norm/head where applicable.
- Swin unfreezes only `layers[3]` plus final norm/head where applicable.
- LLRD parameter groups are verified for ViT/BEiT/Swin fine-tuning.
- Each planned fine-tune run completes on fold 0 and saves `metrics.json`.
- Every run directory contains `metrics.json`, `config.yaml`, `predictions.npz`,
  and `best.pt`.
- Every reported metric traces to
  `results/vit/runs/{experiment_id}/metrics.json`.
- Fine-tune ranking table is ordered by test macro-F1.
- Frozen-vs-fine-tune deltas are reported for the same configs.
- Top 4 configurations for Sprint 4 are identified, but Sprint 4 is not run.
- `uv run pytest tests/` passes after any code/config/doc updates.
- No large artifacts are committed, and `src/models/backbones.py` remains
  untouched.

## 9. Commands to Run

### Branch and git hygiene

```powershell
git status --short
git branch --show-current
git switch -c sprint3/vit-finetune-funnel
```

Only create the branch if it does not already exist and Sprint 2 has been
merged or explicitly approved as the base.

### Verify Sprint 2 inputs

```powershell
Test-Path data/splits/hyperkvasir_official_5fold/fold_0.csv
Test-Path results/vit/tables/frozen_ablation_ranked.csv
Test-Path results/vit/feature_cache/fold_0_vit_b_features.pt
Test-Path results/vit/feature_cache/fold_0_swin_t_features.pt
Test-Path results/vit/feature_cache/fold_0_beit_b_features.pt
```

```powershell
uv run python -c "from pathlib import Path; ids=['02_single_swin_t_frozen_official','11_triple_weighted_frozen_official','09_pair_swin_t_beit_b_weighted_frozen_official','04_pair_vit_b_swin_t_concat_frozen_official','05_pair_vit_b_beit_b_concat_frozen_official']; required=['metrics.json','config.yaml','predictions.npz','best.pt']; root=Path('results/vit/runs'); [print(i, {r:(root/i/r).exists() for r in required}) for i in ids]"
```

### Validate A100 availability before fine-tune

```powershell
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required for Sprint 3 fine-tune'; name=torch.cuda.get_device_name(0); print('cuda_device', name); assert 'A100' in name, f'Sprint 3 fine-tune requires A100 provenance, got {name}'"
```

Run this on the intended Colab/runtime immediately before launching fine-tune
jobs. Do not run frozen cache extraction on this A100 runtime.

### Run the five fine-tune rows

These commands assume the five fine-tune rows have been added to
`configs/vit/experiment_matrix.yaml` and each row points to
`configs/vit/training/vit_finetune.yaml`.

```powershell
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 02_single_swin_t_finetune_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 11_triple_weighted_finetune_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 09_pair_swin_t_beit_b_weighted_finetune_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 04_pair_vit_b_swin_t_concat_finetune_official --device cuda
uv run python scripts/train.py --config configs/vit/experiment_matrix.yaml --experiment 05_pair_vit_b_beit_b_concat_finetune_official --device cuda
```

### Inspect required run artifacts

```powershell
uv run python -c "from pathlib import Path; ids=['02_single_swin_t_finetune_official','11_triple_weighted_finetune_official','09_pair_swin_t_beit_b_weighted_finetune_official','04_pair_vit_b_swin_t_concat_finetune_official','05_pair_vit_b_beit_b_concat_finetune_official']; required=['metrics.json','config.yaml','predictions.npz','best.pt']; root=Path('results/vit/runs'); [print(i, {r:(root/i/r).exists() for r in required}) for i in ids]"
```

### Aggregate and rank fine-tune metrics

```powershell
uv run python scripts/generate_report_tables.py --runs-dir results/vit/runs --output-dir results/vit/tables
```

If the existing table script does not produce the Sprint 3 ranking in the
required order, add a minimal helper or one-off inspection step that reads only
`results/vit/runs/{experiment_id}/metrics.json` and writes under
`results/vit/tables/`.

```powershell
uv run python -c "from pathlib import Path; import json; ids=['02_single_swin_t_finetune_official','11_triple_weighted_finetune_official','09_pair_swin_t_beit_b_weighted_finetune_official','04_pair_vit_b_swin_t_concat_finetune_official','05_pair_vit_b_beit_b_concat_finetune_official']; root=Path('results/vit/runs'); rows=[]; [rows.append((json.load((root/i/'metrics.json').open())['test']['macro_f1'], json.load((root/i/'metrics.json').open())['test']['accuracy'], i)) for i in ids]; [print(f'{f1:.10f} {acc:.10f} {i}') for f1,acc,i in sorted(rows, reverse=True)]"
```

### Compare frozen vs fine-tune metrics

```powershell
uv run python -c "from pathlib import Path; import json; pairs=[('02_single_swin_t_frozen_official','02_single_swin_t_finetune_official'),('11_triple_weighted_frozen_official','11_triple_weighted_finetune_official'),('09_pair_swin_t_beit_b_weighted_frozen_official','09_pair_swin_t_beit_b_weighted_finetune_official'),('04_pair_vit_b_swin_t_concat_frozen_official','04_pair_vit_b_swin_t_concat_finetune_official'),('05_pair_vit_b_beit_b_concat_frozen_official','05_pair_vit_b_beit_b_concat_finetune_official')]; root=Path('results/vit/runs'); [print(ft, 'frozen_f1', json.load((root/fr/'metrics.json').open())['test']['macro_f1'], 'finetune_f1', json.load((root/ft/'metrics.json').open())['test']['macro_f1'], 'delta', json.load((root/ft/'metrics.json').open())['test']['macro_f1']-json.load((root/fr/'metrics.json').open())['test']['macro_f1']) for fr,ft in pairs]"
```

### Tests and final hygiene

```powershell
uv run pytest tests/
git status --short
git diff -- src/models/backbones.py
```

## 10. Documentation Updates Required

- `docs/vit/experiment_log.md`:
  - commands run.
  - A100 device/provenance confirmation.
  - fine-tune training config and key hyperparameters.
  - run artifact status.
  - fine-tune metrics and source paths.
  - frozen-vs-fine-tune comparison.
  - `uv run pytest tests/` result.
  - selected top 4 for Sprint 4.
- `docs/vit/results_progress.md`:
  - Sprint 3 fine-tune ranking table with config, backbones, fusion, transfer,
    fold, accuracy, macro-F1, macro precision, macro recall, and metrics source.
  - frozen-vs-fine-tune delta table.
  - selected Sprint 4 top 4.
- `docs/vit/known_issues.md`:
  - update only for observed issues, not anticipated risks.
- `docs/exec-plans/active/007-vit-finetune.md`:
  - keep active until Sprint 3 completion is explicitly approved.
- After Sprint 3 completion:
  - move this plan from `active/` to `completed/` only when explicitly approved.
