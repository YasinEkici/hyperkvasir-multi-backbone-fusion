"""Leakage-free TTA + seed-ensemble evaluation (009-vit-cv.md, Slice 5).

Recomputes test softmax probabilities from a finished run's `best.pt` (no change
to the training pipeline) and reports, per config, the out-of-fold (OOF)
macro-F1 + bootstrap 95% CI for:

  * baseline (single-view) vs **TTA** (original + horizontal-flip softmax average),
  * **top-1 seed ensemble** (average softmax across seed checkpoints of the same
    config/fold).

All averaging is **within each fold's own held-out test set**, then folds are
concatenated for the OOF estimate (folds disjoint, cover the dataset once). We
never average fold *models* (VLD-13 / anti-hallucination §5.1.7). Inference uses
`num_workers=0` (avoids the Windows KI-VIT-002 crash). Every number traces to
`results/vit/runs/{run_id}/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import scripts.train as T  # noqa: E402
from src.models.full_model import MultiCNNFusionClassifier  # noqa: E402
from src.utils.checkpointing import load_checkpoint  # noqa: E402
from summarize_vit_cv import bootstrap_macro_f1_ci, macro_f1  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def average_probs(prob_arrays: list[np.ndarray]) -> np.ndarray:
    """Element-wise mean of a list of (N, C) softmax arrays (ensemble / TTA)."""
    stacked = np.stack([np.asarray(p, dtype=float) for p in prob_arrays], axis=0)
    return stacked.mean(axis=0)


def preds_from_probs(probs: np.ndarray) -> np.ndarray:
    """Arg-max class predictions from (N, C) probabilities."""
    return np.asarray(probs).argmax(axis=1)


def run_name(cv_id: str, fold: int, seed: int = 42) -> str:
    name = cv_id
    if seed != 42:
        name = f"{name}_seed{seed}"
    if fold != 0:
        name = f"{name}_fold_{fold}"
    return name


# ---------------------------------------------------------------------------
# Inference (recompute test softmax from a checkpoint)
# ---------------------------------------------------------------------------

def _build_model_and_test_loader(run_dir: Path, device: str):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    method, dataset = cfg["method"], cfg["dataset"]
    training = dict(cfg["training"])
    fold = int(json.loads((run_dir / "metrics.json").read_text())["fold"])
    # force single-process inference (no worker spawn -> avoids KI-VIT-002)
    training["dataloader"] = {"num_workers": 0, "pin_memory": False,
                              "persistent_workers": False}
    split = dataset.get("split_protocol", "hyperkvasir_official_5fold")
    manifest = _ROOT / "data" / "splits" / split / f"fold_{fold}.csv"
    _, _, test_loader = T._make_image_loaders(
        fold_manifest=manifest, dataset_cfg=dataset, training_cfg=training,
        root=_ROOT, batch_size=64,
        interpolation=transforms.InterpolationMode.BICUBIC,
    )
    model = MultiCNNFusionClassifier(
        backbone_names=method["backbone_names"],
        unfreeze_blocks=int(training.get("unfreeze_blocks", 3)),
        projection_dim=int(method.get("projection_dim", 512)),
        fusion_type=method.get("fusion_type", "none"),
        num_classes=int(dataset["num_classes"]),
        mlp_hidden=method.get("mlp_hidden", [256]),
        dropout=float(method.get("dropout", 0.3)),
        drop_path_rate=float(training.get("drop_path_rate", 0.05)),
    )
    load_checkpoint(model, run_dir / "best.pt")
    model.to(device).eval()
    return model, test_loader


def predict_proba(run_dir: Path, device: str, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs [N,C], labels [N]) on the run's fold test set.

    TTA averages softmax of the original and the horizontally-flipped input
    (a within-sample, leakage-free augmentation)."""
    model, loader = _build_model_and_test_loader(run_dir, device)
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = torch.softmax(model(x), dim=1)
            if tta:
                p = (p + torch.softmax(model(torch.flip(x, dims=[3])), dim=1)) / 2.0
            probs.append(p.cpu().numpy())
            labels.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(labels)


def _oof(per_fold_probs: list[np.ndarray], per_fold_labels: list[np.ndarray]):
    probs = np.concatenate(per_fold_probs)
    labels = np.concatenate(per_fold_labels)
    preds = preds_from_probs(probs)
    point, lo, hi = bootstrap_macro_f1_ci(labels, preds)
    return point, lo, hi


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="results/vit/runs")
    p.add_argument("--mode", choices=["tta", "ensemble"], required=True)
    p.add_argument("--configs", nargs="+",
                   default=["11_triple_weighted_cv"],
                   help="CV ids (tta mode). Default: best config only.")
    p.add_argument("--config", default="11_triple_weighted_cv",
                   help="single CV id for ensemble mode")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024],
                   help="seeds to ensemble (ensemble mode)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    runs = _ROOT / args.runs_dir

    if args.mode == "tta":
        print("config                                base_OOF  ->  TTA_OOF  [95% CI]")
        for cid in args.configs:
            base_p, base_l, tta_p = [], [], []
            for f in FOLDS:
                rd = runs / run_name(cid, f)
                pr, lb = predict_proba(rd, args.device, tta=False)
                tpr, _ = predict_proba(rd, args.device, tta=True)
                base_p.append(pr); base_l.append(lb); tta_p.append(tpr)
            b_pt, _, _ = _oof(base_p, base_l)
            t_pt, t_lo, t_hi = _oof(tta_p, base_l)
            print(f"{cid:36}  {b_pt:.4f}   ->  {t_pt:.4f}  [{t_lo:.4f}, {t_hi:.4f}]  "
                  f"(delta {t_pt - b_pt:+.4f})")
    else:  # ensemble
        cid = args.config
        ens_p, ens_l, single_p = [], [], []
        for f in FOLDS:
            seed_probs = []
            for s in args.seeds:
                rd = runs / run_name(cid, f, s)
                pr, lb = predict_proba(rd, args.device, tta=False)
                seed_probs.append(pr)
                if s == args.seeds[0]:
                    fold_labels = lb
            ens_p.append(average_probs(seed_probs))
            ens_l.append(fold_labels)
            single_p.append(seed_probs[0])
        s_pt, _, _ = _oof(single_p, ens_l)
        e_pt, e_lo, e_hi = _oof(ens_p, ens_l)
        print(f"{cid} seed-ensemble {args.seeds}:")
        print(f"  single(seed {args.seeds[0]}) OOF macro-F1: {s_pt:.4f}")
        print(f"  ensemble OOF macro-F1: {e_pt:.4f} [{e_lo:.4f}, {e_hi:.4f}] "
              f"(delta {e_pt - s_pt:+.4f})")


if __name__ == "__main__":
    main()
