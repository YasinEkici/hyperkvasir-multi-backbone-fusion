"""UMAP feature-space maps for the frozen final model (011-vit-report.md, Slice 2).

Projects the 512-d fused and per-branch features (VLD-05) of ``11_triple_weighted``
to 2-D with UMAP (McInnes et al. 2018) and colours by the 23 HyperKvasir classes,
to visualise class separability and the rare-class crowding that caps macro-F1
(report §5.5).  Inference-only from ``best.pt`` over the fold-0 OOF test split
(leakage-free, VLD-13); no retraining, 0 A100 units.

UMAP is an *unsupervised* embedding for visualisation only — it is not a model
metric and never re-selects the champion.  ``random_state`` is fixed for
reproducibility.

Usage:
    uv run python scripts/interpret_umap.py --run 11_triple_weighted_cv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.interpret_common import (  # noqa: E402
    build_model,
    extract_features,
    oof_test_dataset,
    subsample_indices_per_class,
)


def _palette(n: int):
    cmap = matplotlib.colormaps["nipy_spectral"].resampled(n)
    return [cmap(i) for i in range(n)]


def _fit_umap(feats: np.ndarray, seed: int) -> np.ndarray:
    import umap  # local import — heavy dependency

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        metric="euclidean", random_state=seed,
    )
    return reducer.fit_transform(feats)


def _scatter(ax, emb, labels, label_to_name, palette, title):
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=6, color=palette[int(c)],
                   label=label_to_name[int(c)], linewidths=0, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="results/vit/runs")
    p.add_argument("--run", default="11_triple_weighted_cv")
    p.add_argument("--out", default="reports/vit/figures")
    p.add_argument("--max-per-class", type=int, default=150,
                   help="cap points per class for readability (rare classes kept in full)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    run_dir = _ROOT / args.runs_dir / args.run
    out_dir = _ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = build_model(run_dir, args.device)
    ds, _, label_to_name, fold = oof_test_dataset(run_dir, cfg)
    print(f"[umap] run={args.run} fold={fold} OOF={len(ds)} device={args.device}")

    branch_feats, fused, labels = extract_features(model, ds, args.device)
    keep = subsample_indices_per_class(labels, args.max_per_class, args.seed)
    labels_k = labels[keep]
    n_classes = int(cfg["dataset"]["num_classes"])
    palette = _palette(n_classes)
    print(f"[umap] points plotted: {len(keep)}/{len(labels)} "
          f"(cap {args.max_per_class}/class)")

    # --- Fused UMAP (primary) ---
    emb = _fit_umap(fused[keep], args.seed)
    fig, ax = plt.subplots(figsize=(8, 7))
    _scatter(ax, emb, labels_k, label_to_name, palette,
             f"UMAP of fused features ({model.fusion_type}, 512-d) — OOF fold {fold}")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6,
              markerscale=1.6, ncol=1, frameon=False)
    fig.tight_layout()
    fpath = out_dir / "umap_fused.png"
    fig.savefig(fpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[umap] wrote {fpath.relative_to(_ROOT)}")

    # --- Per-branch UMAP panel ---
    names = list(model.backbone_names)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        emb_b = _fit_umap(branch_feats[n][keep], args.seed)
        _scatter(ax, emb_b, labels_k, label_to_name, palette,
                 f"{n} projection (512-d)")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="center right", fontsize=6, markerscale=1.6,
               frameon=False)
    fig.suptitle(f"UMAP of per-branch projected features — OOF fold {fold}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.9, 0.97))
    fpath = out_dir / "umap_branches.png"
    fig.savefig(fpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[umap] wrote {fpath.relative_to(_ROOT)}")

    print("[umap] done — figures trace to "
          f"{run_dir.relative_to(_ROOT)}/best.pt (OOF fold {fold})")


if __name__ == "__main__":
    main()
