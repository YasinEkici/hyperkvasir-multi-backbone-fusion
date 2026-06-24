"""Grad-CAM++ panels for the frozen final model (011-vit-report.md, Slice 1).

Grad-CAM++ (Chattopadhyay et al. 2018) for each backbone branch of
``11_triple_weighted``, via the standard ``pytorch-grad-cam`` library with a
transformer ``reshape_transform`` (token sequence -> spatial grid).  Inference
from ``best.pt`` over the OOF test split (leakage-free, VLD-13); no retraining.

Target layer = the last block's ``norm1`` of each branch's timm model.  Token
layout differs by family:
  * ViT-B / BEiT-B (CLS-token, isotropic): ``(B, 1+196, C)`` -> drop CLS ->
    14×14 grid.
  * **Swin-T (CAVEAT):** timm Swin uses a *hierarchical / shifted-window*,
    channels-last spatial layout — its block output is ``(B, 7, 7, C)``, NOT a
    flat ``(B, N, C)`` CLS sequence.  We reshape that explicitly (permute), but
    a windowed Grad-CAM++ map is a coarser 7×7 and less faithful than the ViT
    maps; it is shown for completeness and flagged.  If it degenerates (all-zero
    / NaN), the branch is skipped with a printed caveat (no misleading map).

Usage:
    uv run python scripts/interpret_gradcam.py --run 11_triple_weighted_cv
"""

from __future__ import annotations

import argparse
import json
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

from pytorch_grad_cam import GradCAMPlusPlus  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402

from scripts.interpret_common import (  # noqa: E402
    build_model,
    denormalize,
    oof_test_dataset,
    pick_examples,
    predict_all,
    select_classes_by_f1,
)

BRANCHES = ("vit_b", "swin_t", "beit_b")


def _vit_reshape(num_prefix: int):
    def fn(tensor: torch.Tensor) -> torch.Tensor:
        # (B, num_prefix + P, C) -> (B, C, side, side)
        patches = tensor[:, num_prefix:, :]
        side = int(round(patches.shape[1] ** 0.5))
        result = patches.reshape(patches.shape[0], side, side, patches.shape[2])
        return result.permute(0, 3, 1, 2)
    return fn


def _swin_reshape(tensor: torch.Tensor) -> torch.Tensor:
    # timm Swin block output is channels-last spatial (B, H, W, C) -> (B, C, H, W).
    if tensor.dim() == 4:
        return tensor.permute(0, 3, 1, 2)
    # Defensive fallback if a flat (B, N, C) ever appears.
    side = int(round(tensor.shape[1] ** 0.5))
    return tensor.reshape(tensor.shape[0], side, side, tensor.shape[2]).permute(0, 3, 1, 2)


def _branch_target_and_reshape(model, branch: str):
    timm_model = model.backbones[branch].model
    if branch == "swin_t":
        target_layer = timm_model.layers[-1].blocks[-1].norm1
        return target_layer, _swin_reshape
    num_prefix = int(getattr(timm_model, "num_prefix_tokens", 1))
    target_layer = timm_model.blocks[-1].norm1
    return target_layer, _vit_reshape(num_prefix)


def gradcam_for_branch(model, branch: str, img: torch.Tensor, target_class: int,
                       device: str) -> np.ndarray | None:
    """Return a HxW Grad-CAM++ map in [0,1], or None if it degenerates (caveat)."""
    target_layer, reshape = _branch_target_and_reshape(model, branch)
    x = img.unsqueeze(0).to(device)
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer],
                          reshape_transform=reshape)
    grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(target_class))])
    grayscale = grayscale[0]
    if not np.isfinite(grayscale).all() or float(grayscale.max()) <= 0.0:
        return None
    return grayscale


def _overlay(ax, rgb: np.ndarray, heat: np.ndarray | None, title: str,
             note_if_missing: bool = False) -> None:
    ax.imshow(rgb)
    if heat is not None:
        h, w = rgb.shape[:2]
        heat_up = np.asarray(
            torch.nn.functional.interpolate(
                torch.tensor(heat)[None, None].float(), size=(h, w),
                mode="bilinear", align_corners=False,
            )[0, 0]
        )
        ax.imshow(heat_up, cmap="jet", alpha=0.45)
    elif note_if_missing:
        # Only a degenerate CAM map gets the caveat note — never the input panel.
        ax.text(0.5, 0.5, "n/a\n(caveat)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="white")
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="results/vit/runs")
    p.add_argument("--run", default="11_triple_weighted_cv")
    p.add_argument("--out", default="reports/vit/figures")
    p.add_argument("--n-best", type=int, default=3)
    p.add_argument("--n-worst", type=int, default=3)
    p.add_argument("--n-correct", type=int, default=2)
    p.add_argument("--n-failure", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    run_dir = _ROOT / args.runs_dir / args.run
    out_dir = _ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = build_model(run_dir, args.device)
    ds, _, label_to_name, fold = oof_test_dataset(run_dir, cfg)
    per_class = json.loads((run_dir / "metrics.json").read_text())["test"]["per_class"]
    best, worst = select_classes_by_f1(per_class, args.n_best, args.n_worst)
    classes = best + worst

    print(f"[gradcam++] run={args.run} fold={fold} OOF={len(ds)} device={args.device}")
    print(f"[gradcam++] best={[label_to_name[c] for c in best]} "
          f"worst={[label_to_name[c] for c in worst]}")

    preds, labels = predict_all(model, ds, args.device)
    picks = pick_examples(preds, labels, classes, args.n_correct, args.n_failure)

    swin_caveat = False
    for c in classes:
        sel = picks[c]["correct"] + picks[c]["failure"]
        if not sel:
            print(f"[gradcam++] class {label_to_name[c]}: no OOF samples, skipped")
            continue
        ncol = 1 + len(BRANCHES)
        fig, axes = plt.subplots(len(sel), ncol, figsize=(2.2 * ncol, 2.2 * len(sel)))
        axes = np.atleast_2d(axes)
        for r, idx in enumerate(sel):
            img, label, _ = ds[idx]
            rgb = denormalize(img)
            pred_c = int(preds[idx])
            tag = "OK" if preds[idx] == labels[idx] else "MISS"
            _overlay(axes[r, 0], rgb, None,
                     f"[{tag}] true={label_to_name[int(labels[idx])]}\n"
                     f"pred={label_to_name[pred_c]}")
            for k, branch in enumerate(BRANCHES, start=1):
                heat = gradcam_for_branch(model, branch, img, pred_c, args.device)
                if heat is None and branch == "swin_t":
                    swin_caveat = True
                cap = "" if branch != "swin_t" else " (7×7*)"
                _overlay(axes[r, k], rgb, heat, f"GradCAM++ {branch}{cap}",
                         note_if_missing=True)
        fig.suptitle(f"Grad-CAM++ — class '{label_to_name[c]}' "
                     f"(F1={float(per_class[str(c)]['f1']):.2f}, "
                     f"n={int(per_class[str(c)]['support'])})  "
                     f"*Swin = coarse windowed map", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fpath = out_dir / f"gradcam_{label_to_name[c]}.png"
        fig.savefig(fpath, dpi=110)
        plt.close(fig)
        print(f"[gradcam++] wrote {fpath.relative_to(_ROOT)}")

    if swin_caveat:
        print("[gradcam++] CAVEAT: some Swin-T maps degenerated and were marked "
              "n/a — windowed layout; ViT-B/BEiT-B maps are the faithful ones.")
    print("[gradcam++] done — figures trace to "
          f"{run_dir.relative_to(_ROOT)}/best.pt (OOF fold {fold})")


if __name__ == "__main__":
    main()
