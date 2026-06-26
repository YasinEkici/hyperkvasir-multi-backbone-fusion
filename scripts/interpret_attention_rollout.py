"""Attention-rollout panels for the frozen final model (011-vit-report.md, Slice 1).

Transformer-native interpretability (Abnar & Zuidema 2020) for the CLS-token
ViT-B/16 and BEiT-B/16 branches of ``11_triple_weighted``.  Inference-only from
``best.pt`` over the OOF test split (leakage-free, VLD-13); no retraining.

Rollout (paper §3): per layer average heads, ``A = 0.5 W_att + 0.5 I``,
re-normalise rows, recursively matmul; take the CLS row over patches, reshape to
14×14, upsample to 224, overlay.  Swin-T is excluded here (hierarchical / shifted
windows make CLS-token rollout ill-defined — see Grad-CAM++ and the documented
caveat); rollout is shown for the two isotropic CLS-token branches.

timm computes attention with a fused SDPA kernel that does not expose the
attention matrix.  We temporarily set ``attn.fused_attn = False`` and hook each
block's ``attn_drop`` (identity in eval) to capture the post-softmax weights,
then restore the original flags — no edit to ``vit_backbones.py``.

Usage:
    uv run python scripts/interpret_attention_rollout.py --run 11_triple_weighted_cv
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
    cls_map_to_grid,
    denormalize,
    oof_test_dataset,
    pick_examples,
    predict_all,
    rollout_from_attentions,
    select_classes_by_f1,
)

ROLLOUT_BRANCHES = ("vit_b", "beit_b")  # CLS-token isotropic branches only


class _AttentionCapture:
    """Context manager: capture post-softmax attention from a timm blocks model.

    Disables the fused SDPA path and hooks each block's ``attn_drop``; restores
    the original ``fused_attn`` flags and removes hooks on exit.
    """

    def __init__(self, timm_model: torch.nn.Module) -> None:
        self.blocks = timm_model.blocks
        self._handles: list = []
        self._saved_fused: list[bool] = []
        self.attentions: list[torch.Tensor] = []

    def __enter__(self) -> "_AttentionCapture":
        for blk in self.blocks:
            self._saved_fused.append(getattr(blk.attn, "fused_attn", False))
            blk.attn.fused_attn = False
            self._handles.append(
                blk.attn.attn_drop.register_forward_hook(self._hook)
            )
        return self

    def _hook(self, _module, _inp, out) -> None:
        self.attentions.append(out.detach())

    def __exit__(self, *exc) -> bool:
        for h in self._handles:
            h.remove()
        for blk, saved in zip(self.blocks, self._saved_fused):
            blk.attn.fused_attn = saved
        return False


def rollout_for_branch(
    model, branch: str, img: torch.Tensor, device: str
) -> np.ndarray:
    """Return the 14×14 CLS→patch rollout map (normalised to [0,1]) for one image."""
    extractor = model.backbones[branch]
    timm_model = extractor.model
    num_prefix = int(getattr(timm_model, "num_prefix_tokens", 1))
    x = img.unsqueeze(0).to(device)
    with _AttentionCapture(timm_model) as cap, torch.no_grad():
        timm_model(x)  # populate per-layer attention via hooks
        attns = [a[0].cpu().numpy() for a in cap.attentions]  # list of (heads,N,N)
    rolled = rollout_from_attentions(attns, add_residual=True)
    grid = cls_map_to_grid(rolled[0], num_prefix=num_prefix)
    gmin, gmax = float(grid.min()), float(grid.max())
    if gmax > gmin:
        grid = (grid - gmin) / (gmax - gmin)
    return grid


def _overlay(ax, rgb: np.ndarray, heat: np.ndarray | None, title: str) -> None:
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
    import json

    per_class = json.loads((run_dir / "metrics.json").read_text())["test"]["per_class"]
    best, worst = select_classes_by_f1(per_class, args.n_best, args.n_worst)
    classes = best + worst

    print(f"[rollout] run={args.run} fold={fold} OOF={len(ds)} "
          f"device={args.device}")
    print(f"[rollout] best classes={[label_to_name[c] for c in best]}")
    print(f"[rollout] worst classes={[label_to_name[c] for c in worst]}")

    preds, labels = predict_all(model, ds, args.device)
    picks = pick_examples(preds, labels, classes, args.n_correct, args.n_failure)

    # One panel per class: rows = examples, cols = [input, ViT-B, BEiT-B].
    for c in classes:
        sel = picks[c]["correct"] + picks[c]["failure"]
        if not sel:
            print(f"[rollout] class {label_to_name[c]}: no OOF samples, skipped")
            continue
        ncol = 1 + len(ROLLOUT_BRANCHES)
        fig, axes = plt.subplots(len(sel), ncol, figsize=(2.2 * ncol, 2.2 * len(sel)))
        axes = np.atleast_2d(axes)
        for r, idx in enumerate(sel):
            img, label, _ = ds[idx]
            rgb = denormalize(img)
            pred_name = label_to_name[int(preds[idx])]
            true_name = label_to_name[int(labels[idx])]
            tag = "OK" if preds[idx] == labels[idx] else "MISS"
            _overlay(axes[r, 0], rgb, None,
                     f"[{tag}] true={true_name}\npred={pred_name}")
            for k, branch in enumerate(ROLLOUT_BRANCHES, start=1):
                heat = rollout_for_branch(model, branch, img, args.device)
                _overlay(axes[r, k], rgb, heat, f"rollout {branch}")
        fig.suptitle(f"Attention rollout — class '{label_to_name[c]}' "
                     f"(F1={float(per_class[str(c)]['f1']):.2f}, "
                     f"n={int(per_class[str(c)]['support'])})", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fpath = out_dir / f"rollout_{label_to_name[c]}.png"
        fig.savefig(fpath, dpi=110)
        plt.close(fig)
        print(f"[rollout] wrote {fpath.relative_to(_ROOT)}")

    print("[rollout] done — figures trace to "
          f"{run_dir.relative_to(_ROOT)}/best.pt (OOF fold {fold})")


if __name__ == "__main__":
    main()
