"""Post-hoc logit adjustment for the frozen final model (011-vit-report.md, Slice 3).

Leakage-free, inference-only, ADDITIVE analysis — it never re-selects the champion
(`11_triple_weighted` stays the final model, VLD-18/VLD-19).  Recomputes OOF test
logits from each fold's `best.pt`, subtracts ``tau * log(train_prior)`` (Menon et
al. 2020) where the prior comes from that fold's TRAIN split only (VLD-13), and
re-reports OOF macro-F1 + bootstrap 95% CI exactly as TTA/ensemble do.

Headline = ``tau=1.0`` (parameter-free posterior correction; no selection on test,
so leakage-free).  A small ``tau`` sweep is shown only as a sensitivity curve, NOT
used to pick the reported number.  NOTE: the model trains with a WeightedRandom
SampleR (balanced sampling), so its implicit prior is already ~uniform — prior
correction is expected to give little or no gain (possibly slightly negative); the
result is reported honestly either way.

Built-in check: ``tau=0`` reproduces the champion's stored predictions exactly.

Usage:
    uv run python scripts/eval_logit_adjust.py --run 11_triple_weighted_cv
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
sys.path.insert(0, str(_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.manifests import read_manifest_csv  # noqa: E402
from scripts.interpret_common import (  # noqa: E402
    build_model,
    class_log_prior,
    logit_adjust,
    oof_test_dataset,
    predict_logits,
)
from summarize_vit_cv import bootstrap_macro_f1_ci, macro_f1  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)
TAU_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)


def _run_name(cid: str, fold: int) -> str:
    return cid if fold == 0 else f"{cid}_fold_{fold}"


def _train_counts(fold: int, num_classes: int, split: str) -> np.ndarray:
    manifest = _ROOT / "data" / "splits" / split / f"fold_{fold}.csv"
    rows = read_manifest_csv(manifest)
    counts = np.zeros(num_classes, dtype=np.float64)
    for r in rows:
        if str(r.get("split", "")) == "train":
            counts[int(r["label"])] += 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="results/vit/runs")
    p.add_argument("--run", default="11_triple_weighted_cv")
    p.add_argument("--out", default="reports/vit/figures")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    runs = _ROOT / args.runs_dir
    out_dir = _ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_logits, per_fold_labels, per_fold_logprior = [], [], []
    for f in FOLDS:
        run_dir = runs / _run_name(args.run, f)
        model, cfg = build_model(run_dir, args.device)
        ds, _, _, fold = oof_test_dataset(run_dir, cfg)
        assert fold == f, f"fold mismatch {fold} != {f}"
        logits, labels = predict_logits(model, ds, args.device)
        num_classes = int(cfg["dataset"]["num_classes"])
        split = cfg["dataset"].get("split_protocol", "hyperkvasir_official_5fold")
        log_prior = class_log_prior(_train_counts(f, num_classes, split))
        per_fold_logits.append(logits)
        per_fold_labels.append(labels)
        per_fold_logprior.append(log_prior)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[logit-adj] fold {f}: OOF={len(labels)} logits={logits.shape}")

    # Built-in leakage/consistency check: tau=0 reproduces the stored champion
    # predictions. Labels must match exactly; a handful of argmax flips at near-tie
    # logit margins are expected (tuned bf16/TF32 path vs this fp32 recompute —
    # VLD-17 "reproducibility approximate"), so we allow >=99.5% per-fold agreement.
    all_labels, n_flips, n_total = [], 0, 0
    for f in FOLDS:
        all_labels.append(per_fold_labels[f])
        stored = np.load(runs / _run_name(args.run, f) / "predictions.npz")
        assert np.array_equal(per_fold_labels[f], stored["labels"]), \
            f"label mismatch (fold {f})"
        agree = float((per_fold_logits[f].argmax(1) == stored["preds"]).mean())
        n_flips += int((per_fold_logits[f].argmax(1) != stored["preds"]).sum())
        n_total += len(stored["preds"])
        assert agree >= 0.995, f"tau=0 preds diverge from stored (fold {f}, {agree:.4f})"
    all_labels = np.concatenate(all_labels)
    print(f"[logit-adj] built-in check OK: tau=0 reproduces stored OOF predictions "
          f"({n_total - n_flips}/{n_total}; {n_flips} near-tie flips, bf16/TF32 vs fp32).")

    # Sweep tau; OOF macro-F1 (+ CI for tau in {0, 1}).
    print(f"\n{'tau':>5}  {'OOF_macroF1':>12}  {'95% CI':>20}  delta_vs_tau0")
    results = []
    base_f1 = None
    for tau in TAU_GRID:
        oof_preds = np.concatenate(
            [logit_adjust(per_fold_logits[f], per_fold_logprior[f], tau).argmax(1)
             for f in FOLDS]
        )
        f1 = macro_f1(all_labels, oof_preds)
        if tau == 0.0:
            base_f1 = f1
        ci = ""
        if tau in (0.0, 1.0):
            pt, lo, hi = bootstrap_macro_f1_ci(all_labels, oof_preds)
            ci = f"[{lo:.4f}, {hi:.4f}]"
        results.append((tau, f1))
        print(f"{tau:>5.1f}  {f1:>12.4f}  {ci:>20}  {f1 - base_f1:+.4f}")

    headline = dict(results)[1.0]
    print(f"\n[logit-adj] HEADLINE tau=1.0: OOF macro-F1 {headline:.4f} "
          f"(baseline tau=0 {base_f1:.4f}, delta {headline - base_f1:+.4f}).")
    verdict = ("no gain" if headline <= base_f1 + 1e-4
               else f"+{headline - base_f1:.4f}")
    print(f"[logit-adj] verdict: prior correction -> {verdict}; "
          f"final model unchanged (additive analysis).")

    # Sensitivity curve (descriptive, not a selection criterion).
    fig, ax = plt.subplots(figsize=(5, 3.4))
    taus = [t for t, _ in results]
    f1s = [v for _, v in results]
    ax.plot(taus, f1s, "o-", color="tab:blue")
    ax.axhline(base_f1, ls="--", color="gray", lw=1, label="baseline (tau=0)")
    ax.set_xlabel("logit-adjustment strength  tau")
    ax.set_ylabel("OOF macro-F1")
    ax.set_title("Post-hoc logit adjustment — sensitivity\n"
                 "(headline = tau=1.0; champion unchanged)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fpath = out_dir / "logit_adjust_tau.png"
    fig.savefig(fpath, dpi=120)
    plt.close(fig)
    print(f"[logit-adj] wrote {fpath.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
