"""McNemar significance test for the frozen final model (011-vit-report.md, Slice 3).

Tests whether the champion `11_triple_weighted` differs *significantly* from a
baseline on paired OOF predictions (Dietterich 1998, "Approximate Statistical
Tests for Comparing Supervised Classification Learning Algorithms").  Leakage-free
and ADDITIVE: it reads the stored `predictions.npz` of both runs (no recompute, no
model averaging), concatenates the 5 disjoint OOF folds (covers the dataset once),
verifies the per-fold labels align, and reports the 2×2 table + statistic + exact
two-sided binomial p-value.  The champion is unchanged regardless of the outcome.

Default comparison: champion (triple weighted fusion) vs the best single backbone
(`02_single_swin_t_cv`) — i.e. "does multi-ViT fusion help significantly?" (§5.5).

Usage:
    uv run python scripts/stats_mcnemar.py --champion 11_triple_weighted_cv \
        --baseline 02_single_swin_t_cv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.interpret_common import mcnemar_contingency, mcnemar_pvalue  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)


def _run_name(cid: str, fold: int) -> str:
    return cid if fold == 0 else f"{cid}_fold_{fold}"


def _load_oof(runs: Path, cid: str):
    """Return (concat preds, concat labels, per-fold label arrays) over the 5 OOF folds."""
    preds, labels, fold_labels = [], [], []
    for f in FOLDS:
        d = np.load(runs / _run_name(cid, f) / "predictions.npz")
        preds.append(d["preds"])
        labels.append(d["labels"])
        fold_labels.append(d["labels"])
    return np.concatenate(preds), np.concatenate(labels), fold_labels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="results/vit/runs")
    p.add_argument("--champion", default="11_triple_weighted_cv")
    p.add_argument("--baseline", default="02_single_swin_t_cv")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()
    runs = _ROOT / args.runs_dir

    champ_preds, champ_labels, champ_fold_labels = _load_oof(runs, args.champion)
    base_preds, base_labels, base_fold_labels = _load_oof(runs, args.baseline)

    # Leakage / pairing guard: same OOF samples in the same per-fold order.
    aligned = all(np.array_equal(a, b)
                  for a, b in zip(champ_fold_labels, base_fold_labels))
    if not aligned or not np.array_equal(champ_labels, base_labels):
        raise SystemExit("ERROR: champion/baseline OOF labels do not align — "
                         "cannot pair predictions for McNemar.")
    labels = champ_labels

    table = mcnemar_contingency(champ_preds, base_preds, labels)
    stat = mcnemar_pvalue(table["a_only"], table["b_only"])

    champ_acc = float((champ_preds == labels).mean())
    base_acc = float((base_preds == labels).mean())

    print(f"McNemar — champion vs baseline (paired OOF, N={len(labels)})")
    print(f"  champion: {args.champion}  (OOF acc {champ_acc:.4f})")
    print(f"  baseline: {args.baseline}  (OOF acc {base_acc:.4f})")
    print("  2×2 paired-correctness table:")
    print(f"    both correct      : {table['both_correct']}")
    print(f"    champion only     : {table['a_only']}  (champ right, base wrong)")
    print(f"    baseline only     : {table['b_only']}  (champ wrong, base right)")
    print(f"    both wrong        : {table['both_wrong']}")
    print(f"  discordant pairs    : {stat['n_discordant']}")
    print(f"  chi-square (cc)     : {stat['chi2_cc']:.4f}")
    print(f"  exact two-sided p   : {stat['p_exact']:.4g}")
    sig = "SIGNIFICANT" if stat["p_exact"] < args.alpha else "not significant"
    direction = ("champion better" if table["a_only"] > table["b_only"]
                 else "baseline better" if table["b_only"] > table["a_only"]
                 else "tie")
    print(f"  verdict (alpha={args.alpha}): {sig} ({direction}). "
          f"Final model unchanged (additive analysis).")


if __name__ == "__main__":
    main()
