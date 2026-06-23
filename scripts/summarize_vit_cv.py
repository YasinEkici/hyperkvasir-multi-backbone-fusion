"""Aggregate Sprint 4 ViT 5-fold CV (009-vit-cv.md, Slice 4).

For each top-4 `_cv` config: per-fold + mean +/- std (Acc / macro-F1 / macro-P /
macro-R) from the fold metrics.json, and a **bootstrap 95% CI on macro-F1** over
the concatenated out-of-fold test predictions. The five official folds' test sets
are disjoint and cover the dataset once (verified), so concatenating their
predictions scores each sample exactly once -- leakage-free (VLD-13): we pool
held-out *predictions*, never average fold *models*.

Every number traces to results/vit/runs/{run_id}/{metrics.json,predictions.npz}.
Tables are written under results/vit/tables/ (gitignored); the committed record
of numbers lives in docs/vit/results_progress.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

_ROOT = Path(__file__).resolve().parents[1]

# (cv_id, backbones, fusion) — the locked Sprint 4 top-4.
CV_CONFIGS: list[tuple[str, str, str]] = [
    ("02_single_swin_t_cv", "S", "none"),
    ("09_pair_swin_t_beit_b_weighted_cv", "S+B", "weighted"),
    ("11_triple_weighted_cv", "V+S+B", "weighted"),
    ("05_pair_vit_b_beit_b_concat_cv", "V+B", "concat"),
]
METRIC_KEYS = ("accuracy", "macro_f1", "macro_precision", "macro_recall")
FOLDS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, sample std with ddof=1) of *values*."""
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    """macro-F1 matching src/evaluation/metrics.compute_metrics (zero_division=0)."""
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def bootstrap_macro_f1_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for macro-F1 over (labels, preds). Returns
    (point, lo, hi). Resamples sample indices with replacement (leakage-free:
    the inputs are concatenated out-of-fold held-out test predictions)."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    n = len(labels)
    rng = np.random.default_rng(seed)
    point = macro_f1(labels, preds)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = macro_f1(labels[idx], preds[idx])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def rank_by_mean_macro_f1(rows: list[dict]) -> list[dict]:
    ranked = sorted(rows, key=lambda r: r["macro_f1_mean"], reverse=True)
    return [{**r, "rank": i} for i, r in enumerate(ranked, start=1)]


# ---------------------------------------------------------------------------
# IO / aggregation
# ---------------------------------------------------------------------------

def _run_name(cv_id: str, fold: int) -> str:
    return cv_id if fold == 0 else f"{cv_id}_fold_{fold}"


def summarize_config(runs_root: Path, cv_id: str, backbones: str, fusion: str,
                     folds: tuple[int, ...] = FOLDS, n_boot: int = 2000) -> dict:
    per_fold = {k: [] for k in METRIC_KEYS}
    all_preds, all_labels, sources = [], [], []
    for f in folds:
        d = runs_root / _run_name(cv_id, f)
        m = json.loads((d / "metrics.json").read_text())["test"]
        for k in METRIC_KEYS:
            per_fold[k].append(float(m[k]))
        npz = np.load(d / "predictions.npz")
        all_preds.append(np.asarray(npz["preds"]))
        all_labels.append(np.asarray(npz["labels"]))
        sources.append((d / "metrics.json").as_posix())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    point, lo, hi = bootstrap_macro_f1_ci(labels, preds, n_boot=n_boot)
    row = {"cv_id": cv_id, "backbones": backbones, "fusion": fusion,
           "n_oof": int(len(preds))}
    for k in METRIC_KEYS:
        mu, sd = mean_std(per_fold[k])
        row[f"{k}_mean"], row[f"{k}_std"] = mu, sd
    row["macro_f1_pooled"], row["macro_f1_ci_lo"], row["macro_f1_ci_hi"] = point, lo, hi
    row["per_fold_macro_f1"] = per_fold["macro_f1"]
    row["sources"] = sources
    return row


def _fmt(v: float) -> str:
    return f"{v:.10f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="results/vit/runs")
    p.add_argument("--output-dir", default="results/vit/tables")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    runs_root = _ROOT / args.runs_dir
    out_dir = _ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize_config(runs_root, cid, bb, fus, n_boot=args.n_boot)
            for cid, bb, fus in CV_CONFIGS]
    ranked = rank_by_mean_macro_f1(rows)

    # --- CSV ---
    fields = ["rank", "cv_id", "backbones", "fusion",
              "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std",
              "macro_precision_mean", "macro_precision_std",
              "macro_recall_mean", "macro_recall_std",
              "macro_f1_pooled", "macro_f1_ci_lo", "macro_f1_ci_hi", "n_oof"]
    with (out_dir / "cv_fold5_ranked.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(ranked)

    # --- Markdown ---
    md = ["# Sprint 4 5-fold CV ranking (by mean macro-F1)\n\n",
          "_All numbers from `results/vit/runs/{id}/metrics.json` + `predictions.npz` "
          "(VLD-10/13). Bootstrap 95% CI over concatenated out-of-fold test "
          "predictions (folds disjoint, cover dataset once)._\n\n",
          "| rank | config | bb | fusion | Acc (mean±std) | macro-F1 (mean±std) | "
          "macro-F1 pooled [95% CI] | macro-P | macro-R |\n",
          "|---:|---|---|---|---|---|---|---|---|\n"]
    for r in ranked:
        md.append(
            f"| {r['rank']} | `{r['cv_id']}` | {r['backbones']} | {r['fusion']} "
            f"| {r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f} "
            f"| {r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f} "
            f"| {r['macro_f1_pooled']:.4f} [{r['macro_f1_ci_lo']:.4f}, {r['macro_f1_ci_hi']:.4f}] "
            f"| {r['macro_precision_mean']:.4f}±{r['macro_precision_std']:.4f} "
            f"| {r['macro_recall_mean']:.4f}±{r['macro_recall_std']:.4f} |\n"
        )
    (out_dir / "cv_fold5_ranked.md").write_text("".join(md), encoding="utf-8")

    print("Sprint 4 5-fold CV (by mean macro-F1):")
    for r in ranked:
        print(f"  {r['rank']}. {r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f} "
              f"pooled {r['macro_f1_pooled']:.4f} "
              f"[{r['macro_f1_ci_lo']:.4f},{r['macro_f1_ci_hi']:.4f}]  {r['cv_id']}")
    best = ranked[0]
    print(f"\nBest config (mean macro-F1): {best['cv_id']} "
          f"({best['macro_f1_mean']:.4f}±{best['macro_f1_std']:.4f})")
    print(f"Wrote tables -> {out_dir}")


if __name__ == "__main__":
    main()
