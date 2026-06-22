"""Tests for scripts/summarize_vit_cv.py (Sprint 4 Slice 4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import summarize_vit_cv as svc  # noqa: E402


def test_mean_std_sample_ddof1() -> None:
    mu, sd = svc.mean_std([0.5, 0.6, 0.7])
    assert abs(mu - 0.6) < 1e-12
    assert abs(sd - 0.1) < 1e-12  # sample std (ddof=1)


def test_macro_f1_perfect_and_matches_sklearn() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2])
    assert svc.macro_f1(labels, labels) == 1.0
    preds = np.array([0, 1, 2, 0, 1, 1])  # one wrong
    assert 0.0 < svc.macro_f1(labels, preds) < 1.0


def test_bootstrap_ci_brackets_point_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 400)
    preds = labels.copy()
    preds[:40] = (preds[:40] + 1) % 5  # 10% errors
    p1, lo1, hi1 = svc.bootstrap_macro_f1_ci(labels, preds, n_boot=200, seed=42)
    p2, lo2, hi2 = svc.bootstrap_macro_f1_ci(labels, preds, n_boot=200, seed=42)
    assert (p1, lo1, hi1) == (p2, lo2, hi2)          # seeded → reproducible
    assert lo1 <= p1 <= hi1                          # CI brackets the point
    assert p1 == svc.macro_f1(labels, preds)         # point == full-sample macro-F1


def test_bootstrap_ci_perfect_predictor() -> None:
    labels = np.array([0, 1, 2, 3, 4] * 20)
    p, lo, hi = svc.bootstrap_macro_f1_ci(labels, labels, n_boot=100, seed=1)
    assert p == 1.0 and lo == 1.0 and hi == 1.0


def test_rank_by_mean_macro_f1() -> None:
    rows = [{"cv_id": "a", "macro_f1_mean": 0.50},
            {"cv_id": "b", "macro_f1_mean": 0.61},
            {"cv_id": "c", "macro_f1_mean": 0.55}]
    ranked = svc.rank_by_mean_macro_f1(rows)
    assert [r["cv_id"] for r in ranked] == ["b", "c", "a"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]


def _write_fold(runs: Path, cv_id: str, fold: int, f1: float, n: int = 50) -> None:
    rn = cv_id if fold == 0 else f"{cv_id}_fold_{fold}"
    d = runs / rn
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"fold": fold, "test": {
        "accuracy": 0.8, "macro_f1": f1, "macro_precision": 0.6, "macro_recall": 0.6}}))
    rng = np.random.default_rng(fold)
    labels = rng.integers(0, 5, n)
    np.savez(d / "predictions.npz", preds=labels, labels=labels)  # perfect → pooled f1=1


def test_summarize_config_aggregates_folds(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    f1s = [0.60, 0.65, 0.59, 0.60, 0.59]
    for fold, f1 in zip(range(5), f1s):
        _write_fold(runs, "x_cv", fold, f1)
    row = svc.summarize_config(runs, "x_cv", "S", "none", n_boot=50)
    assert row["per_fold_macro_f1"] == f1s
    assert abs(row["macro_f1_mean"] - float(np.mean(f1s))) < 1e-12
    assert row["n_oof"] == 250  # 5 folds x 50
    assert row["macro_f1_pooled"] == 1.0  # perfect preds in fixtures
