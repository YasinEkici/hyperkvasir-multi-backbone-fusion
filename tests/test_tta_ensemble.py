"""Tests for scripts/eval_tta_ensemble.py pure helpers (Sprint 4 Slice 5)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import eval_tta_ensemble as ete  # noqa: E402


def test_average_probs_midpoint_and_identity() -> None:
    a = np.array([[0.7, 0.3], [0.2, 0.8]])
    b = np.array([[0.5, 0.5], [0.4, 0.6]])
    m = ete.average_probs([a, b])
    assert np.allclose(m, [[0.6, 0.4], [0.3, 0.7]])
    # averaging identical arrays is a no-op
    assert np.allclose(ete.average_probs([a, a, a]), a)


def test_average_probs_rows_still_sum_to_one() -> None:
    # convex combination of valid distributions stays a distribution
    a = np.array([[0.7, 0.2, 0.1]])
    b = np.array([[0.1, 0.1, 0.8]])
    assert np.allclose(ete.average_probs([a, b]).sum(axis=1), 1.0)


def test_preds_from_probs_argmax() -> None:
    probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    assert list(ete.preds_from_probs(probs)) == [1, 0, 1]


def test_run_name_matches_train_py_scheme() -> None:
    assert ete.run_name("x_cv", 0, 42) == "x_cv"
    assert ete.run_name("x_cv", 3, 42) == "x_cv_fold_3"
    assert ete.run_name("x_cv", 0, 123) == "x_cv_seed123"
    assert ete.run_name("x_cv", 2, 123) == "x_cv_seed123_fold_2"
