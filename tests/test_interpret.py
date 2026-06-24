"""Unit tests for Sprint 5 interpretability helpers (011-vit-report.md, Slice 1).

Pure-function tests only — no model load, no disk, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.interpret_common import (  # noqa: E402
    cls_map_to_grid,
    denormalize,
    pick_examples,
    rollout_from_attentions,
    select_classes_by_f1,
    subsample_indices_per_class,
)


# --------------------------------------------------------------------------- #
# rollout_from_attentions
# --------------------------------------------------------------------------- #

def test_rollout_identity_preserves_rows_sum_to_one():
    n = 5
    eye = np.eye(n)
    rolled = rollout_from_attentions([eye, eye, eye], add_residual=True)
    assert rolled.shape == (n, n)
    np.testing.assert_allclose(rolled.sum(axis=-1), np.ones(n), atol=1e-9)
    # 0.5 I + 0.5 I = I, re-normalised stays I across all layers.
    np.testing.assert_allclose(rolled, eye, atol=1e-9)


def test_rollout_uniform_attention_stays_uniform_and_normalised():
    n = 4
    uniform = np.full((n, n), 1.0 / n)
    rolled = rollout_from_attentions([uniform, uniform], add_residual=True)
    np.testing.assert_allclose(rolled.sum(axis=-1), np.ones(n), atol=1e-9)
    # Rows of a product of doubly-balanced row-stochastic matrices stay valid
    # probability rows; every entry remains positive.
    assert (rolled > 0).all()


def test_rollout_averages_heads():
    # (heads, N, N): two heads that average to the identity.
    n = 3
    head_a = np.eye(n)
    head_b = np.eye(n)
    stacked = np.stack([head_a, head_b], axis=0)
    rolled = rollout_from_attentions([stacked], add_residual=False)
    np.testing.assert_allclose(rolled, np.eye(n), atol=1e-9)


def test_rollout_rejects_non_square():
    with pytest.raises(ValueError):
        rollout_from_attentions([np.ones((3, 4))])


def test_rollout_rejects_empty():
    with pytest.raises(ValueError):
        rollout_from_attentions([])


# --------------------------------------------------------------------------- #
# cls_map_to_grid
# --------------------------------------------------------------------------- #

def test_cls_map_to_grid_drops_prefix_and_squares():
    row = np.arange(197, dtype=float)  # 1 CLS + 196 patches
    grid = cls_map_to_grid(row, num_prefix=1)
    assert grid.shape == (14, 14)
    # first patch value preserved (index 1 of the original row)
    assert grid[0, 0] == 1.0


def test_cls_map_to_grid_rejects_non_square():
    # 11 - 1 prefix = 10 patches, not a perfect square.
    with pytest.raises(ValueError):
        cls_map_to_grid(np.arange(11, dtype=float), num_prefix=1)


# --------------------------------------------------------------------------- #
# denormalize
# --------------------------------------------------------------------------- #

def test_denormalize_shape_and_range():
    t = torch.randn(3, 16, 16)
    rgb = denormalize(t)
    assert rgb.shape == (16, 16, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_denormalize_inverts_normalize():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    raw = torch.rand(3, 8, 8)  # already in [0,1]
    norm = (raw - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    rgb = denormalize(norm, mean, std)
    np.testing.assert_allclose(rgb, raw.permute(1, 2, 0).numpy(), atol=1e-5)


# --------------------------------------------------------------------------- #
# class / example selection
# --------------------------------------------------------------------------- #

def _per_class(f1_support: dict[int, tuple[float, int]]) -> dict:
    return {str(k): {"precision": 0.0, "recall": 0.0, "f1": f1, "support": s}
            for k, (f1, s) in f1_support.items()}


def test_select_classes_by_f1_best_worst_split():
    pc = _per_class({0: (0.1, 8), 1: (0.9, 100), 2: (0.5, 40),
                     3: (0.95, 200), 4: (0.2, 12)})
    best, worst = select_classes_by_f1(pc, n_best=2, n_worst=2)
    assert best[0] == 3 and 1 in best        # highest F1 first
    assert worst[0] == 0 and 4 in worst      # lowest F1 first
    assert set(best).isdisjoint(worst)


def test_select_classes_skips_zero_support():
    pc = _per_class({0: (0.0, 0), 1: (0.5, 10), 2: (0.7, 20)})
    best, worst = select_classes_by_f1(pc, n_best=3, n_worst=3)
    assert 0 not in best and 0 not in worst


def test_pick_examples_correct_and_failure():
    labels = np.array([0, 0, 0, 1, 1, 2])
    preds = np.array([0, 0, 1, 1, 0, 2])  # class0: idx0,1 correct, idx2 miss
    picks = pick_examples(preds, labels, classes=[0, 1], n_correct=2, n_failure=1)
    assert picks[0]["correct"] == [0, 1]
    assert picks[0]["failure"] == [2]
    assert picks[1]["correct"] == [3]
    assert picks[1]["failure"] == [4]


def test_pick_examples_deterministic_lowest_first():
    labels = np.zeros(6, dtype=int)
    preds = np.zeros(6, dtype=int)
    picks = pick_examples(preds, labels, classes=[0], n_correct=3, n_failure=0)
    assert picks[0]["correct"] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# subsample_indices_per_class (UMAP point cap)
# --------------------------------------------------------------------------- #

def test_subsample_caps_each_class():
    labels = np.array([0] * 200 + [1] * 50 + [2] * 5)
    keep = subsample_indices_per_class(labels, max_per_class=30, seed=0)
    kept = labels[keep]
    assert (kept == 0).sum() == 30   # capped
    assert (kept == 1).sum() == 30   # capped
    assert (kept == 2).sum() == 5    # rare class kept in full


def test_subsample_is_sorted_and_deterministic():
    labels = np.array([0] * 100 + [1] * 100)
    a = subsample_indices_per_class(labels, max_per_class=20, seed=42)
    b = subsample_indices_per_class(labels, max_per_class=20, seed=42)
    np.testing.assert_array_equal(a, b)                 # deterministic
    np.testing.assert_array_equal(a, np.sort(a))        # sorted
    assert set(np.unique(labels)).issubset(set(labels[a]))  # all classes present
