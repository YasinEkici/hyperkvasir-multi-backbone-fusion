"""Tests for scripts/summarize_vit_finetune.py (Sprint 3 Slice 5)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import summarize_vit_finetune as svf  # noqa: E402


def test_rank_by_macro_f1_descending_with_ranks() -> None:
    rows = [
        {"finetune_id": "a", "macro_f1": 0.50},
        {"finetune_id": "b", "macro_f1": 0.59},
        {"finetune_id": "c", "macro_f1": 0.55},
    ]
    ranked = svf.rank_by_macro_f1(rows)
    assert [r["finetune_id"] for r in ranked] == ["b", "c", "a"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]
    # Pure: input not mutated with a rank key.
    assert "rank" not in rows[0]


def test_delta_is_finetune_minus_frozen() -> None:
    frozen = {"accuracy": 0.80, "macro_f1": 0.50, "macro_precision": 0.51, "macro_recall": 0.52}
    finetune = {"accuracy": 0.83, "macro_f1": 0.56, "macro_precision": 0.55, "macro_recall": 0.60}
    d = svf.delta(frozen, finetune)
    assert d["macro_f1"] == 0.56 - 0.50
    assert d["accuracy"] == 0.83 - 0.80
    assert set(d) == set(svf.METRIC_KEYS)


def test_select_top_k() -> None:
    ranked = [{"finetune_id": x} for x in ["b", "c", "a", "d", "e"]]
    assert svf.select_top_k(ranked, 4) == ["b", "c", "a", "d"]


def _write_metrics(run_dir: Path, **test_vals) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps({"fold": 0, "test": test_vals}))


def test_build_ranks_and_deltas_from_metrics(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    # Two-row mini funnel; finetune beats frozen.
    funnel = [
        ("fz_a", "ft_a", "S", "none"),
        ("fz_b", "ft_b", "S+B", "weighted"),
    ]
    _write_metrics(runs / "fz_a", accuracy=0.80, macro_f1=0.50, macro_precision=0.5, macro_recall=0.5)
    _write_metrics(runs / "ft_a", accuracy=0.86, macro_f1=0.59, macro_precision=0.6, macro_recall=0.6)
    _write_metrics(runs / "fz_b", accuracy=0.85, macro_f1=0.56, macro_precision=0.5, macro_recall=0.5)
    _write_metrics(runs / "ft_b", accuracy=0.88, macro_f1=0.58, macro_precision=0.6, macro_recall=0.6)

    ranked, deltas = svf.build(funnel, runs)
    # ranked by macro_f1 desc: ft_a (0.59) before ft_b (0.58)
    assert [r["finetune_id"] for r in ranked] == ["ft_a", "ft_b"]
    # delta rows keep source paths and correct macro-F1 deltas
    by_id = {d["finetune_id"]: d for d in deltas}
    assert round(by_id["ft_a"]["delta_macro_f1"], 4) == 0.09
    assert round(by_id["ft_b"]["delta_macro_f1"], 4) == 0.02
    assert by_id["ft_a"]["finetune_source"].endswith("runs/ft_a/metrics.json")
