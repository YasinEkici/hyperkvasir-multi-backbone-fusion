"""Aggregate Sprint 3 ViT fine-tune metrics (007-vit-finetune.md, Slice 5).

Ranks the five fold-0 fine-tune candidates by test macro-F1, compares each to its
frozen Sprint 2 counterpart, and reports the Sprint 4 top-4. Every number traces
to ``results/vit/runs/{id}/metrics.json`` (VLD-10); fold 0 only (VLD-12). Tables
are written under ``results/vit/tables/`` (gitignored); the committed record of
numbers lives in ``docs/vit/results_progress.md``.

This helper exists because ``scripts/generate_report_tables.py`` sorts by id and
does not rank-by-F1 or compute frozen-vs-fine-tune deltas (exec-plan §5/§9).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# The locked Sprint 3 funnel: (frozen_id, finetune_id, backbones, fusion).
FUNNEL: list[tuple[str, str, str, str]] = [
    ("02_single_swin_t_frozen_official",
     "02_single_swin_t_finetune_official", "S", "none"),
    ("11_triple_weighted_frozen_official",
     "11_triple_weighted_finetune_official", "V+S+B", "weighted"),
    ("09_pair_swin_t_beit_b_weighted_frozen_official",
     "09_pair_swin_t_beit_b_weighted_finetune_official", "S+B", "weighted"),
    ("04_pair_vit_b_swin_t_concat_frozen_official",
     "04_pair_vit_b_swin_t_concat_finetune_official", "V+S", "concat"),
    ("05_pair_vit_b_beit_b_concat_frozen_official",
     "05_pair_vit_b_beit_b_concat_finetune_official", "V+B", "concat"),
]

METRIC_KEYS: tuple[str, ...] = (
    "accuracy", "macro_f1", "macro_precision", "macro_recall",
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def rank_by_macro_f1(rows: list[dict]) -> list[dict]:
    """Return *rows* ranked by ``macro_f1`` descending (stable), with 1-based rank."""
    ranked = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)
    return [{**row, "rank": i} for i, row in enumerate(ranked, start=1)]


def delta(frozen: dict, finetune: dict) -> dict:
    """Per-metric ``finetune - frozen`` over METRIC_KEYS."""
    return {k: finetune[k] - frozen[k] for k in METRIC_KEYS}


def select_top_k(ranked_rows: list[dict], k: int) -> list[str]:
    """Return the finetune ids of the top *k* ranked rows."""
    return [r["finetune_id"] for r in ranked_rows[:k]]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_test_metrics(runs_root: Path, run_id: str) -> tuple[dict, str]:
    """Return ({metric: value} over METRIC_KEYS, source_path) from metrics.json."""
    path = runs_root / run_id / "metrics.json"
    test = json.loads(path.read_text())["test"]
    return {k: float(test[k]) for k in METRIC_KEYS}, path.as_posix()


def build(funnel: list[tuple[str, str, str, str]], runs_root: Path) -> tuple[list[dict], list[dict]]:
    """Return (ranked finetune rows, delta rows) for the funnel."""
    finetune_rows: list[dict] = []
    delta_rows: list[dict] = []
    for frozen_id, finetune_id, backbones, fusion in funnel:
        fr, fr_src = load_test_metrics(runs_root, frozen_id)
        ft, ft_src = load_test_metrics(runs_root, finetune_id)
        finetune_rows.append({
            "finetune_id": finetune_id, "backbones": backbones, "fusion": fusion,
            **ft, "source": ft_src,
        })
        d = delta(fr, ft)
        delta_rows.append({
            "finetune_id": finetune_id, "frozen_id": frozen_id,
            "backbones": backbones, "fusion": fusion,
            **{f"frozen_{k}": fr[k] for k in METRIC_KEYS},
            **{f"finetune_{k}": ft[k] for k in METRIC_KEYS},
            **{f"delta_{k}": d[k] for k in METRIC_KEYS},
            "frozen_source": fr_src, "finetune_source": ft_src,
        })
    return rank_by_macro_f1(finetune_rows), delta_rows


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: float) -> str:
    return f"{v:.10f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="results/vit/runs")
    parser.add_argument("--output-dir", default="results/vit/tables")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    runs_root = _ROOT / args.runs_dir
    out_dir = _ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked, deltas = build(FUNNEL, runs_root)

    # --- ranked table ---
    rank_fields = ["rank", "finetune_id", "backbones", "fusion",
                   "accuracy", "macro_f1", "macro_precision", "macro_recall", "source"]
    _write_csv(out_dir / "finetune_fold0_ranked.csv", ranked, rank_fields)
    md = ["# Sprint 3 fine-tune fold-0 ranking (by test macro-F1)\n\n",
          "_All numbers from `results/vit/runs/{id}/metrics.json` (VLD-10), fold 0._\n\n",
          "| rank | config | backbones | fusion | Acc | macro-F1 | macro-P | macro-R | source |\n",
          "|---:|---|---|---|---:|---:|---:|---:|---|\n"]
    for r in ranked:
        md.append(
            f"| {r['rank']} | `{r['finetune_id']}` | {r['backbones']} | {r['fusion']} "
            f"| {_fmt(r['accuracy'])} | {_fmt(r['macro_f1'])} | {_fmt(r['macro_precision'])} "
            f"| {_fmt(r['macro_recall'])} | `{r['source']}` |\n"
        )
    (out_dir / "finetune_fold0_ranked.md").write_text("".join(md), encoding="utf-8")

    # --- delta table ---
    delta_fields = (["finetune_id", "frozen_id", "backbones", "fusion"]
                    + [f"frozen_{k}" for k in METRIC_KEYS]
                    + [f"finetune_{k}" for k in METRIC_KEYS]
                    + [f"delta_{k}" for k in METRIC_KEYS]
                    + ["frozen_source", "finetune_source"])
    _write_csv(out_dir / "frozen_vs_finetune_fold0_delta.csv", deltas, delta_fields)
    dmd = ["# Sprint 3 frozen vs fine-tune fold-0 deltas (finetune - frozen)\n\n",
           "_All numbers from `results/vit/runs/{id}/metrics.json` (VLD-10), fold 0._\n\n",
           "| config | backbones | fusion | Δacc | Δmacro-F1 | Δmacro-P | Δmacro-R |\n",
           "|---|---|---|---:|---:|---:|---:|\n"]
    for r in deltas:
        dmd.append(
            f"| `{r['finetune_id']}` | {r['backbones']} | {r['fusion']} "
            f"| {r['delta_accuracy']:+.10f} | {r['delta_macro_f1']:+.10f} "
            f"| {r['delta_macro_precision']:+.10f} | {r['delta_macro_recall']:+.10f} |\n"
        )
    (out_dir / "frozen_vs_finetune_fold0_delta.md").write_text("".join(dmd), encoding="utf-8")

    top = select_top_k(ranked, args.top_k)

    print("Fine-tune fold-0 ranking (by test macro-F1):")
    for r in ranked:
        print(f"  {r['rank']}. {r['macro_f1']:.10f}  {r['finetune_id']}")
    print(f"\nSprint 4 top-{args.top_k} (carry forward): {top}")
    dropped = [r["finetune_id"] for r in ranked[args.top_k:]]
    print(f"Dropped: {dropped}")
    print(f"\nWrote tables -> {out_dir}")


if __name__ == "__main__":
    main()
