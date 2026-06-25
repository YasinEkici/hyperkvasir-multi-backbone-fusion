"""Report §5.4 figures + per-class table for the ViT final model (011, Slice 4).

INFERENCE-FREE: reads the champion `11_triple_weighted` per-fold `predictions.npz`
(preds + labels) and `metrics.json` (training `history`) — no model load, no
training, 0 A100.  All figures are pooled over the 5 disjoint OOF folds
(leakage-free, VLD-13; covers the dataset once, n=10662), so they match the
headline pooled macro-F1 0.6119.

Outputs (small PNG -> reports/vit/figures/, tables -> results/vit/tables/):
  * confusion_matrix.png      — row-normalised 23×23 pooled OOF confusion matrix
  * per_class_f1.png          — per-class F1 (sorted), coloured by support
  * per_class_champion.{csv,md} — per-class P/R/F1/support table (pooled OOF)
  * training_curves.png       — fold-0 train loss + val macro-F1 vs epoch
  * cv_macrof1_bar.png        — the 4 CV configs' macro-F1 (mean ± std)

Usage:
    uv run python scripts/make_vit_report_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.manifests import read_manifest_csv  # noqa: E402
from src.evaluation.visualization import plot_confusion_matrix  # noqa: E402

RUNS = _ROOT / "results" / "vit" / "runs"
FIG = _ROOT / "reports" / "vit" / "figures"
TBL = _ROOT / "results" / "vit" / "tables"
CHAMP = "11_triple_weighted_cv"
NUM = 23

# The 4 CV configs (id, label) for the comparison bar — from cv_fold5_ranked.
CV_CONFIGS = [
    ("11_triple_weighted_cv", "V+S+B weighted"),
    ("02_single_swin_t_cv", "Swin-T (tek)"),
    ("09_pair_swin_t_beit_b_weighted_cv", "S+B weighted"),
    ("05_pair_vit_b_beit_b_concat_cv", "V+B concat"),
]


def _run_name(cid: str, fold: int) -> str:
    return cid if fold == 0 else f"{cid}_fold_{fold}"


def _pooled_oof(cid: str) -> tuple[np.ndarray, np.ndarray]:
    preds, labels = [], []
    for f in range(5):
        d = np.load(RUNS / _run_name(cid, f) / "predictions.npz")
        preds.append(d["preds"])
        labels.append(d["labels"])
    return np.concatenate(preds), np.concatenate(labels)


def _class_names() -> list[str]:
    rows = read_manifest_csv(_ROOT / "data" / "splits"
                             / "hyperkvasir_official_5fold" / "fold_0.csv")
    label_to_name = {int(r["label"]): str(r["class_name"]) for r in rows}
    return [label_to_name[i] for i in range(NUM)]


def _fold_macrof1_mean_std(cid: str) -> tuple[float, float]:
    vals = []
    for f in range(5):
        m = json.loads((RUNS / _run_name(cid, f) / "metrics.json").read_text())
        vals.append(float(m["test"]["macro_f1"]))
    return float(np.mean(vals)), float(np.std(vals))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TBL.mkdir(parents=True, exist_ok=True)
    names = _class_names()
    preds, labels = _pooled_oof(CHAMP)
    print(f"[fig] pooled OOF n={len(labels)} (expect 10662)")

    # --- 1. Confusion matrix (pooled OOF, row-normalised) ---
    cm = confusion_matrix(labels, preds, labels=list(range(NUM)))
    plot_confusion_matrix(
        cm.tolist(), names, FIG / "confusion_matrix.png",
        title="Nihai model — havuzlanmış OOF karışıklık matrisi (satır-normalize)",
    )
    print(f"[fig] wrote {(FIG / 'confusion_matrix.png').relative_to(_ROOT)}")

    # --- 2. Per-class P/R/F1/support (pooled OOF) -> table + bar ---
    p, r, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM)), zero_division=0)
    rows_tbl = [
        {"class": names[i], "precision": round(float(p[i]), 4),
         "recall": round(float(r[i]), 4), "f1": round(float(f1[i]), 4),
         "support": int(sup[i])}
        for i in range(NUM)
    ]
    with open(TBL / "per_class_champion.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["class", "precision", "recall", "f1", "support"])
        w.writeheader()
        w.writerows(rows_tbl)
    md = ["# Nihai model — sınıf bazında metrikler (havuzlanmış OOF, n=10662)", "",
          "_Kaynak: `results/vit/runs/11_triple_weighted_cv*/predictions.npz` (VLD-10/13)._",
          "", "| Sınıf | Precision | Recall | F1 | Destek |", "|---|---:|---:|---:|---:|"]
    for row in sorted(rows_tbl, key=lambda x: x["f1"]):
        md.append(f"| {row['class']} | {row['precision']:.4f} | {row['recall']:.4f} "
                  f"| {row['f1']:.4f} | {row['support']} |")
    (TBL / "per_class_champion.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[fig] wrote {(TBL / 'per_class_champion.md').relative_to(_ROOT)}")

    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.clip(sup[order] / max(sup.max(), 1), 0, 1))
    ax.barh([names[i] for i in order], f1[order], color=colors)
    ax.set_xlabel("Sınıf bazında F1 (havuzlanmış OOF)")
    ax.set_title("Nihai model — sınıf bazında F1 (renk = destek; nadir sınıflar F1'i sınırlar)",
                 fontsize=9)
    ax.tick_params(axis="y", labelsize=6.5)
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=0, vmax=float(sup.max())))
    fig.colorbar(sm, ax=ax, label="destek (test örnek sayısı)", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "per_class_f1.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {(FIG / 'per_class_f1.png').relative_to(_ROOT)}")

    # --- 3. Training curves (champion fold 0) ---
    hist = json.loads((RUNS / CHAMP / "metrics.json").read_text())["history"]
    ep = [h["epoch"] for h in hist]
    tr_loss = [h["train_loss"] for h in hist]
    val_f1 = [h["val_macro_f1"] for h in hist]
    best_i = int(np.argmax(val_f1))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ep, tr_loss, "o-", color="tab:red", label="eğitim kaybı")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("eğitim kaybı", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(ep, val_f1, "s-", color="tab:blue", label="doğrulama makro-F1")
    ax2.axvline(ep[best_i], ls="--", color="gray", lw=1)
    ax2.set_ylabel("doğrulama makro-F1", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title(f"Nihai model eğitim eğrileri (fold 0; en iyi epoch={ep[best_i]})",
                  fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {(FIG / 'training_curves.png').relative_to(_ROOT)}")

    # --- 4. CV macro-F1 bar (mean ± std) ---
    labels_cv, means, stds = [], [], []
    for cid, lab in CV_CONFIGS:
        mean, std = _fold_macrof1_mean_std(cid)
        labels_cv.append(lab)
        means.append(mean)
        stds.append(std)
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(labels_cv))
    ax.bar(xs, means, yerr=stds, capsize=4, color="tab:purple", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels_cv, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("5-katlı makro-F1 (ortalama ± std)")
    ax.set_ylim(0.5, 0.65)
    ax.set_title("Seçili yapılandırmalar — 5-katlı CV makro-F1", fontsize=9)
    for x, m in zip(xs, means):
        ax.text(x, m + 0.005, f"{m:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "cv_macrof1_bar.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {(FIG / 'cv_macrof1_bar.png').relative_to(_ROOT)}")

    print("[fig] done — all figures trace to results/vit/runs/11_triple_weighted_cv*/")


if __name__ == "__main__":
    main()
