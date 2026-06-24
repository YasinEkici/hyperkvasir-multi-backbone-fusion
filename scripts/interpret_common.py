"""Shared inference-only helpers for Sprint 5 interpretability (011-vit-report.md).

All figures are produced from a *finished* run's ``best.pt`` over its
out-of-fold (OOF) test split — leakage-free (VLD-13), no retraining, no change
to the training pipeline (mirrors ``scripts/eval_tta_ensemble.py``).  The frozen
final model is ``11_triple_weighted`` (VLD-18); these scripts EXPLAIN it, they
never change it.

Pure, unit-tested helpers live here (rollout math, token→grid reshape,
denormalisation, class selection); the rollout / Grad-CAM++ drivers import them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.datasets import HyperKvasirImageDataset  # noqa: E402
from src.data.manifests import read_manifest_csv  # noqa: E402
from src.models.full_model import MultiCNNFusionClassifier  # noqa: E402
from src.utils.checkpointing import load_checkpoint  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ROOT = _ROOT


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested — no model / disk access)
# ---------------------------------------------------------------------------

def rollout_from_attentions(
    attentions: list[np.ndarray], add_residual: bool = True
) -> np.ndarray:
    """Attention rollout for one sample (Abnar & Zuidema 2020, Eq. 1).

    Parameters
    ----------
    attentions : list of per-layer attention matrices, each ``(heads, N, N)``
        (averaged over heads internally) or already ``(N, N)``.  Ordered from
        the first (closest to input) to the last layer.
    add_residual : bool
        Account for residual connections via ``A = 0.5 W_att + 0.5 I`` and
        re-normalise rows, exactly as in the paper (§3).

    Returns
    -------
    np.ndarray ``(N, N)`` rolled-out attention.  Row ``0`` (the CLS token) over
    columns ``1:`` gives the CLS→patch importance map.
    """
    result: np.ndarray | None = None
    for att in attentions:
        a = np.asarray(att, dtype=np.float64)
        if a.ndim == 3:  # (heads, N, N) -> average heads (paper §3, App. A.1)
            a = a.mean(axis=0)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"each attention must be square (N,N); got {a.shape}")
        if add_residual:
            a = 0.5 * a + 0.5 * np.eye(a.shape[0])
        a = a / a.sum(axis=-1, keepdims=True)  # re-normalise rows to sum 1
        result = a if result is None else a @ result  # Ã(l_i)=A(l_i)·Ã(l_{i-1})
    if result is None:
        raise ValueError("attentions must be non-empty")
    return result


def cls_map_to_grid(cls_row: np.ndarray, num_prefix: int = 1) -> np.ndarray:
    """Drop the prefix (CLS) token(s) and reshape the patch row to a square grid.

    ``cls_row`` is the length-``N`` rolled-out attention of the CLS token.  The
    remaining ``N - num_prefix`` patch tokens must form a perfect square (196 ->
    14×14 for ViT-B/16 / BEiT-B/16 at 224²).
    """
    patches = np.asarray(cls_row, dtype=np.float64)[num_prefix:]
    side = int(round(patches.size ** 0.5))
    if side * side != patches.size:
        raise ValueError(f"{patches.size} patch tokens is not a perfect square")
    return patches.reshape(side, side)


def denormalize(t: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """``(3,H,W)`` ImageNet-normalised tensor -> ``(H,W,3)`` float in ``[0,1]``."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    x = (t.detach().cpu() * s + m).clamp(0.0, 1.0)
    return x.permute(1, 2, 0).numpy()


def select_classes_by_f1(
    per_class: dict, n_best: int = 4, n_worst: int = 4
) -> tuple[list[int], list[int]]:
    """Pick representative classes from the final model's per-class F1.

    Returns ``(best_classes, worst_classes)`` as label indices, considering only
    classes with non-zero support.  Worst classes (low F1, typically rare) tie
    the qualitative panels to the macro-F1 ceiling story (report §5.5).
    """
    scored = [
        (int(k), float(v["f1"]), int(v["support"]))
        for k, v in per_class.items()
        if int(v["support"]) > 0
    ]
    by_f1 = sorted(scored, key=lambda r: (r[1], -r[2]))
    worst = [c for c, _, _ in by_f1[:n_worst]]
    best = [c for c, _, _ in reversed(by_f1[-n_best:])]
    return best, worst


# ---------------------------------------------------------------------------
# Model + OOF dataset loading (inference-only, reuses the training graph)
# ---------------------------------------------------------------------------

def load_run_config(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "config.yaml").read_text())


def build_model(run_dir: Path, device: str) -> tuple[MultiCNNFusionClassifier, dict]:
    """Rebuild the trained model from ``config.yaml`` and load ``best.pt`` (eval)."""
    cfg = load_run_config(run_dir)
    method, dataset, training = cfg["method"], cfg["dataset"], cfg["training"]
    model = MultiCNNFusionClassifier(
        backbone_names=method["backbone_names"],
        unfreeze_blocks=int(training.get("unfreeze_blocks", 3)),
        projection_dim=int(method.get("projection_dim", 512)),
        fusion_type=method.get("fusion_type", "none"),
        num_classes=int(dataset["num_classes"]),
        mlp_hidden=method.get("mlp_hidden", [256]),
        dropout=float(method.get("dropout", 0.3)),
        drop_path_rate=float(training.get("drop_path_rate", 0.05)),
    )
    load_checkpoint(model, run_dir / "best.pt")
    model.to(device).eval()
    return model, cfg


def oof_test_dataset(run_dir: Path, cfg: dict):
    """Build the deterministic OOF test dataset for this run's fold (VLD-13).

    Returns ``(dataset, test_rows, label_to_name, fold)``.  Preprocessing matches
    the val/test transform of training (Resize 256 BICUBIC -> CenterCrop 224 ->
    ImageNet norm; VLD-09).
    """
    dataset_cfg = cfg["dataset"]
    fold = int(json.loads((run_dir / "metrics.json").read_text())["fold"])
    split = dataset_cfg.get("split_protocol", "hyperkvasir_official_5fold")
    manifest = _ROOT / "data" / "splits" / split / f"fold_{fold}.csv"
    rows = read_manifest_csv(manifest)
    test_rows = [r for r in rows if str(r.get("split", "")) == "test"]
    label_to_name = {int(r["label"]): str(r["class_name"]) for r in rows}
    mean = dataset_cfg.get("normalize_mean", list(IMAGENET_MEAN))
    std = dataset_cfg.get("normalize_std", list(IMAGENET_STD))
    image_size = int(dataset_cfg.get("image_size", 224))
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = HyperKvasirImageDataset(test_rows, transform=tfm, project_root=_ROOT)
    return ds, test_rows, label_to_name, fold


@torch.no_grad()
def predict_all(model, ds, device: str, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Single forward pass over ``ds`` -> ``(preds, labels)`` aligned to ds order."""
    preds, labels = [], []
    batch_imgs, batch_lbls = [], []

    def _flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(np.asarray(batch_lbls))
        batch_imgs.clear()
        batch_lbls.clear()

    for i in range(len(ds)):
        img, label, _ = ds[i]
        batch_imgs.append(img)
        batch_lbls.append(int(label))
        if len(batch_imgs) == batch_size:
            _flush()
    _flush()
    return np.concatenate(preds), np.concatenate(labels)


def pick_examples(
    preds: np.ndarray,
    labels: np.ndarray,
    classes: list[int],
    n_correct: int = 2,
    n_failure: int = 1,
) -> dict[int, dict[str, list[int]]]:
    """For each class, return dataset indices of correct + misclassified samples.

    Deterministic (lowest indices first) so figures are reproducible.
    """
    out: dict[int, dict[str, list[int]]] = {}
    for c in classes:
        idx_class = np.where(labels == c)[0]
        correct = [int(i) for i in idx_class if preds[i] == c][:n_correct]
        failure = [int(i) for i in idx_class if preds[i] != c][:n_failure]
        out[c] = {"correct": correct, "failure": failure}
    return out
