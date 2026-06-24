"""Train one configured experiment from the experiment matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision import transforms

from src.data.datasets import HyperKvasirImageDataset
from src.data.feature_cache import cache_frozen_features
from src.data.manifests import read_manifest_csv
from src.models.classifiers import MLPClassifier
from src.models.full_model import MultiCNNFusionClassifier
from src.models.projections import BranchProjection
from src.models.vit_backbones import is_vit_backbone
from src.training.ema import EMA
from src.training.losses import build_loss
from src.training.optimizers import build_adamw_with_llrd, build_optimizer
from src.training.schedulers import build_scheduler
from src.training.trainer import Trainer
from src.utils.checkpointing import load_checkpoint
from src.utils.logging import get_logger, save_metrics
from src.utils.paths import project_root
from src.utils.reproducibility import seed_all

logger = get_logger(__name__)

BACKBONE_DIMS: dict[str, int] = {
    "resnet50": 2048,
    "mobilenetv2": 1280,
    "efficientnetb0": 1280,
}

VIT_FEATURE_DIM = 768


def _backbone_feature_dim(name: str) -> int:
    """Return cached feature dimension for a CNN or ViT backbone."""
    if name in BACKBONE_DIMS:
        return BACKBONE_DIMS[name]
    if is_vit_backbone(name):
        return VIT_FEATURE_DIM
    raise KeyError(f"Unknown backbone feature dimension for {name!r}")


def _all_vit_backbones(backbone_names: list[str]) -> bool:
    return bool(backbone_names) and all(is_vit_backbone(name) for name in backbone_names)


def require_a100_for_finetune(
    device: str,
    allow_non_a100: bool,
    *,
    device_name: str | None = None,
) -> str | None:
    """A100 provenance gate for fine-tune runs (VLD-11; reuses CNN D-09 intent).

    Fine-tuning is the only stage allowed to spend A100 units, and frozen
    caches/heads must never run on the A100 runtime.  This gate fails early
    unless CUDA is active on an A100 device.  ``allow_non_a100`` is an explicit,
    loudly-logged escape hatch for off-A100 dev/smoke runs.

    ``device_name`` may be injected for testing; otherwise it is read from the
    live CUDA device when ``device == "cuda"``.

    Returns the resolved device name (or the bypass device) for logging.
    Raises ``SystemExit`` when the gate fails and the bypass is not set.
    """
    if device_name is None and device == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    if device_name is not None and "A100" in device_name:
        return device_name

    if allow_non_a100:
        logger.warning(
            "A100 provenance gate BYPASSED via --allow-non-a100 "
            "(device=%s, cuda_device_name=%s). VLD-11: A100 units are for "
            "fine-tuning only; off-A100 runs must be flagged.",
            device, device_name,
        )
        return device_name or device

    raise SystemExit(
        "A100 PROVENANCE GATE FAILED: Sprint 3 fine-tune requires an A100 "
        f"(VLD-11). device={device!r}, cuda_device_name={device_name!r}. "
        "Re-run on an A100 runtime, or pass --allow-non-a100 for an explicit "
        "off-A100 run."
    )


def _run_dir(exp_id: str, *, vit: bool) -> Path:
    root = project_root()
    path = root / "results" / "vit" / "runs" / exp_id if vit else root / "results" / "runs" / exp_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _feature_cache_dir(*, vit: bool) -> Path:
    root = project_root()
    path = root / "results" / "vit" / "feature_cache" if vit else root / "results" / "feature_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Lightweight frozen head model
# ---------------------------------------------------------------------------

class FrozenHeadModel(nn.Module):
    """Projection + optional fusion + MLP classifier over cached features."""

    def __init__(
        self,
        backbone_names: list[str],
        projection_dim: int,
        fusion_type: str,
        num_classes: int,
        mlp_hidden: list[int],
        dropout: float,
        fusion_kwargs: dict | None = None,
    ):
        super().__init__()
        self.fusion_kwargs = fusion_kwargs or {}
        self.backbone_names = backbone_names
        self.fusion_type = fusion_type
        self.backbone_dims = [_backbone_feature_dim(n) for n in backbone_names]

        self.projections = nn.ModuleDict(
            {
                name: BranchProjection(in_dim=dim, out_dim=projection_dim)
                for name, dim in zip(backbone_names, self.backbone_dims)
            }
        )

        num_branches = len(backbone_names)
        if num_branches == 1 or fusion_type == "none":
            self.fusion = None
            fusion_out_dim = projection_dim
        elif fusion_type == "concat":
            from src.models.fusion.concat import FusionModule
            self.fusion = FusionModule(num_branches=num_branches, feature_dim=projection_dim)
            fusion_out_dim = self.fusion.output_dim
        elif fusion_type == "weighted":
            from src.models.fusion.weighted import FusionModule
            self.fusion = FusionModule(num_branches=num_branches, feature_dim=projection_dim)
            fusion_out_dim = self.fusion.output_dim
        elif fusion_type == "gmu":
            from src.models.fusion.gmu import FusionModule
            self.fusion = FusionModule(num_branches=num_branches, feature_dim=projection_dim, **self.fusion_kwargs)
            fusion_out_dim = self.fusion.output_dim
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.classifier = MLPClassifier(
            input_dim=fusion_out_dim,
            num_classes=num_classes,
            hidden_dims=mlp_hidden,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, sum(backbone_dims)]
        splits = torch.split(x, self.backbone_dims, dim=1)
        projected = [self.projections[name](feat) for name, feat in zip(self.backbone_names, splits)]

        if self.fusion is None:
            fused = projected[0]
        else:
            fused = self.fusion(projected)

        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Feature loading helpers (frozen path)
# ---------------------------------------------------------------------------

def _load_split_features(
    backbone_names: list[str],
    fold: int,
    cache_dir: Path,
    fold_manifest: Path,
    dataset_config: dict,
    batch_size: int,
    device: str,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return train/val/test (features, labels) tensors, building cache if needed."""
    rows = read_manifest_csv(fold_manifest)
    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, row in enumerate(rows):
        split = str(row.get("split", ""))
        if split in split_indices:
            split_indices[split].append(i)

    missing_backbones = [
        name
        for name in backbone_names
        if not (cache_dir / f"{fold_manifest.stem}_{name}_features.pt").exists()
    ]
    if missing_backbones:
        logger.info("Building feature cache for backbones: %s", missing_backbones)
        mean = dataset_config.get("normalize_mean", dataset_config.get("mean", [0.485, 0.456, 0.406]))
        std = dataset_config.get("normalize_std", dataset_config.get("std", [0.229, 0.224, 0.225]))
        cache_config = {
            "mean": mean,
            "std": std,
            "image_size": dataset_config.get("image_size", 224),
        }
        if _all_vit_backbones(missing_backbones):
            cache_config["crop_pct"] = dataset_config.get("crop_pct", 0.9)
        cache_frozen_features(
            backbones=missing_backbones,
            dataset_config=cache_config,
            split_manifest=fold_manifest,
            output_dir=cache_dir,
            batch_size=batch_size,
            device=device,
        )

    per_backbone: list[torch.Tensor] = []
    labels_tensor: torch.Tensor | None = None

    for name in backbone_names:
        cache_path = cache_dir / f"{fold_manifest.stem}_{name}_features.pt"
        cache = torch.load(cache_path, weights_only=True)
        per_backbone.append(cache["features"])
        if labels_tensor is None:
            labels_tensor = cache["labels"]

    assert labels_tensor is not None
    all_features = torch.cat(per_backbone, dim=1)

    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split_name, idxs in split_indices.items():
        if not idxs:
            continue
        idx_tensor = torch.tensor(idxs, dtype=torch.long)
        result[split_name] = (all_features[idx_tensor], labels_tensor[idx_tensor])

    return result


def _make_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    *,
    weighted_sampler: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TensorDataset(features, labels)
    sampler = None
    if weighted_sampler:
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float().clamp(min=1)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Image DataLoader helpers (fine-tune path)
# ---------------------------------------------------------------------------

class _PairDataset(Dataset):
    """Strips the row-dict from HyperKvasirImageDataset, returning (image, label)."""

    def __init__(self, base: HyperKvasirImageDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> tuple[object, int]:
        img, label, _ = self.base[i]
        return img, label


def _make_image_loaders(
    fold_manifest: Path,
    dataset_cfg: dict,
    training_cfg: dict,
    root: Path,
    batch_size: int,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test image DataLoaders for fine-tuning.

    Train loader: RandAugment + optional horizontal flip + WeightedRandomSampler.
    Val / test loaders: deterministic centre-crop, no augmentation.

    Preprocessing follows project_structure.md §2.3:
        Resize shortest edge to 256 → RandomCrop(224) train / CenterCrop(224) val.
        ImageNet normalisation: mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225].

    ``interpolation`` controls the resize kernel: ViT fine-tune runs must pass
    BICUBIC (VLD-09 — timm ViT/Swin/BEiT weights were trained with bicubic);
    the default BILINEAR preserves the existing CNN image-loader behaviour.
    """
    rows = read_manifest_csv(fold_manifest)
    split_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = str(row.get("split", ""))
        if split in split_rows:
            split_rows[split].append(row)

    mean: list[float] = dataset_cfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std: list[float] = dataset_cfg.get("normalize_std", [0.229, 0.224, 0.225])
    image_size: int = int(dataset_cfg.get("image_size", 224))
    resize_to: int = 256  # per project_structure.md §2.3

    aug_cfg = training_cfg.get("augmentation", {})
    ra_cfg = aug_cfg.get("rand_augment", {})
    N: int = int(ra_cfg.get("N", 2))
    M: int = int(ra_cfg.get("M", 9))
    hflip: bool = bool(aug_cfg.get("horizontal_flip", True))

    train_tfm_list: list = [
        transforms.Resize(resize_to, interpolation=interpolation),
        transforms.RandomCrop(image_size),
    ]
    if hflip:
        train_tfm_list.append(transforms.RandomHorizontalFlip())
    train_tfm_list += [
        transforms.RandAugment(num_ops=N, magnitude=M),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    train_transform = transforms.Compose(train_tfm_list)

    val_transform = transforms.Compose([
        transforms.Resize(resize_to, interpolation=interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    use_sampler = dataset_cfg.get("class_imbalance_handling", "none") == "weighted_sampler"

    # DataLoader performance knobs (Sprint 3.5). Defaults (0 workers, no pin)
    # preserve the original behavior when a config omits the `dataloader` section,
    # so frozen/CNN runs are unaffected.
    dl_cfg = training_cfg.get("dataloader", {})
    num_workers = int(dl_cfg.get("num_workers", 0))
    pin_memory = bool(dl_cfg.get("pin_memory", False))
    persistent = bool(dl_cfg.get("persistent_workers", False)) and num_workers > 0
    prefetch = int(dl_cfg.get("prefetch_factor", 2))

    def _loader(rows_subset: list[dict], transform, weighted: bool) -> DataLoader:
        base = HyperKvasirImageDataset(rows_subset, transform=transform, project_root=root)
        ds = _PairDataset(base)
        sampler = None
        if weighted:
            label_list = [int(r["label"]) for r in rows_subset]
            labels_t = torch.tensor(label_list, dtype=torch.long)
            class_counts = torch.bincount(labels_t)
            class_weights = 1.0 / class_counts.float().clamp(min=1)
            sample_weights = class_weights[labels_t]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        loader_kwargs = dict(batch_size=batch_size, sampler=sampler,
                             num_workers=num_workers, pin_memory=pin_memory)
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent
            loader_kwargs["prefetch_factor"] = prefetch
        return DataLoader(ds, **loader_kwargs)

    train_loader = _loader(split_rows["train"], train_transform, weighted=use_sampler)
    val_loader = _loader(split_rows["val"], val_transform, weighted=False)
    test_loader = _loader(split_rows["test"], val_transform, weighted=False)

    logger.info(
        "Image loaders — train: %d | val: %d | test: %d",
        len(split_rows["train"]), len(split_rows["val"]), len(split_rows["test"]),
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train one experiment from the matrix.")
    parser.add_argument("--config", required=True, help="Path to experiment_matrix.yaml")
    parser.add_argument("--experiment", required=True, help="Experiment ID to run")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--fold", type=int, default=None, help="Override fold index from experiment config")
    parser.add_argument(
        "--training",
        default=None,
        help="Override the experiment's training config without mutating the matrix",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override reproducibility seed; seeds other than 42 write to a "
        "seed-tagged run dir ({exp}_seed{S}[_fold_{k}]) to avoid overwriting.",
    )
    parser.add_argument(
        "--allow-non-a100",
        action="store_true",
        help="Bypass the A100 provenance gate for fine-tune runs (VLD-11). "
        "Use only for explicit off-A100 dev/smoke runs; logged loudly.",
    )
    args = parser.parse_args()

    root = project_root()

    with open(args.config, "r") as f:
        matrix = yaml.safe_load(f)

    experiments = {exp["id"]: exp for exp in matrix["experiments"]}
    if args.experiment not in experiments:
        raise SystemExit(f"Experiment '{args.experiment}' not found in {args.config}")
    exp = experiments[args.experiment]

    with open(root / exp["dataset"], "r") as f:
        dataset_cfg = yaml.safe_load(f)
    with open(root / exp["method"], "r") as f:
        method_cfg = yaml.safe_load(f)
    training_path = args.training or exp["training"]
    with open(root / training_path, "r") as f:
        training_cfg = yaml.safe_load(f)

    fold: int = args.fold if args.fold is not None else int(exp.get("fold", 0))
    device: str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    repro_cfg = training_cfg.setdefault("reproducibility", {})
    seed: int = args.seed if args.seed is not None else int(repro_cfg.get("seed", 42))
    repro_cfg["seed"] = seed  # record the effective seed in the saved config
    seed_all(
        seed,
        deterministic=repro_cfg.get("deterministic", True),
        cudnn_benchmark=repro_cfg.get("cudnn_benchmark", False),
    )
    # Performance knobs (Sprint 3.5); all default-off so frozen/CNN runs are
    # unchanged. `amp_dtype` flows to the Trainer's autocast.
    perf_cfg = training_cfg.get("performance", {})
    amp_dtype = str(perf_cfg.get("amp_dtype", "float16"))
    if perf_cfg.get("tf32", False) and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 matmul enabled (Sprint 3.5 perf)")
    logger.info(
        "Device: %s | Experiment: %s | Fold: %d | Seed: %d", device, args.experiment, fold, seed
    )

    split_protocol = dataset_cfg.get("split_protocol", "hyperkvasir_official_5fold")
    fold_manifest = root / "data" / "splits" / split_protocol / f"fold_{fold}.csv"
    if not fold_manifest.exists():
        raise FileNotFoundError(f"Fold manifest not found: {fold_manifest}")

    # Run-dir naming (backward compatible):
    #   seed 42  -> canonical {exp}[_fold_{k}]   (existing Week 3 runs)
    #   seed !=42 -> {exp}_seed{S}[_fold_{k}]    (new ensemble seeds, no overwrite)
    #   fold 0 uses no _fold suffix; fold k>=1 appends _fold_{k}.
    CANONICAL_SEED = 42
    run_name = args.experiment
    if seed != CANONICAL_SEED:
        run_name = f"{run_name}_seed{seed}"
    if args.fold is not None and args.fold > 0:
        run_name = f"{run_name}_fold_{args.fold}"

    backbone_names: list[str] = method_cfg["backbone_names"]
    is_vit_run = _all_vit_backbones(backbone_names)
    run_dir = _run_dir(run_name, vit=is_vit_run)
    projection_dim: int = int(method_cfg.get("projection_dim", 512))
    fusion_type: str = method_cfg.get("fusion_type", "none")
    mlp_hidden: list[int] = method_cfg.get("mlp_hidden", [256])
    dropout: float = float(method_cfg.get("dropout", 0.3))
    fusion_kwargs: dict = method_cfg.get("fusion_kwargs", {}) or {}
    num_classes: int = int(dataset_cfg["num_classes"])

    epochs: int = int(training_cfg.get("epochs", 60))
    batch_size: int = int(training_cfg.get("batch_size", 64))
    mixed_precision: bool = bool(training_cfg.get("mixed_precision", False))
    early_stopping_patience: int = int(training_cfg.get("early_stopping", {}).get("patience", 8))
    unfreeze_blocks: int = int(training_cfg.get("unfreeze_blocks", 0))

    # ------------------------------------------------------------------
    # Branch: frozen feature extraction vs. end-to-end fine-tuning
    # ------------------------------------------------------------------

    if unfreeze_blocks > 0:
        # ---- Fine-tune path ----
        logger.info("Fine-tune mode: unfreeze_blocks=%d", unfreeze_blocks)

        # A100 provenance gate (VLD-11): fine-tuning is the only A100-eligible
        # stage; abort early on non-A100 unless explicitly allowed.
        resolved_device_name = require_a100_for_finetune(device, args.allow_non_a100)
        logger.info(
            "A100 provenance: device=%s cuda_device_name=%s",
            device, resolved_device_name,
        )

        # VLD-09: ViT runs use bicubic resize; CNN runs keep bilinear.
        interpolation = (
            transforms.InterpolationMode.BICUBIC
            if is_vit_run
            else transforms.InterpolationMode.BILINEAR
        )
        train_loader, val_loader, test_loader = _make_image_loaders(
            fold_manifest=fold_manifest,
            dataset_cfg=dataset_cfg,
            training_cfg=training_cfg,
            root=root,
            batch_size=batch_size,
            interpolation=interpolation,
        )

        # drop_path (VLD-15) is plumbed to the timm ViT backbones; default 0.0
        # leaves CNN backbones unaffected (they ignore it).
        drop_path_rate = float(training_cfg.get("drop_path_rate", 0.0))
        model = MultiCNNFusionClassifier(
            backbone_names=backbone_names,
            unfreeze_blocks=unfreeze_blocks,
            projection_dim=projection_dim,
            fusion_type=fusion_type,
            num_classes=num_classes,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            fusion_kwargs=fusion_kwargs,
        )

        opt_cfg = training_cfg.get("optimizer", {})
        if opt_cfg.get("backbone_lr"):
            optimizer = build_adamw_with_llrd(
                model,
                head_lr=float(opt_cfg["head_lr"]),
                backbone_lr=float(opt_cfg["backbone_lr"]),
                weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
                llrd_decay=float(opt_cfg.get("llrd_decay", 0.75)),
            )
        else:
            optimizer = build_optimizer(model, opt_cfg)

        ema_cfg = training_cfg.get("ema", {})
        ema: EMA | None = None
        if ema_cfg.get("enabled", False):
            ema = EMA(
                model,
                decay=float(ema_cfg.get("decay", 0.999)),
                start_epoch=int(ema_cfg.get("start_epoch", 0)),
            )
            logger.info(
                "EMA enabled: decay=%.4f, start_epoch=%d",
                ema.decay, ema.start_epoch,
            )

        aug_cfg = training_cfg.get("augmentation", {})
        cutmix_cfg = aug_cfg.get("cutmix", {})
        cutmix_prob: float = float(cutmix_cfg.get("prob", 0.0))
        cutmix_alpha: float = float(cutmix_cfg.get("alpha", 1.0))
        mixup_cfg = aug_cfg.get("mixup", {})
        mixup_prob: float = float(mixup_cfg.get("prob", 0.0))
        mixup_alpha: float = float(mixup_cfg.get("alpha", 0.0))

    else:
        # ---- Frozen feature extraction path (original, untouched) ----
        logger.info("Frozen mode: using cached features")

        cache_dir = _feature_cache_dir(vit=is_vit_run)
        logger.info("Loading features for fold %d from %s", fold, cache_dir)
        split_data = _load_split_features(
            backbone_names=backbone_names,
            fold=fold,
            cache_dir=cache_dir,
            fold_manifest=fold_manifest,
            dataset_config=dataset_cfg,
            batch_size=batch_size,
            device=device,
        )

        train_features, train_labels = split_data["train"]
        val_features, val_labels = split_data["val"]
        test_features, test_labels = split_data["test"]

        logger.info(
            "Split sizes — train: %d | val: %d | test: %d",
            len(train_labels), len(val_labels), len(test_labels),
        )

        class_imbalance: str = dataset_cfg.get("class_imbalance_handling", "none")
        use_sampler = class_imbalance == "weighted_sampler"
        train_loader = _make_loader(train_features, train_labels, batch_size, weighted_sampler=use_sampler)
        val_loader = _make_loader(val_features, val_labels, batch_size)
        test_loader = _make_loader(test_features, test_labels, batch_size)

        model = FrozenHeadModel(
            backbone_names=backbone_names,
            projection_dim=projection_dim,
            fusion_type=fusion_type,
            num_classes=num_classes,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
            fusion_kwargs=fusion_kwargs,
        )

        optimizer = build_optimizer(model, training_cfg.get("optimizer", {}))
        ema = None
        cutmix_prob = 0.0
        cutmix_alpha = 0.0
        mixup_prob = 0.0
        mixup_alpha = 0.0

    # ------------------------------------------------------------------
    # Shared: scheduler, criterion, trainer, training loop
    # ------------------------------------------------------------------

    steps_per_epoch = max(1, len(train_loader))
    # Intra-epoch progress heartbeat (~20 lines/epoch) for the fine-tune path
    # only; frozen cached-feature epochs stay silent (interval 0).
    progress_log_interval = max(1, steps_per_epoch // 20) if unfreeze_blocks > 0 else 0
    scheduler = build_scheduler(
        optimizer,
        training_cfg.get("scheduler", {}),
        steps_per_epoch=steps_per_epoch,
        total_epochs=epochs,
    )
    criterion = build_loss(training_cfg.get("loss", {}))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        run_dir=run_dir,
        device=device,
        num_classes=num_classes,
        early_stopping_patience=early_stopping_patience,
        mixed_precision=mixed_precision,
        ema=ema,
        cutmix_alpha=cutmix_alpha,
        cutmix_prob=cutmix_prob,
        mixup_alpha=mixup_alpha,
        mixup_prob=mixup_prob,
        progress_log_interval=progress_log_interval,
        amp_dtype=amp_dtype,
    )

    logger.info("Starting training — %d epochs, patience %d", epochs, early_stopping_patience)
    history = trainer.fit(train_loader, val_loader, epochs)

    # Evaluate on test set using best checkpoint
    load_checkpoint(model, run_dir / "best.pt")
    model.to(device)
    test_metrics = trainer.evaluate(test_loader)
    preds, labels_arr = trainer.predict(test_loader)

    logger.info(
        "Test — accuracy: %.4f  macro_f1: %.4f  zero_support_classes: %s",
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
        test_metrics["zero_support_classes"],
    )

    final_metrics = {
        "experiment_id": args.experiment,
        "fold": fold,
        "device": device,
        "best_val_macro_f1": trainer.best_val_f1,
        "test": test_metrics,
        "history": history,
    }
    save_metrics(run_dir, final_metrics)

    np.savez(
        run_dir / "predictions.npz",
        preds=preds,
        labels=labels_arr,
    )

    combined_cfg = {
        "experiment": exp,
        "training_config_path": training_path,
        "dataset": dataset_cfg,
        "method": method_cfg,
        "training": training_cfg,
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(combined_cfg, f, allow_unicode=True)

    logger.info("Results saved to %s", run_dir)


if __name__ == "__main__":
    main()
