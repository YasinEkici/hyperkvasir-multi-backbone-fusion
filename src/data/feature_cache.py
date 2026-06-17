"""Frozen feature cache utilities."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.datasets import HyperKvasirImageDataset
from src.models.backbones import BackboneFeatureExtractor
from src.models.vit_backbones import ViTFeatureExtractor, is_vit_backbone
from src.utils.reproducibility import seed_all, seed_worker


def _all_same_backbone_family(backbones: list[str]) -> bool:
    """Return True when all requested backbones are CNNs or all are ViTs."""
    if not backbones:
        return True
    first_is_vit = is_vit_backbone(backbones[0])
    return all(is_vit_backbone(name) == first_is_vit for name in backbones)


def _build_transform(dataset_config: dict, *, is_vit: bool) -> transforms.Compose:
    """Build the cache transform without changing the legacy CNN path."""
    mean = dataset_config.get("mean", [0.485, 0.456, 0.406])
    std = dataset_config.get("std", [0.229, 0.224, 0.225])
    size = dataset_config.get("image_size", (224, 224))
    if isinstance(size, int):
        size = (size, size)

    if not is_vit:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    if size[0] != size[1]:
        raise ValueError(f"ViT cache expects square image_size, got {size}")
    image_size = int(size[0])
    crop_pct = float(dataset_config.get("crop_pct", 0.9))
    resize_size = int(image_size / crop_pct)

    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _build_extractor(name: str) -> BackboneFeatureExtractor | ViTFeatureExtractor:
    """Build the correct frozen extractor for a CNN or ViT backbone."""
    if is_vit_backbone(name):
        return ViTFeatureExtractor(name=name, pretrained=True, unfreeze_blocks=0)
    return BackboneFeatureExtractor(name=name, pretrained=True, unfreeze_blocks=0)


def cache_frozen_features(
    backbones: list[str],
    dataset_config: dict,
    split_manifest: Path,
    output_dir: Path,
    batch_size: int = 64,
    device: str = "cuda",
) -> dict[str, Path]:
    """Cache frozen backbone features for a split manifest."""
    if not _all_same_backbone_family(backbones):
        raise ValueError(
            "cache_frozen_features expects all backbones in one call to use the "
            "same preprocessing family. Call CNN and ViT caches separately."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    is_vit_cache = any(is_vit_backbone(name) for name in backbones)
    
    # Determinism
    seed_all(42)
    g = torch.Generator()
    g.manual_seed(42)

    transform = _build_transform(dataset_config, is_vit=is_vit_cache)

    dataset = HyperKvasirImageDataset(manifest=split_manifest, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # CRITICAL: MUST BE FALSE
        num_workers=0, # 0 for MVP to ensure ordering and avoid IPC issues
        worker_init_fn=seed_worker,
        generator=g,
    )

    models = {}
    for name in backbones:
        model = _build_extractor(name)
        model = model.to(device)
        model.eval()
        models[name] = model

    all_features = {name: [] for name in backbones}
    all_labels = []
    all_paths = []
    all_indices = []

    with torch.no_grad():
        for images, labels, rows in dataloader:
            images = images.to(device)
            for name, model in models.items():
                feats = model(images)
                all_features[name].append(feats.cpu())
            
            all_labels.extend(labels.tolist())
            all_paths.extend(rows["path"])
            # The collate_fn might return strings for indices depending on CSV format
            all_indices.extend([int(idx) for idx in rows.get("original_index", [-1] * len(labels))])

    # Config hash for cache invalidation
    config_str = json.dumps(dataset_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()

    saved_paths = {}
    for name in backbones:
        cat_features = torch.cat(all_features[name], dim=0)
        
        # Validation
        if cat_features.shape[0] != len(dataset):
            raise ValueError(f"Alignment error: extracted {cat_features.shape[0]} features, but dataset has {len(dataset)} samples.")
        if len(all_labels) != len(dataset):
            raise ValueError("Alignment error: label count mismatch")
            
        cache_data = {
            "features": cat_features,
            "labels": torch.tensor(all_labels),
            "paths": all_paths,
            "indices": all_indices,
            "config_hash": config_hash,
            "backbone": name
        }
        
        out_path = output_dir / f"{split_manifest.stem}_{name}_features.pt"
        torch.save(cache_data, out_path)
        saved_paths[name] = out_path

    return saved_paths
