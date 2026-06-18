"""Extract frozen backbone features."""

import argparse
from pathlib import Path
import yaml
import torch

from src.data.feature_cache import cache_frozen_features
from src.models.vit_backbones import is_vit_backbone


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen features.")
    parser.add_argument("--config", required=True, type=str, help="Path to dataset config YAML")
    parser.add_argument("--backbones", nargs="+", default=["resnet50", "mobilenetv2", "efficientnetb0"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # In feature_cache, it expects "mean" and "std" instead of normalize_mean and normalize_std
    if "normalize_mean" in config:
        config["mean"] = config["normalize_mean"]
    if "normalize_std" in config:
        config["std"] = config["normalize_std"]

    split_dir = Path(config.get("split_manifest_dir", "data/splits"))
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    elif all(is_vit_backbone(name) for name in args.backbones):
        out_dir = Path("results/vit/feature_cache")
    else:
        out_dir = Path("results/feature_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    for split in splits:
        manifest_path = split_dir / f"{split}.csv"
        if not manifest_path.exists():
            print(f"Warning: Manifest not found: {manifest_path}")
            continue
        
        print(f"Extracting features for {split} split...")
        cache_frozen_features(
            backbones=args.backbones,
            dataset_config=config,
            split_manifest=manifest_path,
            output_dir=out_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(f"Finished extracting features for {split} split.")


if __name__ == "__main__":
    main()
