import json
import torch
import pytest
from pathlib import Path
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.feature_cache import _build_transform, cache_frozen_features


@pytest.fixture
def dummy_dataset(tmp_path: Path) -> Path:
    """Create a dummy dataset with 2 images and a manifest."""
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    
    # Create dummy images
    img1_path = data_dir / "img1.jpg"
    img2_path = data_dir / "img2.jpg"
    Image.new("RGB", (64, 64), color="red").save(img1_path)
    Image.new("RGB", (64, 64), color="blue").save(img2_path)
    
    # Create manifest
    manifest_path = tmp_path / "test_manifest.csv"
    with open(manifest_path, "w") as f:
        f.write("path,label,original_index\n")
        f.write(f"{img1_path},0,10\n")
        f.write(f"{img2_path},1,20\n")
        
    return manifest_path


def test_cache_frozen_features_e2e(tmp_path: Path, dummy_dataset: Path) -> None:
    # We use a fast backbone like mobilenetv2
    out_dir = tmp_path / "cache"
    
    config = {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "image_size": 224,
    }
    
    saved_paths = cache_frozen_features(
        backbones=["mobilenetv2"],
        dataset_config=config,
        split_manifest=dummy_dataset,
        output_dir=out_dir,
        batch_size=2,
        device="cpu", # Use CPU for tests
    )
    
    assert "mobilenetv2" in saved_paths
    cache_file = saved_paths["mobilenetv2"]
    assert cache_file.exists()
    
    # Verify cached data
    data = torch.load(cache_file)
    
    assert "features" in data
    assert data["features"].shape == (2, 1280)
    
    assert "labels" in data
    assert torch.equal(data["labels"], torch.tensor([0, 1]))
    
    assert "indices" in data
    assert data["indices"] == [10, 20]
    
    assert "config_hash" in data
    assert isinstance(data["config_hash"], str)


def test_cache_frozen_features_vit_uses_vit_extractor(
    tmp_path: Path,
    dummy_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyViTFeatureExtractor(nn.Module):
        feature_dim = 768

        def __init__(self, name: str, pretrained: bool, unfreeze_blocks: int) -> None:
            super().__init__()
            self.name = name
            self.pretrained = pretrained
            self.unfreeze_blocks = unfreeze_blocks

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones(x.shape[0], self.feature_dim)

    monkeypatch.setattr(
        "src.data.feature_cache.ViTFeatureExtractor",
        DummyViTFeatureExtractor,
    )

    out_dir = tmp_path / "results" / "vit" / "feature_cache"
    config = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "image_size": 224,
        "crop_pct": 0.9,
    }

    saved_paths = cache_frozen_features(
        backbones=["vit_b"],
        dataset_config=config,
        split_manifest=dummy_dataset,
        output_dir=out_dir,
        batch_size=2,
        device="cpu",
    )

    cache_file = saved_paths["vit_b"]
    assert cache_file.parent == out_dir
    data = torch.load(cache_file)
    assert data["features"].shape == (2, 768)
    assert data["backbone"] == "vit_b"
    assert len(data["paths"]) == 2


def test_cache_frozen_features_rejects_mixed_preprocessing_families(
    tmp_path: Path,
    dummy_dataset: Path,
) -> None:
    with pytest.raises(ValueError, match="same preprocessing family"):
        cache_frozen_features(
            backbones=["mobilenetv2", "vit_b"],
            dataset_config={"image_size": 224},
            split_manifest=dummy_dataset,
            output_dir=tmp_path / "cache",
            batch_size=2,
            device="cpu",
        )


def test_vit_transform_uses_bicubic_center_crop() -> None:
    transform = _build_transform(
        {
            "image_size": 224,
            "crop_pct": 0.9,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        is_vit=True,
    )

    resize, crop, *_ = transform.transforms
    assert isinstance(resize, transforms.Resize)
    assert resize.size == 248
    assert resize.interpolation == InterpolationMode.BICUBIC
    assert isinstance(crop, transforms.CenterCrop)
    assert crop.size == (224, 224)
