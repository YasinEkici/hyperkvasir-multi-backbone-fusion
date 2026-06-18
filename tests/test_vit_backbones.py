"""Tests for ViT backbone factory, dispatch, and fusion integration.

Sprint 1 Slice 1 acceptance tests:
- (2, 768) shape per backbone alias
- block/stage count verification
- freeze logic
- CNN full_model regression
- ViT full_model forward (single + multi-branch)
"""

import pytest
import torch
import torch.nn as nn

from src.models.vit_backbones import (
    ViTFeatureExtractor,
    is_vit_backbone,
    resolve_timm_name,
    _ALIAS_TO_TIMM,
    _EXPECTED_COUNTS,
    _BLOCK_CONTAINER,
)


# ── Alias resolution ─────────────────────────────────────────────────────

class TestAliasResolution:
    """Test that short aliases and full timm strings resolve correctly."""

    @pytest.mark.parametrize("alias", ["vit_b", "swin_t", "beit_b"])
    def test_alias_resolves(self, alias: str) -> None:
        resolved_alias, timm_str = resolve_timm_name(alias)
        assert resolved_alias == alias
        assert timm_str == _ALIAS_TO_TIMM[alias]

    @pytest.mark.parametrize("timm_str", list(_ALIAS_TO_TIMM.values()))
    def test_full_timm_string_resolves(self, timm_str: str) -> None:
        _, resolved = resolve_timm_name(timm_str)
        assert resolved == timm_str

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported ViT backbone"):
            resolve_timm_name("nonexistent_model")

    @pytest.mark.parametrize("name", ["vit_b", "swin_t", "beit_b"])
    def test_is_vit_backbone_aliases(self, name: str) -> None:
        assert is_vit_backbone(name) is True

    @pytest.mark.parametrize("name", ["resnet50", "mobilenetv2", "efficientnetb0"])
    def test_is_vit_backbone_cnn_names(self, name: str) -> None:
        assert is_vit_backbone(name) is False


# ── Shape tests (pretrained=False for speed) ─────────────────────────────

class TestViTFeatureExtractorShapes:
    """Core acceptance: ViTFeatureExtractor(alias)(x).shape == (2, 768)."""

    @pytest.mark.parametrize("alias", ["vit_b", "swin_t", "beit_b"])
    def test_output_shape_is_2x768(self, alias: str) -> None:
        extractor = ViTFeatureExtractor(name=alias, pretrained=False, unfreeze_blocks=0)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = extractor(x)
        assert out.shape == (2, 768), f"{alias}: expected (2, 768), got {out.shape}"

    @pytest.mark.parametrize("alias", ["vit_b", "swin_t", "beit_b"])
    def test_feature_dim_property(self, alias: str) -> None:
        extractor = ViTFeatureExtractor(name=alias, pretrained=False, unfreeze_blocks=0)
        assert extractor.feature_dim == 768


# ── Block-count verification ─────────────────────────────────────────────

class TestBlockCounts:
    """Verify timm block containers match documented structure (§2.1)."""

    @pytest.mark.parametrize("alias", ["vit_b", "beit_b"])
    def test_isotropic_block_count_is_12(self, alias: str) -> None:
        extractor = ViTFeatureExtractor(name=alias, pretrained=False, unfreeze_blocks=0)
        container = getattr(extractor.model, "blocks")
        assert len(container) == 12, f"{alias}: expected 12 blocks, got {len(container)}"

    def test_swin_stage_depths(self) -> None:
        extractor = ViTFeatureExtractor(name="swin_t", pretrained=False, unfreeze_blocks=0)
        layers = getattr(extractor.model, "layers")
        depths = [len(stage.blocks) for stage in layers]
        assert depths == [2, 2, 6, 2], f"swin_t: expected [2,2,6,2], got {depths}"

    @pytest.mark.parametrize("alias", ["vit_b", "beit_b"])
    def test_last_3_blocks_slice_correct(self, alias: str) -> None:
        """blocks[9:] should yield exactly 3 blocks (VLD-08 unfreeze target)."""
        extractor = ViTFeatureExtractor(name=alias, pretrained=False, unfreeze_blocks=0)
        container = getattr(extractor.model, "blocks")
        last_3 = container[9:]
        assert len(last_3) == 3, f"{alias}: blocks[9:] has {len(last_3)} blocks, expected 3"


# ── Freeze logic ─────────────────────────────────────────────────────────

class TestFreezeLogic:
    """All parameters should be frozen when unfreeze_blocks=0."""

    @pytest.mark.parametrize("alias", ["vit_b", "swin_t", "beit_b"])
    def test_all_params_frozen(self, alias: str) -> None:
        extractor = ViTFeatureExtractor(name=alias, pretrained=False, unfreeze_blocks=0)
        for name, param in extractor.named_parameters():
            assert not param.requires_grad, f"{alias}: param {name} is not frozen"


# ── full_model.py dispatch tests ─────────────────────────────────────────

class TestFullModelDispatch:
    """Verify that full_model.py correctly dispatches CNN vs ViT."""

    def test_cnn_full_model_still_works(self) -> None:
        """Regression: CNN backbone construction and forward must not break."""
        from src.models.full_model import MultiCNNFusionClassifier

        model = MultiCNNFusionClassifier(
            backbone_names=["resnet50"],
            unfreeze_blocks=0,
            projection_dim=512,
            fusion_type="none",
            num_classes=23,
        )
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 23)
        assert torch.isfinite(logits).all()

    def test_single_vit_full_model_forward(self) -> None:
        """Single ViT-B through projection → MLP yields finite logits."""
        from src.models.full_model import MultiCNNFusionClassifier

        model = MultiCNNFusionClassifier(
            backbone_names=["vit_b"],
            unfreeze_blocks=0,
            projection_dim=512,
            fusion_type="none",
            num_classes=23,
        )
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 23)
        assert torch.isfinite(logits).all()

    def test_vit_pair_concat_fusion(self) -> None:
        """Two ViT backbones + concat fusion → logits shape correct."""
        from src.models.full_model import MultiCNNFusionClassifier

        model = MultiCNNFusionClassifier(
            backbone_names=["vit_b", "swin_t"],
            unfreeze_blocks=0,
            projection_dim=512,
            fusion_type="concat",
            num_classes=23,
        )
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 23)
        assert torch.isfinite(logits).all()

    def test_vit_pair_weighted_fusion(self) -> None:
        """Two ViT backbones + weighted fusion → logits shape correct."""
        from src.models.full_model import MultiCNNFusionClassifier

        model = MultiCNNFusionClassifier(
            backbone_names=["vit_b", "beit_b"],
            unfreeze_blocks=0,
            projection_dim=512,
            fusion_type="weighted",
            num_classes=23,
        )
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 23)
        assert torch.isfinite(logits).all()
