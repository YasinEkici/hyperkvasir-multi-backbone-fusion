"""Sprint 3 Slice 2 — ViT fine-tune path implementation audit.

Covers the fixes made for VLD-08 (unfreeze scope), VLD-09 (bicubic), and
VLD-15 (drop_path, LLRD, MixUp), plus the A100 provenance gate (VLD-11).

Backbones are built with ``pretrained=False`` for speed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from src.data.augmentation import apply_mixup
from src.models.full_model import MultiCNNFusionClassifier
from src.models.vit_backbones import ViTFeatureExtractor
from src.training.optimizers import build_adamw_with_llrd
from src.training.trainer import Trainer

HEAD_LR = 1e-3
BACKBONE_LR = 5e-5
DECAY = 0.7


# ── VLD-08: unfreeze scope ────────────────────────────────────────────────

class TestUnfreezeScope:
    @pytest.mark.parametrize("alias", ["vit_b", "beit_b"])
    def test_isotropic_unfreezes_last_3_blocks_only(self, alias: str) -> None:
        ext = ViTFeatureExtractor(name=alias, pretrained=False, unfreeze_blocks=3)
        blocks = ext.model.blocks
        # blocks[:9] frozen, blocks[9:] trainable
        for i, blk in enumerate(blocks):
            trainable = any(p.requires_grad for p in blk.parameters())
            if i >= 9:
                assert trainable, f"{alias}: block {i} should be trainable"
            else:
                assert not trainable, f"{alias}: block {i} should be frozen"

    def test_swin_unfreezes_final_stage_only(self) -> None:
        # VLD-08: Swin = layers[3] only, even with unfreeze_blocks=3.
        ext = ViTFeatureExtractor(name="swin_t", pretrained=False, unfreeze_blocks=3)
        layers = ext.model.layers
        for i, stage in enumerate(layers):
            trainable = any(p.requires_grad for p in stage.parameters())
            if i == len(layers) - 1:
                assert trainable, "swin_t: final stage should be trainable"
            else:
                assert not trainable, f"swin_t: stage {i} should be frozen"

    @pytest.mark.parametrize("alias", ["vit_b", "swin_t", "beit_b"])
    def test_final_norm_is_trainable(self, alias: str) -> None:
        ext = ViTFeatureExtractor(name=alias, pretrained=False, unfreeze_blocks=3)
        norm = getattr(ext.model, "norm", None)
        if norm is not None and any(True for _ in norm.parameters()):
            assert any(p.requires_grad for p in norm.parameters()), (
                f"{alias}: final norm should be trainable when fine-tuning"
            )


# ── VLD-15: drop_path plumbing ────────────────────────────────────────────

class TestDropPath:
    @pytest.mark.parametrize("alias", ["vit_b", "swin_t", "beit_b"])
    def test_drop_path_rate_propagates_to_timm(self, alias: str) -> None:
        ext = ViTFeatureExtractor(
            name=alias, pretrained=False, unfreeze_blocks=3, drop_path_rate=0.1
        )
        drop_probs = [
            float(getattr(m, "drop_prob", 0.0))
            for m in ext.model.modules()
            if type(m).__name__ == "DropPath"
        ]
        assert drop_probs, f"{alias}: no DropPath modules found"
        assert max(drop_probs) > 0.0, f"{alias}: drop_path_rate not applied"

    def test_default_drop_path_is_zero(self) -> None:
        ext = ViTFeatureExtractor(name="vit_b", pretrained=False, unfreeze_blocks=0)
        drop_probs = [
            float(getattr(m, "drop_prob", 0.0))
            for m in ext.model.modules()
            if type(m).__name__ == "DropPath"
        ]
        assert all(p == 0.0 for p in drop_probs), "default drop_path must be 0.0"

    def test_forward_finite_with_drop_path(self) -> None:
        ext = ViTFeatureExtractor(
            name="vit_b", pretrained=False, unfreeze_blocks=3, drop_path_rate=0.1
        )
        ext.eval()
        with torch.no_grad():
            out = ext(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 768)
        assert torch.isfinite(out).all()


# ── VLD-15: ViT LLRD parameter groups ─────────────────────────────────────

def _vit_model(backbone_names: list[str], fusion: str) -> MultiCNNFusionClassifier:
    return MultiCNNFusionClassifier(
        backbone_names=backbone_names,
        unfreeze_blocks=3,
        projection_dim=512,
        fusion_type=fusion,
        num_classes=23,
        drop_path_rate=0.05,
    )


def _requires_grad_ids(model: nn.Module) -> set[int]:
    return {id(p) for p in model.parameters() if p.requires_grad}


class TestViTLLRD:
    @pytest.mark.parametrize("names,fusion", [
        (["swin_t"], "none"),
        (["vit_b", "swin_t"], "concat"),
        (["swin_t", "beit_b"], "weighted"),
        (["vit_b", "swin_t", "beit_b"], "weighted"),
    ])
    def test_all_requires_grad_params_covered_exactly_once(self, names, fusion):
        model = _vit_model(names, fusion)
        opt = build_adamw_with_llrd(
            model, head_lr=HEAD_LR, backbone_lr=BACKBONE_LR,
            weight_decay=0.05, llrd_decay=DECAY,
        )
        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        assert len(ids) == len(set(ids)), "param double-counted across groups"
        assert set(ids) == _requires_grad_ids(model)

    def test_no_empty_groups(self):
        model = _vit_model(["vit_b", "swin_t"], "concat")
        opt = build_adamw_with_llrd(
            model, head_lr=HEAD_LR, backbone_lr=BACKBONE_LR,
            weight_decay=0.05, llrd_decay=DECAY,
        )
        for i, g in enumerate(opt.param_groups):
            assert len(g["params"]) > 0, f"group {i} empty"

    def test_not_flat_lr_real_llrd(self):
        """Backbone groups must span more than one LR (would be flat if buggy)."""
        model = _vit_model(["vit_b"], "none")
        opt = build_adamw_with_llrd(
            model, head_lr=HEAD_LR, backbone_lr=BACKBONE_LR,
            weight_decay=0.05, llrd_decay=DECAY,
        )
        backbone_lrs = {g["lr"] for g in opt.param_groups if g["lr"] != HEAD_LR}
        assert len(backbone_lrs) >= 3, (
            f"expected per-layer LLRD (norm + 3 blocks), got LRs {backbone_lrs}"
        )
        assert max(backbone_lrs) == pytest.approx(BACKBONE_LR)
        assert min(backbone_lrs) == pytest.approx(BACKBONE_LR * DECAY ** 3)

    def test_head_params_at_head_lr(self):
        model = _vit_model(["vit_b"], "none")
        opt = build_adamw_with_llrd(
            model, head_lr=HEAD_LR, backbone_lr=BACKBONE_LR,
            weight_decay=0.05, llrd_decay=DECAY,
        )
        head_ids = (
            {id(p) for p in model.projections.parameters()}
            | {id(p) for p in model.classifier.parameters()}
        )
        for g in opt.param_groups:
            if head_ids & {id(p) for p in g["params"]}:
                assert g["lr"] == pytest.approx(HEAD_LR)


# ── VLD-15: MixUp ─────────────────────────────────────────────────────────

class TestMixUp:
    def test_apply_mixup_shapes_and_lam(self):
        x = torch.randn(8, 3, 16, 16)
        y = torch.randint(0, 4, (8,))
        mixed, la, lb, lam = apply_mixup(x, y, alpha=0.2)
        assert mixed.shape == x.shape
        assert la.shape == y.shape and lb.shape == y.shape
        assert 0.0 <= lam <= 1.0
        assert torch.equal(la, y)

    def test_apply_mixup_alpha_zero_is_identity(self):
        x = torch.randn(4, 3, 8, 8)
        y = torch.randint(0, 4, (4,))
        mixed, _, _, lam = apply_mixup(x, y, alpha=0.0)
        assert lam == 1.0
        assert torch.allclose(mixed, x)

    def test_trainer_mixup_loss_finite(self):
        class _FlattenLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 16 * 16, 4)

            def forward(self, t):
                return self.fc(t.flatten(1))

        model = _FlattenLinear()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                model=model,
                optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
                criterion=nn.CrossEntropyLoss(),
                scheduler=None,
                run_dir=Path(tmp),
                device="cpu",
                num_classes=4,
                mixup_alpha=0.2,
                mixup_prob=1.0,
            )
            x = torch.randn(4, 3, 16, 16)
            y = torch.randint(0, 4, (4,))
            for _ in range(5):
                loss = trainer._compute_loss(x, y)
                assert torch.isfinite(loss)

    def test_trainer_default_no_mixup(self):
        model = nn.Linear(8, 4)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                model=model,
                optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
                criterion=nn.CrossEntropyLoss(),
                scheduler=None,
                run_dir=Path(tmp),
                device="cpu",
                num_classes=4,
            )
            assert trainer.mixup_prob == pytest.approx(0.0)
            assert trainer.mixup_alpha == pytest.approx(0.0)


# ── VLD-11: A100 provenance gate ──────────────────────────────────────────

class TestA100Gate:
    def test_passes_on_a100(self):
        from scripts.train import require_a100_for_finetune
        name = require_a100_for_finetune(
            "cuda", allow_non_a100=False, device_name="NVIDIA A100-SXM4-40GB"
        )
        assert "A100" in name

    def test_fails_on_non_a100(self):
        from scripts.train import require_a100_for_finetune
        with pytest.raises(SystemExit, match="A100 PROVENANCE GATE FAILED"):
            require_a100_for_finetune(
                "cuda", allow_non_a100=False, device_name="Tesla T4"
            )

    def test_fails_on_cpu(self):
        from scripts.train import require_a100_for_finetune
        with pytest.raises(SystemExit, match="A100 PROVENANCE GATE FAILED"):
            require_a100_for_finetune("cpu", allow_non_a100=False, device_name=None)

    def test_bypass_allows_non_a100(self):
        from scripts.train import require_a100_for_finetune
        out = require_a100_for_finetune(
            "cpu", allow_non_a100=True, device_name=None
        )
        assert out == "cpu"


# ── VLD-09: bicubic image loaders ─────────────────────────────────────────

def test_image_loader_accepts_bicubic_interpolation():
    """_make_image_loaders must accept BICUBIC without error (VLD-09)."""
    import inspect

    from scripts.train import _make_image_loaders

    sig = inspect.signature(_make_image_loaders)
    assert "interpolation" in sig.parameters
    default = sig.parameters["interpolation"].default
    # Default preserves CNN behaviour (bilinear); ViT path passes bicubic.
    assert default == transforms.InterpolationMode.BILINEAR
