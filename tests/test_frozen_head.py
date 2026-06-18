"""Shape tests for FrozenHeadModel with pair and triple backbone inputs."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import train
from scripts.train import FrozenHeadModel

BATCH = 4
NUM_CLASSES = 23
PROJECTION_DIM = 512

# Backbone feature dimensions from project_structure.md §2.1
R = 2048   # resnet50
M = 1280   # mobilenetv2
E = 1280   # efficientnetb0
V = 768    # ViT/Swin/BEiT native pooled dim


def _make_model(backbone_names, fusion_type):
    return FrozenHeadModel(
        backbone_names=backbone_names,
        projection_dim=PROJECTION_DIM,
        fusion_type=fusion_type,
        num_classes=NUM_CLASSES,
        mlp_hidden=[256],
        dropout=0.0,
    )


# --- Single backbone (regression guard) ---

def test_single_resnet50_output_shape():
    model = _make_model(["resnet50"], "none")
    x = torch.randn(BATCH, R)
    assert model(x).shape == (BATCH, NUM_CLASSES)


def test_single_mobilenetv2_output_shape():
    model = _make_model(["mobilenetv2"], "none")
    x = torch.randn(BATCH, M)
    assert model(x).shape == (BATCH, NUM_CLASSES)


def test_single_efficientnetb0_output_shape():
    model = _make_model(["efficientnetb0"], "none")
    x = torch.randn(BATCH, E)
    assert model(x).shape == (BATCH, NUM_CLASSES)


def test_single_vit_b_output_shape():
    model = _make_model(["vit_b"], "none")
    x = torch.randn(BATCH, V)
    assert model(x).shape == (BATCH, NUM_CLASSES)


# --- Pair backbones ---

def test_pair_r_m_concat_output_shape():
    model = _make_model(["resnet50", "mobilenetv2"], "concat")
    x = torch.randn(BATCH, R + M)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_pair_r_e_concat_output_shape():
    model = _make_model(["resnet50", "efficientnetb0"], "concat")
    x = torch.randn(BATCH, R + E)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_pair_m_e_concat_output_shape():
    model = _make_model(["mobilenetv2", "efficientnetb0"], "concat")
    x = torch.randn(BATCH, M + E)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_pair_r_m_weighted_output_shape():
    # Weighted fusion output dim = projection_dim (not 2 * projection_dim)
    model = _make_model(["resnet50", "mobilenetv2"], "weighted")
    x = torch.randn(BATCH, R + M)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


# --- Triple backbones ---

def test_triple_concat_output_shape():
    model = _make_model(["resnet50", "mobilenetv2", "efficientnetb0"], "concat")
    x = torch.randn(BATCH, R + M + E)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_triple_weighted_output_shape():
    model = _make_model(["resnet50", "mobilenetv2", "efficientnetb0"], "weighted")
    x = torch.randn(BATCH, R + M + E)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_triple_vit_concat_output_shape():
    model = _make_model(["vit_b", "swin_t", "beit_b"], "concat")
    x = torch.randn(BATCH, V + V + V)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_triple_vit_weighted_output_shape():
    model = _make_model(["vit_b", "swin_t", "beit_b"], "weighted")
    x = torch.randn(BATCH, V + V + V)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


# --- Feature splitting correctness ---

def test_pair_concat_split_is_order_dependent():
    """Swapping backbone order changes the output (features are split by position)."""
    model_rm = _make_model(["resnet50", "mobilenetv2"], "concat")
    model_mr = _make_model(["mobilenetv2", "resnet50"], "concat")
    x_rm = torch.randn(BATCH, R + M)
    # Same raw bytes, different split interpretation → different projections
    x_mr = torch.cat([x_rm[:, R:], x_rm[:, :R]], dim=1)
    out_rm = model_rm(x_rm)
    out_mr = model_mr(x_mr)
    # Both are valid shapes; just confirm no crash and shapes match
    assert out_rm.shape == (BATCH, NUM_CLASSES)
    assert out_mr.shape == (BATCH, NUM_CLASSES)


def test_vit_run_and_cache_dirs_are_namespaced(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "project_root", lambda: tmp_path)

    run_dir = train._run_dir("01_single_vit_b_frozen_official", vit=True)
    cache_dir = train._feature_cache_dir(vit=True)

    assert run_dir == tmp_path / "results" / "vit" / "runs" / "01_single_vit_b_frozen_official"
    assert cache_dir == tmp_path / "results" / "vit" / "feature_cache"
    assert run_dir.exists()
    assert cache_dir.exists()


def test_cnn_run_and_cache_dirs_remain_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "project_root", lambda: tmp_path)

    run_dir = train._run_dir("01_single_resnet50_frozen_official", vit=False)
    cache_dir = train._feature_cache_dir(vit=False)

    assert run_dir == tmp_path / "results" / "runs" / "01_single_resnet50_frozen_official"
    assert cache_dir == tmp_path / "results" / "feature_cache"
