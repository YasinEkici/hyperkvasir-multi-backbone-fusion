"""GMU gate_mode fidelity tests (Sprint 4.5 Slice 0, VLD-19)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.fusion.gmu import FusionModule  # noqa: E402


def _feats(n: int, b: int = 4, d: int = 8) -> list[torch.Tensor]:
    return [torch.randn(b, d) for _ in range(n)]


def test_scalar_is_default_and_backcompat() -> None:
    m = FusionModule(num_branches=3, feature_dim=8)
    assert m.gate_mode == "scalar"
    assert m.gate.out_features == 3          # one logit per branch (legacy/CNN)


def test_elementwise_gate_layer_is_per_feature() -> None:
    m = FusionModule(num_branches=3, feature_dim=8, gate_mode="elementwise")
    assert m.gate.out_features == 3 * 8      # N*D logits (per-feature gate)


def test_invalid_gate_mode_raises() -> None:
    with pytest.raises(ValueError):
        FusionModule(num_branches=2, feature_dim=8, gate_mode="bogus")


@pytest.mark.parametrize("mode", ["scalar", "elementwise"])
def test_output_shape(mode: str) -> None:
    out = FusionModule(num_branches=3, feature_dim=8, gate_mode=mode)(_feats(3))
    assert out.shape == (4, 8)


def test_scalar_gate_weights_sum_to_one_over_branches() -> None:
    z = FusionModule(num_branches=3, feature_dim=8, gate_mode="scalar").gate_weights(_feats(3))
    assert z.shape == (4, 3)
    assert torch.allclose(z.sum(dim=1), torch.ones(4), atol=1e-5)


def test_elementwise_gate_weights_sum_to_one_per_feature() -> None:
    z = FusionModule(num_branches=3, feature_dim=8, gate_mode="elementwise").gate_weights(_feats(3))
    assert z.shape == (4, 3, 8)              # (B, N, D)
    assert torch.allclose(z.sum(dim=1), torch.ones(4, 8), atol=1e-5)


def test_bimodal_elementwise_reduces_to_tied_gate() -> None:
    # N=2: per-feature softmax over 2 branches => z0 + z1 = 1 per (B, D),
    # i.e. the paper's tied z ⊙ h_v + (1−z) ⊙ h_t.
    z = FusionModule(num_branches=2, feature_dim=8, gate_mode="elementwise").gate_weights(_feats(2))
    assert torch.allclose(z[:, 0, :] + z[:, 1, :], torch.ones(4, 8), atol=1e-5)


def test_fusion_kwargs_wired_through_frozenheadmodel() -> None:
    import scripts.train as T
    m = T.FrozenHeadModel(
        backbone_names=["vit_b", "swin_t", "beit_b"], projection_dim=512,
        fusion_type="gmu", num_classes=23, mlp_hidden=[256], dropout=0.3,
        fusion_kwargs={"gate_mode": "elementwise"},
    )
    assert m.fusion.gate_mode == "elementwise"
    # default (no fusion_kwargs) keeps the legacy scalar gate
    m2 = T.FrozenHeadModel(["vit_b", "swin_t"], 512, "gmu", 23, [256], 0.3)
    assert m2.fusion.gate_mode == "scalar"
