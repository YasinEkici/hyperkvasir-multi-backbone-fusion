"""Gated Multimodal Unit (GMU) fusion.

Implements the Gated Multimodal Unit from:
    Arevalo et al. (2017), "Gated Multimodal Units for Information Fusion",
    ICLR Workshop. Section 3.1.

Bimodal formulation (verbatim from paper §3.1):

    h_v = tanh(W_v · x_v)
    h_t = tanh(W_t · x_t)
    z   = σ(W_z · [x_v, x_t])
    h   = z * h_v + (1 − z) * h_t
    Θ   = {W_v, W_t, W_z}

    with [·,·] the concatenation operator. The gate z is **element-wise** — the
    same dimensionality as h: the paper "[uses] multiplicative gates that assign
    importance to various features simultaneously" (§3.1), i.e. a per-feature
    gate, not a single scalar per modality.

Gate modes (``gate_mode``):

  * ``"elementwise"`` (faithful to the paper): a per-feature weight per branch.
        z = softmax_over_branches( reshape(W_z · concat(x), (N, D)) )   → (B, N, D)
        h = Σ_i  z_i ⊙ h_i                                              → (B, D)
    For N=2 this reduces exactly to the paper's tied bimodal
    ``z ⊙ h_v + (1 − z) ⊙ h_t`` (a per-dimension softmax over two branches is
    σ / (1 − σ)). The multimodal (N > 2) case generalises the tie to a normalised
    per-feature competition across branches.

  * ``"scalar"`` (default, legacy): one scalar weight per branch,
        z = softmax(W_z · concat(x))   → (B, N);   h = Σ_i z_i · h_i.
    This is a simplification that **loses** the paper's element-wise gating
    (it behaves like an input-gated weighted sum). Kept as the default for
    backward compatibility with the CNN project's earlier GMU run; ViT GMU uses
    ``gate_mode="elementwise"`` (docs/vit/decisions.md VLD-19).

Do **not** hard-code ``forward_features(x)[:,0]`` anywhere upstream — fusion
operates on the 512-d branch projections (VLD-05).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_GATE_MODES = ("scalar", "elementwise")


class FusionModule(nn.Module):
    """Gated Multimodal Unit fusion for N input branches.

    Args:
        num_branches: Number of input feature branches (N ≥ 1).
        feature_dim:  Dimensionality of each branch's projected vector.
        gate_mode:    "elementwise" (paper-faithful per-feature gate) or
                      "scalar" (legacy per-branch scalar gate; default).
        **kwargs:     Ignored; present for interface compatibility.

    Input:  list of N tensors, each (B, feature_dim).
    Output: fused tensor of shape (B, feature_dim).
    """

    def __init__(
        self, num_branches: int, feature_dim: int, gate_mode: str = "scalar", **kwargs
    ) -> None:
        super().__init__()
        if gate_mode not in _GATE_MODES:
            raise ValueError(f"gate_mode must be one of {_GATE_MODES}, got {gate_mode!r}")
        self.num_branches = num_branches
        self.feature_dim = feature_dim
        self.gate_mode = gate_mode

        # W_i: per-branch linear transform — h_i = tanh(W_i · x_i)
        self.branch_transforms = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim) for _ in range(num_branches)]
        )
        # W_z: gate. scalar -> N logits; elementwise -> N*D logits (per feature).
        gate_out = num_branches if gate_mode == "scalar" else num_branches * feature_dim
        self.gate = nn.Linear(num_branches * feature_dim, gate_out)

    def gate_weights(self, features: list[Tensor]) -> Tensor:
        """Normalised gate weights from the raw (pre-transform) branch features.

        scalar      -> (B, N)     summing to 1 over the branch axis.
        elementwise -> (B, N, D)  summing to 1 over the branch axis per feature.
        """
        gate_input = torch.cat(features, dim=-1)                 # (B, N*D)
        logits = self.gate(gate_input)
        if self.gate_mode == "scalar":
            return F.softmax(logits, dim=-1)                     # (B, N)
        z = logits.view(-1, self.num_branches, self.feature_dim)  # (B, N, D)
        return F.softmax(z, dim=1)                                # per-feature competition

    def forward(self, features: list[Tensor]) -> Tensor:
        if len(features) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} feature tensors, got {len(features)}."
            )
        # h_i = tanh(W_i · x_i)  ->  (B, N, D)
        h_stack = torch.stack(
            [torch.tanh(self.branch_transforms[i](features[i]))
             for i in range(self.num_branches)],
            dim=1,
        )
        z = self.gate_weights(features)
        if self.gate_mode == "scalar":
            return (z.unsqueeze(-1) * h_stack).sum(dim=1)        # (B, D)
        return (z * h_stack).sum(dim=1)                          # (B, D)

    @property
    def output_dim(self) -> int:
        return self.feature_dim
