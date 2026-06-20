"""timm-based ViT feature extractors for the multi-ViT fusion project.

Backbone factory for ViT-B/16, Swin-T, and BEiT-B/16.  All three are loaded
via ``timm.create_model(name, pretrained=True, num_classes=0)`` and forward
with ``model(x)`` to obtain the *native pooled* pre-logits vector (B, 768).

Feature-extraction rule (VLD-04):
    m = timm.create_model(name, pretrained=True, num_classes=0)
    feat = m(x)   # native pooled pre-logits, (B, 768)

Do **not** hard-code ``forward_features(x)[:,0]`` — that is wrong for
Swin (global avg pool, no CLS) and BEiT (timm default may differ).
``num_classes=0`` + ``m(x)`` is architecture-correct everywhere.

See:
    - docs/vit/project_structure.md §2.1 (verified timm strings + 768-d)
    - docs/vit/decisions.md VLD-03, VLD-04, VLD-08, VLD-09
"""

from __future__ import annotations

import timm
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Alias → verified timm string mapping (§2.1 VERIFIED 2026-06-16)
# ---------------------------------------------------------------------------
_ALIAS_TO_TIMM: dict[str, str] = {
    "vit_b": "vit_base_patch16_224.orig_in21k_ft_in1k",
    "swin_t": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "beit_b": "beit_base_patch16_224.in22k_ft_in22k_in1k",
}

# Reverse lookup: allow full timm string as input too
_KNOWN_TIMM_STRINGS: set[str] = set(_ALIAS_TO_TIMM.values())

# Which backbone alias uses which block container (for unfreeze logic)
_BLOCK_CONTAINER: dict[str, str] = {
    "vit_b": "blocks",   # 12 blocks
    "swin_t": "layers",  # 4 stages, depths [2,2,6,2]
    "beit_b": "blocks",  # 12 blocks
}

# Expected block/stage counts for verification
_EXPECTED_COUNTS: dict[str, int | list[int]] = {
    "vit_b": 12,
    "swin_t": [2, 2, 6, 2],  # per-stage depths
    "beit_b": 12,
}


def resolve_timm_name(name: str) -> tuple[str, str]:
    """Resolve a backbone name to (alias, timm_string).

    Accepts either a short alias (``vit_b``, ``swin_t``, ``beit_b``) or a
    full timm model string.  Returns the canonical alias and the verified
    timm string.

    Raises ``ValueError`` for unrecognised names.
    """
    key = name.lower().strip()

    if key in _ALIAS_TO_TIMM:
        return key, _ALIAS_TO_TIMM[key]

    if key in _KNOWN_TIMM_STRINGS:
        # Reverse-map to alias
        for alias, timm_str in _ALIAS_TO_TIMM.items():
            if timm_str == key:
                return alias, timm_str

    raise ValueError(
        f"Unsupported ViT backbone: {name!r}. "
        f"Supported aliases: {list(_ALIAS_TO_TIMM.keys())}; "
        f"or full timm strings: {list(_KNOWN_TIMM_STRINGS)}"
    )


def is_vit_backbone(name: str) -> bool:
    """Return True if *name* is a recognised ViT alias or timm string."""
    key = name.lower().strip()
    return key in _ALIAS_TO_TIMM or key in _KNOWN_TIMM_STRINGS


class ViTFeatureExtractor(nn.Module):
    """Wraps a timm ViT/Swin/BEiT model and exposes native pooled features.

    Contract (docs/vit/project_structure.md §6):
        - ``.forward(x) -> (B, feature_dim)`` native pooled pre-logits
          via ``num_classes=0``.
        - ``.feature_dim -> int`` (768 for all three verified backbones).
        - ``.trainable_param_groups(head_lr, backbone_lr, llrd_decay)``
          for fine-tune (Sprint 3+).

    Parameters
    ----------
    name : str
        Short alias (``vit_b``, ``swin_t``, ``beit_b``) or full timm string.
    pretrained : bool
        Load ImageNet-pretrained weights (default True).
    unfreeze_blocks : int
        Number of trailing blocks to unfreeze for fine-tuning.  Default 0
        (fully frozen).  For ViT/BEiT this is the count of trailing
        ``model.blocks`` to unfreeze (VLD-08 uses 3 → ``blocks[9:]``); for
        Swin it is ignored beyond ``> 0`` because VLD-08 locks Swin to its
        final stage (``layers[3]``) only.
    drop_path_rate : float
        Stochastic-depth rate passed to timm at model creation (VLD-15 uses
        ~0.05 for fine-tuning).  Default 0.0 leaves frozen / feature-cache
        behaviour unchanged (and a fully frozen branch runs in ``.eval()``,
        disabling drop_path regardless).
    """

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        unfreeze_blocks: int = 0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.alias, self.timm_name = resolve_timm_name(name)
        self.pretrained = pretrained
        self.unfreeze_blocks = unfreeze_blocks
        self.drop_path_rate = drop_path_rate

        # Build model — num_classes=0 removes the classification head and
        # returns the native pooled pre-logits vector (VLD-04).  drop_path_rate
        # is plumbed through to timm here (VLD-15); default 0.0 is a no-op.
        self.model: nn.Module = timm.create_model(
            self.timm_name,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        # ── Verify block counts (exec-plan Step 2) ────────────────────
        self._verify_block_counts()

        # ── Freeze / unfreeze logic ───────────────────────────────────
        self._freeze_parameters()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def _verify_block_counts(self) -> None:
        """Assert that the timm model has the expected block structure.

        Raises ``RuntimeError`` if counts are unexpected (stop-and-ask
        trigger per AGENTS.md).
        """
        expected = _EXPECTED_COUNTS[self.alias]
        container_attr = _BLOCK_CONTAINER[self.alias]

        container = getattr(self.model, container_attr, None)
        if container is None:
            raise RuntimeError(
                f"timm model {self.timm_name!r} has no attribute "
                f"'{container_attr}' — expected for {self.alias}. "
                f"Stop-and-ask: timm attribute path mismatch."
            )

        if isinstance(expected, int):
            # ViT / BEiT: flat list of blocks
            actual = len(container)
            if actual != expected:
                raise RuntimeError(
                    f"{self.alias}: expected {expected} blocks in "
                    f"model.{container_attr}, got {actual}. "
                    f"Stop-and-ask: timm block count mismatch."
                )
        else:
            # Swin: stages with per-stage depths
            actual_depths = []
            for stage in container:
                # Each Swin stage has a `.blocks` sub-container
                stage_blocks = getattr(stage, "blocks", None)
                if stage_blocks is None:
                    raise RuntimeError(
                        f"Swin stage has no 'blocks' attribute. "
                        f"Stop-and-ask: Swin stage structure unexpected."
                    )
                actual_depths.append(len(stage_blocks))
            if actual_depths != expected:
                raise RuntimeError(
                    f"swin_t: expected stage depths {expected}, "
                    f"got {actual_depths}. "
                    f"Stop-and-ask: Swin depth mismatch."
                )

    # ------------------------------------------------------------------
    # Freeze logic (VLD-08)
    # ------------------------------------------------------------------
    def _freeze_parameters(self) -> None:
        """Freeze all parameters, then optionally unfreeze trailing blocks.

        In Sprint 1 (frozen regime), ``unfreeze_blocks=0`` so everything
        stays frozen.  Fine-tune unfreezing is stored for Sprint 3+.

        ViTs use LayerNorm (no BatchNorm).  A frozen branch is ``.eval()``
        to disable dropout/stochastic-depth (drop_path).
        """
        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False

        if self.unfreeze_blocks <= 0:
            # Fully frozen — put in eval to disable drop_path
            self.model.eval()
            return

        # ── Unfreeze trailing blocks (Sprint 3+, not exercised in S1) ─
        container_attr = _BLOCK_CONTAINER[self.alias]
        container = getattr(self.model, container_attr)

        if self.alias in ("vit_b", "beit_b"):
            # Unfreeze last N blocks from model.blocks (VLD-08: 3 → blocks[9:]).
            start_idx = max(0, len(container) - self.unfreeze_blocks)
            for block in container[start_idx:]:
                for param in block.parameters():
                    param.requires_grad = True
        elif self.alias == "swin_t":
            # VLD-08: Swin fine-tunes the FINAL STAGE ONLY (layers[3]),
            # regardless of the ViT-calibrated unfreeze_blocks count.  Using a
            # stage *count* here would over-unfreeze (e.g. unfreeze_blocks=3
            # would open the last 3 of 4 stages).
            for param in container[-1].parameters():
                param.requires_grad = True

        # Always unfreeze final norm + head (if they exist)
        for attr_name in ("norm", "head", "fc_norm"):
            module = getattr(self.model, attr_name, None)
            if module is not None and isinstance(module, nn.Module):
                for param in module.parameters():
                    param.requires_grad = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Extract native pooled pre-logits.

        Input:  (B, 3, 224, 224)
        Output: (B, feature_dim)  — (B, 768) for all verified backbones.

        Uses ``self.model(x)`` with ``num_classes=0`` (VLD-04).
        """
        return self.model(x)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Return the native pooled feature dimension (768 for all three)."""
        return self.model.num_features

    # ------------------------------------------------------------------
    # Param groups for fine-tuning (Sprint 3+)
    # ------------------------------------------------------------------
    def trainable_param_groups(
        self,
        head_lr: float,
        backbone_lr: float,
        llrd_decay: float = 0.75,
    ) -> list[dict]:
        """Per-layer LLRD parameter groups for this backbone's unfrozen params.

        Returns one group per transformer layer that contains trainable
        parameters, with a layer-wise decayed learning rate (VLD-15):

            lr(d) = backbone_lr * llrd_decay ** d

        where ``d=0`` is nearest the output (the final norm) and ``d``
        increases toward the stem.  Only ``requires_grad`` parameters are
        included, so under VLD-08 this yields groups for ViT/BEiT
        ``blocks[9:]`` + final norm, or Swin's final stage + final norm.

        ``head_lr`` is accepted for interface symmetry
        (``project_structure.md §6``) but not applied here: projection /
        fusion / classifier ("head") parameters live outside the backbone and
        are grouped by the caller at ``head_lr``.

        Reference: Howard & Ruder (2018), ULMFiT §3.3 — discriminative
        fine-tuning (LLRD).
        """
        groups: list[dict] = []
        seen: set[int] = set()

        def _add(candidate_params, depth: int) -> None:
            ps = [
                p for p in candidate_params
                if p.requires_grad and id(p) not in seen
            ]
            if ps:
                seen.update(id(p) for p in ps)
                groups.append(
                    {"params": ps, "lr": backbone_lr * (llrd_decay ** depth)}
                )

        # depth 0 — final norm(s) nearest the output (head is Identity here
        # because num_classes=0, but include it defensively).
        head_like: list = []
        for attr in ("norm", "fc_norm", "head"):
            module = getattr(self.model, attr, None)
            if isinstance(module, nn.Module):
                head_like.extend(module.parameters())
        _add(head_like, depth=0)

        container = getattr(self.model, _BLOCK_CONTAINER[self.alias])
        if self.alias in ("vit_b", "beit_b"):
            n = len(container)
            # deepest block (last) -> depth 1, increasing toward the stem.
            for i in range(n - 1, -1, -1):
                _add(container[i].parameters(), depth=n - i)
        else:  # swin_t — final stage only (VLD-08)
            _add(container[-1].parameters(), depth=1)

        return groups
