"""Optimizer builders."""

import torch
import torch.nn as nn

from src.models.vit_backbones import ViTFeatureExtractor

# ---------------------------------------------------------------------------
# Per-backbone LLRD block specification
# Verified against project_structure.md §2.1 and §2.2.
#
# Each entry is a list of (depth, [sequential child indices]) tuples.
# depth=0 is the deepest block (adjacent to the head); LR = backbone_lr * decay^depth.
#
# BackboneFeatureExtractor stores:
#   ResNet50:      self.backbone = nn.Sequential(*list(model.children())[:-1])
#                  children: conv1(0) bn1(1) relu(2) maxpool(3) layer1(4)
#                            layer2(5) layer3(6) layer4(7) avgpool(8)
#   MobileNetV2:   self.backbone = model.features   (19 items, indices 0-18)
#   EfficientNetB0:self.backbone = model.features   (9 items, indices 0-8)
# ---------------------------------------------------------------------------
_BACKBONE_BLOCKS: dict[str, list[tuple[int, list[int]]]] = {
    "resnet50": [
        (0, [7]),           # layer4 (3 bottlenecks) — deepest
        (1, [6]),           # layer3 (6 bottlenecks)
        (2, [5]),           # layer2
        (3, [4]),           # layer1
        (4, [0, 1, 2, 3]), # stem: conv1, bn1, relu, maxpool
    ],
    "mobilenetv2": [
        (0, list(range(15, 19))),  # features[15:] — last IR blocks + 1×1 conv
        (1, list(range(13, 15))),  # features[13:15]
        (2, list(range(1, 13))),   # features[1:13]
        (3, [0]),                  # features[0] stem
    ],
    "efficientnetb0": [
        (0, list(range(6, 9))),    # features[6:9]
        (1, list(range(4, 6))),    # features[4:6]
        (2, list(range(1, 4))),    # features[1:4]
        (3, [0]),                  # features[0] stem
    ],
}


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build AdamW optimizer from config dict.

    Config keys: head_lr, weight_decay.
    """
    lr = float(config.get("head_lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )


def build_adamw_with_llrd(
    model: nn.Module,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    llrd_decay: float = 0.75,
) -> torch.optim.Optimizer:
    """AdamW with layer-wise learning rate decay (LLRD) for fine-tuning.

    Walks ``model.backbones`` (a nn.ModuleDict of BackboneFeatureExtractor
    instances) and assigns each backbone block its own parameter group with
    an exponentially decayed learning rate:

        lr(d) = backbone_lr * (llrd_decay ** d)

    where d=0 is the deepest block (adjacent to the head) and d increases
    toward the stem. Non-backbone parameters (projections, fusion, classifier)
    are collected in a single head group at ``head_lr``.

    Empty groups (no requires_grad parameters in a block) are omitted so the
    returned optimizer always has at least one non-empty group.

    Reference: Howard & Ruder (2018), "Universal Language Model Fine-Tuning"
    (ULMFiT), §3.3 — Discriminative fine-tuning (LLRD).

    Args:
        model:       Full model with ``model.backbones`` as nn.ModuleDict.
        head_lr:     Learning rate for projection / fusion / classifier params.
        backbone_lr: Learning rate for the deepest backbone block (depth=0).
        weight_decay:AdamW weight decay applied to every group.
        llrd_decay:  Multiplicative decay per depth level (default 0.75).

    Returns:
        AdamW optimizer with per-block parameter groups.
    """
    backbone_param_ids: set[int] = set()
    backbone_groups: list[dict] = []

    backbones_dict: dict = getattr(model, "backbones", {})

    for bb_name, bb_extractor in backbones_dict.items():
        name_lower = bb_name.lower()

        # ViT/BEiT/Swin: per-layer LLRD comes from the extractor itself, which
        # knows its transformer-layer structure (VLD-08/VLD-15).  The CNN
        # _BACKBONE_BLOCKS path below does not apply to timm ViT models.
        if isinstance(bb_extractor, ViTFeatureExtractor):
            for group in bb_extractor.trainable_param_groups(
                head_lr=head_lr, backbone_lr=backbone_lr, llrd_decay=llrd_decay
            ):
                params = [
                    p for p in group["params"]
                    if id(p) not in backbone_param_ids
                ]
                if params:
                    backbone_param_ids.update(id(p) for p in params)
                    backbone_groups.append({
                        "params": params,
                        "lr": group["lr"],
                        "weight_decay": weight_decay,
                    })
            continue

        block_spec = _BACKBONE_BLOCKS.get(name_lower)

        if block_spec is None:
            # Unknown backbone — one flat group at backbone_lr
            params = [
                p for p in bb_extractor.parameters()
                if p.requires_grad and id(p) not in backbone_param_ids
            ]
            if params:
                backbone_param_ids.update(id(p) for p in params)
                backbone_groups.append({
                    "params": params,
                    "lr": backbone_lr,
                    "weight_decay": weight_decay,
                })
            continue

        backbone_seq = bb_extractor.backbone  # nn.Sequential

        for depth, indices in block_spec:
            block_params: list[torch.Tensor] = []
            for idx in indices:
                for p in backbone_seq[idx].parameters():
                    if p.requires_grad and id(p) not in backbone_param_ids:
                        block_params.append(p)
                        backbone_param_ids.add(id(p))
            if block_params:
                backbone_groups.append({
                    "params": block_params,
                    "lr": backbone_lr * (llrd_decay ** depth),
                    "weight_decay": weight_decay,
                })

    # Head: all remaining requires_grad params not claimed by any backbone block
    head_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_param_ids
    ]

    param_groups: list[dict] = []
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
        })
    param_groups.extend(backbone_groups)

    if not param_groups:
        # Model has no trainable parameters — return a valid but empty optimizer
        param_groups = [{"params": [], "lr": head_lr, "weight_decay": weight_decay}]

    return torch.optim.AdamW(param_groups)
