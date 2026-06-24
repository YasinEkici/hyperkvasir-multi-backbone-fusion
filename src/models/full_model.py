"""End-to-end multi-backbone fusion classifier.

Supports both CNN backbones (torchvision, via ``BackboneFeatureExtractor``)
and ViT backbones (timm, via ``ViTFeatureExtractor``).  Dispatch is by
backbone name: recognised ViT aliases / timm strings route to the ViT
extractor; everything else routes to the CNN extractor.  The projection →
fusion → MLP pipeline is backbone-agnostic.
"""

from torch import Tensor, nn

from src.models.backbones import BackboneFeatureExtractor
from src.models.projections import BranchProjection
from src.models.classifiers import MLPClassifier
from src.models.vit_backbones import ViTFeatureExtractor, is_vit_backbone


def _build_extractor(
    name: str, pretrained: bool, unfreeze_blocks: int, drop_path_rate: float = 0.0
) -> BackboneFeatureExtractor | ViTFeatureExtractor:
    """Dispatch to CNN or ViT feature extractor by backbone name.

    ``drop_path_rate`` (VLD-15) only applies to the timm ViT extractor; the
    frozen CNN ``BackboneFeatureExtractor`` signature is left untouched.
    """
    if is_vit_backbone(name):
        return ViTFeatureExtractor(
            name=name,
            pretrained=pretrained,
            unfreeze_blocks=unfreeze_blocks,
            drop_path_rate=drop_path_rate,
        )
    return BackboneFeatureExtractor(
        name=name, pretrained=pretrained, unfreeze_blocks=unfreeze_blocks
    )


class MultiCNNFusionClassifier(nn.Module):
    """End-to-end model: backbones -> projections -> fusion -> MLP head.

    Works with both CNN (torchvision) and ViT (timm) backbones.  The class
    name is kept as ``MultiCNNFusionClassifier`` for backward compatibility
    with existing CNN configs and checkpoints.
    """

    def __init__(
        self,
        backbone_names: list[str],
        unfreeze_blocks: int,
        projection_dim: int,
        fusion_type: str,
        num_classes: int,
        mlp_hidden: list[int] = [256],
        dropout: float = 0.3,
        drop_path_rate: float = 0.0,
        fusion_kwargs: dict | None = None,
    ):
        super().__init__()
        self.fusion_kwargs = fusion_kwargs or {}
        self.backbone_names = backbone_names
        self.unfreeze_blocks = unfreeze_blocks
        self.projection_dim = projection_dim
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        self.mlp_hidden = mlp_hidden
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate

        # Initialize backbones — dispatch CNN vs ViT by name
        self.backbones = nn.ModuleDict()
        self.projections = nn.ModuleDict()

        for name in backbone_names:
            bb = _build_extractor(
                name=name,
                pretrained=True,
                unfreeze_blocks=unfreeze_blocks,
                drop_path_rate=drop_path_rate,
            )
            self.backbones[name] = bb
            self.projections[name] = BranchProjection(in_dim=bb.feature_dim, out_dim=projection_dim)

        num_branches = len(backbone_names)
        
        # Initialize fusion module
        if num_branches == 1 or fusion_type == "none":
            self.fusion = nn.Identity()
            fusion_out_dim = projection_dim
        elif fusion_type == "concat":
            from src.models.fusion.concat import FusionModule as ConcatFusion
            self.fusion = ConcatFusion(num_branches=num_branches, feature_dim=projection_dim)
            fusion_out_dim = self.fusion.output_dim
        elif fusion_type == "weighted":
            from src.models.fusion.weighted import FusionModule as WeightedFusion
            self.fusion = WeightedFusion(num_branches=num_branches, feature_dim=projection_dim)
            fusion_out_dim = self.fusion.output_dim
        elif fusion_type == "gmu":
            from src.models.fusion.gmu import FusionModule as GMUFusion
            self.fusion = GMUFusion(num_branches=num_branches, feature_dim=projection_dim, **self.fusion_kwargs)
            fusion_out_dim = self.fusion.output_dim
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

        # Initialize MLP head
        self.classifier = MLPClassifier(
            input_dim=fusion_out_dim,
            num_classes=num_classes,
            hidden_dims=mlp_hidden,
            dropout=dropout
        )

    def extract_features(self, x: Tensor) -> dict[str, Tensor]:
        """Extract raw features from all backbones."""
        features = {}
        for name, backbone in self.backbones.items():
            features[name] = backbone(x)
        return features

    def forward(self, x: Tensor) -> Tensor:
        # Extract features and project
        projected = []
        for name in self.backbone_names:
            raw_feat = self.backbones[name](x)
            proj_feat = self.projections[name](raw_feat)
            projected.append(proj_feat)
        
        # Fuse
        if len(self.backbone_names) == 1 or self.fusion_type == "none":
            fused = projected[0]
        else:
            fused = self.fusion(projected)
            
        # Classify
        return self.classifier(fused)
