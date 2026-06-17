from pathlib import Path

import yaml

from src.models.vit_backbones import is_vit_backbone


ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def test_vit_experiment_matrix_configs_parse() -> None:
    matrix_path = ROOT / "configs" / "vit" / "experiment_matrix.yaml"
    matrix = _read_yaml(matrix_path)
    experiments = matrix["experiments"]

    assert len(experiments) == 11
    assert experiments[0]["id"] == "01_single_vit_b_frozen_official"

    expected_fusions = {"none", "concat", "weighted"}
    for exp in experiments:
        dataset_path = ROOT / exp["dataset"]
        method_path = ROOT / exp["method"]
        training_path = ROOT / exp["training"]

        assert dataset_path.exists()
        assert method_path.exists()
        assert training_path.exists()

        method = _read_yaml(method_path)
        assert method["classifier"] == "mlp"
        assert method["projection_dim"] == 512
        assert method["fusion_type"] in expected_fusions
        assert all(is_vit_backbone(name) for name in method["backbone_names"])

        if len(method["backbone_names"]) == 1:
            assert method["fusion_type"] == "none"
        else:
            assert method["fusion_type"] in {"concat", "weighted"}


def test_vit_frozen_training_config_is_frozen_mlp_only() -> None:
    cfg = _read_yaml(ROOT / "configs" / "vit" / "training" / "vit_frozen.yaml")

    assert cfg["unfreeze_blocks"] == 0
    assert cfg["optimizer"]["type"] == "adamw"
    assert cfg["loss"]["type"] == "ce_label_smooth"
    assert cfg["augmentation"]["cutmix"]["prob"] == 0.0
    assert cfg["augmentation"]["mixup"]["prob"] == 0.0
