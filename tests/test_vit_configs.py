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

    # 11 Sprint 2 frozen rows + 5 Sprint 3 fine-tune rows.
    assert len(experiments) == 16
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


def test_sprint3_finetune_rows() -> None:
    """Exactly the five locked Sprint 3 candidates, each on the FT recipe."""
    matrix = _read_yaml(ROOT / "configs" / "vit" / "experiment_matrix.yaml")
    experiments = {e["id"]: e for e in matrix["experiments"]}

    expected = {
        "02_single_swin_t_finetune_official": "single_swin_t",
        "04_pair_vit_b_swin_t_concat_finetune_official": "pair_vit_b_swin_t_concat",
        "05_pair_vit_b_beit_b_concat_finetune_official": "pair_vit_b_beit_b_concat",
        "09_pair_swin_t_beit_b_weighted_finetune_official": "pair_swin_t_beit_b_weighted",
        "11_triple_weighted_finetune_official": "triple_vit_swin_beit_weighted",
    }

    finetune_ids = {e["id"] for e in matrix["experiments"] if "finetune" in e["id"]}
    assert finetune_ids == set(expected), "Exactly the 5 locked FT candidates only"

    for exp_id, method_stem in expected.items():
        row = experiments[exp_id]
        assert row["training"] == "configs/vit/training/vit_finetune.yaml"
        assert row["method"] == f"configs/vit/method/{method_stem}.yaml"
        assert row["fold"] == 0
        # No GMU or non-MLP arm sneaks in (VLD-06/VLD-07).
        method = _read_yaml(ROOT / row["method"])
        assert method["classifier"] == "mlp"
        assert method["fusion_type"] in {"none", "concat", "weighted"}


def test_vit_frozen_training_config_is_frozen_mlp_only() -> None:
    cfg = _read_yaml(ROOT / "configs" / "vit" / "training" / "vit_frozen.yaml")

    assert cfg["unfreeze_blocks"] == 0
    assert cfg["optimizer"]["type"] == "adamw"
    assert cfg["loss"]["type"] == "ce_label_smooth"
    assert cfg["augmentation"]["cutmix"]["prob"] == 0.0
    assert cfg["augmentation"]["mixup"]["prob"] == 0.0
