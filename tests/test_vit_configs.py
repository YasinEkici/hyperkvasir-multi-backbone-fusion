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

    # 11 Sprint 2 frozen + 5 Sprint 3 fine-tune + 4 Sprint 4 CV + 4 Sprint 4.5 GMU.
    assert len(experiments) == 24
    assert experiments[0]["id"] == "01_single_vit_b_frozen_official"

    expected_fusions = {"none", "concat", "weighted", "gmu"}
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
            assert method["fusion_type"] in {"concat", "weighted", "gmu"}


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


def test_sprint4_cv_rows() -> None:
    """Exactly the four Sprint 4 CV candidates (top-4), tuned config, fold 0."""
    matrix = _read_yaml(ROOT / "configs" / "vit" / "experiment_matrix.yaml")
    experiments = {e["id"]: e for e in matrix["experiments"]}
    expected = {
        "02_single_swin_t_cv": "single_swin_t",
        "09_pair_swin_t_beit_b_weighted_cv": "pair_swin_t_beit_b_weighted",
        "11_triple_weighted_cv": "triple_vit_swin_beit_weighted",
        "05_pair_vit_b_beit_b_concat_cv": "pair_vit_b_beit_b_concat",
    }
    cv_ids = {e["id"] for e in matrix["experiments"]
              if e["id"].endswith("_cv") and "gmu" not in e["id"]}
    assert cv_ids == set(expected), "exactly the four top-4 CV rows"
    # the dropped Sprint 3 config 04 must not appear as a CV row
    assert not any("04_pair_vit_b_swin_t_concat" in i for i in cv_ids)
    for exp_id, method_stem in expected.items():
        row = experiments[exp_id]
        assert row["training"] == "configs/vit/training/vit_finetune.yaml"
        assert row["method"] == f"configs/vit/method/{method_stem}.yaml"
        assert row["fold"] == 0
        method = _read_yaml(ROOT / row["method"])
        assert method["classifier"] == "mlp"
        assert method["fusion_type"] in {"none", "concat", "weighted"}


def test_sprint45_gmu_rows() -> None:
    """The four Sprint 4.5 GMU ablation rows (multi-backbone, faithful gate)."""
    matrix = _read_yaml(ROOT / "configs" / "vit" / "experiment_matrix.yaml")
    experiments = {e["id"]: e for e in matrix["experiments"]}
    expected = {
        "pair_vit_b_swin_t_gmu_cv": ("pair_vit_b_swin_t_gmu", ["vit_b", "swin_t"]),
        "pair_vit_b_beit_b_gmu_cv": ("pair_vit_b_beit_b_gmu", ["vit_b", "beit_b"]),
        "pair_swin_t_beit_b_gmu_cv": ("pair_swin_t_beit_b_gmu", ["swin_t", "beit_b"]),
        "triple_vit_swin_beit_gmu_cv": ("triple_vit_swin_beit_gmu",
                                        ["vit_b", "swin_t", "beit_b"]),
    }
    gmu_ids = {e["id"] for e in matrix["experiments"] if "gmu" in e["id"]}
    assert gmu_ids == set(expected), "exactly the four GMU ablation rows"
    for exp_id, (method_stem, backbones) in expected.items():
        row = experiments[exp_id]
        assert row["training"] == "configs/vit/training/vit_finetune.yaml"
        assert row["method"] == f"configs/vit/method/{method_stem}.yaml"
        assert row["fold"] == 0
        method = _read_yaml(ROOT / row["method"])
        assert method["fusion_type"] == "gmu"
        # faithful element-wise gate (VLD-19) — not the legacy scalar default
        assert method["fusion_kwargs"]["gate_mode"] == "elementwise"
        assert method["classifier"] == "mlp"
        assert method["projection_dim"] == 512
        assert method["backbone_names"] == backbones
        assert len(backbones) >= 2  # GMU is multi-backbone only (no singles)


def test_vit_finetune_perf_config_sprint35() -> None:
    """Sprint 3.5 throughput knobs are present and well-formed (VLD-17)."""
    cfg = _read_yaml(ROOT / "configs" / "vit" / "training" / "vit_finetune.yaml")
    dl = cfg["dataloader"]
    assert int(dl["num_workers"]) > 0
    assert dl["pin_memory"] is True
    assert dl["persistent_workers"] is True
    perf = cfg["performance"]
    assert perf["amp_dtype"] in {"float16", "bfloat16"}
    assert isinstance(perf["tf32"], bool)
    repro = cfg["reproducibility"]
    # autotuner enabled for speed; determinism relaxed (seeds still set).
    assert repro["cudnn_benchmark"] is True
    assert repro["deterministic"] is False
    # batch size unchanged from Sprint 3 for result comparability.
    assert int(cfg["batch_size"]) == 32


def test_vit_frozen_training_config_is_frozen_mlp_only() -> None:
    cfg = _read_yaml(ROOT / "configs" / "vit" / "training" / "vit_frozen.yaml")

    assert cfg["unfreeze_blocks"] == 0
    assert cfg["optimizer"]["type"] == "adamw"
    assert cfg["loss"]["type"] == "ce_label_smooth"
    assert cfg["augmentation"]["cutmix"]["prob"] == 0.0
    assert cfg["augmentation"]["mixup"]["prob"] == 0.0
