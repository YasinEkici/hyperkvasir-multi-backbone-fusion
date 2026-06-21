"""Measure fine-tune training throughput for a ViT experiment (Sprint 3.5).

Runs a short, timed fine-tune loop for one experiment under two configurations
and prints images/sec, steps/sec, ms/step and peak GPU memory so we can quantify
A100 under-utilisation BEFORE changing the training pipeline:

  - ``current``: the present settings (num_workers=0, no pin_memory, fp16,
    cudnn.benchmark=False / deterministic, TF32 off).
  - ``fast``:    tuned settings (num_workers, pin_memory, persistent_workers,
    cudnn.benchmark=True, TF32 on, bf16).

This is a measurement tool only — it does NOT touch scripts/train.py, the saved
configs, or any run artifacts. Run it on the A100 after staging the dataset.

Example:
    uv run --no-sync python scripts/benchmark_vit_throughput.py \
        --experiment 02_single_swin_t_finetune_official --modes both --max-steps 100
    uv run --no-sync python scripts/benchmark_vit_throughput.py \
        --experiment 11_triple_weighted_finetune_official --modes both --max-steps 60
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.datasets import HyperKvasirImageDataset
from src.data.manifests import read_manifest_csv
from src.models.full_model import MultiCNNFusionClassifier
from src.training.losses import build_loss
from src.training.optimizers import build_adamw_with_llrd, build_optimizer


class _PairDataset(Dataset):
    def __init__(self, base: HyperKvasirImageDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        img, label, _ = self.base[i]
        return img, label


# Per-mode knobs. "current" mirrors today's pipeline; "fast" is the proposal.
MODES: dict[str, dict] = {
    "current": dict(num_workers=0, pin_memory=False, persistent_workers=False,
                    cudnn_benchmark=False, deterministic=True, tf32=False,
                    amp_dtype="float16"),
    "fast": dict(num_workers=min(8, os.cpu_count() or 8), pin_memory=True,
                 persistent_workers=True, cudnn_benchmark=True, deterministic=False,
                 tf32=True, amp_dtype="bfloat16"),
}


def _build_train_loader(rows, dataset_cfg, training_cfg, batch_size, knobs):
    mean = dataset_cfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std = dataset_cfg.get("normalize_std", [0.229, 0.224, 0.225])
    image_size = int(dataset_cfg.get("image_size", 224))
    ra = training_cfg.get("augmentation", {}).get("rand_augment", {})
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=int(ra.get("N", 2)), magnitude=int(ra.get("M", 9))),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    base = HyperKvasirImageDataset(rows, transform=tfm, project_root=_ROOT)
    ds = _PairDataset(base)
    nw = knobs["num_workers"]
    kwargs = dict(batch_size=batch_size, shuffle=True, num_workers=nw,
                  pin_memory=knobs["pin_memory"], drop_last=True)
    if nw > 0:
        kwargs.update(persistent_workers=knobs["persistent_workers"], prefetch_factor=4)
    return DataLoader(ds, **kwargs)


def _run_mode(mode, exp, dataset_cfg, method_cfg, training_cfg, rows, device,
              batch_size, warmup, max_steps):
    knobs = MODES[mode]
    torch.backends.cudnn.benchmark = knobs["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = knobs["deterministic"]
    # TF32 on Ampere+ (A100): big FP32-matmul speedup, off by default in our code.
    torch.backends.cuda.matmul.allow_tf32 = knobs["tf32"]
    torch.backends.cudnn.allow_tf32 = knobs["tf32"]

    loader = _build_train_loader(rows, dataset_cfg, training_cfg, batch_size, knobs)

    model = MultiCNNFusionClassifier(
        backbone_names=method_cfg["backbone_names"],
        unfreeze_blocks=int(training_cfg.get("unfreeze_blocks", 3)),
        projection_dim=int(method_cfg.get("projection_dim", 512)),
        fusion_type=method_cfg.get("fusion_type", "none"),
        num_classes=int(dataset_cfg["num_classes"]),
        mlp_hidden=method_cfg.get("mlp_hidden", [256]),
        dropout=float(method_cfg.get("dropout", 0.3)),
        drop_path_rate=float(training_cfg.get("drop_path_rate", 0.05)),
    ).to(device)
    model.train()

    opt_cfg = training_cfg.get("optimizer", {})
    if opt_cfg.get("backbone_lr"):
        optimizer = build_adamw_with_llrd(
            model, head_lr=float(opt_cfg["head_lr"]),
            backbone_lr=float(opt_cfg["backbone_lr"]),
            weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
            llrd_decay=float(opt_cfg.get("llrd_decay", 0.7)))
    else:
        optimizer = build_optimizer(model, opt_cfg)
    criterion = build_loss(training_cfg.get("loss", {}))

    amp_dtype = getattr(torch, knobs["amp_dtype"])
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    seen = 0
    t0 = None
    step = 0
    for x, y in loader:
        x = x.to(device, non_blocking=knobs["pin_memory"])
        y = y.to(device, non_blocking=knobs["pin_memory"])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step == warmup and device == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            seen = 0
            continue
        if step > warmup:
            seen += x.size(0)
        if step >= warmup + max_steps:
            break

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0 if t0 else float("nan")
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else float("nan")
    return {
        "mode": mode, "elapsed_s": elapsed, "images": seen,
        "img_per_s": seen / elapsed if elapsed else float("nan"),
        "steps_per_s": (seen / batch_size) / elapsed if elapsed else float("nan"),
        "ms_per_step": 1000.0 * elapsed / (seen / batch_size) if seen else float("nan"),
        "peak_mem_gb": peak_gb, "num_workers": knobs["num_workers"],
        "amp": knobs["amp_dtype"], "tf32": knobs["tf32"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/vit/experiment_matrix.yaml")
    p.add_argument("--experiment", required=True)
    p.add_argument("--training", default=None, help="Override training config path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--modes", choices=["current", "fast", "both"], default="both")
    p.add_argument("--fast-workers", type=int, default=None,
                   help="Override num_workers for the 'fast' mode (data-ceiling sweep)")
    args = p.parse_args()
    if args.fast_workers is not None:
        MODES["fast"]["num_workers"] = args.fast_workers

    matrix = yaml.safe_load((_ROOT / args.config).read_text())
    exp = {e["id"]: e for e in matrix["experiments"]}[args.experiment]
    dataset_cfg = yaml.safe_load((_ROOT / exp["dataset"]).read_text())
    method_cfg = yaml.safe_load((_ROOT / exp["method"]).read_text())
    training_cfg = yaml.safe_load((_ROOT / (args.training or exp["training"])).read_text())
    batch_size = args.batch_size or int(training_cfg.get("batch_size", 32))

    fold = int(exp.get("fold", 0))
    split = dataset_cfg.get("split_protocol", "hyperkvasir_official_5fold")
    manifest = _ROOT / "data" / "splits" / split / f"fold_{fold}.csv"
    rows = [r for r in read_manifest_csv(manifest) if str(r.get("split")) == "train"]

    if args.device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"exp={args.experiment} backbones={method_cfg['backbone_names']} "
          f"bs={batch_size} warmup={args.warmup} steps={args.max_steps} train_rows={len(rows)}\n")

    modes = ["current", "fast"] if args.modes == "both" else [args.modes]
    results = []
    for m in modes:
        print(f"--- running mode: {m} ({MODES[m]['num_workers']} workers, "
              f"{MODES[m]['amp_dtype']}, tf32={MODES[m]['tf32']}, "
              f"cudnn.benchmark={MODES[m]['cudnn_benchmark']}) ---", flush=True)
        r = _run_mode(m, exp, dataset_cfg, method_cfg, training_cfg, rows,
                      args.device, batch_size, args.warmup, args.max_steps)
        results.append(r)
        print(f"  {r['img_per_s']:.1f} img/s | {r['steps_per_s']:.2f} steps/s | "
              f"{r['ms_per_step']:.1f} ms/step | peak {r['peak_mem_gb']:.1f} GB\n", flush=True)

    if len(results) == 2:
        speedup = results[1]["img_per_s"] / results[0]["img_per_s"]
        print(f"==> fast is {speedup:.2f}x current "
              f"({results[0]['img_per_s']:.1f} -> {results[1]['img_per_s']:.1f} img/s)")


if __name__ == "__main__":
    main()
