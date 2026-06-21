"""Trainer AMP dtype wiring (Sprint 3.5, VLD-17)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn

from src.training.trainer import Trainer


def _trainer(tmp_path, **kw) -> Trainer:
    model = nn.Linear(4, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return Trainer(model, opt, nn.CrossEntropyLoss(), None, tmp_path, "cpu", 3, **kw)


def test_amp_dtype_defaults_to_float16(tmp_path) -> None:
    assert _trainer(tmp_path).amp_dtype == torch.float16


def test_amp_dtype_bfloat16(tmp_path) -> None:
    assert _trainer(tmp_path, amp_dtype="bfloat16").amp_dtype == torch.bfloat16


def test_cpu_run_has_no_grad_scaler(tmp_path) -> None:
    # device="cpu" forces mixed_precision off -> no GradScaler regardless of dtype.
    t = _trainer(tmp_path, mixed_precision=True, amp_dtype="bfloat16")
    assert t.scaler is None
