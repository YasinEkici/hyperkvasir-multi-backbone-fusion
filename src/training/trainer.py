"""Training orchestration."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.augmentation import apply_cutmix, apply_mixup
from src.evaluation.metrics import compute_metrics
from src.utils.checkpointing import save_checkpoint
from src.utils.logging import get_logger, log_epoch

if TYPE_CHECKING:
    from src.training.ema import EMA

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler,
        run_dir: Path,
        device: str,
        num_classes: int,
        early_stopping_patience: int = 8,
        mixed_precision: bool = False,
        ema: "EMA | None" = None,
        cutmix_alpha: float = 0.0,
        cutmix_prob: float = 0.0,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
        progress_log_interval: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.run_dir = Path(run_dir)
        self.device = device
        self.num_classes = num_classes
        self.patience = early_stopping_patience
        self.mixed_precision = mixed_precision and (device == "cuda")
        self.ema = ema
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        # When > 0, log a heartbeat every N steps inside an epoch (fine-tune
        # path only; 0 keeps frozen/cached-feature epochs silent).
        self.progress_log_interval = progress_log_interval

        self.scaler = torch.amp.GradScaler("cuda") if self.mixed_precision else None

        self.best_val_f1 = -1.0
        self.epochs_no_improve = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> list[dict]:
        self.model.to(self.device)
        ema_warm_started = False

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)

            # EMA: update shadow weights every epoch once past start_epoch;
            # apply shadow weights only for evaluation (not for gradient updates).
            # On first activation, warm-start shadow from current trained weights
            # so we don't apply near-random-init shadow during evaluation.
            ema_active = self.ema is not None and (epoch - 1) >= self.ema.start_epoch
            if self.ema is not None:
                if ema_active and not ema_warm_started:
                    self.ema.reset_shadow(self.model)
                    ema_warm_started = True
                self.ema.update(self.model, epoch - 1)
            if ema_active:
                self.ema.apply(self.model)

            val_metrics = self.evaluate(val_loader)
            val_f1 = val_metrics["macro_f1"]

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, float)},
            }
            self.history.append(epoch_record)
            log_epoch(self.run_dir, epoch_record)

            logger.info(
                "Epoch %d/%d — loss: %.4f  val_f1: %.4f  val_acc: %.4f",
                epoch, epochs, train_loss, val_f1, val_metrics["accuracy"],
            )

            improved = val_f1 > self.best_val_f1
            if improved:
                self.best_val_f1 = val_f1
                self.epochs_no_improve = 0
                # best.pt contains EMA weights when ema_active, live weights otherwise.
                save_checkpoint(self.model, self.run_dir / "best.pt")
                if ema_active:
                    # Persist shadow weights separately for inspection / resuming.
                    torch.save(self.ema._shadow, self.run_dir / "ema.pt")
            else:
                self.epochs_no_improve += 1

            if ema_active:
                self.ema.restore(self.model)

            if not improved and self.epochs_no_improve >= self.patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

            # last.pt always contains live (non-EMA) weights for training resumption.
            save_checkpoint(self.model, self.run_dir / "last.pt")

        return self.history

    # ------------------------------------------------------------------
    def _compute_loss(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with optional MixUp / CutMix mixing.

        At most one mixing op fires per batch: MixUp is tried first with
        probability ``mixup_prob``, then CutMix with probability
        ``cutmix_prob`` (VLD-15 keeps CutMix off/small for endoscopy).  Either
        way the mixed loss is
        ``lam * L(logits, label_a) + (1-lam) * L(logits, label_b)``.  Falls
        back to standard cross-entropy when neither triggers.
        """
        if self.mixup_prob > 0.0 and random.random() < self.mixup_prob:
            features, label_a, label_b, lam = apply_mixup(
                features, labels, alpha=self.mixup_alpha
            )
        elif self.cutmix_prob > 0.0 and random.random() < self.cutmix_prob:
            features, label_a, label_b, lam = apply_cutmix(
                features, labels, alpha=self.cutmix_alpha
            )
        else:
            logits = self.model(features)
            return self.criterion(logits, labels)

        logits = self.model(features)
        return (
            lam * self.criterion(logits, label_a)
            + (1.0 - lam) * self.criterion(logits, label_b)
        )

    # ------------------------------------------------------------------
    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(loader)

        for step, (features, labels) in enumerate(loader, start=1):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.amp.autocast("cuda"):
                    loss = self._compute_loss(features, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._compute_loss(features, labels)
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item() * len(labels)

            if self.progress_log_interval and (
                step % self.progress_log_interval == 0 or step == n_batches
            ):
                logger.info(
                    "    step %d/%d — loss: %.4f", step, n_batches, loss.item()
                )

        return total_loss / len(loader.dataset)

    # ------------------------------------------------------------------
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                logits = self.model(features)
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        return compute_metrics(
            np.array(all_preds),
            np.array(all_labels),
            num_classes=self.num_classes,
        )

    # ------------------------------------------------------------------
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                logits = self.model(features)
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        return np.array(all_preds), np.array(all_labels)
