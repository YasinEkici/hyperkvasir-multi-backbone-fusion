"""Augmentation helpers for fine-tuning training configs.

RandAugment reference: Cubuk et al. 2020 (NeurIPS).
CutMix reference:      Yun et al. 2019 (ICCV), equations (1)–(3).
MixUp reference:       Zhang et al. 2018 (ICLR), equation (1).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms


def apply_rand_augment(img: Image.Image, N: int, M: int) -> Image.Image:
    """Apply RandAugment to a PIL image.

    Args:
        img: Input PIL image (mode RGB).
        N:   Number of augmentation operations to apply (num_ops).
        M:   Magnitude of each operation (0–31 scale in torchvision).

    Returns:
        Augmented PIL image with the same spatial size and mode.
    """
    return transforms.RandAugment(num_ops=N, magnitude=M)(img)


def apply_cutmix(
    images: Tensor,
    labels: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Apply CutMix to a batch of images (Yun et al. 2019, ICCV).

    Samples λ ~ Beta(alpha, alpha), derives a rectangular cut box from λ,
    pastes that region from a shuffled copy of the batch, and returns the
    adjusted λ based on the actual mixed area (accounting for border clipping).

    The calling code is responsible for computing the mixed loss:
        loss = lam * criterion(logits, label_a) + (1 - lam) * criterion(logits, label_b)

    Args:
        images: Float tensor of shape [B, C, H, W].
        labels: Long tensor of shape [B].
        alpha:  Beta distribution concentration parameter (config: cutmix.alpha).

    Returns:
        mixed_images: Tensor [B, C, H, W] — images with CutMix applied.
        label_a:      Tensor [B] — original labels.
        label_b:      Tensor [B] — labels from the shuffled pair.
        lam:          float — actual mix ratio (area of kept region / total area).
    """
    B, C, H, W = images.shape

    # Sample lambda from Beta(alpha, alpha)
    lam: float = float(np.random.beta(alpha, alpha))

    # Derive cut box size from lam (Yun et al. eq. 3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    # Random centre, then clip to image boundary
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    # Adjust lam to reflect actual clipped area (so loss weighting is correct)
    actual_cut_area = (x2 - x1) * (y2 - y1)
    lam = 1.0 - actual_cut_area / (H * W)

    # Shuffle batch to get pairs
    rand_index = torch.randperm(B, device=images.device)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]

    return mixed, labels, labels[rand_index], lam


def apply_mixup(
    images: Tensor,
    labels: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Apply MixUp to a batch of images (Zhang et al. 2018, ICLR).

    Verbatim from Zhang et al. 2018 §2 (ICLR), Equation 1 — vicinal
    risk-minimization construction of virtual training examples:

        x̃ = λ x_i + (1 − λ) x_j,   where x_i, x_j are raw input vectors
        ỹ = λ y_i + (1 − λ) y_j,   where y_i, y_j are one-hot label encodings

    with λ ∼ Beta(α, α), λ ∈ [0, 1].  (x_i, y_i) and (x_j, y_j) are two
    examples drawn at random from the training data.

    This function mixes the whole image (unlike CutMix's rectangular paste).
    As with CutMix, the caller computes the mixed loss:
        loss = lam * criterion(logits, label_a) + (1 - lam) * criterion(logits, label_b)

    Args:
        images: Float tensor of shape [B, C, H, W].
        labels: Long tensor of shape [B].
        alpha:  Beta distribution concentration parameter (config: mixup.alpha).

    Returns:
        mixed_images: Tensor [B, C, H, W] — λ·images + (1−λ)·shuffled.
        label_a:      Tensor [B] — original labels (weight λ).
        label_b:      Tensor [B] — labels from the shuffled pair (weight 1−λ).
        lam:          float — mix coefficient λ ~ Beta(alpha, alpha).
    """
    B = images.shape[0]

    # λ ~ Beta(alpha, alpha). alpha<=0 disables mixing (lam=1.0).
    lam: float = float(np.random.beta(alpha, alpha)) if alpha > 0.0 else 1.0

    rand_index = torch.randperm(B, device=images.device)

    mixed = lam * images + (1.0 - lam) * images[rand_index]

    return mixed, labels, labels[rand_index], lam
