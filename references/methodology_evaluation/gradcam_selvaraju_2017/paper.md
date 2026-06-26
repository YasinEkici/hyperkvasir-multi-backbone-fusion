# Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

**Authors:** Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna
Vedantam, Devi Parikh, Dhruv Batra
**Venue:** ICCV 2017
**Reference stub** (for the Multi-ViT report, Sprint 5 / interpretability).

## Why we cite it

Grad-CAM is the foundational gradient-based class-activation method that
Grad-CAM++ generalises. It produces class-discriminative localisation maps by
weighting the last feature-map channels with the global-average of the gradients
of the target class score w.r.t. those activations:

`w_k^c = (1/Z) Σ_i Σ_j ∂y^c / ∂A_{ij}^k`,  `L^c = ReLU(Σ_k w_k^c A^k)`.

It works on any differentiable architecture without retraining or architectural
change. We cite it as the basis for the Grad-CAM++ maps used on the ViT/BEiT/Swin
branches of the final model.

## How it maps to our use

We use the improved variant Grad-CAM++ (Chattopadhyay et al. 2018, pixel-wise
weighted positive gradients) via a transformer `reshape_transform`; Grad-CAM is
cited as the method it builds on. Inference-only from `best.pt`; no model change.
