# Universal Language Model Fine-tuning for Text Classification (ULMFiT)

**Authors:** Jeremy Howard, Sebastian Ruder
**Venue:** ACL 2018
**Reference stub** (for the Multi-ViT report, Sprint 5 / fine-tuning recipe).

## Why we cite it

ULMFiT introduces **discriminative fine-tuning** (a.k.a. layer-wise learning-rate
decay, LLRD): different layers of a pretrained model are fine-tuned with different
learning rates, lower for the earlier (more general) layers and higher for the
later (more task-specific) layers. The rate at depth `d` follows
`lr(d) = lr_base · decay^d`.

## How it maps to our use

We apply LLRD when fine-tuning the last transformer blocks of ViT-B/BEiT-B and the
final Swin stage: a base backbone learning rate of `5e-5` with per-layer decay
`0.7`, while the fresh projection/fusion/MLP head trains at `1e-3`. This protects
the general pretrained features from catastrophic forgetting on the small
endoscopy dataset while still adapting the top layers to the domain.
