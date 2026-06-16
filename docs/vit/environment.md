# Environment — ViT Fusion

Records the compute environments for the ViT project. Local `uv` env is shared
with the CNN project (already validated in root `docs/environment.md`); the new
surface is **Google Colab Pro (A100)**, validated at the first Colab session (S1+).

## Local development (shared with CNN project)

- Managed by `uv` (`pyproject.toml` + `uv.lock`); `timm>=1.0` already pinned.
- timm model strings verified loadable on 2026-06-16 (local), 768-d pooled output:
  `vit_base_patch16_224.orig_in21k_ft_in1k`,
  `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`,
  `beit_base_patch16_224.in22k_ft_in22k_in1k`.
- Use local (any GPU / CPU) for: smoke tests, feature caching, frozen-path ablation.

## Colab Pro (A100) — primary training compute (VLD-11)

*To be filled at the first Colab session (Sprint 1/2).* Record:

- CUDA device name (must contain `A100` — D-09 gate), VRAM, driver/CUDA version.
- torch / torchvision / timm versions in the Colab runtime.
- Dataset staging method (Drive → local `/content`), dataset SHA256.
- Compute-unit burn rate observed per fine-tune fold (sanity-check the ~65-unit budget).
- Any version drift vs the local `uv.lock` stack.
