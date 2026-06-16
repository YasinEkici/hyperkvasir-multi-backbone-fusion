# Experiment Log — ViT Fusion

Record successful runs and environment notes here (chronological, append-only).
Each reportable run must also have a `metrics.json` under `results/vit/runs/{id}/`
and trace to a resolved config (provenance gate, CNN D-09 reused).

## 2026-06-16 — Project scaffolding (Sprint 0 / setup)

- `docs/vit/` created: `final_assignment.md`, `project_structure.md`,
  `project_plan.md`, `decisions.md` (VLD-01…16), `results_progress.md`,
  `known_issues.md`, `environment.md`.
- References: 10 ViT folders added (`INDEX.md` Section H, #32–41); `dropblock`
  and `swa` removed. timm strings verified loadable (768-d) on 2026-06-16.
- `AGENTS.md` ViT addendum added (timm/LayerNorm/path overrides).
- No code or runs yet — implementation starts in Sprint 1.
