# Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms

**Author:** Thomas G. Dietterich
**Venue:** Neural Computation 10(7):1895–1923, 1998
**Reference stub** (for the Multi-ViT report, Sprint 5 Slice 3 / VLD-20).

## Why we cite it

Dietterich compares statistical tests for deciding whether two classifiers differ
on the same test set, and recommends **McNemar's test** when each algorithm is
trained once and evaluated on a shared test set (low Type-I error, no model
retraining required). This is exactly our setting: the champion
`11_triple_weighted` and a baseline (best single backbone / best pair) are each
evaluated once on the same disjoint 5-fold OOF test split.

## Method we use

On the paired predictions over the same N samples, build the 2×2 discordant table:
- `b` = champion correct, baseline wrong
- `c` = champion wrong, baseline correct

Continuity-corrected statistic: `χ² = (|b − c| − 1)² / (b + c)` (≈ χ² with df=1).
For moderate discordant counts we report the **exact two-sided binomial** p-value
(`Binom(min(b,c); b+c, 0.5)`), which is robust for small `b+c`.

## How it maps to our result (VLD-20)

- Champion vs best single (Swin-T): b=422, c=315, χ²cc=15.25, exact p=9.2e-5.
- Champion vs best pair (S+B weighted): b=423, c=288, χ²cc=25.25, exact p=4.7e-7.
- Accuracy-level test — complements (does not contradict) the CI-overlapping
  macro-F1 headline. Implemented in `scripts/stats_mcnemar.py` (leakage-free,
  reads stored `predictions.npz`; never averages fold models).
