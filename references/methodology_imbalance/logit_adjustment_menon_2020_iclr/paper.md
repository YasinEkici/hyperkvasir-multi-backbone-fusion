# Long-tail Learning via Logit Adjustment

**Authors:** Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu
Jain, Andreas Veit, Sanjiv Kumar
**Venue:** ICLR 2021 (arXiv:2007.07314, 2020)
**Reference stub** (for the Multi-ViT report, Sprint 5 Slice 3 / VLD-20).

## Why we cite it

Menon et al. formalise **logit adjustment** for class-imbalanced (long-tail)
classification: shifting the model logits by the (log) class priors yields a
Bayes-consistent estimator for the *balanced* error, equivalently optimising a
macro-averaged objective. The post-hoc variant subtracts `τ·log(prior)` from the
logits at inference time, with the prior estimated from the **training**
distribution.

## Method we use (post-hoc, leakage-free)

For each fold's OOF test logits, subtract `τ·log(train_prior)` where the prior is
computed from that fold's TRAIN split only (no test leakage, VLD-13), then
recompute macro-F1 over the concatenated OOF predictions. `τ=0` is the unadjusted
champion; `τ=1` is the parameter-free posterior correction.

## How it maps to our result (VLD-20)

Logit adjustment is a **clean honest negative** here: OOF macro-F1 falls
monotonically with τ (0.6118 → 0.4911 at τ=1 → collapse). The model is trained
with a `WeightedRandomSampler` (balanced sampling), so its implicit prior is
already ~uniform; subtracting the train prior **double-corrects** and over-shifts
toward rare classes. Conclusion: the balanced sampler is the correct imbalance
mechanism for this setup; post-hoc prior correction does not help. Implemented in
`scripts/eval_logit_adjust.py` (additive analysis — champion unchanged).
