# Experiment registry

## A — Original V8 (historical main study)
- Location: `experiments/<run>/result.json`.
- 36 main runs, seeds 42/43/44, 12 model/graph configurations.
- Balanced inverse-frequency sampling with replacement and unweighted label-smoothed cross-entropy.
- Original config: `configs/wireflynet_g1_seed42.json`.
- Pre-weighted implementation: `models/wireflynet_before_weighted_loss_20261013_181355.py`.
- Results include **validation and test**. The test partition has already been inspected.

## B — Weighted-loss V8 follow-up
- Location: `experiments_weighted_loss/<run>/validation_result.json`.
- 36 main runs, same seed/configuration layout.
- Shuffled sampling and inverse-frequency weighted loss; weights account for expected labels after the existing 50% LEFT/RIGHT-swapping horizontal flip.
- Config: `configs/wireflynet_g1_weighted_loss.json`; implementation: `models/wireflynet.py`.
- Raw training class counts: 47,283 / 2,931 / 2,340.
- Expected post-flip counts: 47,283 / 2,635.5 / 2,635.5.
- Loss weights approximately 0.37049 / 6.64694 / 6.64694.
- **Validation-only** selection and reporting; do not reinterpret missing test results as zero.

## Shared layout
Bio and CNN each have three seeds. Rewired and random families each have five graph realizations × three seeds. Keep graph realization, training seed, protocol, and evaluation split identifiable. Do not collapse these into 72 independent routes or compare one protocol's validation scores to the other's test scores.

CSV metric units: accuracy and balanced accuracy are percent; macro-F1 and class recalls are fractions. Group means and sample SD use stored rounded per-run metrics. The verification script checks confusion-matrix agreement within rounding tolerance.

`experiments_pilot/` contains preliminary diagnostics and is not part of the 72 main runs. Historical CSVs elsewhere are retained but may be stale; regenerate `results/` using the supplied script.
