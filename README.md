# WireFlyNet V8 — Fly-Connectivity-Constrained UAV Maneuver Classification

Research code, saved checkpoints, predictions, experiment records, and provenance for two distinct V8 protocols. The task is offline classification of recorded UAV image pairs as **FORWARD, LEFT, or RIGHT**, not closed-loop navigation or collision avoidance.

## Two experiments — do not mix these results

| Protocol | Saved results | Training imbalance handling | Evaluation available |
|---|---|---|---|
| **Original V8** | `experiments/` — 36 `result.json` files | Inverse-frequency balanced sampling with replacement; unweighted cross-entropy | Validation and previously inspected test results |
| **Weighted-loss V8** | `experiments_weighted_loss/` — 36 `validation_result.json` files | Shuffled sampling; inverse-frequency weighted cross-entropy | **Validation only**; test evaluation disabled |

Each protocol contains Bio (3 seeds), comparable-budget CNN (3), five rewired graphs (15), and five random graphs (15): **36 runs per protocol, 72 main runs total**. Preliminary runs in `experiments_pilot/` are separate and excluded from main summaries. Model weights are retained in each run's `best.pt` where provided.

**Start here:** [Reproducibility](docs/REPRODUCIBILITY.md) · [Protocol map](docs/EXPERIMENTS.md) · [Scientific status](docs/SCIENTIFIC_STATUS.md) · [Provenance](docs/PROVENANCE.md) · [Third-party/data notice](THIRD_PARTY_NOTICES.md).

## Layout

```text
models/                       Current weighted-capable model and frozen pre-weighted source
configs/                      Original and weighted configurations
experiments/                  ORIGINAL 36 runs: val + previously inspected test
experiments_weighted_loss/    WEIGHTED 36 runs: validation only
experiments_pilot/            Preliminary diagnostic runs, not main results
results/                      Fresh protocol/split-labelled CSV summaries
analysis/                     Historical analysis notes plus new reproducibility utilities
connectome/                   G1: 556 nodes, 6,470 directed connections; derivation and metadata
graphs/                       Biological + five degree-preserving rewired + five random graphs
provenance/m2_handoff/         Recovered original M1/M2 handoff, extraction script and M2 export
jobs/                         Historical HPC submission scripts; inspect paths before use
tools/                        Preflight, submission, and historical summary utilities
scripts/                      Safe CPU-only summary and verification utilities added for this release
logs/                         Saved logs; top-level scheduler outputs collected in logs/scheduler/
v7_baseline/                  Historical frozen TinyFlyNet reference, not a third V8 experiment
docs/                         Protocol, provenance, limitations, and archive inventory
v8_weighted_loss_patch/        Historical weighting patch and its tests
v8_weighted_controls_package/ Historical control-submission package and its tests
```

## Biological scope

G1 is an LPLC2-centered selection documented as derived from FlyWire release-783 visual-system tables. It has **556 unique nodes**, **6,470 directed connections**, **210 entry nodes**, and **445 output nodes**; entry and output subsets overlap by **99 nodes**. These are not disjoint layers. Encoder features are projected into entry nodes; a rate-based recurrent core executes four internal updates and a learned readout produces three class scores.

The recovered M2 export (5,750 neurons / 33,071 edges) reproduces G1's node set, edges, synapse counts, and neuron annotation rows. G1 reproduces the saved biological NPZ. **Original raw repository tables are not included and source-to-M2 extraction has not been independently rerun.** External release identity, license, and citation checks remain separate from internal consistency.

Biological anatomy constrains the graph; image encoding, dynamics, weight learning, input projection, and readout are engineered. No physiological equivalence or flight safety is established.

## Weighted validation snapshot

Mean ± sample SD across three training seeds, using the same validation examples:

| Configuration | Macro-F1 | Balanced accuracy (%) | LEFT recall |
|---|---:|---:|---:|
| Bio | 0.7630 ± 0.0148 | 86.692 ± 1.006 | 0.8226 ± 0.0101 |
| Random 4 | 0.7620 ± 0.0239 | 85.672 ± 3.961 | 0.8018 ± 0.0781 |
| Rewired 2 | 0.7616 ± 0.0104 | 84.893 ± 2.622 | 0.8184 ± 0.0302 |
| Comparable-budget CNN | 0.7531 ± 0.0014 | 85.315 ± 3.765 | 0.8151 ± 0.0315 |

Full results for all realizations and both protocols are in [`results/group_summary.csv`](results/group_summary.csv). Bio has the highest observed weighted mean validation macro-F1, but its small lead does **not** establish a reliable biological-wiring advantage. Rewired 1 has higher weighted mean balanced accuracy and LEFT recall. Seed SD is not uncertainty across independent routes.

Graph models have **69,080** trainable parameters; CNN has **83,971** (17.73% fewer for the graph model). "Matched CNN" is the historical run label; the budgets are comparable, not exactly equal. No speed, energy, power, or end-to-end efficiency claim follows from parameter counts.

## Read existing results — no GPU or dataset needed

```bash
python scripts/summarize_results.py
python scripts/verify_artifacts.py
```

These commands read archived results/graph CSVs and write only derived summaries under `results/`; they do not train or evaluate checkpoints. Model training requires separate dataset access and an appropriate PyTorch/CUDA environment. See [reproduction instructions](docs/REPRODUCIBILITY.md) before running any historical job script.

## Availability and licensing

MidAir images, poses, and external raw connectome tables are not included. Derived biological artifacts are accompanied by their source metadata; consult upstream terms before redistribution or reuse. **No blanket open-source license is asserted for the combined package.** See `THIRD_PARTY_NOTICES.md`; rights and required citations remain the authors' responsibility to verify.

This repository contains AI-assisted organization, documentation, and verification utilities. The archived experiment source/results are preserved; no training or new test evaluation was performed during repository preparation. Documentation does not certify journal acceptance, complete source authenticity, or deployment readiness.
