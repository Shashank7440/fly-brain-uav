# V8.1 — WireFlyNet: FlyWire-constrained visual action network

**Central question:** does a selected biological visual subconnectome (G1: LPLC2-centered,
FlyWire release 783) improve action prediction, robustness, or learning efficiency compared
with **matched non-biological networks**? Better accuracy is a possible outcome, not an
assumption. **A negative result will be reported.**

V7 (TinyFlyNet route-split experiment) is the FROZEN baseline — do not retrain or modify it.
V8 uses the same dataset, corrected route-grouped split, flow caches, and training recipe.

## Status (2026-10-10, v8.1.1 final review corrections)

| Milestone | Status |
|---|---|
| M1 dataset audit | Path-level: PASSED (see V7 archive `m1_manifest_audit.json`). Image-existence/pose checks still require the HPC run. |
| M2 circuit export | G1 derived + validated: 556 neurons / 6,470 edges / 1 component / no self-loops. Provenance in `connectome/`. |
| M3 working model | Code complete; **Phase 4 test suite: 66/66 PASSED** + **integration smoke: 17/17 PASSED** (v8.1.1: param counts asserted from instantiation, fan-in init stats, full train/eval/overfit pipeline verified end-to-end). |
| M4 pilot | Pending — run `jobs/RUN_COMMANDS.txt` steps 2–3 on the HPC. |
| M5 controlled study | Pending — graphs pre-generated: bio + 5 rewired (degree-preserving) + 5 random. |

## v8.1 review fixes (before any HPC training)
1. **Matched initialization** — random-sparse controls previously drew synapse counts from
   uniform 1..19; they now use the biological synapse-count **multiset** (randomly permuted),
   like the rewired controls. Bio-vs-control differences are now attributable to topology
   alone. All 11 graphs regenerated and validated by the test suite.
2. **V7-matched preprocessing** — V8 now applies V7 TinyFlyNet's retina front-end verbatim
   (contrast-norm RGB, temporal delta, relative flow, divergence → 11 channels; parameter-free),
   for both wireflynet and cnn_matched. V7 stays the frozen baseline.
3. **Path portability** — env overrides (WF_DATASET_ROOT, WF_IMAGE_SUBROOT, WF_EXP_ROOT,
   WF_V8_ROOT, WF_GRAPHS_DIR) and package-relative graph fallback; no more hard /Data4-only runs.
4. **Overfit gate** — stratified, seeded 64-sample subset (all 3 classes); a failed overfit now
   exits non-zero so PBS reports it.
5. **Pilot hygiene** — pilot.json sets `eval_test=false` (no test metrics from pilots;
   validation-only decisions) and `train_limit_mode=stratified` (8k class-aware subset).
6. **Threshold enforcement** — derive_g1.py now filters `synapses >= 5` in code
   (`--min-synapses`), matching the declared G1 rule.

Details: docs/review_fixes_v8.1.md; run order: jobs/RUN_COMMANDS.txt, docs/RUNBOOK.md.

## v8.1.1 final corrections (second review round)
1. **Structural pilot isolation** — with `eval_test=false`, train() never constructs the
   test loader or opens the test split at all. The integration smoke test proves it by
   making the test manifest unreadable (chmod 000) during a pilot run.
2. **Fan-in initialization accounted for** — the model divides synapse init by
   sqrt(post in-degree), so matched raw counts still yield different final init when
   degree sequences differ. `graphs_manifest.json` now records `init_fanin_stats` for
   all 11 graphs; **rewired (degree-sequence matched) is the PRIMARY topology control**;
   `--no-fanin-norm` provides the raw-init ablation arm for random-sparse sensitivity.
   result.json/compare.csv record `init_fanin_normalize` per run.
3. **Pipeline order verified** — new `tests/integration_smoke.py` (17 checks) runs the
   real train/eval/overfit pipeline end-to-end on a synthetic on-disk dataset in the
   MidAir file formats: checkpoint saved only on val macro-F1 improvement (best_epoch
   == argmax history), test eval only after training, pilot writes no test metrics,
   finished runs refuse retraining, overfit exit codes propagate (0 pass / 1 fail),
   per-class overfit accuracy reported. The REAL-data overfit gate remains HPC step 2.
4. **Param counts reconfirmed from instantiation** — wireflynet 69,080;
   cnn_matched 83,971 (asserted in the test suite for the paper table).

## Package layout
```
connectome/   g1_neurons.csv, g1_edges.csv, selection_g1.json (pre-declared rule), provenance.md, derive_g1.py
graphs/       build_graphs.py + biological/g1_bio.npz + rewired/ (5) + random_sparse/ (5) + graphs_manifest.json
models/       wireflynet.py — model, train/eval/compare/overfit (config-driven, V7-matched recipe)
docs/         novelty_positioning.md, review_fixes_v8.1.md, RUNBOOK.md
v7_baseline/  frozen V7 reference code (retina interface source; audit scripts) — never retrained
configs/      wireflynet_g1_seed42.json (main), pilot.json (pilot)
tests/        test_suite.py — Phase 4 checks (direction, topology, shape, gradient, state-reset,
              numerics, learnability, checkpoint, eval-only) + all-graph validation
jobs/         v8_overfit.pbs, v8_pilot.pbs, v8_train.pbs, RUN_COMMANDS.txt
docs/         novelty_positioning.md (updated for FLYNN + NeurIPS-2026 Garcia-Duran poster)
references_v8.bib
analysis/     post-training analysis plan (see analysis/README.md)
```

## Model (Phase 3, v8.1)
9-ch input (identical to V7) → **V7 retina preprocessing** (parameter-free, 11 ch) → shared CNN
encoder (~33K params) → input projection into the
**210 LPLC2 entry cells** → edge-list rate-based recurrent core (T=4 steps, only observed edges
have learnable weights; synapse-count init normalized by fan-in) → readout from the **445
documented output cells** → 3 logits. **69,080 trainable params** (TinyFlyNet: 88,171).

Controls share the encoder, entry/output counts, dynamics, recipe: `rewired:k` (exact degree
sequences + matched syn-init multiset), `random:k` (matched sparsity + matched syn-init
multiset), `cnn_matched` (83,971 params, no graph; submits with GRAPH=none).
V7 numbers enter the paper table as the frozen baseline.

## Engineering-to-biology interface = assumptions
Entry/output subsets, rate-based dynamics, per-edge learnable weights, and the >=5-synapse
filter are **modelling choices**, documented in `connectome/provenance.md`. Synapse counts are
not physiological strengths. No neurotransmitter signs are used (variant deferred).

## Compute
Core: 6,470 edge params, scatter-add core, batch-64 at 128px — far lighter than V7 CNN.
Pilot first (8 epochs, 8k pairs); main study: 3 seeds x (bio + 10 controls) + cnn_matched.

## Known limitations (carry into the paper)
- G1 omits LPLC2's upstream columnar fan-in (5,312 neurons in the M2 export) — the engineered
  input lands at LPLC2. Whole-brain coverage and neurotransmitter data are not certified.
- Test LEFT support: 78 samples, one route, three correlated weather versions.
- Labels are recorded body-frame motion directions, not verified safe actions.
- Naming "WireFlyNet" is provisional; check availability before camera-ready.
