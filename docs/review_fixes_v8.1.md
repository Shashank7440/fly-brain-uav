# V8.1 — External review fixes (applied 2026-10-10, before any HPC training)

An independent review of the V8 package identified 6 issues. All are fixed below.
Nothing had trained yet, so no results were invalidated; all graph files were
regenerated and the full test suite re-run (61/61 PASSED).

## 1. Random-graph initialization was confounded (HIGH — fixed)
**Issue:** `graphs/build_graphs.py` gave random-sparse controls synapse counts drawn
from uniform 1..19, while bio/rewired used the empirical distribution. With
`init: "synapse"`, bio-vs-random differences could reflect initialization, not topology.
**Fix:** all controls (rewired AND random) now use the biological synapse-count
MULTISET, randomly permuted across edges (`rng.permutation`). `to_npz` still
normalizes to mean edge weight 1.0, so the multiset is preserved exactly. The builder
now self-validates: sorted `syn_init` values of every control must equal the
biological graph's, and the test suite re-checks all 10 controls.
All 11 graphs regenerated (same seed 20261008, k=5; rewired realizations differ from
the v8.0 drafts because rng draws shifted — acceptable pre-training, manifest updated).

## 2. V8/V7 internal preprocessing mismatch (HIGH — fixed)
**Issue:** V7 TinyFlyNet applies retina preprocessing internally (per-sample contrast
normalization on RGB, temporal delta `curr − prev`, relative flow ÷ mean|flow|, flow
divergence, log scale → 11 channels). V8.0 fed raw 9-channel tensors to its encoder.
**Fix:** `models/wireflynet.py` now has a parameter-free `Retina` module reproducing
V7's preprocessing VERBATIM (source: `v7_baseline/flynet_v7_routesplit.py`), applied
identically to wireflynet AND cnn_matched (`input_preprocess: "v7_retina"`; `"raw"`
kept as ablation). External input stays 9 channels; the encoder consumes 11.
Param effect: +288 (first conv 9→11 ch): wireflynet 68,792 → **69,080**;
cnn_matched 83,683 → **83,971**. Test 10 verifies bit-level equivalence with V7's code.

## 3. Hard-coded HPC paths (MEDIUM — fixed)
**Issue:** configs pin `/Data4/...`; the model loaded the configured absolute graph
path unless WF_GRAPHS_DIR was set; tests failed off-HPC.
**Fix:** env overrides WF_V8_ROOT / WF_DATASET_ROOT / WF_IMAGE_SUBROOT / WF_EXP_ROOT;
graphs dir = WF_GRAPHS_DIR → configured dir IF it exists → package-relative `graphs/`
fallback. The test suite now runs with no env setup (61/61 off-HPC, verified).

## 4. Overfit gate could false-pass (MEDIUM — fixed)
**Issue:** `overfit()` returned 0/1 but `__main__` ignored it; a failed gate exited 0.
Subset was the first 64 manifest rows (possibly class/route-biased).
**Fix:** `sys.exit()` propagates the return code (PBS now reports failure); the subset
is a seeded, stratified sample guaranteeing all 3 classes (≥2 each when present);
class counts are recorded in `overfit.json`.

## 5. Pilots exposed test metrics (MEDIUM — fixed)
**Issue:** training always ran test evaluation at the end, so pilots produced test
metrics despite the validation-only protocol.
**Fix:** new `eval_test` config key + `--skip-test-eval` flag; `pilot.json` sets
`eval_test=false` and pilots write `test_not_evaluated.txt` instead. Also
`train_limit_mode: "stratified"` replaces the head-slice 8k pilot subset (subset
selection preserves original row indices — flow-cache keys stay correct).

## 6. Derivation script didn't enforce the ≥5-synapse threshold (MEDIUM — fixed)
**Issue:** `connectome/derive_g1.py` documented ≥5 synapses but did not filter on it
(the packaged CSV happened to contain only ≥5 edges).
**Fix:** `--min-synapses` (default 5) is now enforced in code for both downstream
expansion and induced edges, and recorded in `selection_g1.json` as `min_synapses`.

## Also addressed (reviewer's secondary notes)
- CNN control metadata: `--graph none` is now a validated, explicit value reserved for
  `--model cnn_matched`; wireflynet + none is rejected. RUN_COMMANDS submits all CNN
  seeds with `GRAPH=none`.
- Reporting guidance (macro-F1/balanced-acc first, per-class + confusion matrices,
  mean±sd across seeds, paired differences vs bio, route-aware uncertainty, latency
  incl. flow) is captured in `analysis/README.md` and the RUNBOOK.

## What was NOT changed
- V7 is untouched and remains the frozen baseline (files copied to `v7_baseline/` for
  reference; never retrained).
- The pre-declared study design (3 seeds × 11 graphs + cnn_matched, graph family, K)
  and the G1 selection rule are unchanged.
- The 78-sample LEFT test caveat stands: interpret per-class conclusions accordingly.

---

# V8.1.1 — Second review round (final corrections, 2026-10-10)

The second review accepted V8.1 as a base and required four updates before HPC launch.
All four are applied and verified; nothing else changed.

## A. Pilot test isolation is now structural (REQUIRED)
`train()` no longer constructs the test loader when `eval_test=false` — the test split
is never opened. Proof: `tests/integration_smoke.py` section A chmods the test
`pairs.csv` to 000 before a pilot run; the run succeeds and writes
`test_not_evaluated.txt` with zero test artifacts. Any test access would crash.

## B. Fan-in initialization difference is measured and accounted for (REQUIRED)
The model divides each edge's synapse init by sqrt(post in-degree). Matching the raw
synapse-count multiset therefore does NOT guarantee identically distributed final
weights across graph types (degree sequences differ for random; and even rewired breaks
the count-to-degree correlation). Measured (`graphs_manifest.json: init_fanin_stats`,
fan-in-normalized init mean): bio 0.2346, rewired 0.223–0.237 (within 5%),
random 0.289–0.294 (+23%). Responses:
- **Primary topology test = bio vs rewired** (identical in/out-degree sequences;
  init means within 10% — asserted by the test suite).
- Random-sparse remains a secondary control with the difference documented.
- `--no-fanin-norm` runs the raw synapse-count init ablation arm (recorded in
  result.json and compare.csv as `init_fanin_normalize`).
- The README no longer claims "topology alone"; the claim is now scoped to the
  rewired comparison.

## C. Overfit gate mechanics verified; real-data gate remains (REQUIRED)
`tests/integration_smoke.py` sections D–E run the actual overfit gate end-to-end:
stratified 3-class subset (16/16/16 on the synthetic set), per-class train accuracy in
`overfit.json`, exit 0 on pass and exit 1 on failure. Learnability on the REAL MidAir
data is still only established by HPC RUN_COMMANDS step 2 — intentionally not claimed here.

## D. Checkpoint / selection / eval order verified (REQUIRED)
Integration smoke section B–C: finished runs refuse retraining (FileExistsError,
non-zero exit); `best.pt` epoch equals argmax of validation macro-F1 in history.csv
(selection is validation-only); `result.json` + `test_predictions.csv` appear only in
the post-training evaluation; pilots produce neither. `result.json` also records
`input_preprocess`, `init`, `init_fanin_normalize`, and `steps` for provenance.

## Verification summary (v8.1.1)
- Structural suite: **66/66 PASSED** (61 + param-count assertions + fan-in stats).
- Integration smoke: **17/17 PASSED** (real pipeline on synthetic on-disk data).
- V7 untouched; study design, graph family, and pre-declared rules unchanged.
