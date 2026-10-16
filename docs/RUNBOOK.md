# V8.1.1 RUNBOOK — WireFlyNet on the HPC (final pre-HPC version)

Order matters: audit → freeze V7 → tests → overfit → pilot → main study → compare.
Never skip a gate; never select configs/seeds/graphs on test metrics.

## Step 0 — One-time environment checks (CPU)
```
# M1 audit on the REAL dataset root (path-level leakage check):
python3 v7_baseline/audit_pairs.py   # writes m1_audit.json; expect PATH_CHECKS_PASS
# V7 freeze (if not already saved): pip freeze, copy CNN/SNN checkpoints aside.
```
Why: the packaged audit JSONs were produced from manifests; the HPC run must confirm
against the actual files. V7 stays frozen — every V8-vs-V7 comparison is against
published V7 numbers, never a retrain.

## Step 1 — Extract + tests (CPU, ~2 + ~3 min)
```
# extract package at /Data4/mi_25026207/drone/v8_wireflynet, then:
cd v8_wireflynet/tests && python3 test_suite.py           # expect: 66 passed, 0 failed
cd v8_wireflynet/tests && python3 integration_smoke.py    # expect: 17 passed, 0 failed
```
The integration smoke test runs the real train/eval/overfit pipeline end-to-end on a
tiny synthetic on-disk dataset in the MidAir formats: pilot isolation (test split
chmod-000 during the pilot run — any test access crashes), checkpoint protection,
validation-only model selection (best_epoch == argmax val macro-F1), test eval only
after training, overfit exit-code propagation.
What it covers: all 11 graph files (edge count, no self-loops, degree preservation,
identical node/entry/output sets, **matched syn_init multiset**), edge direction,
topology enforcement, shapes, gradients, state reset, numerics, learnability,
checkpointing, eval-only, **V7-retina equivalence**, graph=none handling.
Why 66 (was 45): v8.1 added matched-init, retina, graph-none checks; v8.1.1 added param-count assertions and fan-in init stats.
Off-HPC: paths fall back to the package; WF_DATASET_ROOT/WF_IMAGE_SUBROOT override.

## Step 2 — Overfit gate (GPU, ~10 min)
```
qsub jobs/v8_overfit.pbs
```
Trains on 64 samples (seeded, stratified, all 3 classes) and expects ≥95% train acc.
Why: catches implementation faults cheaply before any full run. v8.1: failure now
exits non-zero (PBS reports it), and class counts land in `overfit.json`.

## Step 3 — Pilot (GPU, 3 jobs, 8 epochs on 8k stratified pairs)
```
qsub -v GRAPH=bio       jobs/v8_pilot.pbs
qsub -v GRAPH=rewired:0 jobs/v8_pilot.pbs
qsub -v GRAPH=random:0  jobs/v8_pilot.pbs
```
Gate: no NaN, affordable walltime, sane LR/loss range. **Validation metrics only** —
v8.1 pilots write NO test metrics (`eval_test=false` in pilot.json). Lock all config
decisions here, before the main study, on validation data alone.

## Step 4 — Main study (GPU, 36 runs at 3 seeds; 60 at 5 seeds)
```
for g in bio rewired:0 rewired:1 rewired:2 rewired:3 rewired:4 random:0 random:1 random:2 random:3 random:4; do
  for s in 42 43 44; do qsub -v GRAPH=$g,SEED=$s jobs/v8_train.pbs; done; done
for s in 42 43 44; do qsub -v "MODEL=cnn_matched,GRAPH=none,SEED=$s" jobs/v8_train.pbs; done
```
Each run: train on val-macro-F1 checkpointing (EMA, OneCycle, balanced sampler) then
test-eval automatically (main config has `eval_test=true`). GRAPH=none is reserved for
cnn_matched so run metadata is unambiguous.
Existing checkpoints block retraining by design; a new study needs a new exp_root.

## Step 5 — Compare + report
```
cd models && python3 wireflynet.py --mode compare   # -> experiments/compare.csv
```
PRIMARY topology comparison for the paper: bio vs rewired:0-4 (identical degree
sequences; fan-in-normalized init means within 10% of bio — test-verified). Report
random:0-4 as the secondary control (its init distribution differs by +23% mean after
normalization; measured in graphs_manifest.json init_fanin_stats) and, if budget
allows, the --no-fanin-norm raw-init sensitivity arm. Reporting priorities: macro-F1 &
balanced accuracy first; per-class precision/recall/F1
+ confusion matrices; mean±sd across seeds; paired differences (bio − each control);
route-aware uncertainty (frames within a route are correlated — don't treat them as
independent); model-only AND end-to-end latency (incl. optical flow). LEFT has only 78
test samples in one route group — interpret per-class results cautiously.
Add V7 frozen numbers to the paper table from the V7 archive; do not retrain V7.

## File map (what depends on what)
| Path | Role |
|---|---|
| `connectome/derive_g1.py` | G1 derivation; **now enforces `--min-synapses 5` in code** |
| `connectome/g1_*.csv, selection_g1.json, provenance.md` | pre-declared circuit + provenance |
| `graphs/build_graphs.py` | regenerates all 11 graphs; controls use the bio syn-count multiset |
| `graphs/{biological,rewired,random_sparse}/*.npz` | validated graph family (test suite re-checks) |
| `models/wireflynet.py` | model + train/eval/compare/overfit; retina, env-overridable paths |
| `configs/wireflynet_g1_seed42.json` | main study config (`eval_test=true`, `input_preprocess=v7_retina`) |
| `configs/pilot.json` | pilot config (`eval_test=false`, stratified 8k subset) |
| `v7_baseline/` | frozen V7 files — reference only, never retrain |
| `docs/review_fixes_v8.1.md` | what changed in v8.1 and why |
| `docs/novelty_positioning.md, references_v8.bib` | positioning + citations |
