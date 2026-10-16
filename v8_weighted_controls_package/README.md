# V8 weighted-loss graph controls — 30 validation-only jobs

## Scope

- Rewired graphs `rewired:0` to `rewired:4`, each at seeds 42, 43, and 44: 15 jobs.
- Random graphs `random:0` to `random:4`, each at seeds 42, 43, and 44: 15 jobs.
- No Bio or CNN jobs are resubmitted.
- No model script, config, graph file, cache, or old result file is replaced.

This extends the completed six-run weighted pilot under the SAME recipe:
ordinary shuffled sampling; inverse-frequency weighted cross-entropy using
expected post-flip training counts; 40-epoch cap; batch 64; LR 0.002; patience
12; workers 4; validation macro-F1 selection; no test evaluation.

The graph NPZ files already used in the original V8 study are reused.
Do not regenerate graphs, split manifests, or flow caches.

## Where outputs go

```
/Data4/mi_25026207/drone/v8_wireflynet/
├── experiments/                         # original 36 runs: untouched
├── experiments_weighted_loss/
│   ├── wireflynet_bio_seed42/...         # completed Bio/CNN pilot: untouched
│   ├── cnn_matched_none_seed42/...
│   ├── wireflynet_rewired0_seed42/       # new control examples
│   ├── wireflynet_rewired0_seed43/
│   ├── wireflynet_random4_seed44/
│   └── _control_submission/
│       ├── inventory.json               # frozen input hashes and all 30 targets
│       └── jobs.csv                     # accepted job IDs / submission status
└── logs/weighted_loss_controls/          # separate new logs
```

The PBS file is NEW: `jobs/v8_weighted_controls.pbs`. The completed Bio/CNN PBS
file remains unchanged. Scheduler resources match those pilot jobs: workq,
8 CPUs, 32 GB, 1 GPU, and 24-hour walltime.

## Install and run preflight (CPU-only)

Upload `v8_weighted_controls_package.zip` into
`/Data4/mi_25026207/drone/v8_wireflynet/`, then run:

```bash
cd /Data4/mi_25026207/drone/v8_wireflynet
conda activate flyuav
unzip -n v8_weighted_controls_package.zip
bash v8_weighted_controls_package/install_and_check.sh
```

Stop on any error. This installer only copies the three NEW job/helper files.
A different existing file is not overwritten. It runs seven tests, syntax
checks, and preflight; it does NOT submit any job.

Preflight verifies:

1. The current model/config hashes and six completed pilot metadata files
   agree with the exact frozen weighted pilot.
2. The pilot's resolved training settings include workers 4 and the same
   training budget, label weights, and train/validation manifests.
3. All 10 control NPZ files exist, have matching node/edge counts and
   entry/output mappings, and have finite initial edge weights.
4. All 30 control targets are fresh.
5. Training/validation manifest and flow-cache bindings are consistent.

It records hashes of the frozen inputs in `inventory.json`. Test samples,
test manifests, and test predictions are not read. Every job verifies the
inventory again before training, so do not edit the model/config/graphs or
pilot metadata while control jobs are queued or running.

Graph topology/initialization provenance is inherited from the existing
graph-building study; these shape/hash checks do not independently prove all
properties of the original graph-generation algorithm.

If preflight reports an environment override such as `WF_EXP_ROOT`, inspect
it before proceeding. An override pointing at the original baseline is not
permitted. Do not bypass a failed check.

## Preview the 30 planned commands (optional)

After preflight passes:

```bash
python3 tools/submit_weighted_controls.py
```

This prints the planned qsub commands but does not submit them.

## Submit all 30 with one command

Only after preflight passes:

```bash
python3 tools/submit_weighted_controls.py --submit
```

Each accepted job prints its ID and exact run name. The helper records
submission intent before contacting PBS and then records accepted IDs in:

```
experiments_weighted_loss/_control_submission/jobs.csv
```

It uses an exclusive submission lock and skips already recorded accepted
jobs on a later invocation, rather than blindly resubmitting 30 jobs.
If PBS rejects a job, returns an ambiguous response, or submission is
interrupted, the uncertain record blocks automatic retry. Check qstat and
send the error/ledger before changing anything. Do not delete the ledger or
repeat raw qsub commands to bypass it.

The job script also uses an atomic per-run lock and refuses existing
checkpoint/history/metadata/results. A failed run's lock/partial files remain
for review rather than being overwritten.

## Queue and logs

```bash
qstat -u "$USER"
```

Queued jobs are normal. Existing V7 jobs can continue independently; do not
cancel them or resubmit queued jobs.

Logs are named:

```
logs/weighted_loss_controls/wireflynet_rewired0_seed42_JOBID.log
```

Replace `JOBID` with the actual ID printed by the submitter or saved in
`jobs.csv`. Use `tail -F` on the exact filename once the job has started.
Confirm startup shows `sampler=shuffle`, `class_weighting=inverse_frequency`,
and `test ISOLATED`. Ctrl+C only stops watching the log.

## After all jobs finish

The weighted study then contains 36 runs: 3 Bio, 3 CNN, 15 random, 15 rewired.
Compare their VALIDATION results first, preserving graph realization and
training-seed identifiers. Do not treat 15 controls as 15 independent
training-seed replicates and do not pick a winning realization/seed.

Do not use the old `--mode compare` command for this stage: it reads test
`result.json`, whereas these validation-only runs save
`validation_result.json`. Do not enable `--mode eval`.

The original test set was already inspected. Future test reporting must
acknowledge this is an adaptive follow-up, not fresh untouched confirmation.
This expansion tests whether the biological topology helps under the new
balancing strategy; it does not guarantee a positive result.

## Testing scope

Seven local tests cover scope, protocol mismatches, existing-output
protection, prepared-input verification, submission intent/accepted-job
records, and interrupted/rejected submissions. Shell and embedded Python
syntax are checked. Test scheduler calls are mocked; no real qsub call or
GPU training was executed on the packaging computer.