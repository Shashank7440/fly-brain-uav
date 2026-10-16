# V7 route-grouped retraining — safe standalone package
## What changes
Only route split, dataset/output paths, and safety checks. All three model classes
and train() are AST-identical to the checkpoint-fixed V7 code. Labels remain
recorded body-frame dominant motion directions, NOT verified safe avoidance actions.
Quaternion and synchronization provenance are still unresolved; this package is
not a claim that those questions have been fixed.

## Install location
Extract this folder so it is located at:
/Data4/mi_25026207/drone/v7_route_retrain
Keep original dataset/images and ExperimentV7 intact.
Uses your existing flyuav environment (PyTorch, torchvision, numpy, OpenCV, PIL,
matplotlib). No new framework needed. Split and audit use Python standard library.

## 1. Prepare and check new split (no GPU; no training)
cd /Data4/mi_25026207/drone/v7_route_retrain
python prepare_route_split.py
python audit_pairs.py --root /Data4/mi_25026207/drone/data/flyuav_midair_3class_route_split --out route_audit.json
cat route_audit.json

Splitter refuses to overwrite an existing output directory. If already prepared,
read its split_metadata.json and audit it; do not rerun with an overwrite flag.
Default image existence checking scans the original image root. If it fails,
correct the --images argument and IMG_SUBROOT in flynet_v7_routesplit.py together.
It copies only CSV records, not images and NOT old flow caches.

Expected counts for the supplied manifests / seed 42:
           routes recordings pairs  FORWARD LEFT RIGHT
train      8      24          52554  47283   2931 2340
val        2      6           13221  12348    402  471
test       2      6           13236  12846     78  312
Only 12 underlying route groups are available. Test LEFT support is small and
correlated across three weather versions. Report this limitation; do not choose
another split after seeing model test performance. With more compute, grouped
cross-validation is a useful future extension.

Group IDs use exact full retained motion/label sequence signatures; naming suffix
correspondence is checked as corroboration. Any same suffix/different signature
conflict stops preparation. This is supported by uploaded manifests; unfiltered
pose identity and content duplicates still need source-record validation.

## 2. Precompute new index-aligned flow (recommended)
qsub v7r_flow.pbs
Wait for SUCCESSFUL completion before submitting GPU jobs.
The job can resume an interrupted cache only if its metadata match the manifest.
Read errors stop the job; it never substitutes zero flow.
If you skip precomputation, training computes flow on demand and will be slower.
Do not copy the original row-indexed caches to this dataset.
Do not run two flow jobs on the same cache simultaneously.

## 3. Train from scratch (three separate GPU jobs)
qsub v7r_train_flynet.pbs
qsub v7r_train_cnn.pbs
qsub v7r_train_snn.pbs
Each trains seed 42 and evaluates its best EMA checkpoint automatically.
Output: /Data4/mi_25026207/drone/ExperimentV7RouteSplit/<model>/seed42
Never use old best.pt to initialize these runs.
Existing checkpoints block retraining; use a new experiment root for a new study.
Jobs preserve original files. Logs include PBS job ID.
Scheduler allocation: workq, 8 CPUs, 1 GPU, 32 GB, 48-hour limit. This is a requested
limit, not runtime estimate; adapt to your cluster's allowed walltime if necessary.

## 4. More seeds after the pipeline works
qsub -v SEED=123 v7r_train_flynet.pbs
qsub -v SEED=123 v7r_train_cnn.pbs
qsub -v SEED=123 v7r_train_snn.pbs
qsub -v SEED=456 v7r_train_flynet.pbs
qsub -v SEED=456 v7r_train_cnn.pbs
qsub -v SEED=456 v7r_train_snn.pbs
SEED controls training only; all use the same corrected seed-42 split.
Do not submit concurrent jobs for the same model and seed.

## 5. Compare after jobs finish
python flynet_v7_routesplit.py --model compare --seed 42
cat /Data4/mi_25026207/drone/ExperimentV7RouteSplit/Compare/summary.md
For another seed, change --seed. The Compare folder is shared; running another
seed's comparison overwrites the comparison artifacts (not per-model results).
Archive comparison outputs before changing the comparison seed.
Auto-comparison may run when all model results exist; run compare explicitly once
all jobs are done to produce a consistent final set.

## Evaluation-only recovery if a job fails AFTER training
qsub v7r_eval_flynet.pbs
qsub v7r_eval_cnn.pbs
qsub v7r_eval_snn.pbs
Run only the needed model after its training job has finished. Default mode is eval;
missing checkpoints fail rather than train. Trusted local checkpoint loading uses
weights_only=False; never load untrusted checkpoint files.

## Local validation performed
- Built corrected manifests from all uploaded records.
- Route signatures do not cross splits; path audit passed.
- Total retained rows unchanged; all three classes present in each split.
- Model definitions and training recipe checked unchanged by AST comparison.
- Python compilation and bash -n passed.
Real GPU training and flow cache generation were NOT executed here: images and
HPC hardware are not present. Expected split files are audit examples only;
prepare_route_split.py generates the active manifests on your cluster.
