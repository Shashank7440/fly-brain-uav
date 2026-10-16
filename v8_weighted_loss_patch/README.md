# V8 weighted-loss follow-up — six validation-only runs

This is a new balancing-strategy ablation, not a replacement for the frozen
36-run study and not a claim that class weighting will fix test generalization.

## What changes

- Original/default behavior: balanced inverse-frequency sampler + unweighted,
  label-smoothed cross-entropy.
- New config: ordinary shuffled sampling, without replacement within an epoch,
  + inverse-frequency weighted, label-smoothed cross-entropy.
- Same model definitions, parameter counts, data manifests, augmentation,
  optimizer, schedule, maximum epochs, patience, and validation macro-F1
  checkpoint selection.
- The weighting/sampling choices are both changed: this compares balancing
  strategies, not the isolated effect of adding loss weights.
- Training weights use selected **training labels only**. The existing 50%
  horizontal flip swaps LEFT and RIGHT, so inverse weights use their expected
  post-augmentation counts. LEFT/RIGHT weights are consequently equal. No
  validation/test counts, synthetic examples, or direction-specific tuning.
- The original `drop_last=True` behavior remains unchanged.
- This patch refuses balanced sampling plus class weighting to prevent double
  minority-class emphasis.

## Installation

Run from `~/drone/v8_wireflynet`. Keep the uploaded ZIP outside the project if
preferred; adjust its path below. The ZIP extracts into a staging folder.

```bash
unzip v8_weighted_loss_patch.zip
backup="models/wireflynet_before_weighted_loss_$(date +%Y%m%d_%H%M%S).py"
cp -n models/wireflynet.py "$backup"
cp v8_weighted_loss_patch/models/wireflynet.py models/wireflynet.py
cp -n v8_weighted_loss_patch/configs/wireflynet_g1_weighted_loss.json configs/
mkdir -p tools
cp -n v8_weighted_loss_patch/tools/summarize_weighted_validation.py tools/
python3 v8_weighted_loss_patch/tests/test_weighted_patch.py
```

If files with these new names already exist, compare them to the staged copies
before proceeding. The `cp -n` commands deliberately do not overwrite them.
The timestamped backup preserves the original model script.
Do not overwrite or remove any existing experiment outputs.

## Output isolation and test guard

- Original experiment root remains `experiments/`.
- New config writes to:
  `/Data4/mi_25026207/drone/v8_wireflynet/experiments_weighted_loss`.
- Check environment overrides before starting. In particular `WF_EXP_ROOT`
  overrides the config; unset it for these jobs, or explicitly point it at
  the new weighted-loss root. Never point it at the frozen baseline.
- The config sets `eval_test: false`; the job commands also set
  `--skip-test-eval`. Training does not construct the test loader.
- `--mode eval` now fails while test evaluation is disabled. Do not switch
  evaluation on during the follow-up's model selection.
- Training refuses to overwrite an existing `best.pt`. Do not resume old
  checkpoints or copy baseline checkpoints into the new root.

## Submit six jobs using the existing GPU/PBS workflow

The six command specifications are in `jobs/weighted_loss_commands.txt`.
They cover Bio and CNN, seeds 42, 43, and 44.

Do **not** execute the commands as GPU training on the login node, and do not
launch all six on one GPU. Reuse the main study's GPU queue/resource requests
and worker setting. The PBS script was not supplied, so this package does not
invent queue names, resource directives, or submit jobs.

The intended budget remains 40 epochs maximum, batch 64, LR 0.002, patience 12.
Preserve the main jobs' actual CLI settings if they differ, and record them.
Send the original main-study PBS script before finalizing submissions.

## Outputs and validation comparison

Each new run saves:

- `best.pt`: EMA checkpoint selected using validation macro-F1.
- `history.csv`, `val_predictions.csv`, and `curves.png`.
- `validation_result.json`: selected validation metrics, epoch, and protocol.
- `run_metadata.json`: resolved CLI arguments, config, label counts, weights,
  sampling choice, and script/config/train/validation-manifest hashes.
- `test_not_evaluated.txt`.

After all six jobs finish, run this CPU-only audit:

```bash
python3 tools/summarize_weighted_validation.py \
  --baseline experiments \
  --weighted experiments_weighted_loss \
  --out experiments_weighted_loss/validation_comparison.csv
```

It verifies selected epochs and recomputes validation metrics from prediction
files. It compares all six new runs to the six original Bio/CNN validation
results, requires the same true-label order, rejects test outputs in the
follow-up root, and does not use test scores.

Judge paired-seed validation macro-F1, balanced accuracy, and precision/recall
for all three classes. Do not pick a winning seed. Do not expand to random or
rewired controls until the six-run evidence justifies the extra compute.

The original test scores have already been inspected. Any later use of that
test set is a disclosed follow-up, not fresh untouched confirmation.

## Testing status

This package has standard-library-only syntax, configuration, class-weight,
sampler-routing, test-guard, and validation-summary tests. PyTorch/Torchvision,
CUDA, and the HPC dataset are not available on the packaging computer, so no
actual training or GPU integration run is claimed. First confirm these files
and the original PBS settings on the HPC host.