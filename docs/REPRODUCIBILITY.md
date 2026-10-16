# Reproducibility guide

## 1. Inspect archived metrics first (safe; standard Python only)
```bash
python scripts/summarize_results.py
python scripts/verify_artifacts.py
```
No model imports, checkpoint loading, GPU, dataset access, or new evaluation are required. Derived CSVs are labelled by protocol and split; original JSONs are never overwritten.

## 2. Environments
Model code uses Python, NumPy, pandas (extraction only), PyTorch, torchvision, Pillow, OpenCV and matplotlib. `requirements.txt` lists unpinned dependencies, **not** a recovered exact training environment. Install PyTorch/torchvision compatible with your CUDA driver. Exact cluster package versions were not included. Historical PBS scripts assume the author's queue, conda environment and paths; inspect/adapt copies rather than run blindly.

## 3. Dataset paths and model use
The dataset is not redistributed. You need the same fixed split manifests, split metadata, images, and correctly bound flow caches. Model code supports environment path overrides:
```bash
export WF_V8_ROOT="$PWD"
export WF_GRAPHS_DIR="$PWD/graphs"
export WF_DATASET_ROOT=/your/path/to/flyuav_midair_3class_route_split
export WF_IMAGE_SUBROOT=/your/path/to/MidAir/Kite_training
export WF_EXP_ROOT="$PWD/reproduction_runs/weighted_loss"
```
Always use a NEW output folder; never overwrite archived runs. Example **new training**, only after arranging data and compute:
```bash
python models/wireflynet.py --mode train --model wireflynet --graph bio \
  --seed 42 --config configs/wireflynet_g1_weighted_loss.json --skip-test-eval
```
The historical original config enables test evaluation. Do not use it for development against a claimed new final holdout. Original-source snapshot is preserved separately for inspection. Do not blindly rerun historical submission ledgers: existing-artifact protection and recorded job IDs are intentionally retained.

## 4. Reconstruct G1 from recovered M2 without changing the archive
```bash
mkdir -p reproduction_runs/g1
python connectome/derive_g1.py \
  --src provenance/m2_handoff/connectome --out reproduction_runs/g1
```
This produces only derived graph CSVs, not training results. Output row-order differences may change byte hashes; compare node/edge content. The verification utility checks the archived M2-to-G1 rule without rewriting graph inputs.

For source-to-M2 reproduction, first obtain and verify the original raw files named in the handoff metadata. Preserve the original script; use a separate copy in a workspace containing `raw/` and an existing `connectome/` output directory. The extraction script makes a network query for repository commit metadata. Do not claim successful raw-source reproduction merely because M2-to-G1 passes.

## 5. Checkpoints
Saved `.pt` files are supplied research artifacts. Loading pickle-based PyTorch files can execute code; load only trusted artifacts and prefer an appropriate `weights_only=True` workflow where supported. Repository preparation did not execute saved model checkpoints.

## 6. Efficiency
Parameter counts are available. Timing/energy are not established. If measured later, document hardware/software, precision policy, batch size, warm-up, synchronization, repetitions, variability and whether loading, preprocessing and flow are included. The recurrent graph core explicitly uses FP32 while the encoder may use AMP. No energy claim without power measurement.
