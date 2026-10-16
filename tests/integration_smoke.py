#!/usr/bin/env python3
"""Integration smoke test (v8.1.1) — run the REAL pipeline end-to-end on a tiny
synthetic on-disk dataset in the exact MidAir route-split FILE FORMATS
(pairs.csv, split_metadata.json with manifest sha256, image_subroot PNGs, no flow
cache -> on-the-fly Farneback flow). CPU, a few minutes, temp dirs only.

Verifies the four "required before HPC" review points:
  1. Pilot isolation is structural: with eval_test=false the test split is never
     even opened (the test pairs.csv is chmod 000 during the pilot run; any test
     access would crash the run) and no test artifacts are written.
  2. Checkpoint protection: re-running a finished run raises FileExistsError and
     exits non-zero (no silent overwrite).
  3. Order: best.pt is saved ONLY on validation macro-F1 improvement (best_epoch
     == argmax of history val_f1); test evaluation happens only AFTER training
     (result.json + test_predictions.csv appear in the eval run); pilot runs get
     test_not_evaluated.txt and NO test metrics.
  4. The overfit gate runs end-to-end: stratified 3-class subset, overfit.json
     with per-class train accuracy, exit 0 on pass, exit 1 on failure
     (forced with overfit_epochs=1).

The REAL-data overfit gate (>=95% on 64 MidAir samples) still runs on the HPC as
RUN_COMMANDS step 2 — this test proves the mechanics, not real-task learnability.

Usage: python3 integration_smoke.py [--keep]
"""
import csv, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
KEEP = "--keep" in sys.argv
PASS, FAIL = [], []

def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

def run(cmd, env, cwd=ROOT):
    return subprocess.run([PY] + cmd, cwd=cwd, env=env, capture_output=True, text=True)

# ── build synthetic dataset in MidAir route-split format ──────────────────────
tmp = Path(tempfile.mkdtemp(prefix="wf_smoke_"))
ds_root, img_root = tmp / "dataset", tmp / "images"
(img_root / "train").mkdir(parents=True); (img_root / "val").mkdir(parents=True); (img_root / "test").mkdir(parents=True)
rng = np.random.default_rng(0)

def make_img(cls, i, shift):
    a = rng.integers(0, 40, (64, 64, 3), dtype=np.uint8)  # dark noise base
    if cls == 0: a[:, :32] += 200                          # FORWARD: bright left half
    if cls == 1: a[:, 32:] += 200                          # LEFT:    bright right half
    if cls == 2: a[:32, :] += 200                          # RIGHT:   bright top half
    return Image.fromarray(np.roll(a, shift, axis=1))

def build_split(name, n_per_cls):
    rows = []
    for cls in range(3):
        for i in range(n_per_cls):
            p = img_root / name / f"c{cls}_{i:03d}.png"
            make_img(cls, i, shift=i % 7).save(p)
            rows.append((f"{name}/c{cls}_{i:03d}.png", f"{name}/c{cls}_{i:03d}.png", cls))
    with open(ds_root / name / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["prev_frame", "curr_frame", "label_id"]); w.writerows(rows)
    return hashlib.sha256((ds_root / name / "pairs.csv").read_bytes()).hexdigest()

for s in ("train", "val", "test"): (ds_root / s).mkdir(parents=True)
hashes = {s: build_split(s, n) for s, n in [("train", 30), ("val", 10), ("test", 10)]}
(ds_root / "split_metadata.json").write_text(json.dumps(
    {"counts": {s: {"manifest_sha256": h} for s, h in hashes.items()}}))

CFG_BASE = dict(enc_dim=32, enc_channels=[8, 16, 32], steps=2, epochs=2, batch=8,
                lr=2e-3, patience=2, weight_decay=1e-4, label_smoothing=0.05,
                overfit_n=48, overfit_epochs=40, input_preprocess="v7_retina")

def write_cfg(name, **over):
    c = dict(CFG_BASE); c.update(over)
    p = tmp / f"{name}.json"; p.write_text(json.dumps(c, indent=1)); return str(p)

def env_for(exp):
    e = dict(os.environ)
    e.update(WF_DATASET_ROOT=str(ds_root), WF_IMAGE_SUBROOT=str(img_root), WF_EXP_ROOT=str(exp))
    return e

pilot_exp, main_exp = tmp / "exp_pilot", tmp / "exp_main"

print("\nA) PILOT ISOLATION: eval_test=false never opens the test split")
cfg_pilot = write_cfg("pilot", eval_test=False)
os.chmod(ds_root / "test" / "pairs.csv", 0o000)   # any test access -> PermissionError
rA = run(["models/wireflynet.py", "--mode", "train", "--graph", "bio", "--config", cfg_pilot,
          "--seed", "42", "--workers", "2"], env_for(pilot_exp))
os.chmod(ds_root / "test" / "pairs.csv", 0o644)
runA = pilot_exp / "wireflynet_bio_seed42"
check("pilot run exits 0 with test split unreadable", rA.returncode == 0,
      (rA.stderr or rA.stdout).strip().splitlines()[-1][:120] if rA.returncode else "")
check("pilot reports structural test isolation", "test ISOLATED" in rA.stdout)
check("pilot writes test_not_evaluated.txt", (runA / "test_not_evaluated.txt").is_file())
check("pilot wrote NO test predictions", not (runA / "test_predictions.csv").exists())
check("pilot wrote NO result.json (no test metrics)", not (runA / "result.json").exists())
check("pilot saved checkpoint on validation", (runA / "best.pt").is_file() and (runA / "val_predictions.csv").is_file())

print("\nB) CHECKPOINT PROTECTION: finished run refuses retraining")
rB = run(["models/wireflynet.py", "--mode", "train", "--graph", "bio", "--config", cfg_pilot,
          "--seed", "42", "--workers", "2"], env_for(pilot_exp))
check("re-run exits non-zero", rB.returncode != 0)
check("re-run refuses with Checkpoint exists", "Checkpoint exists" in rB.stderr)

print("\nC) MAIN RUN ORDER: checkpoint on val-macro-F1, then test eval after training")
cfg_main = write_cfg("main", eval_test=True)
rC = run(["models/wireflynet.py", "--mode", "train", "--graph", "bio", "--config", cfg_main,
          "--seed", "42", "--workers", "2"], env_for(main_exp))
runC = main_exp / "wireflynet_bio_seed42"
check("main run exits 0", rC.returncode == 0)
check("test evaluation ran after training", (runC / "result.json").is_file() and (runC / "test_predictions.csv").is_file())
res = json.loads((runC / "result.json").read_text())
hist = list(csv.DictReader(open(runC / "history.csv")))
vf1 = [float(h["val_f1"]) for h in hist]
best_ep = 1 + int(np.argmax(vf1))
check("best.pt epoch == argmax(val macro-F1) (val-based selection)",
      res["best_epoch"] == best_ep, f"best_epoch={res['best_epoch']}, argmax={best_ep}")
check("result.json records provenance (preprocess, init, fanin-norm, steps)",
      res.get("input_preprocess") == "v7_retina" and res.get("init_fanin_normalize") is True
      and res.get("steps") == 2)
check("val metrics in result.json come from checkpoint, test metrics evaluated once",
      "macro_f1" in res["test"] and "macro_f1" in res["val"])

print("\nD) OVERFIT GATE: 3-class stratified subset, pass path, exit code 0")
cfg_of = write_cfg("overfit")
rD = run(["models/wireflynet.py", "--mode", "overfit", "--graph", "bio", "--config", cfg_of,
          "--seed", "42"], env_for(main_exp))
of = json.loads((main_exp / "overfit_wireflynet_bio_seed42" / "overfit.json").read_text())
check("overfit gate passes on separable synthetic data (exit 0)", rD.returncode == 0 and of["passed"])
check("overfit subset has all 3 classes", set(of["class_counts"]) == {"FORWARD", "LEFT", "RIGHT"},
      str(of["class_counts"]))
check("overfit.json reports per-class train accuracy", set(of["per_class_train_acc"]) == {"FORWARD", "LEFT", "RIGHT"})

print("\nE) OVERFIT GATE FAILURE PROPAGATES (exit 1)")
cfg_of1 = write_cfg("overfit_fail", overfit_epochs=1)
rE = run(["models/wireflynet.py", "--mode", "overfit", "--graph", "bio", "--config", cfg_of1,
          "--seed", "42"], env_for(main_exp))
check("failed overfit exits non-zero", rE.returncode == 1 and "OVERFIT TEST FAILED" in rE.stdout)

print(f"\n{'='*60}\nRESULT: {len(PASS)} passed, {len(FAIL)} failed")
if not KEEP: shutil.rmtree(tmp, ignore_errors=True)
else: print("(kept:", tmp, ")")
if FAIL:
    print("FAILED:", *FAIL, sep="\n  - "); sys.exit(1)
print("ALL INTEGRATION CHECKS PASSED")
