#!/usr/bin/env python3
"""Prepare/verify the frozen 30-control plan; never train or open test data."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/Data4/mi_25026207/drone/v8_wireflynet")
REFERENCE = ROOT / "v8_weighted_controls_package" / "protocol_reference.json"
ARMS = [(f"{kind}:{k}", seed) for kind in ["rewired", "random"] for k in range(5) for seed in [42,43,44]]
ARTIFACTS = ["best.pt", "history.csv", "run_metadata.json", "validation_result.json",
             "result.json", "test_predictions.csv", ".weighted_control_lock", "control_job_manifest.json"]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run_name(graph, seed):
    if (graph, int(seed)) not in ARMS:
        raise ValueError("Only rewired:0..4/random:0..4, seeds42/43/44 are allowed")
    return f"wireflynet_{graph.replace(':','')}_seed{int(seed)}"

def graph_path(directory, graph):
    kind, k = graph.split(":")
    return directory / ("rewired" if kind == "rewired" else "random_sparse") / f"g1_{kind}_{k}.npz"

def paths():
    out = ROOT / "experiments_weighted_loss"
    return out, out / "_control_submission" / "inventory.json"

def protocol():
    out, _ = paths()
    cfgpath = ROOT / "configs/wireflynet_g1_weighted_loss.json"
    source = ROOT / "models/wireflynet.py"
    cfg = json.loads(cfgpath.read_text())
    ref = json.loads(REFERENCE.read_text())
    if digest(cfgpath) != ref["config_sha256"] or digest(source) != ref["script_sha256"]:
        raise RuntimeError("Config/model script differs from the completed six-run weighted pilot. Stop and review.")
    if Path(cfg["exp_root"]).resolve() != out.resolve():
        raise RuntimeError("Unexpected weighted output root")
    for key, value in dict(eval_test=False, train_sampler="shuffle", class_weighting="inverse_frequency",
                           epochs=40, batch=64, lr=0.002, patience=12, init_fanin_normalize=True).items():
        if cfg.get(key) != value:
            raise RuntimeError(f"Unexpected frozen setting: {key}={cfg.get(key)}")
    if cfg.get("train_limit") is not None:
        raise RuntimeError("Controls must use the full training set, not a training subset")
    for env, key in {
        "WF_EXP_ROOT":"exp_root", "WF_DATASET_ROOT":"dataset_root",
        "WF_IMAGE_SUBROOT":"image_subroot", "WF_GRAPHS_DIR":"graphs_dir", "WF_V8_ROOT":"v8_root"
    }.items():
        if os.environ.get(env) and Path(os.environ[env]).resolve() != Path(cfg[key]).resolve():
            raise RuntimeError(f"{env} overrides the frozen config. Unset/review it before preparing controls.")
    return cfg, cfgpath, source

def verify():
    """Check a prepared inventory; safe after some jobs have started."""
    cfg, _, _ = protocol()
    _, inventory = paths()
    plan = json.loads(inventory.read_text())
    if plan["runs"] != [dict(graph=g, seed=s, run=run_name(g,s)) for g,s in ARMS]:
        raise RuntimeError("Unexpected control plan")
    for filename, expected in plan["sha256"].items():
        if not Path(filename).is_file() or digest(filename) != expected:
            raise RuntimeError(f"Frozen input changed/missing while controls were queued: {filename}")
    return plan

def prepare():
    cfg, cfgpath, source = protocol()
    out, inventory = paths()
    if inventory.exists():
        verify()
        print("Existing control inventory verified. Use the submission helper to resume; do not delete the ledger.")
        return
    files = [cfgpath, source]
    data, images, graphs = map(Path, [cfg["dataset_root"], cfg["image_subroot"], cfg["graphs_dir"]])
    metadata_path = data / "split_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    files.append(metadata_path)
    hashes = {}
    for split in ["train", "val"]:  # No test manifest/predictions/cache is opened.
        p = data / split / "pairs.csv"
        hashes[split] = digest(p)
        if hashes[split] != metadata["counts"][split]["manifest_sha256"]:
            raise RuntimeError(f"Changed {split} manifest")
        files.append(p)
        cache = data / split / "flow"
        if cache.exists():
            marker = cache / "cache_metadata.json"
            cm = json.loads(marker.read_text())
            if cm["manifest_sha256"] != hashes[split] or cm["image_root"] != str(images.resolve()):
                raise RuntimeError(f"Stale {split} flow-cache metadata")
            files.append(marker)

    # The six completed pilot runs establish the exact source/config/data/weights.
    common_weights = None
    for model, graph in [("wireflynet","bio"), ("cnn_matched","none")]:
        for seed in [42,43,44]:
            folder = out / f"{model}_{graph}_seed{seed}"
            mp, rp = folder / "run_metadata.json", folder / "validation_result.json"
            m, r = json.loads(mp.read_text()), json.loads(rp.read_text())
            if m["config_sha256"] != digest(cfgpath) or m["script_sha256"] != digest(source) or m.get("config") != cfg:
                raise RuntimeError(f"Pilot source/config mismatch: {folder}")
            if m["eval_test"] or m["train_sampler"] != "shuffle" or m["class_weighting"] != "inverse_frequency":
                raise RuntimeError(f"Pilot protocol mismatch: {folder}")
            args = m["resolved_arguments"]
            expected_args = dict(model=model, graph=graph, seed=seed, epochs=40, batch=64,
                                 lr=0.002, patience=12, workers=4)
            if any(args.get(k) != v for k,v in expected_args.items()):
                raise RuntimeError(f"Pilot CLI/budget mismatch: {folder}")
            if m["train_manifest_sha256"] != hashes["train"] or m["val_manifest_sha256"] != hashes["val"]:
                raise RuntimeError(f"Pilot data split mismatch: {folder}")
            if common_weights is None:
                common_weights = m["loss_class_weights"]
            elif m["loss_class_weights"] != common_weights:
                raise RuntimeError("Pilot class weights differ across runs")
            if r["run"] != folder.name or r["seed"] != seed or r["model"] != model or r["graph"] != graph:
                raise RuntimeError(f"Pilot identity mismatch: {folder}")
            if (folder/"test_predictions.csv").exists() or (folder/"result.json").exists():
                raise RuntimeError(f"Pilot contains test outputs; review before proceeding: {folder}")
            files.extend([mp,rp])

    import numpy as np
    bio_path = graphs / "biological/g1_bio.npz"
    with np.load(bio_path, allow_pickle=False) as bio:
        bm = np.asarray(bio["mask"], dtype=bool)
        be, bo = np.array(bio["entry_idx"]), np.array(bio["output_idx"])
    if bm.ndim != 2 or bm.shape[0] != bm.shape[1]:
        raise RuntimeError("Invalid biological graph shape")
    n, edges = bm.shape[0], int(bm.sum())
    files.append(bio_path)
    graph_info = []
    for kind in ["rewired","random"]:
        for k in range(5):
            graph = f"{kind}:{k}"
            path = graph_path(graphs, graph)
            with np.load(path, allow_pickle=False) as g:
                mask, syn = np.asarray(g["mask"], dtype=bool), np.asarray(g["syn_init"])
                entry, output = np.array(g["entry_idx"]), np.array(g["output_idx"])
            if mask.shape != bm.shape or int(mask.sum()) != edges or syn.shape != mask.shape:
                raise RuntimeError(f"Graph node/edge/initialization shape mismatch: {path}")
            if not np.array_equal(entry, be) or not np.array_equal(output, bo):
                raise RuntimeError(f"Entry/output mapping differs from Bio: {path}")
            if not np.isfinite(syn[mask]).all():
                raise RuntimeError(f"Nonfinite initial edge weights: {path}")
            graph_info.append(dict(graph=graph, nodes=n, edges=edges, sha256=digest(path)))
            files.append(path)
    for graph, seed in ARMS:
        folder = out / run_name(graph,seed)
        if any((folder/a).exists() for a in ARTIFACTS):
            raise RuntimeError(f"A control run already exists/started: {folder}. Stop; do not duplicate it.")
    plan = dict(
        runs=[dict(graph=g,seed=s,run=run_name(g,s)) for g,s in ARMS],
        sha256={str(p):digest(p) for p in files}, graphs=graph_info,
        loss_class_weights=common_weights,
        note="Same weighted pilot protocol; only train/val data inspected; no graph regeneration"
    )
    inventory.parent.mkdir(parents=True, exist_ok=True)
    with inventory.open("x") as f:
        json.dump(plan,f,indent=2)
    print("PREFLIGHT PASSED: six pilot runs match; 10 existing graphs verified; 30 fresh control targets.")
    print(f"Graph geometry: N={n}, E={edges}; matching Bio entry/output mappings.")
    print(f"Class weights: {common_weights}")
    print(f"Saved frozen inventory: {inventory}")
    print("No model training, graph rebuilding, or test evaluation was performed.")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="Verify prepared inventory without requiring fresh targets")
    args = ap.parse_args()
    if args.verify:
        verify()
        print("Frozen control inventory verified.")
    else:
        prepare()

if __name__ == "__main__":
    main()