#!/usr/bin/env python3
"""Derive candidate circuit G1 from the M2 connectome export (pre-declared rule).

G1 rule (declared BEFORE any V8 training):
  - seed cells: all 210 LPLC2 cells, both hemispheres (is_seed=True in source export)
  - expansion: DIRECT postsynaptic partners of seeds on >=5-synapse edges
    (downstream only; upstream columnar fan-in is excluded)
  - edges: all induced edges (>=5 synapses) among selected nodes
  - entry subset: the 210 LPLC2 seeds (sensory input lands here)
  - output subset: the 445 direct downstream partners (readout taken here)

Source: murthylab/visual-system-parts-list export, FlyWire release 783.
Run from the folder containing the source neurons.csv/edges.csv:
  python3 derive_g1.py --src <source_dir> --out <this_dir>
"""
import csv, json, hashlib, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="directory with source neurons.csv/edges.csv/selection.json")
ap.add_argument("--out", default=".", help="output directory for g1_* files")
ap.add_argument("--min-synapses", type=int, default=5,
                help="synapse threshold for the pre-declared G1 rule (default 5)")
a = ap.parse_args()
src, out = Path(a.src), Path(a.out)
MIN_SYN = a.min_synapses

neurons = {}
with (src / "neurons.csv").open() as f:
    for r in csv.DictReader(f):
        neurons[r["root_id"]] = r
edges = []
with (src / "edges.csv").open() as f:
    for r in csv.DictReader(f):
        edges.append((r["pre_id"], r["post_id"], int(r["synapses"])))

seeds = {n for n, r in neurons.items() if r.get("is_seed") in ("True", "true")}
assert len(seeds) == 210, f"expected 210 LPLC2 seeds, got {len(seeds)}"
down = set()
for pre, post, syn in edges:
    if pre in seeds and syn >= MIN_SYN:   # enforce the declared threshold in code
        down.add(post)
sel = seeds | down
ind = [(p, q, s) for p, q, s in edges if p in sel and q in sel and s >= MIN_SYN]

with (out / "g1_neurons.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(next(iter(neurons.values())).keys()))
    w.writeheader()
    for n in sel:
        w.writerow(neurons[n])
with (out / "g1_edges.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pre_id", "post_id", "synapses"])
    for p, q, s in sorted(ind, key=lambda e: (-e[2], e[0], e[1])):
        w.writerow([p, q, s])

entry = sorted(seeds)
output_ = sorted(down)
node_ids = sorted(sel)
meta = {
    "graph": "G1",
    "status": "PREDECLARED_CANDIDATE_V8_FIRST_TRAINING_GRAPH",
    "source_release": 783,
    "source_repository": "https://github.com/murthylab/visual-system-parts-list",
    "seed_cells": "210 LPLC2 (both hemispheres, is_seed=True)",
    "min_synapses": MIN_SYN,
    "expansion_rule": "direct postsynaptic partners of seeds on >=5-synapse edges; upstream excluded",
    "edge_rule": "induced edges among selected nodes (>=5 synapses)",
    "n_nodes": len(sel), "n_edges": len(ind),
    "n_entry": len(entry), "n_output": len(output_),
    "entry_subset_rule": "the 210 LPLC2 seeds (sensory encoder projects here)",
    "output_subset_rule": "445 direct downstream partners (readout source)",
    "self_loops": 0, "edge_orientation": "pre_id -> post_id",
    "g1_neurons_sha256": hashlib.sha256((out / "g1_neurons.csv").read_bytes()).hexdigest(),
    "g1_edges_sha256": hashlib.sha256((out / "g1_edges.csv").read_bytes()).hexdigest(),
    "notes": ("Downstream-only expansion declared before training because the source export is "
              "dominated by LPLC2 upstream columnar fan-in (5312 neurons) with no principled "
              "engineered input interface for them; LPLC2 is the documented entry stage of this "
              "candidate circuit. This is an engineered selection, not a claim of completeness."),
}
(out / "selection_g1.json").write_text(json.dumps(meta, indent=2))
print(json.dumps({k: meta[k] for k in ["n_nodes", "n_edges", "n_entry", "n_output"]}, indent=2))
