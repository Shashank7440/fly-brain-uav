#!/usr/bin/env python3
"""Build the V8 graph family: biological G1 + matched controls (Phase 5).

Controls:
  rewired    — degree-preserving directed rewiring: repeated random 2-edge swaps that
               preserve every node's exact in- and out-degree sequence, no self-loops,
               no duplicate edges. K realizations.
  random     — directed Erdos-Renyi with identical N and edge count, no self-loops. K realizations.

Each graph npz contains:
  mask        — (N,N) bool; mask[i,j]=True iff edge j -> i exists (row=post, col=pre)
  syn_init    — (N,N) float32; normalized synapse-count init (zeros where no edge)
  node_ids    — (N,) str node ordering (sorted root IDs, identical across all graphs)
  entry_idx   — indices of entry subset (LPLC2) in node ordering
  output_idx  — indices of output subset
Random graphs reuse entry/output index subsets (same counts, arbitrary nodes) and get
syn_init drawn from the SAME empirical synapse-count multiset as the biological graph
(randomly permuted across edges), so init: "synapse" initializes every control from a
matched distribution and bio-vs-control differences are attributable to TOPOLOGY alone.

Usage: python3 build_graphs.py --connectome ../connectome --k 5 --seed 20261008
"""
import csv, json, argparse, hashlib
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--connectome", default="../connectome")
ap.add_argument("--k", type=int, default=5, help="realizations per control")
ap.add_argument("--seed", type=int, default=20261008)
a = ap.parse_args()
cdir = Path(a.connectome)
rng = np.random.default_rng(a.seed)

neurons = {}
with (cdir / "g1_neurons.csv").open() as f:
    for r in csv.DictReader(f):
        neurons[r["root_id"]] = r
node_ids = sorted(neurons)
idx = {n: i for i, n in enumerate(node_ids)}
N = len(node_ids)

edges = []
with (cdir / "g1_edges.csv").open() as f:
    for r in csv.DictReader(f):
        edges.append((idx[r["pre_id"]], idx[r["post_id"]], int(r["synapses"])))
E = len(edges)

entry_idx = np.array(sorted(idx[n] for n in neurons if neurons[n].get("is_seed") in ("True", "true")))
sel = json.loads((cdir / "selection_g1.json").read_text())
out_set = set()
with (cdir / "g1_edges.csv").open() as f:
    seed_ids = {n for n in neurons if neurons[n].get("is_seed") in ("True", "true")}
    for r in csv.DictReader(f):
        if r["pre_id"] in seed_ids:
            out_set.add(idx[r["post_id"]])
output_idx = np.array(sorted(out_set))
assert len(entry_idx) == sel["n_entry"] and len(output_idx) == sel["n_output"]


def to_npz(path, edge_list, syn_counts=None):
    mask = np.zeros((N, N), dtype=bool)          # mask[i, j] = edge j -> i
    syn = np.zeros((N, N), dtype=np.float32)
    for pre, post, s in edge_list:
        if pre == post:
            raise ValueError("self-loop in graph")
        if mask[post, pre]:
            raise ValueError("duplicate edge")
        mask[post, pre] = True
        syn[post, pre] = float(s)
    if syn_counts is None:
        syn_counts = syn
    # normalize: mean edge weight 1.0 over existing edges
    init = np.zeros((N, N), dtype=np.float32)
    m = mask & (syn_counts > 0)
    if m.any():
        init[m] = syn_counts[m] / syn_counts[m].mean()
    np.savez_compressed(path, mask=mask, syn_init=init, node_ids=np.array(node_ids),
                        entry_idx=entry_idx, output_idx=output_idx)
    return mask


def edge_list_from_mask(mask):
    posts, pres = np.nonzero(mask)               # edge posts[i] <- pres[i]
    return list(zip(pres.tolist(), posts.tolist()))


def rewire(edge_list, rng, n_swaps=None):
    """Degree-preserving directed double-edge swap. In/out degree sequences preserved exactly."""
    el = [tuple(e) for e in edge_list]
    eset = set(el)
    if n_swaps is None:
        n_swaps = 20 * len(el)
    done = 0
    attempts = 0
    max_attempts = 200 * n_swaps
    while done < n_swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, len(el), size=2)
        if i == j:
            continue
        (a1, b1), (a2, b2) = el[i], el[j]
        # candidate swap: a1->b2 and a2->b1
        if a1 == b2 or a2 == b1:
            continue
        if (a1, b2) in eset or (a2, b1) in eset:
            continue
        eset.discard(el[i]); eset.discard(el[j])
        el[i], el[j] = (a1, b2), (a2, b1)
        eset.add(el[i]); eset.add(el[j])
        done += 1
    if done < n_swaps:
        raise RuntimeError("rewiring stalled (graph too rigid)")
    return el


def random_graph(n_edges, rng):
    """Directed ER, no self-loops, exact edge count by rejection sampling."""
    eset = set()
    while len(eset) < n_edges:
        need = n_edges - len(eset)
        pre = rng.integers(0, N, size=3 * need)
        post = rng.integers(0, N, size=3 * need)
        for p, q in zip(pre, post):
            if p != q:
                eset.add((int(p), int(q)))
            if len(eset) >= n_edges:
                break
    return list(eset)


manifest = {"seed": a.seed, "k": a.k, "n_nodes": N, "n_edges": E,
            "files": {}, "rules": {
                "rewired": "degree-preserving directed double-edge swaps; exact in/out degree sequences; no self-loops; no duplicates; 20*E swap attempts",
                "random": "directed Erdos-Renyi, identical N and E, no self-loops",
                "entry_output": "entry/output index subsets reused (same counts) for all graphs; random graphs assign the same LPLC2 entry set by convention (declared limitation for rewired/random: entry/output subsets are not biologically meaningful there)"}}

bio_path = Path("biological/g1_bio.npz")
to_npz(bio_path, edges)
manifest["files"]["biological"] = str(bio_path)

for k in range(a.k):
    el = edge_list_from_mask(np.load(bio_path)["mask"])
    re = rewire(el, rng)
    # sanity: degree sequences match
    din_b = np.zeros(N, int); dout_b = np.zeros(N, int)
    din_r = np.zeros(N, int); dout_r = np.zeros(N, int)
    for p, q, _s in edges: dout_b[p] += 1; din_b[q] += 1
    for p, q in re: dout_r[p] += 1; din_r[q] += 1
    assert (din_b == din_r).all() and (dout_b == dout_r).all(), "degree sequence not preserved"
    p = Path(f"rewired/g1_rewired_{k}.npz")
    # synapse init for rewired: biological synapse-count MULTISET, randomly permuted
    # across the new edge set (matched initialization distribution)
    counts = rng.permutation(np.array([s for _, _, s in edges], dtype=np.int64))
    re_sorted = sorted(re)
    to_npz(p, [(pre, post, int(c)) for (pre, post), c in zip(re_sorted, counts)])
    manifest["files"][f"rewired_{k}"] = str(p)

    ra = random_graph(E, rng)
    rc = rng.permutation(np.array([s for _, _, s in edges], dtype=np.int64))
    # same empirical synapse-count multiset, randomly assigned to random-sparse edges
    # (fix: previously uniform 1..19, which confounded topology with init distribution)
    p2 = Path(f"random_sparse/g1_random_{k}.npz")
    to_npz(p2, [(pre, post, int(c)) for (pre, post), c in zip(sorted(ra), rc)])
    manifest["files"][f"random_{k}"] = str(p2)

# ── validation: every control must carry the biological syn_init multiset ─────
bio_init = np.sort(np.load(bio_path)["syn_init"][np.load(bio_path)["mask"]])
for key, rel in manifest["files"].items():
    if key == "biological":
        continue
    g = np.load(rel)
    ctrl_init = np.sort(g["syn_init"][g["mask"]])
    assert np.allclose(ctrl_init, bio_init), f"{rel}: syn_init multiset != biological"
manifest["rules"]["syn_init"] = ("all controls (rewired + random) initialize from the biological "
                                 "synapse-count multiset, randomly permuted across edges; verified "
                                 "identical (sorted) to g1_bio.npz syn_init values")
manifest["rules"]["rewired"] += "; synapse counts randomly permuted"
manifest["rules"]["random"] += "; synapse counts randomly permuted from the biological multiset"

# ── fan-in normalization stats (v8.1.1) ──────────────────────────────────────
# The model divides each edge's syn_init by sqrt(in-degree of its post node)
# (config init_fanin_normalize=true). Matching the RAW multiset therefore does not
# guarantee identically distributed FINAL weights when degree sequences differ.
# Rewired controls share the biological in-degree sequence -> near-identical stats;
# random-sparse graphs differ by construction. These stats make the residual
# difference measurable. PRIMARY topology test: bio vs rewired (degree-matched).
# Optional ablation arm: random:k with --no-fanin-norm (raw synapse-count init).
init_stats = {}
for key, rel in manifest["files"].items():
    g = np.load(rel)
    m, s = g["mask"], g["syn_init"]
    posts, _pres = np.nonzero(m)
    indeg = m.sum(1)
    w0 = s[m] / np.sqrt(np.maximum(indeg[posts], 1.0))
    init_stats[key] = dict(mean=round(float(w0.mean()), 6), std=round(float(w0.std()), 6),
                           min=round(float(w0.min()), 6), max=round(float(w0.max()), 6))
manifest["init_fanin_stats"] = init_stats
manifest["rules"]["fanin_norm_note"] = (
    "model init divides syn_init by sqrt(post in-degree); final init distributions "
    "are near-identical for rewired (degree-sequence matched) and can differ for "
    "random (degree sequence differs) — see init_fanin_stats; rewired is the primary "
    "topology control; --no-fanin-norm provides the raw-init ablation arm")

Path("graphs_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({"nodes": N, "edges": E, "k": a.k, "files": len(manifest["files"])}))
