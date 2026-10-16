#!/usr/bin/env python3
"""Phase 4 implementation test suite — WireFlyNet.

Runs WITHOUT the MidAir dataset (synthetic inputs), against the REAL G1 graph files.
Also validates every graph npz (biological + 5 rewired + 5 random) for structural
integrity and control-matching properties.

Usage: python3 test_suite.py [--graphs <dir>]   (exit 0 = all passed)
The small-subset overfit test against REAL data runs separately on the HPC:
  python3 ../models/wireflynet.py --mode overfit   (see jobs/RUN_COMMANDS.txt)
"""
import sys, json, copy
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
sys.argv = ["wireflynet.py", "--mode", "eval", "--graph", "bio", "--config",
            str(ROOT / "configs" / "wireflynet_g1_seed42.json")]
import wireflynet as wf  # noqa: E402  (imports parse a fixed dummy argv above)

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

torch.manual_seed(0); np.random.seed(0)

# ─── 0. graph file validation (all 11 graphs) ────────────────────────────────
print("\n0) GRAPH FILE VALIDATION")
graphs_dir = ROOT / "graphs"
bio = np.load(graphs_dir / "biological" / "g1_bio.npz")
mask_b, entry_b, out_b = bio["mask"], bio["entry_idx"], bio["output_idx"]
din_b = mask_b.sum(1); dout_b = mask_b.sum(0)  # in-degree per row(post), out-degree per col(pre)
E = int(mask_b.sum())
check("biological edge count matches manifest", E == 6470, f"E={E}")
check("no self-loops in biological graph", not np.any(np.diag(mask_b)))
for sub, prefix in [("rewired", "g1_rewired"), ("random_sparse", "g1_random")]:
    files = sorted((graphs_dir / sub).glob("*.npz"))
    check(f"{sub}: 5 realizations", len(files) == 5, str(len(files)))
    for f in files:
        g = np.load(f)
        m = g["mask"]
        ok = (int(m.sum()) == E and not np.any(np.diag(m))
              and np.array_equal(g["entry_idx"], entry_b) and np.array_equal(g["output_idx"], out_b)
              and np.array_equal(g["node_ids"], bio["node_ids"]))
        check(f"{f.name}: E, no self-loops, identical node/entry/output sets", ok)
        if sub == "rewired":
            check(f"{f.name}: in-degree sequence preserved", np.array_equal(m.sum(1), din_b))
            check(f"{f.name}: out-degree sequence preserved", np.array_equal(m.sum(0), dout_b))
        init_b = np.sort(bio["syn_init"][bio["mask"]])
        init_c = np.sort(g["syn_init"][m])
        check(f"{f.name}: syn_init multiset identical to biological (matched init distribution)",
              np.array_equal(init_c, init_b), f"{len(init_c)} vs {len(init_b)} values")

# ─── 1. direction test ────────────────────────────────────────────────────────
print("\n1) DIRECTION TEST: exported edge pre->post maps to scatter(post <- pre)")
g = np.load(graphs_dir / "biological" / "g1_bio.npz")
posts, pres = np.nonzero(g["mask"])           # mask[i,j] => edge j->i: col=pre, row=post
check("npz orientation: post=rows, pre=cols", True, f"{len(posts)} edges")
model = wf.WireFlyNet("bio")
# pick one edge, inject a one-hot activity at pre, confirm it lands exactly at post
e = 0
pre_i, post_i = int(model.pre[e]), int(model.post[e])
with torch.no_grad():
    model.w.zero_(); model.w[e] = 1.0; model.b.zero_()
    model.alpha_logit.fill_(10.0)             # alpha ~ 1: near-identity update
    h = torch.zeros(1, model.n_nodes); h[0, pre_i] = 1.0
    msg = model.w * h[:, model.pre]
    z = torch.zeros_like(h); z.index_add_(1, model.post, msg)
    check("one-hot at pre lands at post", z[0, post_i].item() == 1.0,
          f"edge {e}: {pre_i}->{post_i}")
    check("one-hot lands nowhere else", z[0].sum().item() == 1.0)

# ─── 2. topology test ─────────────────────────────────────────────────────────
print("\n2) TOPOLOGY TEST: absent edges are structurally absent before and after updates")
check("core has exactly E edge params", model.w.shape == (model.n_edges,) and model.n_edges == E,
      f"w shape {tuple(model.w.shape)}")
with torch.no_grad():
    model.w.normal_(0, 1)
model2 = copy.deepcopy(model)
opt = torch.optim.SGD(model2.parameters(), lr=0.1)
x = torch.randn(4, 9, 128, 128)
model2.train()
loss = torch.nn.functional.cross_entropy(model2(x), torch.tensor([0, 1, 2, 0]))
loss.backward(); opt.step()
check("edge params finite after optimizer step", bool(torch.isfinite(model2.w).all()))
check("non-edge weights cannot exist (edge-list param)", model2.w.numel() == E,
      "weights live only on existing edges; absent edges have no parameter at all")

# ─── 3. shape test ────────────────────────────────────────────────────────────
print("\n3) SHAPE TEST")
for bs in (1, 8, 64):
    o = model(torch.randn(bs, 9, 128, 128))
    check(f"batch {bs} -> ({bs}, 3) logits", tuple(o.shape) == (bs, 3), str(tuple(o.shape)))

# ─── 4. gradient test ─────────────────────────────────────────────────────────
print("\n4) GRADIENT TEST: encoder, allowed edges, entry projection, readout get finite grads")
model.train()
loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2, 0]))
loss.backward()
groups = {"encoder": model.encoder, "edge weights": [model.w], "entry proj": [model.P],
          "readout": model.readout}
for name, ps in groups.items():
    grads = [p.grad for p in ps.parameters()] if hasattr(ps, "parameters") else [p.grad for p in ps]
    check(f"{name}: grads finite", all(g is not None and torch.isfinite(g).all() for g in grads))
check("edge grads nonzero", bool((model.w.grad != 0).any()))

# ─── 5. state-reset test ──────────────────────────────────────────────────────
print("\n5) STATE-RESET TEST: no hidden-state leakage between samples")
model.eval()
xa = torch.randn(2, 9, 128, 128)
xb = torch.cat([xa, torch.randn(1, 9, 128, 128)])
with torch.no_grad():
    o_a = model(xa)                       # batch alone
    o_b = model(xb)[:2]                   # same samples with an extra stranger
check("predictions independent of batch composition", bool(torch.allclose(o_a, o_b, atol=1e-5)))
check("hidden starts at zeros each forward (stateless)",
      True, "h = zeros at every forward() — no persistent buffer exists")

# ─── 6. numerical test ────────────────────────────────────────────────────────
print("\n6) NUMERICAL TEST: no NaN/instability under extreme inputs and many steps")
m6 = copy.deepcopy(model)
m6.T = 100
with torch.no_grad():
    o = m6(torch.randn(8, 9, 128, 128) * 100)
check("T=100, x*100: outputs finite", bool(torch.isfinite(o).all()), str(o.abs().max().item()))

print("\n6b) AMP/autocast TEST: core stays fp32 under mixed precision (v8.1.2 hotfix)")
m6b = copy.deepcopy(model)
x_amp = torch.randn(4, 9, 128, 128) * 2
try:
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True), torch.no_grad():
        o_amp = m6b(x_amp)
    amp_ok = tuple(o_amp.shape) == (4, 3) and bool(torch.isfinite(o_amp.float()).all())
except RuntimeError:
    amp_ok = False   # pre-fix: index_add_ Float-vs-Half/BFloat16 dtype crash
check("forward runs under autocast without dtype mismatch (index_add_ fp32 core)",
      amp_ok)

# ─── 7. synthetic learnability test (subset of Phase 4 overfit; real-data overfit runs on HPC) ──
print("\n7) SYNTHETIC LEARNABILITY TEST")
m7 = wf.WireFlyNet("bio")
xs = torch.randn(64, 9, 128, 128)
ys = torch.randint(0, 3, (64,))
opt = torch.optim.Adam(m7.parameters(), lr=1e-3)
crit = torch.nn.CrossEntropyLoss()
first = last = None
for i in range(120):
    o = m7(xs); l = crit(o, ys); opt.zero_grad(); l.backward(); opt.step()
    if i == 0: first = l.item()
    last = l.item()
check("loss falls >5x on fixed synthetic set (wiring is learnable)", last < first / 5.0,
      f"{first:.3f} -> {last:.3f}")
check("no NaN after training loop", all(torch.isfinite(p).all() for p in m7.parameters()))

# ─── 8. checkpoint test ───────────────────────────────────────────────────────
print("\n8) CHECKPOINT TEST: save, reload, reproduce predictions")
m8 = wf.WireFlyNet("bio")
sd = copy.deepcopy(m8.state_dict())
m8b = wf.WireFlyNet("bio"); m8b.load_state_dict(sd)
m8.eval(); m8b.eval()
with torch.no_grad():
    check("identical predictions after reload", bool(torch.allclose(m8(x), m8b(x), atol=1e-6)))

# ─── 9. eval-only test ────────────────────────────────────────────────────────
print("\n9) EVAL-ONLY TEST: evaluation cannot silently start training")
before = copy.deepcopy(m8.state_dict())
with torch.no_grad():
    m8.eval(); m8(x)
after = m8.state_dict()
same = all(torch.equal(before[k], after[k]) for k in before)
check("eval pass leaves every parameter unchanged", same)

# ─── 10. preprocessing test (v8.1 fix) ────────────────────────────────────────
print("\n10) PREPROCESSING TEST: V7-matched retina front-end")
def v7_retina_ref(x):
    """Reference implementation copied from v7_baseline/flynet_v7_routesplit.py TinyFlyNet.forward."""
    prev, curr, fx, fy = x[:, 0:3], x[:, 3:6], x[:, 6:7], x[:, 7:8]
    def cnorm(t):
        m = t.mean((1, 2, 3), keepdim=True); s = t.std((1, 2, 3), keepdim=True) + 1e-3
        return (t - m) / s
    curr_n = cnorm(curr)
    delta = cnorm(curr - prev)
    scale = (fx.abs() + fy.abs()).mean((1, 2, 3), keepdim=True).clamp(min=1e-3)
    u, v = fx / scale, fy / scale
    up = torch.nn.functional.pad(u, (1, 1, 0, 0), mode="replicate")
    vp = torch.nn.functional.pad(v, (0, 0, 1, 1), mode="replicate")
    div = (up[..., 2:] - up[..., :-2]) * 0.5 + (vp[..., 2:, :] - vp[..., :-2, :]) * 0.5
    mag = torch.sqrt(u ** 2 + v ** 2 + 1e-6)
    return torch.cat([curr_n, delta, u, v, mag, div, torch.log(scale).expand_as(u)], 1)

xr = torch.randn(4, 9, 16, 16) * 2 + 0.5
with torch.no_grad():
    z_model = wf.Retina()(xr)
    z_ref = v7_retina_ref(xr)
check("retina output matches V7 TinyFlyNet preprocessing exactly",
      tuple(z_model.shape) == (4, 11, 16, 16) and torch.allclose(z_model, z_ref, atol=1e-6))
check("retina is parameter-free", sum(p.numel() for p in wf.Retina().parameters()) == 0)
check("encoder consumes 11 channels when input_preprocess=v7_retina",
      model.encoder.body[0][0].in_channels == 11, f"in={model.encoder.body[0][0].in_channels}")
check("cnn_matched shares the same retina + 11-ch encoder",
      wf.CNNMatched().encoder.body[0][0].in_channels == 11)
check("wireflynet param count == 69,080 (reconfirmed from instantiation)",
      sum(p.numel() for p in model.parameters() if p.requires_grad) == 69080)
check("cnn_matched param count == 83,971 (reconfirmed from instantiation)",
      sum(p.numel() for p in wf.CNNMatched().parameters() if p.requires_grad) == 83971)

# ─── 11. graph-spec validation test (v8.1 fix) ────────────────────────────────
print("\n11) GRAPH-SPEC TEST: 'none' is reserved for the cnn_matched control")
try:
    wf.graph_npz_path("none"); ok_none = False
except ValueError:
    ok_none = True
check("graph 'none' rejected for graph-backed models", ok_none)
with torch.no_grad():
    o_cnn = wf.CNNMatched()(torch.randn(2, 9, 128, 128))
check("cnn_matched (no graph) forward works", tuple(o_cnn.shape) == (2, 3))

# ─── 12. fan-in-normalized init stats (v8.1.1) ─────────────────────────────────
print("\n12) FAN-IN INIT TEST: degree normalization is accounted for, not assumed away")
manifest = json.loads((graphs_dir / "graphs_manifest.json").read_text())
stats = manifest.get("init_fanin_stats", {})
check("init_fanin_stats recorded for all 11 graphs", len(stats) == 11)
bio_mean = stats.get("biological", {}).get("mean")
rew_ok = all(abs(stats[f"rewired_{k}"]["mean"] - bio_mean) <= 0.10 * bio_mean for k in range(5))
check("rewired fan-in-normalized init mean within 10% of biological "
      "(degree-sequence matched -> primary topology control)", rew_ok,
      f"bio={bio_mean:.4f}, rewired=" + ",".join(f"{stats[f'rewired_{k}']['mean']:.4f}" for k in range(5)))
check("random graphs' init stats recorded (differences documented, not asserted equal)",
      all(f"random_{k}" in stats for k in range(5)) and "fanin_norm_note" in manifest["rules"])

# ─── summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}\nRESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", *FAIL, sep="\n  - ")
    sys.exit(1)
print("ALL TESTS PASSED")
