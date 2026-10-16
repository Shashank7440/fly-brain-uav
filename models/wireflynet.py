#!/usr/bin/env python3
"""
wireflynet.py — V8: FlyWire-constrained visual action network ("WireFlyNet", provisional name)
=========================================================================================
Question: does a selected biological visual subconnectome (G1: LPLC2-centered, FlyWire r783)
improve action prediction / robustness / learning efficiency vs MATCHED non-biological graphs?
Better accuracy is a possible outcome, not an assumption. A negative result will be reported.

Model (Phase 3, engineered-to-biology interface; all assumptions documented in ../connectome/provenance.md):
  9-ch input [prev_rgb|curr_rgb|flow_x|flow_y|flow_mag] (IDENTICAL to V7)
    -> V7 retina preprocessing (per-sample contrast-norm RGB, temporal delta,
       relative flow, divergence; reproduced verbatim from flynet_v7_routesplit.py
       TinyFlyNet, parameter-free, -> 11 channels)
    -> small shared sensory encoder (CNN)
    -> input projection P restricted to the 210 LPLC2 entry cells
    -> edge-list recurrent core, T steps:
         h_i^{t+1} = (1-alpha_i) h_i^t + alpha_i * phi( sum_{j->i in G} w_e h_j^t + b_i + (P x)_i )
       only observed edges carry learnable weights (structurally absent elsewhere)
    -> readout from the 445 documented output cells (direct LPLC2 targets)
    -> FORWARD/LEFT/RIGHT logits

Controls (same encoder, entry/output counts, dynamics, training):
  --graph bio | rewired:k | random:k    (built by ../graphs/build_graphs.py)
  --model cnn_matched                   (encoder + linear head, comparable budget)
V7 results are the frozen baseline (route-grouped split; do not retrain V7).

Recipe (matched to V7 for comparability): AdamW + OneCycle, label smoothing 0.05,
AMP, balanced sampler, EMA 0.999, checkpoint on VALIDATION macro-F1, early stop.
Dataset/flow caches: SAME corrected route-grouped split as V7 (caches reused, hashes checked).

Modes:  --mode train | eval | compare | overfit
Config keys added in the v8.1 review fix: input_preprocess ("v7_retina"|"raw"),
eval_test (bool), train_limit_mode ("stratified"|"head").
Usage examples in ../jobs/RUN_COMMANDS.txt
"""
import argparse, csv, json, math, random, time, copy
import os, sys, hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
cv2.setNumThreads(0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── args ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["train", "eval", "compare", "overfit"])
ap.add_argument("--graph", default="bio", help="bio | rewired:k | random:k (e.g. rewired:0)")
ap.add_argument("--skip-test-eval", action="store_true",
                help="train without touching the test set (pilot protocol; config eval_test=false does the same)")
ap.add_argument("--no-fanin-norm", action="store_true",
                help="disable fan-in normalization of synapse init (init-distribution ablation arm; see docs/review_fixes_v8.1.md)")
ap.add_argument("--model", default="wireflynet", choices=["wireflynet", "cnn_matched"])
ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "wireflynet_g1_seed42.json"))
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--batch", type=int, default=None)
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--patience", type=int, default=None)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()

CFG = json.loads(Path(args.config).read_text())
if args.no_fanin_norm:
    CFG["init_fanin_normalize"] = False   # ablation arm: raw synapse-count init
for k, dflt in [("epochs", CFG.get("epochs", 40)), ("batch", CFG.get("batch", 64)),
                ("lr", CFG.get("lr", 2e-3)), ("patience", CFG.get("patience", 12))]:
    if getattr(args, k) is None:
        setattr(args, k, dflt)

# ─── paths (config-driven; env-overridable; package-relative fallbacks) ────────
# Portability fix (v8.1): absolute /Data4/... paths stay as HPC defaults, but env vars
# WF_DATASET_ROOT / WF_IMAGE_SUBROOT / WF_EXP_ROOT / WF_V8_ROOT override them, and the
# graphs dir falls back to this package's graphs/ when the configured path is absent,
# so tests and local runs work off-HPC without editing configs.
_PKG_ROOT = Path(__file__).resolve().parents[1]
V8_ROOT = Path(os.environ.get("WF_V8_ROOT") or CFG.get("v8_root") or str(_PKG_ROOT))
_ds = os.environ.get("WF_DATASET_ROOT") or CFG.get("dataset_root")
_img = os.environ.get("WF_IMAGE_SUBROOT") or CFG.get("image_subroot")
if not _ds or not _img:
    raise RuntimeError("dataset_root/image_subroot must be set in the config or via "
                       "WF_DATASET_ROOT / WF_IMAGE_SUBROOT environment variables")
DATASET_ROOT = Path(_ds)
IMG_SUBROOT = Path(_img)
_CFG_GRAPHS = CFG.get("graphs_dir")
GRAPHS_DIR = Path(os.environ.get("WF_GRAPHS_DIR",
                                 _CFG_GRAPHS if _CFG_GRAPHS and Path(_CFG_GRAPHS).is_dir()
                                 else str(_PKG_ROOT / "graphs")))
EXP_ROOT = Path(os.environ.get("WF_EXP_ROOT", CFG.get("exp_root", str(V8_ROOT / "experiments"))))
IMG_SIZE, NUM_CLS, IN_CH = 128, 3, 9
INPUT_PRE = CFG.get("input_preprocess", "v7_retina")   # "v7_retina" (default) | "raw"
ENC_IN = 9 if INPUT_PRE == "raw" else 11
CLASSES = ["FORWARD", "LEFT", "RIGHT"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Follow-up ablation: defaults retain the original sampler and unweighted loss.
# Share the existing flip probability with the weighting calculation, so LEFT
# and RIGHT weights reflect the labels actually expected after augmentation.
TRAIN_HFLIP_P = 0.5


def validate_balancing_config(cfg):
    sampler = cfg.get("train_sampler", "balanced")
    weighting = cfg.get("class_weighting", "none")
    if sampler not in {"balanced", "shuffle"}:
        raise ValueError("train_sampler must be 'balanced' or 'shuffle'")
    if weighting not in {"none", "inverse_frequency"}:
        raise ValueError("class_weighting must be 'none' or 'inverse_frequency'")
    if sampler == "balanced" and weighting != "none":
        raise ValueError("Do not combine balanced sampling with class-weighted loss. "
                         "Set train_sampler='shuffle' for the weighted-loss arm.")
    return sampler, weighting


def training_loss_weights(labels, mode, flip_probability=TRAIN_HFLIP_P):
    """Derive weights ONLY from selected training labels; never val/test.

    The existing training augmentation swaps LEFT/RIGHT on horizontal flips.
    Use expected augmented counts, not a test-driven direction preference.
    Counts are expectations, not realized counts in a finite training epoch.
    """
    if mode not in {"none", "inverse_frequency"}:
        raise ValueError("Unknown class weighting mode")
    if not 0 <= flip_probability <= 1:
        raise ValueError("Invalid horizontal flip probability")
    cnt = Counter(labels)
    if any(label not in range(NUM_CLS) for label in cnt):
        raise ValueError("Training labels must be 0=FORWARD, 1=LEFT, 2=RIGHT")
    raw = [int(cnt.get(i, 0)) for i in range(NUM_CLS)]
    p = flip_probability
    expected = [float(raw[0]), (1-p)*raw[1] + p*raw[2],
                (1-p)*raw[2] + p*raw[1]]
    if mode == "none":
        return None, raw, expected
    if any(n <= 0 for n in expected):
        raise ValueError("Weighted loss requires positive expected training support for every class")
    total = sum(expected)
    weights = [total / (NUM_CLS * n) for n in expected]
    return weights, raw, expected


TRAIN_SAMPLER, CLASS_WEIGHTING = validate_balancing_config(CFG)
if CLASS_WEIGHTING != "none" and EXP_ROOT.resolve() == (V8_ROOT / "experiments").resolve():
    raise ValueError("Weighted-loss runs must not use the frozen baseline experiments/ "
                     "directory. Check exp_root and any WF_EXP_ROOT override.")


def _worker_init(_):
    torch.set_num_threads(1); cv2.setNumThreads(0)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def stratified_indices(labels, n, seed, min_per_class=2):
    """Reproducible class-aware subset: every class is guaranteed >= min_per_class
    representatives (if present), remainder proportional to class frequency.
    v8.1 fix: head-slicing a manifest can be route/class-biased."""
    rng = random.Random(seed)
    by_class = {}
    for i, l in enumerate(labels):
        by_class.setdefault(l, []).append(i)
    total = len(labels)
    sel = []
    for l, idxs in by_class.items():
        k = min(len(idxs), max(min_per_class, round(n * len(idxs) / total)))
        sel += rng.sample(idxs, k)
    sel.sort()
    return sel


def graph_npz_path(spec):
    if spec == "bio":
        return GRAPHS_DIR / "biological" / "g1_bio.npz"
    if spec == "none":
        raise ValueError("--graph none is only valid with --model cnn_matched (no-graph control)")
    kind, k = spec.split(":")
    sub = "rewired" if kind == "rewired" else "random_sparse"
    return GRAPHS_DIR / sub / f"g1_{'rewired' if kind == 'rewired' else 'random'}_{k}.npz"


def run_name():
    return f"{args.model}_{args.graph.replace(':', '')}_seed{args.seed}"


# ═════════════════════════════════════════════════════════════════════════════
# Data — identical pipeline to V7 (copied verbatim so the interface is matched)
# ═════════════════════════════════════════════════════════════════════════════
class MidAirDataset(Dataset):
    FB = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    def __init__(self, split, augment=False, limit=None, stratified=False, seed=42):
        self.flow_dir = DATASET_ROOT / split / "flow"
        self.augment = augment
        self.rows = []
        with open(DATASET_ROOT / split / "pairs.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append((IMG_SUBROOT / r["prev_frame"], IMG_SUBROOT / r["curr_frame"], int(r["label_id"])))
        # subset selection keeps ORIGINAL row indices (flow cache files are keyed
        # by pairs.csv row index; re-slicing rows would silently mismatch caches)
        if limit is not None:
            if stratified:
                self.sel = stratified_indices([r[2] for r in self.rows], limit, seed)
            else:
                self.sel = list(range(min(limit, len(self.rows))))
        else:
            self.sel = list(range(len(self.rows)))
        self.resize = T.Resize((IMG_SIZE, IMG_SIZE), antialias=True)

    def __len__(self): return len(self.sel)
    def labels(self): return [self.rows[j][2] for j in self.sel]

    def _rgb(self, p): return TF.to_tensor(self.resize(Image.open(p).convert("RGB")))

    def _flow(self, idx, pp, cp):
        fp = self.flow_dir / f"{idx:06d}.npz"
        if fp.exists():
            d = np.load(fp)
            return (torch.from_numpy(d["flow_x"].astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(d["flow_y"].astype(np.float32)).unsqueeze(0))
        g = lambda p: np.array(Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.uint8)
        f = cv2.calcOpticalFlowFarneback(g(pp), g(cp), None, **self.FB)
        return torch.from_numpy(f[..., 0]).unsqueeze(0), torch.from_numpy(f[..., 1]).unsqueeze(0)

    def __getitem__(self, i):
        j = self.sel[i]                          # original pairs.csv row (flow-cache key)
        pp, cp, label = self.rows[j]
        prev, curr = self._rgb(pp), self._rgb(cp)
        fx, fy = self._flow(j, pp, cp)
        if self.augment:
            b, c, s = random.uniform(.7, 1.3), random.uniform(.7, 1.3), random.uniform(.7, 1.3)
            def photo(x):
                x = TF.adjust_brightness(x, b); x = TF.adjust_contrast(x, c)
                return TF.adjust_saturation(x, s)
            prev, curr = photo(prev), photo(curr)
            if random.random() < 0.2:
                k = random.uniform(0.2, 0.6); prev, curr = prev * k, curr * k
            if random.random() < 0.2:
                prev = (prev + 0.05 * torch.randn_like(prev)).clamp(0, 1)
                curr = (curr + 0.05 * torch.randn_like(curr)).clamp(0, 1)
            if random.random() < TRAIN_HFLIP_P:
                prev, curr = prev.flip(2), curr.flip(2)
                fx, fy = -fx.flip(2), fy.flip(2)
                label = {1: 2, 2: 1}.get(label, label)
            if random.random() < 0.25:
                h, w = random.randint(16, 48), random.randint(16, 48)
                y0, x0 = random.randint(0, IMG_SIZE - h), random.randint(0, IMG_SIZE - w)
                for t in (prev, curr, fx, fy): t[:, y0:y0 + h, x0:x0 + w] = 0
        mag = torch.sqrt(fx ** 2 + fy ** 2)
        return torch.cat([prev, curr, fx, fy, mag], 0), label


def make_loader(split, augment=False, limit=None, stratified=None, seed=42):
    import hashlib
    metadata_path = DATASET_ROOT / "split_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("Missing corrected split metadata; the V7 route split must exist first")
    metadata = json.loads(metadata_path.read_text())
    manifest_hash = hashlib.sha256((DATASET_ROOT / split / "pairs.csv").read_bytes()).hexdigest()
    if manifest_hash != metadata["counts"][split]["manifest_sha256"]:
        raise RuntimeError("Manifest changed since route split preparation")
    cache = DATASET_ROOT / split / "flow"
    if cache.exists():
        marker = cache / "cache_metadata.json"
        if not marker.is_file():
            raise RuntimeError("Unbound flow cache; refusing old indexed cache")
        cache_meta = json.loads(marker.read_text())
        if cache_meta["manifest_sha256"] != manifest_hash or cache_meta["image_root"] != str(IMG_SUBROOT.resolve()):
            raise RuntimeError("Stale flow cache metadata")
    if stratified is None:
        stratified = CFG.get("train_limit_mode", "head") == "stratified"
    ds = MidAirDataset(split, augment, limit, stratified=stratified, seed=seed)
    if split == "train" and augment:
        kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                  worker_init_fn=_worker_init, persistent_workers=args.workers > 0)
        if TRAIN_SAMPLER == "balanced":
            cnt = Counter(ds.labels())
            w = [1.0 / cnt[l] for l in ds.labels()]
            return DataLoader(ds, sampler=WeightedRandomSampler(w, len(w), replacement=True), drop_last=True, **kw), ds
        # Ordinary shuffled sampling without replacement; augmentation/drop_last
        # are unchanged. Class emphasis is supplied by the weighted loss instead.
        return DataLoader(ds, shuffle=True, drop_last=True, **kw), ds
    return DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=2, pin_memory=False), ds


# ═════════════════════════════════════════════════════════════════════════════
# Model
# ═════════════════════════════════════════════════════════════════════════════
class Retina(nn.Module):
    """V7 TinyFlyNet front-end preprocessing, reproduced VERBATIM from
    v7_baseline/flynet_v7_routesplit.py (parameter-free):
      photoreceptor gain control (per-sample contrast norm on RGB),
      temporal channel (curr - prev), relative flow (/ mean |flow|),
      flow divergence (looming cue). -> 11 channels.
    v8.1 fix: without this, V8's encoder saw raw 9-ch tensors while V7 TinyFlyNet
    saw these 11 processed channels, so V7-vs-V8 differences were not attributable
    to the graph alone. Applies identically to wireflynet AND cnn_matched."""
    @staticmethod
    def _cnorm(x):
        m = x.mean((1, 2, 3), keepdim=True); s = x.std((1, 2, 3), keepdim=True) + 1e-3
        return (x - m) / s

    @staticmethod
    def _divergence(u, v):
        up = F.pad(u, (1, 1, 0, 0), mode="replicate")
        vp = F.pad(v, (0, 0, 1, 1), mode="replicate")
        dudx = (up[..., 2:] - up[..., :-2]) * 0.5
        dvdy = (vp[..., 2:, :] - vp[..., :-2, :]) * 0.5
        return dudx + dvdy

    def forward(self, x):
        prev, curr, fx, fy = x[:, 0:3], x[:, 3:6], x[:, 6:7], x[:, 7:8]
        curr_n = self._cnorm(curr)
        delta = self._cnorm(curr - prev)
        scale = (fx.abs() + fy.abs()).mean((1, 2, 3), keepdim=True).clamp(min=1e-3)
        u, v = fx / scale, fy / scale
        div = self._divergence(u, v)
        mag = torch.sqrt(u ** 2 + v ** 2 + 1e-6)
        return torch.cat([curr_n, delta, u, v, mag, div, torch.log(scale).expand_as(u)], 1)


class Encoder(nn.Module):
    """Small shared sensory encoder — IDENTICAL for wireflynet and cnn_matched.
    An engineered feature extractor, not a reconstructed fly retina."""
    def __init__(self, enc_dim=128):
        super().__init__()
        ch = CFG.get("enc_channels", [16, 32, 64])
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.body = nn.Sequential(blk(ENC_IN, ch[0]), blk(ch[0], ch[1]), blk(ch[1], ch[2]),
                                  nn.Conv2d(ch[2], enc_dim, 1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.out_dim = enc_dim

    def forward(self, x):
        return self.body(x).flatten(1)


class WireFlyNet(nn.Module):
    """Edge-list rate-based recurrent core constrained by the loaded graph.
    Only existing edges have parameters; absent edges are structurally absent."""
    def __init__(self, graph_spec="bio"):
        super().__init__()
        g = np.load(graph_npz_path(graph_spec))
        mask, syn = g["mask"], g["syn_init"]           # mask[i,j]=True iff edge j->i
        entry = torch.from_numpy(g["entry_idx"].astype(np.int64))
        output = torch.from_numpy(g["output_idx"].astype(np.int64))
        posts, pres = np.nonzero(mask)                 # edges: posts[r] <- pres[r]
        order = np.argsort(posts)                     # group by post for scatter_add
        self.register_buffer("pre", torch.from_numpy(pres[order].astype(np.int64)))
        self.register_buffer("post", torch.from_numpy(posts[order].astype(np.int64)))
        self.register_buffer("entry", entry)
        self.register_buffer("output", output)
        E, N = len(self.pre), mask.shape[0]
        self.n_nodes, self.n_edges = N, E

        self.encoder = Encoder()
        self.retina = Retina() if INPUT_PRE != "raw" else nn.Identity()
        D = self.encoder.out_dim
        self.P = nn.Parameter(torch.empty(len(entry), D))
        nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.b = nn.Parameter(torch.zeros(N))

        indeg = torch.zeros(N); indeg.index_add_(0, self.post, torch.ones(E))
        if CFG.get("init", "synapse") == "synapse":
            w0 = torch.from_numpy(syn[posts, pres][order]).float()
        else:  # controlled random init, matched fan-in scale
            w0 = torch.randn(E)
        if CFG.get("init_fanin_normalize", True):  # fan-in stability normalization
            w0 = w0 / torch.sqrt(indeg[self.post].clamp(min=1.0))
        self.w = nn.Parameter(w0)

        if CFG.get("alpha_mode", "learnable") == "learnable":
            self.alpha_logit = nn.Parameter(torch.full((N,), float(CFG.get("alpha_init_logit", 2.0))))
        else:
            self.register_buffer("alpha_logit", torch.full((N,), float(CFG.get("alpha_init_logit", 2.0))))
        self.T = CFG.get("steps", 4)
        self.phi = {"tanh": torch.tanh, "relu": F.relu}[CFG.get("phi", "tanh")]

        self.readout = nn.Linear(len(output), NUM_CLS)  # documented output subset
        self.graph_spec = graph_spec

    def forward(self, x):
        x = self.retina(x)
        z_enc = self.encoder(x)                                  # (B,D)
        B = x.shape[0]
        # v8.1.2 hotfix: run the recurrent core in fp32 with autocast DISABLED.
        # Under CUDA autocast the P @ z_enc matmul returns fp16 while state h is
        # fp32, and index_add_ requires matching dtypes (crash on GPU, invisible
        # on CPU). The core is tiny (scatter-add over E edges) - fp32 costs
        # nothing and is more stable; the CNN encoder still gets AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            z_enc = z_enc.float()
            h = x.new_zeros(B, self.n_nodes)                      # state reset every forward - no leakage
            alpha = torch.sigmoid(self.alpha_logit)
            drive = (self.P @ z_enc.t()).t()                      # (B, n_entry)
            for _ in range(self.T):
                msg = self.w * h[:, self.pre]                     # (B,E) only existing edges
                z = torch.zeros_like(h)
                z.index_add_(1, self.post, msg)                   # recurrent sum
                z = z + self.b
                z.index_add_(1, self.entry, drive)                # input projection into entry cells
                h = (1 - alpha) * h + alpha * self.phi(z)
        return self.readout(h[:, self.output])


class CNNMatched(nn.Module):
    """Phase 5 control: the SAME encoder + MLP head sized to a comparable parameter
    budget (cnn_head_hidden in config); no recurrent core. Tests whether a comparable
    budget without any graph explains the result."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.retina = Retina() if INPUT_PRE != "raw" else nn.Identity()
        h = CFG.get("cnn_head_hidden", 384)
        self.head = nn.Sequential(nn.Linear(self.encoder.out_dim, h), nn.ReLU(inplace=True),
                                  nn.Linear(h, NUM_CLS))

    def forward(self, x):
        x = self.retina(x)
        return self.head(self.encoder(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_model():
    if args.model == "cnn_matched":
        return CNNMatched()
    if args.graph == "none":
        raise ValueError("--graph none is only valid with --model cnn_matched; "
                         "wireflynet requires an actual graph (bio | rewired:k | random:k)")
    return WireFlyNet(args.graph)


# ═════════════════════════════════════════════════════════════════════════════
# Metrics / EMA / plots (V7 protocol)
# ═════════════════════════════════════════════════════════════════════════════
def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    cm = np.zeros((NUM_CLS, NUM_CLS), int)
    for t, q in zip(y, p): cm[t, q] += 1
    acc = (p == y).mean()
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    pre = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = 2 * pre * rec / np.maximum(pre + rec, 1e-9)
    macro_f1 = f1.mean() if (pre + rec > 0).any() else 0.0
    return dict(accuracy=round(100 * acc, 3), balanced_accuracy=round(100 * rec.mean(), 3),
                macro_f1=round(f1.mean(), 4),
                per_class={CLASSES[i]: dict(recall=round(float(rec[i]), 4), precision=round(float(pre[i]), 4),
                                            f1=round(float(f1[i]), 4), support=int(cm[i].sum())) for i in range(NUM_CLS)},
                confusion=cm.tolist())


class EMA:
    def __init__(self, model, decay=0.999):
        self.m = copy.deepcopy(model).eval(); self.d = decay
        for p in self.m.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point: e.mul_(self.d).add_(p.detach(), alpha=1 - self.d)
            else: e.copy_(p)


def plot_curves(H, out, title):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ep = H["epoch"]
    ax[0].plot(ep, H["train_loss"], label="train"); ax[0].plot(ep, H["val_loss"], label="val"); ax[0].set_title("Loss")
    ax[1].plot(ep, H["train_acc"], label="train"); ax[1].plot(ep, H["val_acc"], label="val"); ax[1].set_title("Accuracy (%)")
    ax[2].plot(ep, H["val_f1"], c="g"); ax[2].set_title("Val macro-F1")
    for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend() if a.get_legend_handles_labels()[0] else None
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close()


def plot_cm(cm, out, title):
    cm = np.array(cm); fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLS):
        for j in range(NUM_CLS):
            ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / max(cm[i].sum(), 1):.1%}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(NUM_CLS), CLASSES); ax.set_yticks(range(NUM_CLS), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# Train / eval
# ═════════════════════════════════════════════════════════════════════════════
def train():
    set_seed(args.seed)
    out = EXP_ROOT / run_name(); out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise FileExistsError(f"Checkpoint exists: {out / 'best.pt'}. Refusing to retrain/overwrite; use --mode eval.")
    limit = CFG.get("train_limit")
    tr_loader, tr_ds = make_loader("train", augment=True, limit=limit)   # stratified if config says so
    loss_weights, raw_counts, expected_counts = training_loss_weights(
        tr_ds.labels(), CLASS_WEIGHTING, flip_probability=TRAIN_HFLIP_P)
    va_loader, va_ds = make_loader("val")
    # v8.1.1: in pilot mode (eval_test=false) the test loader is NEVER constructed —
    # the test split is not even opened, so pilot isolation is structural, not conventional.
    do_eval_test = CFG.get("eval_test", True) and not args.skip_test_eval
    te_loader, _ = (make_loader("test") if do_eval_test else (None, None))

    model = build_model().to(DEVICE)
    n_params = count_params(model)
    ginfo = f" | graph {args.graph} N={getattr(model, 'n_nodes', '-')} E={getattr(model, 'n_edges', '-')}" if args.model == "wireflynet" else ""
    print(f"\n{run_name()}: {n_params:,} params{ginfo} | train {len(tr_loader.dataset)} val {len(va_ds)}"
          + (f" test {len(te_loader.dataset)}" if do_eval_test else " test ISOLATED (pilot: eval_test=false)"))

    run_metadata = dict(
        run=run_name(), config=copy.deepcopy(CFG), resolved_arguments=vars(args).copy(),
        train_sampler=TRAIN_SAMPLER, class_weighting=CLASS_WEIGHTING,
        raw_training_class_counts=dict(zip(CLASSES, raw_counts)),
        expected_augmented_training_class_counts=dict(zip(CLASSES, expected_counts)),
        horizontal_flip_probability=TRAIN_HFLIP_P,
        loss_class_weights=None if loss_weights is None else dict(zip(CLASSES, loss_weights)),
        training_rows=len(tr_ds),
        config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        train_manifest_sha256=hashlib.sha256((DATASET_ROOT / "train" / "pairs.csv").read_bytes()).hexdigest(),
        val_manifest_sha256=hashlib.sha256((DATASET_ROOT / "val" / "pairs.csv").read_bytes()).hexdigest(),
        eval_test=bool(do_eval_test))
    (out / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))
    print(f"  sampler={TRAIN_SAMPLER} | class_weighting={CLASS_WEIGHTING} | "
          f"train counts={raw_counts} | expected post-flip counts={expected_counts} | "
          f"loss weights={loss_weights}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=CFG.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr_loader), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    weight_tensor = None if loss_weights is None else torch.tensor(
        loss_weights, dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=CFG.get("label_smoothing", 0.05))
    ema = EMA(model)

    H = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr", "sec"]}
    best, bad = -1, 0
    best_vm, best_epoch = None, None
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train(); L = C = N = 0
        for x, y in tr_loader:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                o = model(x); loss = crit(o, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            L += loss.item() * len(y); C += (o.argmax(1) == y).sum().item(); N += len(y)
        ema.m.eval(); vl = vn = 0; VP, VY = [], []
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                    o = ema.m(x); vl += F.cross_entropy(o.float(), y, reduction="sum").item()
                VP += o.argmax(1).cpu().tolist(); VY += y.cpu().tolist(); vn += len(y)
        vm = metrics(VY, VP)
        for k, v in zip(H, [ep, L / N, 100 * C / N, vl / vn, vm["accuracy"], vm["macro_f1"], sched.get_last_lr()[0], time.time() - t0]):
            H[k].append(round(v, 5) if isinstance(v, float) else v)
        flag = ""
        if vm["macro_f1"] > best:
            best, bad, flag = vm["macro_f1"], 0, " *"
            best_vm, best_epoch = vm, ep
            torch.save({"epoch": ep, "state": ema.m.state_dict(), "val": vm, "config": CFG,
                        "graph": args.graph, "model": args.model, "seed": args.seed,
                        "run_metadata": run_metadata}, out / "best.pt")
            with open(out / "val_predictions.csv", "w", newline="") as f:
                csv.writer(f).writerows([["true", "pred"]] + list(zip(VY, VP)))
        else:
            bad += 1
        print(f"  ep {ep:03d} | loss {L / N:.4f} acc {100 * C / N:.2f} | val loss {vl / vn:.4f} acc {vm['accuracy']:.2f} F1 {vm['macro_f1']:.4f} | {time.time() - t0:.0f}s{flag}", flush=True)
        with open(out / "history.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(H.keys()); w.writerows(zip(*H.values()))
        if bad >= args.patience:
            print(f"  early stop at epoch {ep}"); break

    validation_result = dict(
        run=run_name(), params=n_params, graph=args.graph, model=args.model, seed=args.seed,
        best_epoch=best_epoch, epochs_completed=len(H["epoch"]), val=best_vm,
        train_sampler=TRAIN_SAMPLER, class_weighting=CLASS_WEIGHTING,
        input_preprocess=INPUT_PRE, init=CFG.get("init", "synapse"),
        init_fanin_normalize=bool(CFG.get("init_fanin_normalize", True)),
        steps=CFG.get("steps", 4), run_metadata=run_metadata)
    (out / "validation_result.json").write_text(json.dumps(validation_result, indent=2))
    plot_curves(H, out / "curves.png", f"{run_name()}")
    if do_eval_test:
        evaluate()
    else:
        # v8.1 fix: pilots must not touch the test set — configuration decisions
        # are locked on VALIDATION only; run --mode eval later for the main study.
        (out / "test_not_evaluated.txt").write_text(
            "Pilot protocol: test set untouched. Run --mode eval for this run after "
            "configuration decisions are locked.\n")
        print("Test evaluation SKIPPED (pilot: validation-only decisions; "
              "run --mode eval when the config is locked).")


@torch.no_grad()
def predict(model, loader):
    model.eval(); Y, P = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            o = model(x)
        P += o.argmax(1).cpu().tolist(); Y += y.cpu().tolist()
    return Y, P


@torch.no_grad()
def evaluate():
    """Load saved EMA weights and evaluate test; never trains, never saves weights."""
    if not CFG.get("eval_test", True) or args.skip_test_eval:
        raise RuntimeError("Test evaluation is disabled for this config. This follow-up "
                           "is validation-only; do not use --mode eval during model selection.")
    set_seed(args.seed)
    out = EXP_ROOT / run_name()
    ck_path = out / "best.pt"
    if not ck_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ck_path}. Evaluation stopped; training will NOT run.")
    model = build_model().to(DEVICE)
    # Locally produced checkpoint from this experiment tree. Never load untrusted files.
    ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state"], strict=True); model.eval()
    n_params = count_params(model)
    print(f"Loaded {ck_path} | best epoch {ck['epoch']} | EVALUATION ONLY", flush=True)
    te_loader, _ = make_loader("test")
    Y, P = predict(model, te_loader); tm = metrics(Y, P)
    with open(out / "test_predictions.csv", "w", newline="") as f:
        csv.writer(f).writerows([["true", "pred"]] + list(zip(Y, P)))
    plot_cm(tm["confusion"], out / "confusion.png", f"{run_name()} test")
    result = dict(run=run_name(), params=n_params, graph=args.graph, model=args.model, seed=args.seed,
                  best_epoch=ck["epoch"], val=ck["val"], test=tm,
                  input_preprocess=INPUT_PRE, init=CFG.get("init", "synapse"),
                  init_fanin_normalize=bool(CFG.get("init_fanin_normalize", True)),
                  steps=CFG.get("steps", 4))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: tm[k] for k in ["accuracy", "balanced_accuracy", "macro_f1"]}, indent=2))
    return result


def compare():
    rows = [["run", "params", "graph", "seed", "acc", "bal_acc", "macro_f1",
             "rec_FORWARD", "rec_LEFT", "rec_RIGHT", "input_preprocess", "init_fanin_normalize"]]
    for p in sorted(EXP_ROOT.glob("*/result.json")):
        r = json.loads(p.read_text()); t = r["test"]
        rows.append([r["run"], r["params"], r["graph"], r["seed"], t["accuracy"], t["balanced_accuracy"], t["macro_f1"],
                     t["per_class"]["FORWARD"]["recall"], t["per_class"]["LEFT"]["recall"], t["per_class"]["RIGHT"]["recall"],
                     r.get("input_preprocess", "-"), r.get("init_fanin_normalize", "-")])
    out = EXP_ROOT / "compare.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {out}")
    for r in rows:
        print("  ", *r, sep=" | " if r is not rows[0] else " ")


def overfit():
    """Phase 4 small-subset test: overfit a tiny training subset to catch implementation faults."""
    set_seed(args.seed)
    out = EXP_ROOT / ("overfit_" + run_name()); out.mkdir(parents=True, exist_ok=True)
    # v8.1 fix: stratified, seeded subset (head-slice can be class/route-biased and
    # can miss rare classes entirely); all three classes guaranteed if present.
    ds = MidAirDataset("train", augment=False, limit=CFG.get("overfit_n", 64),
                       stratified=True, seed=args.seed)
    cls_counts = Counter(ds.labels())
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = build_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    y_all, p_all = [], []
    for ep in range(1, CFG.get("overfit_epochs", 30) + 1):
        model.train(); L = C = N = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            o = model(x); loss = crit(o, y)
            opt.zero_grad(); loss.backward(); opt.step()
            L += loss.item() * len(y); C += (o.argmax(1) == y).sum().item(); N += len(y)
            y_all += y.cpu().tolist(); p_all += o.argmax(1).cpu().tolist()
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep} loss {L / N:.4f} acc {100 * C / N:.1f}", flush=True)
    acc = 100 * C / N
    ok = acc >= 95.0
    ya, pa = np.array(y_all), np.array(p_all)
    per_cls = {}
    for i, c in enumerate(CLASSES):
        n_c = cls_counts.get(i, 0)
        per_cls[c] = dict(n=n_c, acc=round(100 * (pa[ya == i] == i).mean(), 1) if n_c else None)
    (out / "overfit.json").write_text(json.dumps(
        {"final_train_acc": acc, "passed": ok, "n": len(ds),
         "class_counts": {CLASSES[k]: v for k, v in sorted(cls_counts.items())},
         "per_class_train_acc": per_cls}, indent=2))
    print(f"OVERFIT TEST {'PASSED' if ok else 'FAILED'} (train acc {acc:.1f}; threshold 95)")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = {"train": train, "eval": evaluate, "compare": compare, "overfit": overfit}[args.mode]()
    # v8.1 fix: propagate overfit()'s failure exit code (a failed gate must not exit 0)
    sys.exit(0 if rc is None else int(rc))
