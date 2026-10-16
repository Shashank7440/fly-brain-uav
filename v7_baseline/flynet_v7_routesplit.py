#!/usr/bin/env python3
"""
flynet_v7.py — Clean, fair comparison: TinyFlyNet vs CNN vs SNN
================================================================
Task   : collision-avoidance action prediction (FORWARD / LEFT / RIGHT)
Input  : 9-ch [prev_rgb(3) | curr_rgb(3) | flow_x | flow_y | flow_mag]  (same for all)

Models
  flynet : TinyFlyNet (~85k params) — retina adaptation, medulla DW columns,
           explicit lobula heads (H1/HS left-right, VS top-bottom, LPLC2
           centre-surround + flow divergence), DNOVS linear decision.
  cnn    : standard 3-block CNN (same as V6 baseline, ~621k params)
  snn    : LIF spiking CNN, T=4, with BN + soft reset (fixed V6 SNN, ~61k params)

Identical recipe for all models (fair):
  - consistent photometric aug on BOTH frames, every class (no RGB-only rotation)
  - label-aware horizontal flip (LEFT<->RIGHT, flow_x sign flip)
  - coherent occlusion (same box on prev, curr and flow)
  - balanced sampler, AdamW + OneCycle, AMP, label smoothing 0.05, EMA weights
  - checkpoint & early stop on VALIDATION macro-F1 (test set never used for selection)

Outputs (no overlap between jobs):
  /Data4/mi_25026207/drone/ExperimentV7/<FlyNet|CNN|SNN>/seed<k>/
      best.pt  history.csv  curves.png  confusion.png  result.json  val_predictions.csv
  /Data4/mi_25026207/drone/ExperimentV7/Compare/   (written when all 3 are done)
      compare.json  compare.csv  summary.md  bars_*.png  curves_all.png  robustness.png  params_vs_f1.png
Usage:
  python flynet_v7.py --model flynet|cnn|snn|compare [--seed 42] [--epochs 40]
"""
import argparse, csv, json, math, random, time, copy
from pathlib import Path
from collections import Counter

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
cv2.setNumThreads(0)          # avoid thread explosion inside DataLoader workers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── args ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["flynet", "cnn", "snn", "compare"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--patience", type=int, default=12)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--mode", choices=["eval", "train"], default="eval",
                help="Default: evaluate existing best.pt only. Training requires explicit --mode train.")
args = ap.parse_args()

# ─── paths ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/Data4/mi_25026207/drone/data/flyuav_midair_3class_route_split")
IMG_SUBROOT = Path("/Data4/mi_25026207/drone/data/flyuav_midair_3class/MidAir_HPC/Kite_training")
EXP_ROOT = Path("/Data4/mi_25026207/drone/ExperimentV7RouteSplit")
NAMES = {"flynet": "FlyNet", "cnn": "CNN", "snn": "SNN"}
COMPARE_DIR = EXP_ROOT / "Compare"

IMG_SIZE, NUM_CLS, IN_CH = 128, 3, 9
CLASSES = ["FORWARD", "LEFT", "RIGHT"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)


def _worker_init(_):
    torch.set_num_threads(1); cv2.setNumThreads(0)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════
class MidAirDataset(Dataset):
    FB = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    def __init__(self, split, augment=False):
        self.flow_dir = DATASET_ROOT / split / "flow"
        self.augment = augment
        self.rows = []
        with open(DATASET_ROOT / split / "pairs.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append((IMG_SUBROOT / r["prev_frame"], IMG_SUBROOT / r["curr_frame"], int(r["label_id"])))
        self.resize = T.Resize((IMG_SIZE, IMG_SIZE), antialias=True)

    def __len__(self): return len(self.rows)
    def labels(self): return [r[2] for r in self.rows]

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

    def __getitem__(self, idx):
        pp, cp, label = self.rows[idx]
        prev, curr = self._rgb(pp), self._rgb(cp)
        fx, fy = self._flow(idx, pp, cp)

        if self.augment:
            # 1) SAME photometric transform on both frames (no fake temporal change)
            b, c, s = random.uniform(.7, 1.3), random.uniform(.7, 1.3), random.uniform(.7, 1.3)
            def photo(x):
                x = TF.adjust_brightness(x, b); x = TF.adjust_contrast(x, c)
                return TF.adjust_saturation(x, s)
            prev, curr = photo(prev), photo(curr)
            # 2) coherent low-light (both frames)
            if random.random() < 0.2:
                k = random.uniform(0.2, 0.6); prev, curr = prev * k, curr * k
            # 3) sensor noise (independent per frame, realistic)
            if random.random() < 0.2:
                prev = (prev + 0.05 * torch.randn_like(prev)).clamp(0, 1)
                curr = (curr + 0.05 * torch.randn_like(curr)).clamp(0, 1)
            # 4) label-aware flip
            if random.random() < 0.5:
                prev, curr = prev.flip(2), curr.flip(2)
                fx, fy = -fx.flip(2), fy.flip(2)
                label = {1: 2, 2: 1}.get(label, label)
            # 5) coherent occlusion (same box on frames + flow)
            if random.random() < 0.25:
                h, w = random.randint(16, 48), random.randint(16, 48)
                y0, x0 = random.randint(0, IMG_SIZE - h), random.randint(0, IMG_SIZE - w)
                for t in (prev, curr, fx, fy): t[:, y0:y0 + h, x0:x0 + w] = 0
        mag = torch.sqrt(fx ** 2 + fy ** 2)
        return torch.cat([prev, curr, fx, fy, mag], 0), label


def make_loader(split, augment=False):
    import hashlib
    metadata_path = DATASET_ROOT / "split_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("Missing corrected split metadata; run prepare_route_split.py first")
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
    ds = MidAirDataset(split, augment)
    if split == "train":
        kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                  worker_init_fn=_worker_init, persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None)
        cnt = Counter(ds.labels())
        w = [1.0 / cnt[l] for l in ds.labels()]
        return DataLoader(ds, sampler=WeightedRandomSampler(w, len(w), replacement=True), drop_last=True, **kw), ds
    # val/test: few, non-persistent workers -> no idle processes holding RAM
    return DataLoader(ds, shuffle=False, batch_size=args.batch, num_workers=2, pin_memory=False), ds


# ═════════════════════════════════════════════════════════════════════════════
# Models
# ═════════════════════════════════════════════════════════════════════════════
class DWBlock(nn.Module):
    """Medulla column: inverted residual, depthwise 3x3, squeeze-excite."""
    def __init__(self, cin, cout, stride, exp=3):
        super().__init__()
        mid, se = cin * exp, max(8, cin * exp // 8)
        self.res = stride == 1 and cin == cout
        self.body = nn.Sequential(
            nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(mid, se, 1), nn.SiLU(inplace=True),
                                nn.Conv2d(se, mid, 1), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = self.body(x); h = self.proj(h * self.se(h))
        return x + h if self.res else h


class TinyFlyNet(nn.Module):
    """
    Retina  : photoreceptor gain control on RGB (per-sample contrast norm),
              temporal channel (curr - prev), relative flow (÷ mean |flow|),
              flow divergence (looming cue, LPLC2 input).
    Medulla : 6 depthwise inverted-residual columns (16→24→40→64).
    Lobula  : parameter-free population read-outs on the final map
              global | H1/HS (L−R) | VS (T−B) | LPLC2 (centre−surround)
              + 8 explicit flow statistics (bounded L/R asymmetry, divergence...).
    DNOVS   : dropout + linear decision.
    """
    def __init__(self, num_cls=NUM_CLS, w=(16, 24, 40, 64)):
        super().__init__()
        c0, c1, c2, c3 = w
        self.stem = nn.Sequential(nn.Conv2d(11, c0, 3, 2, 1, bias=False), nn.BatchNorm2d(c0), nn.SiLU(inplace=True))
        self.medulla = nn.Sequential(
            DWBlock(c0, c1, 2), DWBlock(c1, c1, 1),
            DWBlock(c1, c2, 2), DWBlock(c2, c2, 1),
            DWBlock(c2, c3, 2), DWBlock(c3, c3, 1))
        self.stat_norm = nn.BatchNorm1d(8)
        self.drop = nn.Dropout(0.2)
        self.dnovs = nn.Linear(c3 * 4 + 8, num_cls)

    @staticmethod
    def _cnorm(x):
        m = x.mean((1, 2, 3), keepdim=True); s = x.std((1, 2, 3), keepdim=True) + 1e-3
        return (x - m) / s

    @staticmethod
    def _divergence(u, v):
        """div = du/dx + dv/dy with 2nd-order central differences
        (replicate padding at borders -> one-sided difference there, same size, centred)."""
        up = F.pad(u, (1, 1, 0, 0), mode="replicate")
        vp = F.pad(v, (0, 0, 1, 1), mode="replicate")
        dudx = (up[..., 2:] - up[..., :-2]) * 0.5
        dvdy = (vp[..., 2:, :] - vp[..., :-2, :]) * 0.5
        return dudx + dvdy

    def forward(self, x):
        prev, curr, fx, fy = x[:, 0:3], x[:, 3:6], x[:, 6:7], x[:, 7:8]
        # ── retina ──
        curr_n = self._cnorm(curr)                         # photoreceptor adaptation (lowlight)
        delta = self._cnorm(curr - prev)                   # temporal (lamina L1/L2-like)
        scale = (fx.abs() + fy.abs()).mean((1, 2, 3), keepdim=True).clamp(min=1e-3)
        u, v = fx / scale, fy / scale                      # relative flow
        div = self._divergence(u, v)                       # central differences
        mag = torch.sqrt(u ** 2 + v ** 2 + 1e-6)
        z = torch.cat([curr_n, delta, u, v, mag, div, torch.log(scale).expand_as(u)], 1)   # 11 ch
        # ── medulla ──
        f = self.medulla(self.stem(z))
        H, W = f.shape[-2:]
        # ── lobula read-outs ──
        g = f.mean((2, 3))
        h1hs = f[..., :W // 2].mean((2, 3)) - f[..., W - W // 2:].mean((2, 3))
        vs = f[:, :, :H // 2].mean((2, 3)) - f[:, :, H - H // 2:].mean((2, 3))
        lplc2 = f[:, :, H // 4:H - H // 4, W // 4:W - W // 4].mean((2, 3)) - g
        # explicit flow statistics
        Wf = u.shape[-1] // 2
        uL, uR = u[..., :Wf].mean((1, 2, 3)), u[..., Wf:].mean((1, 2, 3))
        mL, mR = mag[..., :Wf].mean((1, 2, 3)), mag[..., Wf:].mean((1, 2, 3))
        Hf = u.shape[-2]
        dC = div[:, :, Hf // 4:Hf - Hf // 4, Wf // 2:Wf + Wf // 2].mean((1, 2, 3))
        stats = torch.stack([uL, uR, uL - uR, (mL - mR) / (mL + mR + 1e-6),
                             v.mean((1, 2, 3)), div.mean((1, 2, 3)), dC, torch.log(scale).flatten()], 1)
        feat = torch.cat([g, h1hs, vs, lplc2, self.stat_norm(stats)], 1)
        return self.dnovs(self.drop(feat))


class CNNModel(nn.Module):
    """Standard 3-block CNN baseline (identical to V6)."""
    def __init__(self, n=NUM_CLS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Sequential(nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, n))

    def forward(self, x): return self.fc(self.net(x).flatten(1))


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x); return (x >= 0).float()

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        return g * torch.clamp(1 - x.abs(), min=0)   # triangular surrogate


class SNNModel(nn.Module):
    """LIF spiking CNN, T steps, BN + soft reset (membrane -= threshold after spike)."""
    def __init__(self, n=NUM_CLS, T=4, tau=0.9, thr=1.0):
        super().__init__()
        self.T, self.tau, self.thr = T, tau, thr
        self.c1, self.b1 = nn.Conv2d(9, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32)
        self.c2, self.b2 = nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64)
        self.c3, self.b3 = nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(64 * 16, n)

    def _lif(self, m, inp):
        m = self.tau * m + inp
        s = SpikeFn.apply(m - self.thr)
        return m - s * self.thr, s

    def forward(self, x):
        i1 = self.b1(self.c1(x))                 # static input current computed once
        m1 = m2 = m3 = 0.; out = 0.
        for _ in range(self.T):
            m1, s1 = self._lif(m1, i1); s1 = F.avg_pool2d(s1, 2)
            m2, s2 = self._lif(m2, self.b2(self.c2(s1))); s2 = F.avg_pool2d(s2, 2)
            m3, s3 = self._lif(m3, self.b3(self.c3(s2)))
            out = out + self.pool(s3).flatten(1)
        return self.fc(out / self.T)


MODELS = {"flynet": TinyFlyNet, "cnn": CNNModel, "snn": SNNModel}


# ═════════════════════════════════════════════════════════════════════════════
# Metrics / profiling
# ═════════════════════════════════════════════════════════════════════════════
def metrics(true, pred):
    cm = np.zeros((NUM_CLS, NUM_CLS), int)
    for t, p in zip(true, pred): cm[t, p] += 1
    per, rec, f1s = {}, [], []
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; pr = tp / max(cm[:, i].sum(), 1); re = tp / max(cm[i].sum(), 1)
        f1 = 2 * pr * re / (pr + re) if pr + re else 0.
        per[c] = dict(precision=round(pr, 4), recall=round(re, 4), f1=round(f1, 4), support=int(cm[i].sum()))
        rec.append(re); f1s.append(f1)
    return dict(accuracy=round(100 * np.trace(cm) / cm.sum(), 3), balanced_accuracy=round(100 * np.mean(rec), 3),
                macro_f1=round(float(np.mean(f1s)), 4), per_class=per, confusion=cm.tolist())


def count_params(m): return sum(p.numel() for p in m.parameters())


def mflops(model):
    total = 0
    def hook(m, i, o):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            total += (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1] * o.numel()
        elif isinstance(m, nn.Linear):
            total += m.in_features * o.numel()
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad(): model(torch.randn(2, IN_CH, IMG_SIZE, IMG_SIZE, device=DEVICE))
    for h in hs: h.remove()
    return total / 2 / 1e6   # MACs per sample (in millions)


@torch.no_grad()
def fps(model, n=200):
    model.eval(); d = torch.randn(1, IN_CH, IMG_SIZE, IMG_SIZE, device=DEVICE)
    for _ in range(20): model(d)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): model(d)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    return round(n / (time.perf_counter() - t), 1)


@torch.no_grad()
def predict(model, loader, corrupt=None):
    model.eval(); P, Y = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        if corrupt: x = corrupt(x)
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            P += model(x).argmax(1).cpu().tolist()
        Y += y.tolist()
    return Y, P


# robustness: (a) current-frame only (V6 protocol)  (b) coherent sequence-level
def _blur(t): return TF.gaussian_blur(t, 9, 3.0)
def _occ(x, chans):
    x = x.clone(); H = W = IMG_SIZE
    x[:, chans, H // 4:H // 4 + H // 2, W // 4:W // 4 + W // 2] = 0; return x
def _set(x, sl, fn): x = x.clone(); x[:, sl] = fn(x[:, sl]); return x

CORRUPTIONS = {
    "curr_blur": lambda x: _set(x, slice(3, 6), _blur),
    "curr_noise": lambda x: _set(x, slice(3, 6), lambda t: (t + 0.1 * torch.randn_like(t)).clamp(0, 1)),
    "curr_lowlight": lambda x: _set(x, slice(3, 6), lambda t: t * 0.2),
    "curr_occlusion": lambda x: _occ(x, [3, 4, 5]),
    "seq_blur": lambda x: _set(x, slice(0, 6), _blur),
    "seq_noise": lambda x: _set(x, slice(0, 6), lambda t: (t + 0.1 * torch.randn_like(t)).clamp(0, 1)),
    "seq_lowlight": lambda x: _set(x, slice(0, 6), lambda t: t * 0.2),
    "seq_occlusion": lambda x: _occ(x, list(range(9))),
}


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=0.999):
        self.m = copy.deepcopy(model).eval(); self.d = decay
        for p in self.m.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point: e.mul_(self.d).add_(p.detach(), alpha=1 - self.d)
            else: e.copy_(p)


def plot_curves(h, out, title):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ep = h["epoch"]
    ax[0].plot(ep, h["train_loss"], label="train"); ax[0].plot(ep, h["val_loss"], label="val"); ax[0].set_title("Loss")
    ax[1].plot(ep, h["train_acc"], label="train"); ax[1].plot(ep, h["val_acc"], label="val"); ax[1].set_title("Accuracy (%)")
    ax[2].plot(ep, h["val_f1"], c="g"); ax[2].set_title("Val macro-F1")
    for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend() if a.get_legend_handles_labels()[0] else None
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_cm(cm, out, title):
    cm = np.array(cm); fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLS):
        for j in range(NUM_CLS):
            ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / max(cm[i].sum(), 1):.1%}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(NUM_CLS), CLASSES); ax.set_yticks(range(NUM_CLS), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def train(key):
    set_seed(args.seed)
    name = NAMES[key]
    out = EXP_ROOT / name / f"seed{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise FileExistsError(f"Checkpoint already exists: {out / 'best.pt'}. Refusing to retrain or overwrite it; use --mode eval.")
    tr_loader, _ = make_loader("train", augment=True)
    va_loader, va_ds = make_loader("val")
    te_loader, _ = make_loader("test")

    model = MODELS[key]().to(DEVICE).to(memory_format=torch.channels_last)
    n_params = count_params(model)
    print(f"\n{name}: {n_params:,} params | train {len(tr_loader.dataset)} val {len(va_ds)} test {len(te_loader.dataset)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr_loader), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    ema = EMA(model)

    H = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr", "sec"]}
    best, bad = -1, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train(); L = C = N = 0
        for x, y in tr_loader:
            x = x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                o = model(x); loss = crit(o, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            L += loss.item() * len(y); C += (o.argmax(1) == y).sum().item(); N += len(y)
        # validation with EMA weights
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
            torch.save({"epoch": ep, "state": ema.m.state_dict(), "val": vm}, out / "best.pt")
            with open(out / "val_predictions.csv", "w", newline="") as f:
                csv.writer(f).writerows([["true", "pred"]] + list(zip(VY, VP)))
        else:
            bad += 1
        print(f"  ep {ep:03d} | loss {L / N:.4f} acc {100 * C / N:.2f} | val loss {vl / vn:.4f} acc {vm['accuracy']:.2f} "
              f"F1 {vm['macro_f1']:.4f} | {time.time() - t0:.0f}s{flag}", flush=True)
        with open(out / "history.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(H.keys()); w.writerows(zip(*H.values()))
        if bad >= args.patience:
            print(f"  early stop at epoch {ep}"); break

    plot_curves(H, out / "curves.png", f"{name} (seed {args.seed})")
    evaluate(key)


@torch.no_grad()
def evaluate(key):
    """Load saved EMA weights and evaluate test/robustness; never train or save weights."""
    set_seed(args.seed)
    name = NAMES[key]
    out = EXP_ROOT / name / f"seed{args.seed}"
    checkpoint = out / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}. Evaluation stopped; training will NOT run.")
    H = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr", "sec"]}
    history_path = out / "history.csv"
    if history_path.exists():
        with open(history_path, newline="") as f:
            for row in csv.DictReader(f):
                for k in H:
                    H[k].append(int(row[k]) if k == "epoch" else float(row[k]))
    model = MODELS[key]().to(DEVICE).to(memory_format=torch.channels_last)
    n_params = count_params(model)
    # ── test with best checkpoint ──
    # This is a trusted, locally produced checkpoint. Never load untrusted files here.
    ck = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state"], strict=True); model.eval()
    print(f"Loaded {checkpoint} | best epoch {ck['epoch']} | EVALUATION ONLY", flush=True)
    te_loader, _ = make_loader("test")
    Y, P = predict(model, te_loader); tm = metrics(Y, P)
    with open(out / "test_predictions.csv", "w", newline="") as f:
        csv.writer(f).writerows([["true", "pred"]] + list(zip(Y, P)))
    plot_cm(tm["confusion"], out / "confusion.png", f"{name} test")
    rob = {}
    for k, fn in CORRUPTIONS.items():
        y, p = predict(model, te_loader, fn); m = metrics(y, p)
        rob[k] = {kk: m[kk] for kk in ["accuracy", "balanced_accuracy", "macro_f1"]}
        print(f"  robust {k:15s} acc {m['accuracy']:.2f} F1 {m['macro_f1']:.4f}")
    res = dict(model=name, seed=args.seed, params=n_params, mmacs=round(mflops(model), 2), fps=fps(model),
               best_epoch=ck["epoch"], val=ck["val"], test=tm, robustness=rob, history=H,
               config=vars(args))
    res["split_metadata"] = json.loads((DATASET_ROOT / "split_metadata.json").read_text())
    with open(out / "result.json", "w") as f: json.dump(res, f, indent=2)
    print(f"\n  TEST acc {tm['accuracy']} | bal {tm['balanced_accuracy']} | F1 {tm['macro_f1']} | saved → {out}")
    # auto-compare once all three are finished
    if all((EXP_ROOT / n / f"seed{args.seed}" / "result.json").exists() for n in NAMES.values()):
        compare()


# ═════════════════════════════════════════════════════════════════════════════
# Comparison
# ═════════════════════════════════════════════════════════════════════════════
def compare():
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    R = {}
    for n in NAMES.values():
        p = EXP_ROOT / n / f"seed{args.seed}" / "result.json"
        if p.exists(): R[n] = json.load(open(p))
    if not R: print("no results yet"); return
    cols = ["#2DC653", "#E63946", "#457B9D"]
    names = list(R)
    rows = []
    for n in names:
        r = R[n]; t = r["test"]
        row = dict(model=n, params=r["params"], mmacs=r["mmacs"], fps=r["fps"], accuracy=t["accuracy"],
                   balanced_accuracy=t["balanced_accuracy"], macro_f1=t["macro_f1"],
                   **{f"recall_{c}": t["per_class"][c]["recall"] for c in CLASSES},
                   **{f"rob_{k}_f1": v["macro_f1"] for k, v in r["robustness"].items()})
        row["rob_mean_f1"] = round(np.mean([v["macro_f1"] for v in r["robustness"].values()]), 4)
        rows.append(row)
    with open(COMPARE_DIR / "compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(COMPARE_DIR / "compare.json", "w"), indent=2)

    for key, lab in [("macro_f1", "Test macro-F1"), ("balanced_accuracy", "Balanced acc (%)"),
                     ("accuracy", "Accuracy (%)"), ("params", "Parameters"), ("mmacs", "MMACs"), ("rob_mean_f1", "Mean robustness F1")]:
        fig, ax = plt.subplots(figsize=(5, 4))
        v = [r[key] for r in rows]; b = ax.bar(names, v, color=cols[:len(v)])
        ax.bar_label(b, fmt="%.4g"); ax.set_title(lab); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(COMPARE_DIR / f"bars_{key}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for n, c in zip(names, cols):
        h = R[n]["history"]; ax[0].plot(h["epoch"], h["val_loss"], c=c, label=n); ax[1].plot(h["epoch"], h["val_f1"], c=c, label=n)
    ax[0].set_title("Validation loss"); ax[1].set_title("Validation macro-F1")
    for a in ax: a.legend(); a.grid(alpha=.3); a.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(COMPARE_DIR / "curves_all.png", dpi=150); plt.close(fig)

    keys = list(CORRUPTIONS); x = np.arange(len(keys)); wd = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(13, 4))
    for i, (n, c) in enumerate(zip(names, cols)):
        ax.bar(x + i * wd, [R[n]["robustness"][k]["macro_f1"] for k in keys], wd, color=c, label=n)
    ax.set_xticks(x + wd * (len(names) - 1) / 2, keys, rotation=20); ax.set_ylabel("macro-F1"); ax.legend(); ax.grid(axis="y", alpha=.3)
    ax.set_title("Robustness (curr = current frame only, seq = both frames)")
    fig.tight_layout(); fig.savefig(COMPARE_DIR / "robustness.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for r, c in zip(rows, cols):
        ax.scatter(r["params"], r["macro_f1"], s=120, c=c); ax.annotate(r["model"], (r["params"], r["macro_f1"]), xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("parameters (log)"); ax.set_ylabel("test macro-F1"); ax.grid(alpha=.3)
    ax.set_title("Accuracy vs size"); fig.tight_layout(); fig.savefig(COMPARE_DIR / "params_vs_f1.png", dpi=150); plt.close(fig)

    with open(COMPARE_DIR / "summary.md", "w") as f:
        f.write(f"# V7 comparison (seed {args.seed})\n\n| Model | Params | MMACs | FPS | Acc | Bal Acc | Macro-F1 | LEFT rec | RIGHT rec | FWD rec | Robust F1 |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['params']:,} | {r['mmacs']} | {r['fps']} | {r['accuracy']} | {r['balanced_accuracy']} | "
                    f"{r['macro_f1']} | {r['recall_LEFT']} | {r['recall_RIGHT']} | {r['recall_FORWARD']} | {r['rob_mean_f1']} |\n")
    print(open(COMPARE_DIR / "summary.md").read()); print(f"Compare → {COMPARE_DIR}")


if __name__ == "__main__":
    if args.model == "compare": compare()
    elif args.mode == "train": train(args.model)
    else: evaluate(args.model)
