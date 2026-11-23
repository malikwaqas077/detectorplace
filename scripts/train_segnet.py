"""Train the learned drawing reader on randomised synthetic plot plans (CPU is fine).

    python scripts/train_segnet.py [n_drawings] [iterations]

The demo plant is never used for training; it is only used afterwards, in scripts/eval_readers.py.
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detectorplace import scene, segnet  # noqa: E402

CROP = 128
SIGMA = 1.2


def make(seed):
    rng = np.random.default_rng(10_000 + seed)
    spec, style = scene.random_spec(rng), scene.random_style(rng)
    img, lab = scene.render(spec, style, rng)
    x = segnet.to_input(img, style.px_per_m)
    H, W = x.shape[:2]
    eq = cv2.resize(lab["equip"], (W, H), interpolation=cv2.INTER_NEAREST)
    zn = cv2.resize(lab["zone"], (W, H), interpolation=cv2.INTER_NEAREST)
    heat = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for px, py in lab["leaks"] * segnet.IN_PX_PER_M:
        heat = np.maximum(heat, np.exp(-((xx + 0.5 - px) ** 2 + (yy + 0.5 - py) ** 2) / (2 * SIGMA ** 2)))
    return x, eq, zn, heat.astype(np.float16)


def batch(data, rng, bs):
    X, E, Z, L = [], [], [], []
    for _ in range(bs):
        x, eq, zn, ht = data[rng.integers(len(data))]
        H, W = eq.shape
        if H < CROP or W < CROP:
            pad = ((0, max(CROP - H, 0)), (0, max(CROP - W, 0)))
            x = np.pad(x, pad + ((0, 0),), constant_values=255)
            eq, zn, ht = (np.pad(a, pad) for a in (eq, zn, ht))
            H, W = eq.shape
        y0, x0 = rng.integers(0, H - CROP + 1), rng.integers(0, W - CROP + 1)
        sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        a = [x[sl], eq[sl], zn[sl], ht[sl]]
        k = rng.integers(4)
        a = [np.rot90(v, k) for v in a]
        if rng.random() < 0.5:
            a = [v[:, ::-1] for v in a]
        if rng.random() < 0.3:   # small global colour shift
            a[0] = np.clip(a[0].astype(np.int16) + rng.integers(-8, 9, 3), 0, 255).astype(np.uint8)
        X.append(np.ascontiguousarray(a[0])); E.append(a[1].copy()); Z.append(a[2].copy()); L.append(a[3].copy())
    X = torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).float() / 255
    return X, torch.from_numpy(np.stack(E)).long(), torch.from_numpy(np.stack(Z)).long(), \
        torch.from_numpy(np.stack(L).astype(np.float32))


def loss_fn(out, E, Z, L):
    le = F.cross_entropy(out[:, :4], E, weight=torch.tensor([0.5, 1.5, 1.5, 1.5]))
    lz = F.cross_entropy(out[:, 4:8], Z)
    ll = F.binary_cross_entropy_with_logits(out[:, 8], L, pos_weight=torch.tensor(20.0))
    return le + lz + 0.5 * ll, (le.item(), lz.item(), ll.item())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    torch.set_num_threads(2)
    t0 = time.time()
    with Pool(2) as pool:
        data = pool.map(make, range(n), chunksize=20)
    print(f"generated {n} drawings in {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = segnet.UNet()
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), 3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=iters, pct_start=0.1)
    model.train()
    t0, run = time.time(), None
    for it in range(1, iters + 1):
        X, E, Z, L = batch(data, rng, 12)
        out = model(X)
        loss, parts = loss_fn(out, E, Z, L)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = np.array(parts) if run is None else 0.98 * run + 0.02 * np.array(parts)
        if it % 100 == 0:
            print(f"it {it} eq {run[0]:.3f} zone {run[1]:.3f} leak {run[2]:.3f}  {time.time() - t0:.0f}s", flush=True)
        if it % 1000 == 0 or it == iters:
            segnet.WEIGHTS.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), segnet.WEIGHTS)
    print("saved", segnet.WEIGHTS)
