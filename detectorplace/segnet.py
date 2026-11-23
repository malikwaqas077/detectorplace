"""Learned drawing reader: a small U-Net that segments a plot plan into equipment elevation bands,
hazard grades and leak-source heatmaps, with per-cell confidence.

It is the learned alternative to the hand-tuned OpenCV rules in vision.py. Both return the same
Plant model, so everything downstream (coverage, optimiser, report) is unchanged. The network is
trained on randomised synthetic drawings (scene.py, scripts/train_segnet.py) and runs on CPU.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .plant import RES, Obstacle, Plant
from .scene import BAND_HEIGHTS

IN_PX_PER_M = 4.0          # network input scale; 2 px per 0.5 m analysis cell
WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "segnet.pt"
N_EQ, N_ZONE = 4, 4

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:          # the OpenCV reader and CAD reader still work without torch
    torch = None


if torch is not None:
    class _Block(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.f = nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                   nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

        def forward(self, x):
            return self.f(x)

    class UNet(nn.Module):
        """4-level U-Net, about 0.47 M parameters. Output: 4 equipment + 4 zone logits + 1 leak logit."""

        def __init__(self, c=(16, 32, 64, 96)):
            super().__init__()
            self.d = nn.ModuleList([_Block(3, c[0]), _Block(c[0], c[1]), _Block(c[1], c[2]), _Block(c[2], c[3])])
            self.u = nn.ModuleList([_Block(c[3] + c[2], c[2]), _Block(c[2] + c[1], c[1]), _Block(c[1] + c[0], c[0])])
            self.head = nn.Conv2d(c[0], N_EQ + N_ZONE + 1, 1)

        def forward(self, x):
            skips = []
            for i, blk in enumerate(self.d):
                x = blk(x if i == 0 else F.max_pool2d(x, 2))
                skips.append(x)
            for blk, s in zip(self.u, reversed(skips[:-1])):
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
                x = blk(torch.cat([x, s], 1))
            return self.head(x)


def to_input(img_rgb: np.ndarray, px_per_m: float) -> np.ndarray:
    f = IN_PX_PER_M / px_per_m
    H, W = img_rgb.shape[:2]
    size = (max(int(round(W * f)), 8), max(int(round(H * f)), 8))
    return cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)


_MODEL = None


def available() -> bool:
    return torch is not None and WEIGHTS.exists()


def load(path=WEIGHTS):
    global _MODEL
    if _MODEL is None:
        m = UNet()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        _MODEL = m
    return _MODEL


def predict(img_rgb: np.ndarray, px_per_m: float, model=None) -> dict:
    """Per-pixel probabilities at the network scale (4 px/m)."""
    model = model or load()
    x = to_input(img_rgb, px_per_m)
    H, W = x.shape[:2]
    ph, pw = (-H) % 8, (-W) % 8
    t = torch.from_numpy(np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="edge")).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        out = model(t)[0, :, :H, :W]
    return {"eq": torch.softmax(out[:N_EQ], 0).numpy(), "zone": torch.softmax(out[N_EQ:N_EQ + N_ZONE], 0).numpy(),
            "leak": torch.sigmoid(out[-1]).numpy(), "input": x}


def _cells(prob: np.ndarray, shape) -> np.ndarray:
    """Average (C, h, w) probabilities onto the analysis grid."""
    return np.stack([cv2.resize(c, (shape[1], shape[0]), interpolation=cv2.INTER_AREA) for c in prob])


def leak_peaks(heat: np.ndarray, thr=0.35, min_sep_px=4) -> np.ndarray:
    """Local maxima of the leak heatmap, in input pixels (x, y)."""
    m = cv2.dilate(heat, np.ones((2 * min_sep_px + 1,) * 2, np.uint8))
    ys, xs = np.nonzero((heat >= m - 1e-6) & (heat > thr))
    order = np.argsort(-heat[ys, xs])
    pts = []
    for i in order:
        p = np.array([xs[i], ys[i]], float)
        if all(np.hypot(*(p - q)) > min_sep_px for q in pts):
            # sub-pixel centroid in a 5x5 window
            y0, x0 = max(ys[i] - 2, 0), max(xs[i] - 2, 0)
            w = heat[y0:ys[i] + 3, x0:xs[i] + 3]
            yy, xx = np.mgrid[y0:y0 + w.shape[0], x0:x0 + w.shape[1]]
            pts.append(np.array([(xx * w).sum() / w.sum(), (yy * w).sum() / w.sum()]))
    return np.array(pts).reshape(-1, 2)


def extract_plant(img_rgb: np.ndarray, px_per_m: float = 10.0, name="Plant (from drawing, learned reader)",
                  model=None):
    """Same contract as vision.extract_plant: returns (Plant, debug). debug carries confidence maps."""
    pr = predict(img_rgb, px_per_m, model)
    H, W = img_rgb.shape[:2]
    width, depth = W / px_per_m, H / px_per_m
    shape = (int(round(depth / RES)), int(round(width / RES)))
    eq, zn = _cells(pr["eq"], shape), _cells(pr["zone"], shape)
    eq_cls, zn_cls = eq.argmax(0), zn.argmax(0)
    heights = np.where(eq_cls > 0, np.array((0.0,) + BAND_HEIGHTS)[eq_cls], 0).astype(np.float32)
    grades = zn_cls.astype(np.int8)
    leaks = leak_peaks(pr["leak"]) / IN_PX_PER_M + 0.5 / IN_PX_PER_M
    obstacles = []
    n, lab, st, _ = cv2.connectedComponentsWithStats((heights > 0).astype(np.uint8))
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < 3:
            heights[lab == k] = 0     # specks under 0.75 m^2
            continue
        band = float(np.median(heights[lab == k]))
        obstacles.append(Obstacle(f"OBS-{len(obstacles) + 1:02d}", "rect",
                                  (x * RES, y * RES, (x + w) * RES, (y + h) * RES), band))
    plant = Plant(width, depth, heights, grades, leaks.reshape(-1, 2), obstacles=obstacles, name=name,
                  panel=(min(69.0, width - 2), min(39.0, depth - 2)))
    conf = {"equipment": eq.max(0), "zone": zn.max(0), "leak_heat": pr["leak"]}
    return plant, {"confidence": conf, "eq_prob": eq, "zone_prob": zn, "input": pr["input"]}
