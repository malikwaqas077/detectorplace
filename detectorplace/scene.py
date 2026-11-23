"""Random plants and randomised drawing styles, used to train and stress-test the drawing readers.

There is no public dataset of industrial plot plans with labelled equipment heights, hazard
grades and leak sources. So the training data is generated: thousands of invented plants, each
drawn in a different style (shades, tints, outlines, grids, hatching, text, dimension lines,
scan noise, JPEG artefacts) and labelled for free from the geometry it was drawn from. Line art
cropped from real engineering drawings (MIT-licensed P&ID sheets) is pasted in as clutter, so
the model learns to ignore real symbols and text rather than our own fonts only.

The drawing convention itself is kept fixed: fill shade = elevation band, red/orange/yellow
tint = Grade A/B/C, blue marker = leak source. A real deployment retrains on the company's own
CAD standard; the method stays the same.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .plant import RES, Obstacle, Plant, _rect_mask, obstacle_mask

BAND_HEIGHTS = (2.0, 5.0, 9.0)          # label 1, 2, 3
BAND_GREY = {2.0: 170, 5.0: 115, 9.0: 60}
ZONE_HUE = {3: 0, 2: 14, 1: 27}         # OpenCV hue (0-179) of the A/B/C tints
CLUTTER_DIR = Path(__file__).resolve().parent.parent / "data" / "clutter"


@dataclass
class Spec:
    width: float
    depth: float
    obstacles: list
    zones: list                    # (grade, x0, y0, x1, y1)
    leaks: np.ndarray              # (k, 2)
    keepout: tuple | None = None   # (x0, y0, x1, y1) title-block corner, nothing placed there
    name: str = "Random plant"


def to_plant(s: Spec) -> Plant:
    shape = (int(round(s.depth / RES)), int(round(s.width / RES)))
    heights = np.zeros(shape, np.float32)
    for ob in s.obstacles:
        m = obstacle_mask(shape, ob)
        heights[m] = np.maximum(heights[m], ob.height)
    grades = np.zeros(shape, np.int8)
    for g, *r in s.zones:
        m = _rect_mask(shape, *r)
        grades[m] = np.maximum(grades[m], g)
    panel = (s.width - 4.0, s.depth - 4.0)
    return Plant(s.width, s.depth, heights, grades, np.asarray(s.leaks, float).reshape(-1, 2), panel=panel,
                 obstacles=list(s.obstacles), name=s.name)


# ---- random plants ----------------------------------------------------------------------

def _overlaps(box, boxes, gap):
    x0, y0, x1, y1 = box
    return any(x0 < b[2] + gap and b[0] < x1 + gap and y0 < b[3] + gap and b[1] < y1 + gap for b in boxes)


def random_spec(rng: np.random.Generator) -> Spec:
    w = float(rng.integers(50, 111))
    d = float(rng.integers(35, 76))
    keep = None
    if rng.random() < 0.5:   # title block in a corner
        kw, kd = float(rng.integers(12, 20)), float(rng.integers(6, 10))
        keep = (w - kw, d - kd, w, d) if rng.random() < 0.5 else (0.0, d - kd, kw, d)
    boxes = [keep] if keep else []
    obstacles = []
    # pipe racks and walls first: long and thin
    for _ in range(rng.integers(0, 3)):
        if rng.random() < 0.6:
            L, t = rng.uniform(0.4, 0.9) * w, rng.uniform(2.5, 4.0)
            x0, y0 = rng.uniform(2, w - L - 2), rng.uniform(3, d - t - 3)
            box = (x0, y0, x0 + L, y0 + t)
        else:
            L, t = rng.uniform(0.4, 0.8) * d, 0.5
            x0, y0 = rng.uniform(3, w - 3), rng.uniform(2, d - L - 2)
            box = (x0, y0, x0 + t, y0 + L)
        box = tuple(round(v * 2) / 2 for v in box)
        if box[2] - box[0] >= 0.5 and box[3] - box[1] >= 0.5 and not _overlaps(box, boxes, 2.0):
            boxes.append(box)
            obstacles.append(Obstacle(f"R{len(obstacles)}", "rect", box, 5.0 if t > 1 else float(rng.choice([5.0, 9.0]))))
    n = int(rng.integers(5, 17))
    for _ in range(200):
        if len(obstacles) >= n:
            break
        h = float(rng.choice(BAND_HEIGHTS, p=[0.35, 0.4, 0.25]))
        if rng.random() < 0.35:
            r = round(rng.uniform(1.5, 5.0) * 2) / 2
            cx, cy = rng.uniform(r + 2, w - r - 2), rng.uniform(r + 2, d - r - 2)
            cx, cy = round(cx * 2) / 2, round(cy * 2) / 2
            box = (cx - r, cy - r, cx + r, cy + r)
            ob = Obstacle(f"V{len(obstacles)}", "circle", (cx, cy, r), h)
        else:
            bw, bd = rng.uniform(2, 14), rng.uniform(2, 10)
            x0, y0 = rng.uniform(2, w - bw - 2), rng.uniform(2, d - bd - 2)
            box = tuple(round(v * 2) / 2 for v in (x0, y0, x0 + bw, y0 + bd))
            ob = Obstacle(f"E{len(obstacles)}", "rect", box, h)
        if _overlaps(box, boxes, 2.0):
            continue
        boxes.append(box)
        obstacles.append(ob)

    def clip(z):
        g, x0, y0, x1, y1 = z
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, d)
        if keep and x0 < keep[2] and keep[0] < x1 and y0 < keep[3] and keep[1] < y1:
            y1 = min(y1, keep[1])   # keep zones clear of the title block
        return (g, *(round(v * 2) / 2 for v in (x0, y0, x1, y1)))

    zones = [clip((1, rng.uniform(1, 6), rng.uniform(1, 5), w - rng.uniform(1, 6), d - rng.uniform(1, 6)))]
    eq = [o for o in obstacles if o.name[0] != "R"]
    for g, k in ((2, rng.integers(1, 5)), (3, rng.integers(1, 4))):
        for _ in range(k):
            if not eq:
                break
            o = eq[rng.integers(len(eq))]
            b = o.geom if o.kind == "rect" else (o.geom[0] - o.geom[2], o.geom[1] - o.geom[2],
                                                  o.geom[0] + o.geom[2], o.geom[1] + o.geom[2])
            m = rng.uniform(2, 7, 4)
            zones.append(clip((g, b[0] - m[0], b[1] - m[1], b[2] + m[2], b[3] + m[3])))
    zones = [z for z in zones if z[3] - z[1] >= 1 and z[4] - z[2] >= 1]

    plant = to_plant(Spec(w, d, obstacles, zones, np.zeros((0, 2))))
    leaks = []
    for _ in range(300):
        if len(leaks) >= rng.integers(3, 13) or not eq:
            break
        o = eq[rng.integers(len(eq))]
        if o.kind == "rect":
            x0, y0, x1, y1 = o.geom
            side = rng.integers(4)
            off = rng.uniform(0.8, 1.6)
            x = [x0 - off, x1 + off, rng.uniform(x0, x1), rng.uniform(x0, x1)][side]
            y = [rng.uniform(y0, y1), rng.uniform(y0, y1), y0 - off, y1 + off][side]
        else:
            cx, cy, r = o.geom
            a = rng.uniform(0, 2 * np.pi)
            x, y = cx + (r + rng.uniform(0.8, 1.6)) * np.cos(a), cy + (r + rng.uniform(0.8, 1.6)) * np.sin(a)
        iy, ix = int(y / RES), int(x / RES)
        if not (1 < x < w - 1 and 1 < y < d - 1) or plant.heights[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2].max() > 0:
            continue
        if keep and keep[0] - 1 < x < keep[2] + 1 and keep[1] - 1 < y < keep[3] + 1:
            continue
        if leaks and np.min(np.linalg.norm(np.array(leaks) - (x, y), axis=1)) < 2.5:
            continue
        leaks.append((round(x * 2) / 2, round(y * 2) / 2))
    return Spec(w, d, obstacles, zones, np.array(leaks).reshape(-1, 2), keep)


def demo_spec() -> Spec:
    from .plant import DEMO_LEAKS, DEMO_OBSTACLES, DEMO_ZONES
    return Spec(80.0, 50.0, list(DEMO_OBSTACLES), list(DEMO_ZONES), DEMO_LEAKS.copy(),
                name="Synthetic gas compression station")


# ---- drawing styles ---------------------------------------------------------------------

@dataclass
class Style:
    px_per_m: float = 10.0
    band_grey: dict = field(default_factory=lambda: dict(BAND_GREY))
    tints: dict = field(default_factory=lambda: {3: (255, 190, 190), 2: (255, 215, 170), 1: (255, 245, 170)})
    paper: tuple = (255, 255, 255)
    grid_m: float = 10.0
    grid_rgb: tuple = (215, 215, 215)
    grid_px: int = 1
    outline_px: int = 0            # black outline round equipment
    zone_outline_px: int = 0       # dashed zone boundaries
    hatch: bool = False            # diagonal hatching on equipment fills
    leak_shape: str = "dot"        # dot / diamond / cross-ring
    leak_r_m: float = 0.5
    leak_rgb: tuple = (30, 80, 220)
    labels: float = 1.0            # how many tag labels to draw (0-1)
    label_scale: float = 0.32
    dims: int = 0                  # dimension / pipe lines across the drawing
    clutter: int = 0               # pasted P&ID crops
    noise: float = 4.0
    blur: float = 0.0
    jpeg: int = 0                  # JPEG quality, 0 = lossless


def demo_style() -> Style:
    return Style()


def random_style(rng: np.random.Generator) -> Style:
    j = lambda lo, hi: int(rng.integers(lo, hi + 1))
    shift = j(-14, 14)
    greys = {2.0: 170 + shift + j(-10, 10), 5.0: 115 + shift + j(-10, 10), 9.0: 60 + shift + j(-10, 10)}
    tints = {}
    for g, hue in ZONE_HUE.items():
        h = (hue + j(-3, 3)) % 180
        s, v = j(40, 115), j(225, 255)
        rgb = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0, 0]
        tints[g] = tuple(int(c) for c in rgb)
    paper = (255, 255, 255) if rng.random() < 0.6 else tuple([j(238, 252)] * 3)
    g = j(150, 225)
    return Style(px_per_m=float(rng.choice([8, 10, 12, 15])), band_grey=greys, tints=tints, paper=paper,
                 grid_m=float(rng.choice([5, 10, 10, 20])) if rng.random() < 0.85 else 0.0,
                 grid_rgb=(g, g, g), grid_px=j(1, 2), outline_px=j(1, 3) if rng.random() < 0.5 else 0,
                 zone_outline_px=j(1, 2) if rng.random() < 0.4 else 0, hatch=rng.random() < 0.25,
                 leak_shape=str(rng.choice(["dot", "diamond", "ring"])), leak_r_m=float(rng.uniform(0.35, 0.7)),
                 leak_rgb=tuple(int(c) for c in cv2.cvtColor(np.uint8([[[j(103, 122), j(150, 255), j(160, 235)]]]),
                                                          cv2.COLOR_HSV2RGB)[0, 0]),
                 labels=float(rng.uniform(0, 1)), label_scale=float(rng.uniform(0.3, 0.6)), dims=j(0, 6),
                 clutter=j(0, 4), noise=float(rng.uniform(0, 8)), blur=float(rng.choice([0, 0, 0.6, 1.0])),
                 jpeg=int(rng.choice([0, 0, 60, 80, 92])))


def hard_style() -> Style:
    """A plausible 'different company' style the hand-tuned OpenCV rules were never tuned for."""
    return Style(px_per_m=12, band_grey={2.0: 182, 5.0: 128, 9.0: 72}, paper=(246, 246, 243),
                 tints={3: (250, 170, 175), 2: (252, 205, 150), 1: (250, 240, 150)}, grid_m=5.0,
                 grid_rgb=(165, 165, 165), grid_px=1, outline_px=2, zone_outline_px=1, hatch=True,
                 leak_shape="ring", leak_r_m=0.45, labels=1.0, label_scale=0.5, dims=4, clutter=3,
                 noise=5.0, blur=0.6, jpeg=70)


_CLUTTER = None


def _clutter_bank():
    global _CLUTTER
    if _CLUTTER is None:
        _CLUTTER = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in sorted(CLUTTER_DIR.glob("*.png"))]
        _CLUTTER = [c for c in _CLUTTER if c is not None]
    return _CLUTTER


def render(spec: Spec, style: Style, rng: np.random.Generator | None = None):
    """Draw a plant in a style. Returns (rgb image, labels) with labels at the image resolution:
    'equip' 0..3 (none / 2 m / 5 m / 9 m band), 'zone' 0..3 (grade, also under equipment),
    'leaks' (k, 2) in metres."""
    rng = rng or np.random.default_rng(0)
    s = style.px_per_m
    W, H = int(round(spec.width * s)), int(round(spec.depth * s))
    p = lambda v: int(round(v * s))
    img = np.empty((H, W, 3), np.uint8)
    img[:] = style.paper
    zone = np.zeros((H, W), np.uint8)
    equip = np.zeros((H, W), np.uint8)

    if style.grid_m:
        step = max(int(style.grid_m * s), 4)
        for x in range(0, W, step):
            cv2.line(img, (x, 0), (x, H), style.grid_rgb, style.grid_px)
        for y in range(0, H, step):
            cv2.line(img, (0, y), (W, y), style.grid_rgb, style.grid_px)
    for g, x0, y0, x1, y1 in sorted(spec.zones):
        cv2.rectangle(img, (p(x0), p(y0)), (p(x1) - 1, p(y1) - 1), style.tints[g], -1)
        sub = zone[p(y0):p(y1), p(x0):p(x1)]
        np.maximum(sub, g, out=sub)
    if style.grid_m and rng.random() < 0.5:   # grid on top of tints too
        step = max(int(style.grid_m * s), 4)
        for x in range(0, W, step):
            cv2.line(img, (x, 0), (x, H), style.grid_rgb, 1)
        for y in range(0, H, step):
            cv2.line(img, (0, y), (W, y), style.grid_rgb, 1)
    if style.zone_outline_px:
        for g, x0, y0, x1, y1 in spec.zones:
            _dashed_rect(img, p(x0), p(y0), p(x1) - 1, p(y1) - 1, (90, 90, 90), style.zone_outline_px, int(s))

    for ob in spec.obstacles:
        grey = style.band_grey[min(BAND_HEIGHTS, key=lambda b: abs(b - ob.height))]
        lab = 1 + int(np.argmin([abs(b - ob.height) for b in BAND_HEIGHTS]))
        m = np.zeros((H, W), np.uint8)
        if ob.kind == "rect":
            x0, y0, x1, y1 = ob.geom
            cv2.rectangle(m, (p(x0), p(y0)), (p(x1) - 1, p(y1) - 1), 1, -1)
        else:
            cx, cy, r = ob.geom
            cv2.circle(m, (p(cx), p(cy)), p(r), 1, -1)
        mb = m.astype(bool)
        img[mb] = grey
        if style.hatch:
            hatch = np.zeros((H, W), np.uint8)
            for k in range(-H, W, max(int(s * 0.8), 5)):
                cv2.line(hatch, (k, 0), (k + H, H), 1, 1)
            hm = mb & hatch.astype(bool)
            img[hm] = max(grey - 35, 0)
        if style.outline_px:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnts, -1, (25, 25, 25), style.outline_px)
        equip[mb] = lab

    for _ in range(style.dims):    # dimension lines / pipe routes: thin dark lines with ticks
        if rng.random() < 0.5:
            y = int(rng.integers(0, H)); x0, x1 = sorted(rng.integers(0, W, 2))
            cv2.line(img, (int(x0), y), (int(x1), y), (40, 40, 40), 1)
            for x in (x0, x1):
                cv2.line(img, (int(x), y - 4), (int(x), y + 4), (40, 40, 40), 1)
        else:
            x = int(rng.integers(0, W)); y0, y1 = sorted(rng.integers(0, H, 2))
            cv2.line(img, (x, int(y0)), (x, int(y1)), (40, 40, 40), 1)

    r = max(int(style.leak_r_m * s), 2)
    for x, y in spec.leaks:
        c = (p(x), p(y))
        if style.leak_shape == "dot":
            cv2.circle(img, c, r, style.leak_rgb, -1)
        elif style.leak_shape == "diamond":
            pts = np.array([[c[0], c[1] - r - 1], [c[0] + r + 1, c[1]], [c[0], c[1] + r + 1], [c[0] - r - 1, c[1]]])
            cv2.fillPoly(img, [pts], style.leak_rgb)
        else:
            cv2.circle(img, c, r, style.leak_rgb, max(r // 2, 2))
            cv2.circle(img, c, 1, style.leak_rgb, -1)

    words = ["K-101", "V-201", "E-301", "P-401", "TK-6", "PR-1", "EL+5.0", "TOS 9.0", "N 1250", "E 340",
             "HOLD", "B-1", "S-701", "XV-12", "PSV-3", "BOP"]
    for ob in spec.obstacles:
        if rng.random() < style.labels:
            x, y = (ob.geom[0], ob.geom[1]) if ob.kind == "rect" else (ob.geom[0] - ob.geom[2], ob.geom[1] - ob.geom[2])
            cv2.putText(img, str(rng.choice(words)), (p(x) + 2, p(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        style.label_scale, (20, 20, 20), 1, cv2.LINE_AA)
    for _ in range(int(style.labels * 6)):
        cv2.putText(img, str(rng.choice(words)), (int(rng.integers(0, W - 40)), int(rng.integers(12, H))),
                    cv2.FONT_HERSHEY_SIMPLEX, style.label_scale, (20, 20, 20), 1, cv2.LINE_AA)

    bank = _clutter_bank()
    if bank:
        for _ in range(style.clutter):
            src = bank[rng.integers(len(bank))]
            cw, ch = int(rng.integers(40, 140)), int(rng.integers(30, 110))
            if src.shape[0] <= ch or src.shape[1] <= cw:
                continue
            sy, sx = int(rng.integers(0, src.shape[0] - ch)), int(rng.integers(0, src.shape[1] - cw))
            patch = src[sy:sy + ch, sx:sx + cw]
            if W <= cw or H <= ch:
                continue
            ty, tx = int(rng.integers(0, H - ch)), int(rng.integers(0, W - cw))
            ink = patch < 128
            reg = img[ty:ty + ch, tx:tx + cw]
            reg[ink] = (30, 30, 30)
    if spec.keepout:
        x0, y0, x1, y1 = spec.keepout
        cv2.rectangle(img, (p(x0), p(y0)), (p(x1) - 1, p(y1) - 1), style.paper, -1)
        cv2.rectangle(img, (p(x0) + 2, p(y0) + 2), (p(x1) - 3, p(y1) - 3), (20, 20, 20), 2)
        for k in range(1, 4):
            yy = p(y0) + k * (p(y1) - p(y0)) // 4
            cv2.line(img, (p(x0) + 2, yy), (p(x1) - 3, yy), (20, 20, 20), 1)
        cv2.putText(img, "PLOT PLAN  DWG-0" + str(rng.integers(100, 999)), (p(x0) + 6, p(y0) + int(s * 1.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(style.label_scale, 0.35), (20, 20, 20), 1, cv2.LINE_AA)

    if style.blur:
        img = cv2.GaussianBlur(img, (0, 0), style.blur)
    if style.noise:
        img = np.clip(img.astype(np.int16) + rng.normal(0, style.noise, img.shape), 0, 255).astype(np.uint8)
    if style.jpeg:
        ok, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, style.jpeg])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy()
    return img, {"equip": equip, "zone": zone, "leaks": np.asarray(spec.leaks, float).reshape(-1, 2)}


def _dashed_rect(img, x0, y0, x1, y1, col, th, dash):
    dash = max(dash, 4)
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        L = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        for t in range(0, L, 2 * dash):
            u, v = t / max(L, 1), min(t + dash, L) / max(L, 1)
            cv2.line(img, (int(a[0] + (b[0] - a[0]) * u), int(a[1] + (b[1] - a[1]) * u)),
                     (int(a[0] + (b[0] - a[0]) * v), int(a[1] + (b[1] - a[1]) * v)), col, th)
