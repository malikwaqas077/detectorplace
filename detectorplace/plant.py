"""2.5D plant model: an obstacle height map, hazard grades and gas leak sources.

The demo plant is an invented onshore gas compression station. Every name, size and
height is synthetic; nothing here describes a real facility.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

RES = 0.5            # metres per height-map cell
PX_PER_M = 10        # plot-plan drawing scale
GRADES = {0: "Ungraded", 1: "Grade C", 2: "Grade B", 3: "Grade A"}

# Drawing convention for the exported plot plan: fill shade encodes the elevation band.
HEIGHT_BANDS = [(60, 9.0), (115, 5.0), (170, 2.0)]   # (grey level, nominal height m)
ZONE_TINTS = {3: (255, 190, 190), 2: (255, 215, 170), 1: (255, 245, 170)}  # RGB
LEAK_RGB = (30, 80, 220)


@dataclass
class Obstacle:
    name: str
    kind: str          # "rect" or "circle"
    geom: tuple        # rect: (x0, y0, x1, y1); circle: (cx, cy, r)  metres
    height: float


@dataclass
class Plant:
    width: float
    depth: float
    heights: np.ndarray            # (ny, nx) obstacle top height in m, 0 = open ground
    grades: np.ndarray             # (ny, nx) 0..3 fire hazard grade
    leaks: np.ndarray              # (k, 2) gas leak source coordinates in m
    panel: tuple = (69.0, 39.0)    # F&G control panel location (cable routing origin)
    obstacles: list = field(default_factory=list)
    name: str = "Plant"

    @property
    def shape(self):
        return self.heights.shape

    def cell(self, x, y):
        return int(y / RES), int(x / RES)


def _rect_mask(shape, x0, y0, x1, y1):
    m = np.zeros(shape, bool)
    m[int(round(y0 / RES)):int(round(y1 / RES)), int(round(x0 / RES)):int(round(x1 / RES))] = True
    return m


def _circle_mask(shape, cx, cy, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return ((xx + 0.5) * RES - cx) ** 2 + ((yy + 0.5) * RES - cy) ** 2 <= r * r


def obstacle_mask(shape, ob: Obstacle):
    return _rect_mask(shape, *ob.geom) if ob.kind == "rect" else _circle_mask(shape, *ob.geom)


DEMO_OBSTACLES = [
    Obstacle("K-101 Compressor package A", "rect", (12, 10, 20, 16), 5.0),
    Obstacle("K-102 Compressor package B", "rect", (12, 22, 20, 28), 5.0),
    Obstacle("FW-1 Blast/fire wall", "rect", (25, 6, 25.5, 30), 5.0),
    Obstacle("V-201 Inlet separator", "circle", (34, 13, 2.5), 9.0),
    Obstacle("V-202 Suction scrubber", "circle", (34, 27, 2.0), 9.0),
    Obstacle("E-301 Gas cooler", "rect", (44, 8, 52, 17), 5.0),
    Obstacle("P-401/402 Condensate pump skid", "rect", (46, 21, 54, 25), 2.0),
    Obstacle("M-501 Fiscal metering skid", "rect", (60, 10, 68, 16), 2.0),
    Obstacle("T-601 Condensate tank", "circle", (66, 27, 4.0), 9.0),
    Obstacle("PR-1 Pipe rack", "rect", (8, 34, 72, 37), 5.0),
    Obstacle("B-1 Control building", "rect", (62, 40, 76, 48), 5.0),
]

DEMO_ZONES = [  # (grade, x0, y0, x1, y1); later entries win only if higher grade
    (1, 4, 3, 76, 33),
    (2, 40, 5, 58, 30),
    (2, 57, 7, 71, 19),
    (2, 59, 20, 73, 33),
    (3, 9, 7, 23, 31),
    (3, 29, 8, 39, 32),
]

DEMO_LEAKS = np.array([[21.0, 13.0], [21.0, 25.0], [11.0, 19.0], [30.5, 13.0],
                       [30.5, 27.0], [50.0, 20.0], [59.0, 13.0], [61.0, 27.0]])


def demo_plant() -> Plant:
    """The ground-truth plant the synthetic drawing is rendered from."""
    w, d = 80.0, 50.0
    shape = (int(d / RES), int(w / RES))
    heights = np.zeros(shape, np.float32)
    for ob in DEMO_OBSTACLES:
        m = obstacle_mask(shape, ob)
        heights[m] = np.maximum(heights[m], ob.height)
    grades = np.zeros(shape, np.int8)
    for g, *r in DEMO_ZONES:
        m = _rect_mask(shape, *r)
        grades[m] = np.maximum(grades[m], g)
    return Plant(w, d, heights, grades, DEMO_LEAKS.copy(), obstacles=list(DEMO_OBSTACLES),
                 name="Synthetic gas compression station")


def render_plot_plan(plant: Plant, noise: float = 4.0, seed: int = 0) -> np.ndarray:
    """Render the plant as a plot-plan image (RGB uint8), the way a CAD export would look.

    Light grid every 10 m, tinted hazard zones, shaded equipment footprints (shade = elevation
    band), blue leak-source markers, a few labels and a little sensor noise.
    """
    W, H = int(plant.width * PX_PER_M), int(plant.depth * PX_PER_M)
    img = np.full((H, W, 3), 255, np.uint8)
    for x in range(0, W, 10 * PX_PER_M):
        cv2.line(img, (x, 0), (x, H), (215, 215, 215), 1)
    for y in range(0, H, 10 * PX_PER_M):
        cv2.line(img, (0, y), (W, y), (215, 215, 215), 1)
    for g, x0, y0, x1, y1 in sorted(DEMO_ZONES):
        p = lambda v: int(round(v * PX_PER_M))
        cv2.rectangle(img, (p(x0), p(y0)), (p(x1) - 1, p(y1) - 1), ZONE_TINTS[g], -1)
    for ob in plant.obstacles:
        grey = min(HEIGHT_BANDS, key=lambda b: abs(b[1] - ob.height))[0]
        col = (grey, grey, grey)
        if ob.kind == "rect":
            x0, y0, x1, y1 = (int(round(v * PX_PER_M)) for v in ob.geom)
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), col, -1)
        else:
            cx, cy, r = ob.geom
            cv2.circle(img, (int(cx * PX_PER_M), int(cy * PX_PER_M)), int(r * PX_PER_M), col, -1)
    for x, y in plant.leaks:
        cv2.circle(img, (int(x * PX_PER_M), int(y * PX_PER_M)), 5, LEAK_RGB, -1)
    for ob in plant.obstacles:  # small tags: the vision step has to ignore them
        x = ob.geom[0] if ob.kind == "rect" else ob.geom[0] - ob.geom[2]
        y = ob.geom[1] if ob.kind == "rect" else ob.geom[1] - ob.geom[2]
        cv2.putText(img, ob.name.split()[0], (int(x * PX_PER_M) + 2, int(y * PX_PER_M) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (20, 20, 20), 1, cv2.LINE_AA)
    if noise:
        rng = np.random.default_rng(seed)
        img = np.clip(img.astype(np.int16) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img
