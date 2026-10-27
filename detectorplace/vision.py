"""Read a plot-plan drawing back into a 2.5D plant model with classical computer vision.

Pipeline: colour-space segmentation -> morphology -> connected components / contours
-> elevation band from fill shade -> resample to the analysis grid. Text, grid lines and
noise are rejected by saturation, brightness and minimum-area rules.
"""
from __future__ import annotations

import cv2
import numpy as np

from .plant import HEIGHT_BANDS, PX_PER_M, RES, Obstacle, Plant

MIN_OBSTACLE_PX = 120     # ~1.2 m^2; smaller dark blobs are text or noise


def _to_grid(mask_px: np.ndarray, shape) -> np.ndarray:
    small = cv2.resize(mask_px.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return small >= 0.5


def extract_plant(img_rgb: np.ndarray, px_per_m: float = PX_PER_M, name="Plant (from drawing)"):
    """Return (Plant, debug) where debug holds intermediate masks for display."""
    img = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    H, W = s.shape
    shape = (int(round(H / px_per_m / RES)), int(round(W / px_per_m / RES)))

    # Equipment: unsaturated and darker than the grid lines.
    dark = ((s < 45) & (v < 200)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    heights_px = np.zeros((H, W), np.float32)
    obstacles = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < MIN_OBSTACLE_PX:
            continue
        comp = labels == k
        core = cv2.erode(comp.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        grey = float(np.median(v[core if core.any() else comp]))
        band = min(HEIGHT_BANDS, key=lambda b: abs(b[0] - grey))
        heights_px[comp] = band[1]
        x, y, w, hh = stats[k, :4]
        obstacles.append(Obstacle(f"OBS-{len(obstacles) + 1:02d}", "rect",
                                  (x / px_per_m, y / px_per_m, (x + w) / px_per_m, (y + hh) / px_per_m),
                                  band[1]))
    occ_frac = cv2.resize((heights_px > 0).astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    h_sum = cv2.resize(heights_px, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    occ = occ_frac >= 0.5
    mean_h = h_sum / np.maximum(occ_frac, 1e-6)          # mean height of the occupied pixels
    bands = np.array([b[1] for b in HEIGHT_BANDS])
    heights = np.where(occ, bands[np.abs(mean_h[..., None] - bands).argmin(-1)], 0).astype(np.float32)

    # Hazard zones: pale tints separated by hue (OpenCV hue is 0-179).
    tinted = (s > 30) & (s < 140) & (v > 200)
    zone_px = {3: tinted & ((h < 8) | (h > 170)), 2: tinted & (h >= 10) & (h < 21),
               1: tinted & (h >= 21) & (h < 35)}
    grades = np.zeros(shape, np.int8)
    for g in (1, 2, 3):
        m = cv2.morphologyEx(zone_px[g].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        # a zone continues underneath equipment drawn on top of it
        grades[_to_grid(m, shape)] = g
    grades = _fill_under_equipment(grades, occ)

    # Leak sources: saturated blue markers.
    blue = ((h > 100) & (h < 130) & (s > 120)).astype(np.uint8)
    n2, _, st2, cents = cv2.connectedComponentsWithStats(blue)
    leaks = np.array([cents[k] / px_per_m for k in range(1, n2) if st2[k, cv2.CC_STAT_AREA] >= 20])

    plant = Plant(W / px_per_m, H / px_per_m, heights, grades, leaks.reshape(-1, 2), obstacles=obstacles,
                  name=name)
    return plant, {"equipment_px": dark.astype(bool), "zones_px": zone_px, "leak_px": blue.astype(bool)}


def _fill_under_equipment(grades, occ):
    """Assign each covered cell the highest grade among its neighbours (iterative dilation)."""
    g = grades.copy()
    for _ in range(40):
        todo = occ & (g == 0)
        if not todo.any():
            break
        grown = cv2.dilate(g.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.int8)
        g[todo] = grown[todo]
    return g


def score_extraction(truth: Plant, got: Plant) -> dict:
    """Agreement between the extracted model and the ground truth it was drawn from."""
    t_occ, g_occ = truth.heights > 0, got.heights > 0
    iou = (t_occ & g_occ).sum() / max((t_occ | g_occ).sum(), 1)
    both = t_occ & g_occ
    height_ok = float(np.mean(np.abs(truth.heights[both] - got.heights[both]) < 0.5)) if both.any() else 0.0
    zone_acc = float(np.mean(truth.grades[~t_occ] == got.grades[~t_occ]))
    leak_err = []
    for p in truth.leaks:
        if len(got.leaks):
            leak_err.append(float(np.min(np.linalg.norm(got.leaks - p, axis=1))))
    return {"equipment_iou": float(iou), "height_band_accuracy": height_ok, "zone_accuracy": zone_acc,
            "leaks_found": int(len(got.leaks)), "leaks_true": int(len(truth.leaks)),
            "leak_max_error_m": max(leak_err) if leak_err else None}
