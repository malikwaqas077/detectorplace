"""Geometric coverage engine: 3D line-of-sight ray marching over the 2.5D height map.

For every (candidate detector, target cell) pair we test range, the horizontal and
vertical cones of vision, off-axis range loss and obstruction by equipment along the
3D sight line. The result is a sparse-in-practice 0/1 coverage matrix that the
optimisers work on. The inner loops are compiled with Numba and run in parallel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .detectors import Candidates, FlameModel, GasModel
from .plant import RES, Plant

TARGET_STEP = 1.0   # metres between target points


@njit(cache=True, fastmath=True)
def _visible(hmap, res, x0, y0, z0, x1, y1, z1):
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    n = int(np.sqrt(dx * dx + dy * dy) / (res * 0.5)) + 1
    ny, nx = hmap.shape
    for s in range(1, n):
        t = s / n
        ix = int((x0 + dx * t) / res)
        iy = int((y0 + dy * t) / res)
        if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
            return False
        if hmap[iy, ix] >= z0 + dz * t:
            return False
    return True


@njit(parallel=True, cache=True, fastmath=True)
def _cone_coverage(hmap, res, cx, cy, cz, caz, ctilt, tx, ty, tz, rng, half_h, half_v, loss):
    nc, nt = cx.shape[0], tx.shape[0]
    out = np.zeros((nc, nt), np.uint8)
    for i in prange(nc):
        for j in range(nt):
            dx = tx[j] - cx[i]
            dy = ty[j] - cy[i]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-6 or d > rng:
                continue
            a = np.arctan2(dy, dx) - caz[i]
            while a > np.pi:
                a -= 2 * np.pi
            while a < -np.pi:
                a += 2 * np.pi
            if abs(a) > half_h:
                continue
            elev = np.arctan2(tz[j] - cz[i], d)
            if abs(elev + ctilt[i]) > half_v:
                continue
            if half_h < np.pi and d > rng * (1.0 - loss * (a / half_h) ** 2):
                continue
            if _visible(hmap, res, cx[i], cy[i], cz[i], tx[j], ty[j], tz[j]):
                out[i, j] = 1
    return out


@dataclass
class Targets:
    xy: np.ndarray        # (n, 2)
    grade: np.ndarray     # fire grade per target (0 for gas-only points)
    cells: np.ndarray     # (n, 2) row/col in the target raster
    shape: tuple


def fire_targets(plant: Plant, step=TARGET_STEP) -> Targets:
    xs, ys = np.arange(step / 2, plant.width, step), np.arange(step / 2, plant.depth, step)
    pts, g, cells = [], [], []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            iy, ix = int(y / RES), int(x / RES)
            if plant.grades[iy, ix] > 0 and plant.heights[iy, ix] == 0:
                pts.append((x, y)); g.append(plant.grades[iy, ix]); cells.append((r, c))
    return Targets(np.array(pts), np.array(g, np.int8), np.array(cells), (len(ys), len(xs)))


def gas_targets(plant: Plant, radius=6.0, step=TARGET_STEP) -> Targets:
    """Points where a gas cloud from a credible leak source would be expected."""
    xs, ys = np.arange(step / 2, plant.width, step), np.arange(step / 2, plant.depth, step)
    pts, cells = [], []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            iy, ix = int(y / RES), int(x / RES)
            if plant.heights[iy, ix] == 0 and len(plant.leaks) and \
                    np.min(np.hypot(plant.leaks[:, 0] - x, plant.leaks[:, 1] - y)) <= radius:
                pts.append((x, y)); cells.append((r, c))
    return Targets(np.array(pts).reshape(-1, 2), np.zeros(len(pts), np.int8), np.array(cells).reshape(-1, 2),
                   (len(ys), len(xs)))


def flame_matrix(plant: Plant, cand: Candidates, tg: Targets, fm: FlameModel = FlameModel()):
    tz = np.full(len(tg.xy), fm.fire_height_m)
    return _cone_coverage(plant.heights.astype(np.float32), RES, cand.f_x, cand.f_y, cand.f_z, cand.f_az,
                          cand.f_tilt, tg.xy[:, 0].copy(), tg.xy[:, 1].copy(), tz, fm.range_m,
                          np.deg2rad(fm.hfov_deg / 2), np.deg2rad(fm.vfov_deg / 2), fm.offaxis_loss)


def gas_matrix(plant: Plant, cand: Candidates, tg: Targets, gm: GasModel = GasModel()):
    """Point gas detectors: within radius and not shielded by equipment at mounting height."""
    n = len(cand.g_xy)
    z = np.full(n, gm.mount_m)
    return _cone_coverage(plant.heights.astype(np.float32), RES, cand.g_xy[:, 0].copy(), cand.g_xy[:, 1].copy(),
                          z, np.zeros(n), np.zeros(n), tg.xy[:, 0].copy(), tg.xy[:, 1].copy(),
                          np.full(len(tg.xy), gm.mount_m), gm.radius_m, np.pi, np.pi, 0.0)


def flame_wedge(x, y, az_deg, fm: FlameModel = FlameModel(), n=24):
    """Polygon of the nominal (unobstructed) cone footprint, for drawing."""
    half = np.deg2rad(fm.hfov_deg / 2)
    a = np.linspace(-half, half, n)
    r = fm.range_m * (1 - fm.offaxis_loss * (a / half) ** 2)
    az = np.deg2rad(az_deg)
    px = np.concatenate([[x], x + r * np.cos(az + a), [x]])
    py = np.concatenate([[y], y + r * np.sin(az + a), [y]])
    return px, py
