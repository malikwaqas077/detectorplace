"""Detector catalogue, candidate mounting positions and the installed-cost model.

Performance and cost figures are illustrative placeholders, not vendor data. In a real
project they come from the manufacturer's certified performance (e.g. FM 3260 flame
response for a stated fire size) and the company's own estimating rates.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .plant import RES, Plant


@dataclass(frozen=True)
class FlameModel:
    name: str = "IR3 flame detector"
    range_m: float = 20.0           # on-axis detection distance for the design fire
    hfov_deg: float = 90.0          # horizontal cone of vision
    vfov_deg: float = 90.0          # vertical cone of vision
    offaxis_loss: float = 0.45      # range falls to (1 - loss) at the edge of the cone
    fire_height_m: float = 0.5      # height of the design fire above grade
    supply: float = 3200.0
    install: float = 1100.0
    high_mount_extra: float = 700.0  # bracket / access for the upper mounting height


@dataclass(frozen=True)
class GasModel:
    name: str = "Point IR gas detector"
    radius_m: float = 5.0           # coverage radius for the design gas cloud
    mount_m: float = 0.5            # heavier-than-air gas: mounted low
    supply: float = 1500.0
    install: float = 600.0


@dataclass(frozen=True)
class CostModel:
    cable_per_m: float = 28.0       # supply + install, routed to the F&G panel
    tail_m: float = 10.0            # drop / tail allowance per device
    io_per_device: float = 300.0    # panel I/O, termination and commissioning
    fixed: float = 18000.0          # panel, engineering, cause & effect, FAT (not optimised)


AZIMUTHS = np.deg2rad(np.arange(0, 360, 45.0))
MOUNT_HEIGHTS = (3.0, 6.0)
TILTS = {3.0: np.deg2rad(15.0), 6.0: np.deg2rad(30.0)}


@dataclass
class Candidates:
    """Flame candidates are (position x orientation x height); gas candidates are positions."""
    f_xy: np.ndarray     # (P, 2) flame positions
    f_x: np.ndarray      # per candidate (P * n_opt)
    f_y: np.ndarray
    f_z: np.ndarray
    f_az: np.ndarray
    f_tilt: np.ndarray
    f_cost: np.ndarray
    n_opt: int
    g_xy: np.ndarray     # (G, 2) gas positions
    g_cost: np.ndarray

    @property
    def n_flame(self):
        return len(self.f_x)

    def flame_option(self, k):
        o = k % self.n_opt
        return MOUNT_HEIGHTS[o // len(AZIMUTHS)], np.rad2deg(AZIMUTHS[o % len(AZIMUTHS)])


def _free(plant: Plant, x, y, clearance=0.6):
    iy, ix = int(y / RES), int(x / RES)
    r = int(np.ceil(clearance / RES))
    win = plant.heights[max(iy - r, 0):iy + r + 1, max(ix - r, 0):ix + r + 1]
    return win.size and float(win.max()) == 0.0


def cable_length(plant: Plant, x, y, cost: CostModel):
    return abs(x - plant.panel[0]) + abs(y - plant.panel[1]) + cost.tail_m


def make_candidates(plant: Plant, flame=FlameModel(), gas=GasModel(), cost=CostModel(),
                    flame_step=3.0, gas_step=2.0) -> Candidates:
    graded = np.argwhere(plant.grades > 0)
    gy, gx = (graded[:, 0] + 0.5) * RES, (graded[:, 1] + 0.5) * RES
    f_xy = []
    for y in np.arange(1.5, plant.depth - 1, flame_step):
        for x in np.arange(1.5, plant.width - 1, flame_step):
            if _free(plant, x, y) and np.min((gx - x) ** 2 + (gy - y) ** 2) <= flame.range_m ** 2:
                f_xy.append((x, y))
    f_xy = np.array(f_xy)
    rows = []
    for x, y in f_xy:
        cab = cable_length(plant, x, y, cost) * cost.cable_per_m
        for z in MOUNT_HEIGHTS:
            for az in AZIMUTHS:
                c = flame.supply + flame.install + cab + cost.io_per_device + (flame.high_mount_extra if z > 3 else 0)
                rows.append((x, y, z, az, TILTS[z], c))
    rows = np.array(rows)
    g_xy = []
    for y in np.arange(1.0, plant.depth - 1, gas_step):
        for x in np.arange(1.0, plant.width - 1, gas_step):
            if len(plant.leaks) and _free(plant, x, y, 0.3) and \
                    np.min(np.linalg.norm(plant.leaks - (x, y), axis=1)) <= 8.0:
                g_xy.append((x, y))
    g_xy = np.array(g_xy).reshape(-1, 2)
    g_cost = np.array([gas.supply + gas.install + cost.io_per_device +
                       cable_length(plant, x, y, cost) * cost.cable_per_m for x, y in g_xy])
    return Candidates(f_xy, rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5],
                      len(MOUNT_HEIGHTS) * len(AZIMUTHS), g_xy, g_cost)


def catalogue_dict(flame=FlameModel(), gas=GasModel(), cost=CostModel()):
    return {"flame": asdict(flame), "gas": asdict(gas), "cost": asdict(cost)}
