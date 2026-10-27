"""Performance targets and design evaluation.

Targets follow the structure of performance-based F&G mapping (ISA-TR84.00.07 style):
each hazard grade has a minimum fraction of its area that must be seen by at least one
detector (1ooN, alarm) and, for the highest grade, by at least two (2ooN, voted
executive action). The numbers below are illustrative defaults; on a real project they
come from the site's F&G philosophy and hazard analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coverage import Targets


@dataclass(frozen=True)
class Requirement:
    key: str
    label: str
    kind: str          # "fire" or "gas"
    grade: int         # fire grade, 0 for gas
    votes: int         # detectors that must see a point (1 = 1ooN, 2 = 2ooN)
    target: float      # required coverage fraction
    weight: float      # risk weight used in the shortfall objective


DEFAULT_REQS = (
    Requirement("A1", "Grade A fire, 1ooN", "fire", 3, 1, 0.90, 5.0),
    Requirement("A2", "Grade A fire, 2ooN", "fire", 3, 2, 0.75, 4.0),
    Requirement("B1", "Grade B fire, 1ooN", "fire", 2, 1, 0.85, 3.0),
    Requirement("C1", "Grade C fire, 1ooN", "fire", 1, 1, 0.60, 1.0),
    Requirement("G1", "Gas release zones, 1ooN", "gas", 0, 1, 0.80, 5.0),
)


@dataclass
class Problem:
    """Everything the optimisers need, as plain arrays."""
    Mf: np.ndarray            # (n_flame_cand, n_fire_targets) uint8
    Mg: np.ndarray            # (n_gas_cand, n_gas_targets) uint8
    f_cost: np.ndarray
    g_cost: np.ndarray
    n_opt: int                # options per flame position
    fire: Targets
    gas: Targets
    reqs: tuple = DEFAULT_REQS
    margin: float = 0.0       # design margin added to every target while optimising
    masks: dict = field(default_factory=dict)

    def __post_init__(self):
        for r in self.reqs:
            self.masks[r.key] = (self.fire.grade == r.grade) if r.kind == "fire" else np.ones(len(self.gas.xy), bool)
        self.wsum = sum(r.weight for r in self.reqs)
        self.Mf32 = self.Mf.astype(np.float32)
        self.Mg32 = self.Mg.astype(np.float32)

    def goal(self, r: Requirement) -> float:
        return min(r.target + self.margin, 1.0)

    @property
    def n_fpos(self):
        return len(self.f_cost) // self.n_opt


@dataclass
class Design:
    flame: np.ndarray   # selected flame candidate indices
    gas: np.ndarray     # selected gas candidate indices

    def key(self):
        return (tuple(sorted(self.flame.tolist())), tuple(sorted(self.gas.tolist())))


def counts(p: Problem, d: Design):
    cf = p.Mf[d.flame].sum(0, dtype=np.int16) if len(d.flame) else np.zeros(p.Mf.shape[1], np.int16)
    cg = p.Mg[d.gas].sum(0, dtype=np.int16) if len(d.gas) else np.zeros(p.Mg.shape[1], np.int16)
    return cf, cg


def coverage(p: Problem, d: Design, cf=None, cg=None) -> dict:
    if cf is None:
        cf, cg = counts(p, d)
    out = {}
    for r in p.reqs:
        c = cf if r.kind == "fire" else cg
        m = p.masks[r.key]
        out[r.key] = float(np.mean(c[m] >= r.votes)) if m.any() else 1.0
    return out


def cost(p: Problem, d: Design) -> float:
    return float(p.f_cost[d.flame].sum() + p.g_cost[d.gas].sum())


def shortfall(p: Problem, cov: dict) -> float:
    return sum(r.weight * max(0.0, p.goal(r) - cov[r.key]) / p.goal(r) for r in p.reqs) / p.wsum


def need_vectors(p: Problem, cf, cg, cov):
    """Value of newly covering each target point, given what is still short of target."""
    need_f = np.zeros(p.Mf.shape[1], np.float32)
    need_g = np.zeros(p.Mg.shape[1], np.float32)
    for r in p.reqs:
        if cov[r.key] >= p.goal(r):
            continue
        m = p.masks[r.key]
        if r.kind == "fire":
            need_f[m & (cf < r.votes)] += r.weight / m.sum()
        else:
            need_g[m & (cg < r.votes)] += r.weight / m.sum()
    return need_f, need_g


def objectives(p: Problem, d: Design):
    """(installed cost, risk-weighted shortfall against targets, feasible)."""
    cov = coverage(p, d)
    short = shortfall(p, cov)
    # a small pull towards more coverage once targets are met keeps the front informative
    surplus = sum(r.weight * (1 - cov[r.key]) for r in p.reqs) / p.wsum
    return cost(p, d), short + 0.02 * surplus, short == 0.0, cov
