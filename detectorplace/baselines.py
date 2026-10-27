"""Reference methods used to validate the genetic algorithm.

* greedy: the cost-effectiveness heuristic a designer effectively follows by hand
  (place the detector that buys the most missing coverage per pound, repeat).
* exact: a mixed-integer linear program solved with HiGHS that finds the cheapest
  layout meeting every target. It handles the linear version of the problem only, but
  gives a provable optimum (or a bound) to measure the GA against.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .evaluate import Design, Problem, coverage, need_vectors, objectives


def greedy(p: Problem, max_steps=200) -> Design:
    sel_f, sel_g = [], []
    cf = np.zeros(p.Mf.shape[1], np.int16)
    cg = np.zeros(p.Mg.shape[1], np.int16)
    used_pos = set()
    for _ in range(max_steps):
        d = Design(np.array(sel_f, int), np.array(sel_g, int))
        cov = coverage(p, d, cf, cg)
        if all(cov[r.key] >= p.goal(r) for r in p.reqs):
            break
        need_f, need_g = need_vectors(p, cf, cg, cov)
        gain_f = (p.Mf32 @ need_f) / p.f_cost
        if used_pos:
            blocked = np.concatenate([np.arange(q * p.n_opt, (q + 1) * p.n_opt) for q in used_pos])
            gain_f[blocked] = -1
        gain_g = (p.Mg32 @ need_g) / p.g_cost if len(p.g_cost) else np.array([])
        if len(sel_g):
            gain_g[sel_g] = -1
        bf = int(np.argmax(gain_f))
        bg = int(np.argmax(gain_g)) if len(gain_g) else -1
        if bg >= 0 and gain_g[bg] > gain_f[bf]:
            sel_g.append(bg); cg += p.Mg[bg]
        else:
            sel_f.append(bf); cf += p.Mf[bf]; used_pos.add(bf // p.n_opt)
    return _prune_exact(p, Design(np.array(sel_f, int), np.array(sel_g, int)))


def _prune_exact(p: Problem, d: Design) -> Design:
    """Remove any detector whose removal keeps all targets met, most expensive first."""
    f, g = list(d.flame), list(d.gas)
    items = sorted([("f", k, p.f_cost[k]) for k in f] + [("g", k, p.g_cost[k]) for k in g], key=lambda t: -t[2])
    for kind, k, _ in items:
        trial_f = [x for x in f if not (kind == "f" and x == k)]
        trial_g = [x for x in g if not (kind == "g" and x == k)]
        if objectives(p, Design(np.array(trial_f, int), np.array(trial_g, int)))[2]:
            f, g = trial_f, trial_g
    return Design(np.array(f, int), np.array(g, int))


def exact(p: Problem, time_limit=120.0, gap=0.01):
    """Cheapest layout meeting every target. Returns (design, info)."""
    nF, nG = p.Mf.shape[0], p.Mg.shape[0]
    tF, tG = p.Mf.shape[1], p.Mg.shape[1]
    reqs2 = [r for r in p.reqs if r.votes == 2]
    a_idx = np.nonzero(p.masks[reqs2[0].key])[0] if reqs2 else np.array([], int)
    nA = len(a_idx)
    # variables: [x_f (nF) | x_g (nG) | y_f (tF, cont) | y_g (tG, cont) | z_A (nA, bin)]
    n = nF + nG + tF + tG + nA
    off_g, off_yf, off_yg, off_z = nF, nF + nG, nF + nG + tF, nF + nG + tF + tG
    c = np.concatenate([p.f_cost, p.g_cost, np.zeros(tF + tG + nA)])

    Mf = sparse.csr_matrix(p.Mf.T.astype(np.float64))
    Mg = sparse.csr_matrix(p.Mg.T.astype(np.float64))
    blocks, lo, hi = [], [], []
    # y_f <= sum x  (a point counts as seen only if some chosen detector sees it)
    blocks.append(sparse.hstack([-Mf, sparse.csr_matrix((tF, nG)), sparse.eye(tF),
                                 sparse.csr_matrix((tF, tG + nA))]))
    lo.append(np.full(tF, -np.inf)); hi.append(np.zeros(tF))
    blocks.append(sparse.hstack([sparse.csr_matrix((tG, nF)), -Mg, sparse.csr_matrix((tG, tF)), sparse.eye(tG),
                                 sparse.csr_matrix((tG, nA))]))
    lo.append(np.full(tG, -np.inf)); hi.append(np.zeros(tG))
    if nA:  # 2 z <= sum x   (voted coverage)
        blocks.append(sparse.hstack([-Mf[a_idx], sparse.csr_matrix((nA, nG + tF + tG)), 2 * sparse.eye(nA)]))
        lo.append(np.full(nA, -np.inf)); hi.append(np.zeros(nA))
    # coverage targets
    rows = []
    for r in p.reqs:
        row = np.zeros(n)
        m = p.masks[r.key]
        if r.kind == "fire" and r.votes == 1:
            row[off_yf + np.nonzero(m)[0]] = 1
        elif r.kind == "gas":
            row[off_yg + np.nonzero(m)[0]] = 1
        else:
            row[off_z:off_z + nA] = 1
        rows.append(row); lo.append([np.ceil(p.goal(r) * m.sum() - 1e-9)]); hi.append([np.inf])
    blocks.append(sparse.csr_matrix(np.array(rows)))
    # one option per flame mounting position
    P = p.n_fpos
    sel = sparse.kron(sparse.eye(P), np.ones((1, p.n_opt)))
    blocks.append(sparse.hstack([sel, sparse.csr_matrix((P, n - nF))]))
    lo.append(np.full(P, -np.inf)); hi.append(np.ones(P))

    A = sparse.vstack(blocks).tocsr()
    cons = LinearConstraint(A, np.concatenate(lo), np.concatenate(hi))
    integrality = np.concatenate([np.ones(nF + nG), np.zeros(tF + tG), np.ones(nA)])
    ub = np.ones(n)
    t0 = time.perf_counter()
    res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(np.zeros(n), ub),
               options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": False})
    info = {"status": res.message, "seconds": round(time.perf_counter() - t0, 1),
            "variables": int(n), "constraints": int(A.shape[0])}
    if res.x is None:
        return None, info
    x = res.x
    d = Design(np.nonzero(x[:nF] > 0.5)[0], np.nonzero(x[off_g:off_g + nG] > 0.5)[0])
    info["objective"] = float(res.fun)
    info["bound"] = float(getattr(res, "mip_dual_bound", np.nan) or np.nan)
    info["gap"] = float(getattr(res, "mip_gap", np.nan) or 0.0)
    return d, info
