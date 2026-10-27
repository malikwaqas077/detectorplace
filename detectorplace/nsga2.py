"""Multi-objective genetic algorithm (NSGA-II) for detector layout.

Chromosome: one integer gene per flame mounting position (0 = no detector, k = one of
the 16 height/orientation options) and one 0/1 gene per gas position. Objectives are
installed cost and risk-weighted coverage shortfall, both minimised. Offspring are
improved by a cheap redundancy-pruning step (a memetic GA), and the population can be
seeded with greedy solutions.
"""
from __future__ import annotations

import numpy as np

from .evaluate import Design, Problem, counts, coverage, need_vectors, objectives, shortfall


def decode(p: Problem, fg, gg) -> Design:
    pos = np.nonzero(fg)[0]
    return Design(pos * p.n_opt + (fg[pos] - 1), np.nonzero(gg)[0])


def encode(p: Problem, d: Design):
    fg = np.zeros(p.n_fpos, np.int16)
    for k in d.flame:
        fg[k // p.n_opt] = k % p.n_opt + 1
    gg = np.zeros(len(p.g_cost), np.int8)
    gg[d.gas] = 1
    return fg, gg


def non_dominated_sort(F):
    n = len(F)
    dominated_by = [[] for _ in range(n)]
    n_dom = np.zeros(n, int)
    fronts = [[]]
    for i in range(n):
        le = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
        ge = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominated_by[i] = list(np.nonzero(le)[0])
        n_dom[i] = ge.sum()
        if n_dom[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(F):
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        o = np.argsort(F[:, j])
        span = F[o[-1], j] - F[o[0], j] or 1.0
        d[o[0]] = d[o[-1]] = np.inf
        d[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / span
    return d


def prune(p: Problem, fg, gg, rng):
    """Drop detectors whose removal does not increase the shortfall (most expensive first)."""
    d = decode(p, fg, gg)
    cf, cg = counts(p, d)
    base = shortfall(p, coverage(p, d, cf, cg))
    items = [("f", i, p.f_cost[i * p.n_opt + fg[i] - 1]) for i in np.nonzero(fg)[0]] +             [("g", i, p.g_cost[i]) for i in np.nonzero(gg)[0]]
    items.sort(key=lambda t: -t[2] + rng.random())     # small jitter keeps variety
    for kind, i, _ in items:
        row = p.Mf[i * p.n_opt + fg[i] - 1] if kind == "f" else p.Mg[i]
        if kind == "f":
            cf -= row
        else:
            cg -= row
        s = shortfall(p, coverage(p, d, cf, cg))
        if s <= base + 1e-12:
            if kind == "f":
                fg[i] = 0
            else:
                gg[i] = 0
            base = s
        elif kind == "f":
            cf += row
        else:
            cg += row
    return fg, gg


def repair(p: Problem, fg, gg, max_add=40):
    """Greedily add the most cost-effective detectors until every target is met."""
    d = decode(p, fg, gg)
    cf, cg = counts(p, d)
    used = np.repeat(fg > 0, p.n_opt)
    for _ in range(max_add):
        cov = coverage(p, d, cf, cg)
        if all(cov[r.key] >= p.goal(r) for r in p.reqs):
            break
        need_f, need_g = need_vectors(p, cf, cg, cov)
        gf = np.where(used, -1.0, (p.Mf32 @ need_f) / p.f_cost)
        gg_gain = np.where(gg > 0, -1.0, (p.Mg32 @ need_g) / p.g_cost) if len(gg) else np.array([-1.0])
        bf, bg = int(np.argmax(gf)), int(np.argmax(gg_gain))
        if gg_gain[bg] > gf[bf]:
            gg[bg] = 1
            cg += p.Mg[bg]
        else:
            pos = bf // p.n_opt
            fg[pos] = bf % p.n_opt + 1
            used[pos * p.n_opt:(pos + 1) * p.n_opt] = True
            cf += p.Mf[bf]
    return fg, gg


def run(p: Problem, pop_size=120, generations=160, seed=0, seeds=(), p_on=0.03, p_mut=None,
        prune_rate=0.3, repair_rate=0.4, callback=None):
    rng = np.random.default_rng(seed)
    nf, ng = p.n_fpos, len(p.g_cost)
    p_mut = p_mut or 1.5 / (nf + ng)

    pop = [encode(p, d) for d in seeds]
    while len(pop) < pop_size:
        dens = rng.uniform(0.2, 2.0) * p_on
        fg = np.where(rng.random(nf) < dens, rng.integers(1, p.n_opt + 1, nf), 0).astype(np.int16)
        gg = (rng.random(ng) < dens * 3).astype(np.int8)
        pop.append((fg, gg))

    def evaluate(ind):
        c, s, feas, cov = objectives(p, decode(p, *ind))
        return np.array([c, s]), feas, cov

    evals = [evaluate(i) for i in pop]
    F = np.array([e[0] for e in evals])
    FE = np.array([e[1] for e in evals])
    history = []

    for gen in range(generations):
        fronts = non_dominated_sort(F)
        rank = np.empty(len(pop), int)
        crowd = np.empty(len(pop))
        for r, fr in enumerate(fronts):
            rank[fr] = r
            crowd[fr] = crowding(F[fr])

        def tournament():
            a, b = rng.integers(0, len(pop), 2)
            return a if (rank[a], -crowd[a]) < (rank[b], -crowd[b]) else b

        children = []
        while len(children) < pop_size:
            pa, pb = pop[tournament()], pop[tournament()]
            mf, mg = rng.random(nf) < 0.5, rng.random(ng) < 0.5
            fg = np.where(mf, pa[0], pb[0]).astype(np.int16)
            gg = np.where(mg, pa[1], pb[1]).astype(np.int8)
            # mutation: switch a position on/off or re-aim / re-mount an existing detector
            for i in np.nonzero(rng.random(nf) < p_mut * 4)[0]:
                fg[i] = 0 if (fg[i] and rng.random() < 0.5) else rng.integers(1, p.n_opt + 1)
            for i in np.nonzero(rng.random(ng) < p_mut * 4)[0]:
                gg[i] ^= 1
            u = rng.random()
            if u < repair_rate:                      # memetic step: repair, then trim
                fg, gg = prune(p, *repair(p, fg, gg), rng)
            elif u < repair_rate + prune_rate:
                fg, gg = prune(p, fg, gg, rng)
            children.append((fg, gg))

        cevals = [evaluate(c) for c in children]
        allpop = pop + children
        allF = np.vstack([F, [e[0] for e in cevals]])
        allFE = np.concatenate([FE, [e[1] for e in cevals]])
        # remove exact duplicates so the front stays diverse
        _, uniq = np.unique(np.round(allF, 6), axis=0, return_index=True)
        uniq = np.sort(uniq)
        allpop = [allpop[i] for i in uniq]
        allF = allF[uniq]
        allFE = allFE[uniq]

        new_idx = []
        for fr in non_dominated_sort(allF):
            if len(new_idx) + len(fr) <= pop_size:
                new_idx += fr
            else:
                cd = crowding(allF[fr])
                new_idx += [fr[i] for i in np.argsort(-cd)[:pop_size - len(new_idx)]]
                break
        pop = [allpop[i] for i in new_idx]
        F = allF[new_idx]
        FE = allFE[new_idx]

        best = float(F[FE, 0].min()) if FE.any() else float("nan")
        history.append({"generation": gen + 1, "cheapest_feasible": best, "front_size": len(non_dominated_sort(F)[0])})
        if callback:
            callback(gen + 1, generations, best)

    front = non_dominated_sort(F)[0]
    results = []
    for i in front:
        d = decode(p, *pop[i])
        c, s, feas, cov = objectives(p, d)
        results.append({"design": d, "cost": c, "shortfall": s, "feasible": feas, "coverage": cov})
    results.sort(key=lambda r: r["cost"])
    return results, history
