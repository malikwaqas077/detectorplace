"""End to end: drawing -> plant model -> coverage engine -> optimisers -> explainable design."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from . import baselines, explain, nsga2
from .coverage import fire_targets, flame_matrix, gas_matrix, gas_targets
from .detectors import CostModel, FlameModel, GasModel, catalogue_dict, make_candidates
from .evaluate import DEFAULT_REQS, Design, Problem, coverage, objectives
from .plant import demo_plant, render_plot_plan
from .vision import extract_plant, score_extraction


def build(reqs=DEFAULT_REQS, flame=FlameModel(), gas=GasModel(), cost=CostModel(), img=None, margin=0.02):
    t = {}
    t0 = time.perf_counter()
    truth = demo_plant()
    img = render_plot_plan(truth) if img is None else img
    plant, dbg = extract_plant(img)
    t["vision_s"] = time.perf_counter() - t0
    cand = make_candidates(plant, flame, gas, cost)
    ft, gt = fire_targets(plant), gas_targets(plant)
    t0 = time.perf_counter()
    Mf, Mg = flame_matrix(plant, cand, ft, flame), gas_matrix(plant, cand, gt, gas)
    t["coverage_s"] = time.perf_counter() - t0
    p = Problem(Mf, Mg, cand.f_cost, cand.g_cost, cand.n_opt, ft, gt, tuple(reqs), margin)
    return {"truth": truth, "img": img, "plant": plant, "vision_debug": dbg, "cand": cand, "problem": p,
            "vision_score": score_extraction(truth, plant), "timing": t,
            "catalogue": catalogue_dict(flame, gas, cost), "rays": int(Mf.size + Mg.size)}


def verify_on_truth(ctx, d: Design, flame=FlameModel(), gas=GasModel()):
    """Re-check the chosen layout against the true geometry rather than the extracted one."""
    truth, cand = ctx["truth"], ctx["cand"]
    sub = replace(cand, f_x=cand.f_x[d.flame], f_y=cand.f_y[d.flame], f_z=cand.f_z[d.flame],
                  f_az=cand.f_az[d.flame], f_tilt=cand.f_tilt[d.flame], f_cost=cand.f_cost[d.flame],
                  g_xy=cand.g_xy[d.gas], g_cost=cand.g_cost[d.gas])
    ft, gt = fire_targets(truth), gas_targets(truth)
    p = Problem(flame_matrix(truth, sub, ft, flame), gas_matrix(truth, sub, gt, gas), sub.f_cost, sub.g_cost,
                1, ft, gt, ctx["problem"].reqs)
    return coverage(p, Design(np.arange(len(d.flame)), np.arange(len(d.gas))))


def optimise(ctx, generations=80, pop_size=100, seed=0, milp_seconds=0, callback=None):
    p = ctx["problem"]
    t0 = time.perf_counter()
    g = baselines.greedy(p)
    t_greedy = time.perf_counter() - t0
    t0 = time.perf_counter()
    front, history = nsga2.run(p, pop_size=pop_size, generations=generations, seed=seed, callback=callback)
    t_ga = time.perf_counter() - t0
    feas = [r for r in front if r["feasible"]]
    best = feas[0] if feas else min(front, key=lambda r: r["shortfall"])
    out = {"front": front, "history": history, "chosen": best,
           "greedy": dict(zip(("cost", "shortfall", "feasible", "coverage"), objectives(p, g)), design=g),
           "timing": {"greedy_s": t_greedy, "ga_s": t_ga}}
    if milp_seconds:
        d, info = baselines.exact(p, time_limit=milp_seconds)
        out["milp"] = info
        if d is not None:
            out["milp"]["design"] = d
            out["milp"].update(dict(zip(("cost", "shortfall", "feasible", "coverage"), objectives(p, d))))
    return out


def explain_design(ctx, d: Design):
    p, cand = ctx["problem"], ctx["cand"]
    rows = explain.schedule(p, cand, d)
    return {"schedule": rows, "n_minus_1": explain.n_minus_1(p, d, rows), "gaps": explain.gaps(p, d),
            "coverage": coverage(p, d), "truth_coverage": verify_on_truth(ctx, d)}


def run_all(generations=80, milp_seconds=0, seed=0):
    ctx = build()
    opt = optimise(ctx, generations=generations, milp_seconds=milp_seconds, seed=seed)
    ctx["opt"] = opt
    ctx["explain"] = explain_design(ctx, opt["chosen"]["design"])
    return ctx
