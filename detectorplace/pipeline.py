"""End to end: drawing -> plant model -> coverage engine -> optimisers -> explainable design."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from . import baselines, checks, explain, nsga2
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
            "catalogue": catalogue_dict(flame, gas, cost), "rays": int(Mf.size + Mg.size),
            "source": "Demo plot-plan image", "reader": "OpenCV rules", "check_name": "the true geometry",
            "input_flags": []}


def build_from(source, kind="image", px_per_m=10.0, reader="opencv", truth=None, name=None, reqs=DEFAULT_REQS,
               flame=FlameModel(), gas=GasModel(), cost=CostModel(), margin=0.02):
    """Build the problem from any input: a plot-plan image (RGB array) read by the OpenCV rules or the
    learned model, or a DXF file (path or text stream). `truth` (a Plant) is only known for synthetic
    drawings; otherwise the layout is cross-checked on the other reader's model of the same drawing."""
    from . import cad, segnet, vision
    t = {}
    t0 = time.perf_counter()
    img, score, alt, nn_dbg, flags = None, None, None, None, []
    if kind == "dxf":
        plant, rep = cad.read_dxf(source, name=name)
        dbg, src, rdr = {"cad_report": rep}, "CAD drawing (DXF)", "CAD layer mapping"
        flags += checks.cad_flags(rep)
        check, check_name = (truth, "the true geometry") if truth is not None else (plant, "the CAD geometry (exact)")
    else:
        img = source
        src = "Plot-plan image"
        learned = reader == "learned"
        if learned and not segnet.available():
            raise RuntimeError("Learned reader unavailable: install torch and train or download models/segnet.pt")
        if learned:
            plant, dbg = segnet.extract_plant(img, px_per_m, name=name or "Plant (from drawing)")
            nn_dbg = dbg
            alt, _ = vision.extract_plant(img, px_per_m)
            rdr, alt_name = "Learned reader (U-Net)", "the OpenCV reading"
        else:
            plant, dbg = vision.extract_plant(img, px_per_m, name=name or "Plant (from drawing)")
            rdr, alt_name = "OpenCV rules", "the learned reading"
            if segnet.available():
                alt, nn_dbg = segnet.extract_plant(img, px_per_m)
        if not (0 <= plant.panel[0] <= plant.width and 0 <= plant.panel[1] <= plant.depth):
            plant.panel = (plant.width - 2.0, plant.depth - 2.0)
        flags += checks.reader_flags(plant, nn_dbg, alt, alt_name)
        if truth is not None:
            score = score_extraction(truth, plant)
            check, check_name = truth, "the true geometry"
        elif alt is not None:
            check, check_name = alt, alt_name
        else:
            check, check_name = plant, "the same model (no cross-check available)"
    t["vision_s"] = time.perf_counter() - t0
    cand = make_candidates(plant, flame, gas, cost)
    ft, gt = fire_targets(plant), gas_targets(plant)
    t0 = time.perf_counter()
    Mf, Mg = flame_matrix(plant, cand, ft, flame), gas_matrix(plant, cand, gt, gas)
    t["coverage_s"] = time.perf_counter() - t0
    p = Problem(Mf, Mg, cand.f_cost, cand.g_cost, cand.n_opt, ft, gt, tuple(reqs), margin)
    return {"truth": check, "img": img, "plant": plant, "vision_debug": dbg, "nn_debug": nn_dbg, "alt_plant": alt,
            "cand": cand, "problem": p, "vision_score": score, "timing": t,
            "catalogue": catalogue_dict(flame, gas, cost), "rays": int(Mf.size + Mg.size),
            "source": src, "reader": rdr, "check_name": check_name, "input_flags": flags, "px_per_m": px_per_m}


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


def optimise(ctx, generations=80, pop_size=100, seed=0, milp_seconds=0, callback=None, on_step=None):
    p = ctx["problem"]
    t0 = time.perf_counter()
    g = baselines.greedy(p)
    t_greedy = time.perf_counter() - t0
    t0 = time.perf_counter()
    front, history = nsga2.run(p, pop_size=pop_size, generations=generations, seed=seed, callback=callback,
                               on_step=on_step)
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
    cov, tcov = coverage(p, d), verify_on_truth(ctx, d)
    flags = ctx.get("input_flags", []) + checks.design_flags(p, cov, tcov, ctx.get("check_name", "the true geometry")) \
        + checks.data_flags()
    return {"schedule": rows, "n_minus_1": explain.n_minus_1(p, d, rows), "gaps": explain.gaps(p, d),
            "coverage": cov, "truth_coverage": tcov, "flags": flags}


def run_from(source, kind="image", generations=60, seed=0, **kw):
    ctx = build_from(source, kind, **kw)
    opt = optimise(ctx, generations=generations, seed=seed)
    ctx["opt"] = opt
    ctx["explain"] = explain_design(ctx, opt["chosen"]["design"])
    return ctx


def run_all(generations=80, milp_seconds=0, seed=0):
    ctx = build()
    opt = optimise(ctx, generations=generations, milp_seconds=milp_seconds, seed=seed)
    ctx["opt"] = opt
    ctx["explain"] = explain_design(ctx, opt["chosen"]["design"])
    return ctx
