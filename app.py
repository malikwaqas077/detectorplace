"""DetectorPlace dashboard:  streamlit run app.py"""
from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from detectorplace import pipeline, report, viz
from detectorplace.coverage import _cone_coverage, flame_wedge
from detectorplace.detectors import FlameModel
from detectorplace.evaluate import DEFAULT_REQS
from detectorplace.plant import GRADES, RES

st.set_page_config(page_title="DetectorPlace", page_icon="🔥", layout="wide")
CACHE = Path(__file__).parent / "results" / "ctx.pkl"


@st.cache_resource(show_spinner="Running the full pipeline (first start only)...")
def baseline_ctx():
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    ctx = pipeline.run_all(generations=80)
    CACHE.parent.mkdir(exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(ctx, f)
    return ctx


if "ctx" not in st.session_state:
    st.session_state.ctx = baseline_ctx()
ctx = st.session_state.ctx
p, plant, opt = ctx["problem"], ctx["plant"], ctx["opt"]

st.markdown("## 🔥 DetectorPlace")
st.caption("AI-assisted fire & gas detector placement, coverage validation and cost estimation · "
           "synthetic gas compression station · all data and rates illustrative")

chosen = opt["chosen"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Installed cost (excl. fixed)", f"£{chosen['cost']:,.0f}")
c2.metric("Detectors", f"{len(chosen['design'].flame)} flame · {len(chosen['design'].gas)} gas")
c3.metric("vs greedy heuristic", f"£{opt['greedy']['cost'] - chosen['cost']:,.0f} cheaper",
          f"{(opt['greedy']['cost'] - chosen['cost']) / opt['greedy']['cost'] * 100:.1f}%")
c4.metric("Targets met", f"{sum(chosen['coverage'][q.key] >= q.target for q in p.reqs)}/{len(p.reqs)}")
c5.metric("Line-of-sight checks", f"{ctx['rays'] / 1e6:.1f} M", f"{ctx['timing']['coverage_s']:.1f} s")

tabs = st.tabs(["1 · Plot plan → model", "2 · Line of sight", "3 · Optimiser", "4 · Design review", "5 · Report"])


def plant_fig(show_grades=True, height=520):
    fig = go.Figure()
    if show_grades:
        g = np.where(plant.grades > 0, plant.grades, np.nan).astype(float)
        fig.add_heatmap(z=g, x0=RES / 2, dx=RES, y0=RES / 2, dy=RES, zmin=1, zmax=3, showscale=False,
                        colorscale=[[0, "#fff4c2"], [0.5, "#ffd9b0"], [1, "#ffc4c4"]], hoverinfo="skip")
    h = np.where(plant.heights > 0, plant.heights, np.nan)
    fig.add_heatmap(z=h, x0=RES / 2, dx=RES, y0=RES / 2, dy=RES, colorscale="Greys", zmin=-4, zmax=10,
                    showscale=False, hovertemplate="x %{x:.1f} m, y %{y:.1f} m<br>top %{z:.0f} m<extra></extra>")
    fig.update_yaxes(autorange="reversed", scaleanchor="x", title="y (m)", range=[plant.depth, 0])
    fig.update_xaxes(title="x (m)", range=[0, plant.width])
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", showlegend=False)
    return fig


with tabs[0]:
    st.markdown("#### From a drawing to a 2.5D engineering model")
    a, b = st.columns(2)
    a.image(ctx["img"], caption="Input: plot-plan export (shade = elevation band, tint = hazard grade, "
                                "blue = leak source)", use_container_width=True)
    fig = plant_fig()
    fig.add_scatter(x=plant.leaks[:, 0], y=plant.leaks[:, 1], mode="markers",
                    marker=dict(color="#1d4ed8", size=9, symbol="diamond"), name="leak source")
    b.plotly_chart(fig, use_container_width=True)
    b.caption("Extracted: equipment footprints with heights, hazard grades and leak sources, on a 0.5 m grid")
    vs = ctx["vision_score"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Equipment footprint IoU", f"{vs['equipment_iou'] * 100:.1f}%")
    m2.metric("Elevation band correct", f"{vs['height_band_accuracy'] * 100:.0f}%")
    m3.metric("Hazard grade agreement", f"{vs['zone_accuracy'] * 100:.1f}%")
    m4.metric("Leak sources found", f"{vs['leaks_found']}/{vs['leaks_true']}")
    st.info("OpenCV pipeline: HSV segmentation → morphology → connected components → elevation from fill "
            "shade → resample to the analysis grid. Text labels, grid lines and sensor noise are rejected. "
            "Scores compare the extracted model with the geometry the drawing was rendered from.")
    areas = pd.DataFrame([{"Hazard grade": GRADES[g], "Open area (m²)": int(((plant.grades == g) &
                          (plant.heights == 0)).sum() * RES * RES)} for g in (3, 2, 1)])
    st.dataframe(areas, hide_index=True)

with tabs[1]:
    st.markdown("#### What can one detector actually see?")
    st.caption("3D ray marching from the detector to every point at fire height. Equipment taller than the "
               "sight line casts a shadow; the cone narrows with off-axis range loss.")
    l, r = st.columns([1, 3])
    fm = FlameModel()
    x = l.slider("x (m)", 1.0, plant.width - 1, 40.0, 0.5)
    y = l.slider("y (m)", 1.0, plant.depth - 1, 23.0, 0.5)
    z = l.select_slider("Mounting height (m)", [3.0, 6.0], 3.0)
    az = l.slider("Azimuth (°)", 0, 315, 0, 45)
    rng_m = l.slider("Detection range for design fire (m)", 10, 30, int(fm.range_m))
    ny, nx = plant.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    free = plant.heights == 0
    tx, ty = ((xx[free] + 0.5) * RES).astype(np.float64), ((yy[free] + 0.5) * RES).astype(np.float64)
    tilt = np.deg2rad(15.0 if z == 3.0 else 30.0)
    vis = _cone_coverage(plant.heights.astype(np.float32), RES, np.array([x]), np.array([y]), np.array([z]),
                         np.array([np.deg2rad(az)]), np.array([tilt]), tx, ty, np.full(len(tx), fm.fire_height_m),
                         float(rng_m), np.deg2rad(fm.hfov_deg / 2), np.deg2rad(fm.vfov_deg / 2), fm.offaxis_loss)[0]
    ras = np.full(plant.shape, np.nan)
    ras[free] = np.where(vis == 1, 1.0, np.nan)
    fig = plant_fig()
    fig.add_heatmap(z=ras, x0=RES / 2, dx=RES, y0=RES / 2, dy=RES, colorscale=[[0, "#2563eb"], [1, "#2563eb"]],
                    opacity=0.45, showscale=False, hoverinfo="skip")
    wx, wy = flame_wedge(x, y, az, replace(fm, range_m=rng_m))
    fig.add_scatter(x=wx, y=wy, mode="lines", line=dict(color="#1d4ed8", width=1, dash="dot"))
    fig.add_scatter(x=[x], y=[y], mode="markers", marker=dict(symbol="triangle-up", size=14, color="#1d4ed8"))
    r.plotly_chart(fig, use_container_width=True)
    l.metric("Area seen", f"{int(vis.sum() * RES * RES)} m²")
    graded_seen = int((vis.astype(bool) & (plant.grades[free] > 0)).sum() * RES * RES)
    l.metric("of which hazard-graded", f"{graded_seen} m²")
    l.caption("Try 6 m: the detector sees over the 2 m pump skid in front of it. The 9 m vessels still cast shadows.")

with tabs[2]:
    st.markdown("#### Multi-objective search: cost against risk")
    a, b = st.columns([3, 2])
    fr = opt["front"]
    fig = go.Figure()
    fig.add_scatter(x=[q["cost"] / 1e3 for q in fr], y=[q["shortfall"] * 100 for q in fr], mode="lines+markers",
                    name="NSGA-II Pareto front", marker=dict(size=6, color=["#16a34a" if q["feasible"] else "#94a3b8"
                                                                            for q in fr]),
                    customdata=[[len(q["design"].flame), len(q["design"].gas)] for q in fr],
                    hovertemplate="£%{x:.0f}k · shortfall %{y:.2f}%<br>%{customdata[0]} flame, %{customdata[1]} gas"
                                  "<extra></extra>")
    fig.add_scatter(x=[chosen["cost"] / 1e3], y=[chosen["shortfall"] * 100], mode="markers", name="chosen",
                    marker=dict(symbol="star", size=18, color="#d97706"))
    fig.add_scatter(x=[opt["greedy"]["cost"] / 1e3], y=[opt["greedy"]["shortfall"] * 100], mode="markers",
                    name="greedy heuristic", marker=dict(symbol="square", size=11, color="#475569"))
    if opt.get("milp", {}).get("cost"):
        fig.add_scatter(x=[opt["milp"]["cost"] / 1e3], y=[opt["milp"]["shortfall"] * 100], mode="markers",
                        name="exact MILP (time-limited)", marker=dict(symbol="diamond", size=11, color="#7c3aed"))
    fig.update_yaxes(type="log", title="risk-weighted shortfall (%)", tickvals=[0.1, 0.3, 1, 3, 10, 30, 100],
                     ticktext=["0.1", "0.3", "1", "3", "10", "30", "100"])
    fig.update_xaxes(title="installed cost (£k)")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08))
    a.plotly_chart(fig, use_container_width=True)
    a.caption("Green points meet every target. Left of the star, money saved is paid for in uncovered risk.")
    hist = pd.DataFrame(opt["history"])
    f2 = go.Figure(go.Scatter(x=hist["generation"], y=hist["cheapest_feasible"] / 1e3, mode="lines",
                              line=dict(color="#16a34a")))
    f2.add_hline(y=opt["greedy"]["cost"] / 1e3, line_dash="dash", annotation_text="greedy")
    if opt.get("milp", {}).get("bound"):
        f2.add_hline(y=opt["milp"]["bound"] / 1e3, line_dash="dot", line_color="#7c3aed",
                     annotation_text="MILP lower bound")
    f2.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), title="Cheapest compliant layout by generation",
                     xaxis_title="generation", yaxis_title="£k")
    b.plotly_chart(f2, use_container_width=True)
    rows = [{"Method": "Greedy heuristic", "Cost (£)": f"{opt['greedy']['cost']:,.0f}",
             "Detectors": len(opt['greedy']['design'].flame) + len(opt['greedy']['design'].gas),
             "Time (s)": f"{opt['timing']['greedy_s']:.1f}"},
            {"Method": "NSGA-II (memetic)", "Cost (£)": f"{chosen['cost']:,.0f}",
             "Detectors": len(chosen['design'].flame) + len(chosen['design'].gas),
             "Time (s)": f"{opt['timing']['ga_s']:.0f}"}]
    if opt.get("milp", {}).get("cost"):
        m = opt["milp"]
        rows.append({"Method": f"Exact MILP (HiGHS, {m['seconds']:.0f} s limit)", "Cost (£)": f"{m['cost']:,.0f}",
                     "Detectors": len(m['design'].flame) + len(m['design'].gas), "Time (s)": f"{m['seconds']:.0f}"})
        rows.append({"Method": "MILP proven lower bound", "Cost (£)": f"{m['bound']:,.0f}", "Detectors": "-",
                     "Time (s)": "-"})
    b.dataframe(pd.DataFrame(rows).astype(str), hide_index=True)

    with st.expander("Change the targets and re-optimise"):
        cols = st.columns(len(DEFAULT_REQS))
        new_reqs = [replace(q, target=cols[i].slider(q.label, 0.5, 0.99, q.target, 0.01, key=f"t{q.key}"))
                    for i, q in enumerate(p.reqs)]
        g1, g2 = st.columns(2)
        gens = g1.slider("Generations", 10, 150, 40, 10)
        margin = g2.slider("Design margin on every target", 0.0, 0.05, p.margin, 0.01)
        if st.button("Re-optimise", type="primary"):
            bar = st.progress(0.0, "Building coverage model...")
            new = pipeline.build(reqs=new_reqs, margin=margin)
            cb = lambda g, n, best: bar.progress(g / n, f"Generation {g}/{n} · cheapest compliant £{best:,.0f}"
                                                 if best == best else f"Generation {g}/{n}")
            new["opt"] = pipeline.optimise(new, generations=gens, callback=cb)
            new["explain"] = pipeline.explain_design(new, new["opt"]["chosen"]["design"])
            st.session_state.ctx = new
            st.rerun()

with tabs[3]:
    st.markdown("#### The proposed layout, and why")
    ex = ctx["explain"]
    a, b = st.columns([3, 2])
    a.image(viz.coverage_map(ctx, chosen["design"], ex["schedule"], dpi=130), use_container_width=True)
    cov = ex["coverage"]
    tc = ex["truth_coverage"]
    b.dataframe(pd.DataFrame([{"Requirement": q.label, "Target": f"{q.target:.0%}",
                               "Achieved": f"{cov[q.key]:.1%}", "Re-checked on true geometry": f"{tc[q.key]:.1%}",
                               "": "✅" if tc[q.key] >= q.target else "⚠️"} for q in p.reqs]), hide_index=True)
    b.caption(f"Optimised to target + {p.margin:.0%} margin so small drawing-extraction errors cannot "
              "push the built design below target.")
    sched = pd.DataFrame(ex["schedule"]).drop(columns=["cand"])
    b.dataframe(sched[["tag", "type", "x", "y", "mount_m", "azimuth_deg", "unique_points", "cost", "why"]],
                hide_index=True, height=300)
    st.markdown("##### If one detector fails (N-1)")
    n1 = pd.DataFrame(ex["n_minus_1"])
    n1["targets_missed"] = n1["targets_missed"].apply(lambda v: ", ".join(v) or "none")
    st.dataframe(n1, hide_index=True, height=220)
    st.markdown("##### Residual gaps in Grade A/B areas")
    st.dataframe(pd.DataFrame(ex["gaps"]), hide_index=True)

with tabs[4]:
    st.markdown("#### Auto-generated design report")
    st.caption("Filled from a standard company template (templates/fg_design_report.json): scope, method, "
               "targets vs achieved, detector schedule, coverage map, N-1 check, residual gaps, Class 4 cost "
               "estimate, limitations and sign-off.")
    tb = report.tables(ctx)
    a, b = st.columns([2, 3])
    a.markdown("**Cost estimate**")
    a.dataframe(pd.DataFrame(tb["cost"][1:], columns=tb["cost"][0]).astype(str), hide_index=True)
    pdf = report.build_pdf(ctx)
    a.download_button("Download PDF report", pdf, "DetectorPlace_FG_design_report.pdf", "application/pdf",
                      type="primary")
    b.markdown("**Coverage against targets**")
    b.dataframe(pd.DataFrame(tb["coverage"][1:], columns=tb["coverage"][0]), hide_index=True)
    b.warning("Advisory output. A competent F&G engineer must verify and sign off the layout before use.")
