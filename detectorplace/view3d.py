"""Interactive 3D view of the plant model and the detector layout (Plotly).

Equipment is drawn from the same 2.5D height map the coverage engine ray-marches through, so
what you see is exactly what the optimiser saw. The ground is coloured by hazard grade or by
achieved coverage, flame detectors show their mounting pole and cone of vision, gas detectors
their coverage radius, and one detector can be selected to light up the ground it actually sees.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .coverage import flame_wedge
from .detectors import FlameModel, GasModel
from .evaluate import counts
from .plant import RES

BAND_COL = {2.0: "#b8c2cc", 5.0: "#8a97a6", 9.0: "#5b6878"}
GRADE_COL = {0: "#eef0f2", 1: "#fbe9a6", 2: "#f9c98f", 3: "#f3a3a3"}
COV_COL = {"none": "#eef0f2", "unseen": "#e5484d", "one": "#f5a524", "two": "#2a9d8f", "gas_miss": "#8e4ec6"}


def _rects(mask: np.ndarray):
    """Greedy decomposition of a boolean grid into axis-aligned rectangles (row0, col0, row1, col1)."""
    m = mask.copy()
    out = []
    ny, nx = m.shape
    for r in range(ny):
        c = 0
        while c < nx:
            if not m[r, c]:
                c += 1
                continue
            c1 = c
            while c1 < nx and m[r, c1]:
                c1 += 1
            r1 = r + 1
            while r1 < ny and m[r1, c:c1].all():
                r1 += 1
            m[r:r1, c:c1] = False
            out.append((r, c, r1, c1))
            c = c1
    return out


def _box(x0, y0, x1, y1, z0, z1):
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    return xs, ys, zs, i, j, k


def equipment_traces(plant):
    traces = []
    for h in sorted(set(np.unique(plant.heights)) - {0.0}):
        X, Y, Z, I, J, K = [], [], [], [], [], []
        for r0, c0, r1, c1 in _rects(plant.heights == h):
            xs, ys, zs, i, j, k = _box(c0 * RES, r0 * RES, c1 * RES, r1 * RES, 0, float(h))
            n = len(X)
            X += xs; Y += ys; Z += zs
            I += [v + n for v in i]; J += [v + n for v in j]; K += [v + n for v in k]
        traces.append(go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color=BAND_COL.get(float(h), "#7a8794"),
                                flatshading=True, opacity=1.0, name=f"Equipment, top {h:.0f} m",
                                hovertemplate=f"Equipment<br>top {h:.0f} m<extra></extra>",
                                lighting=dict(ambient=0.55, diffuse=0.8, specular=0.1, roughness=0.9),
                                lightposition=dict(x=-1000, y=-2000, z=3000), showlegend=True))
    return traces


def _ground(plant, values, colours, labels, name):
    """Flat surface at z=0 on the 1 m target grid, coloured by integer category."""
    ny, nx = values.shape
    xs = np.arange(nx + 1) * (plant.width / nx)
    ys = np.arange(ny + 1) * (plant.depth / ny)
    v = np.pad(values, ((0, 1), (0, 1)), mode="edge").astype(float)
    n = len(colours)
    scale = []
    for q, c in enumerate(colours):
        scale += [[q / n, c], [(q + 1) / n, c]]
    text = np.vectorize(lambda q: labels[int(q)])(v)
    return go.Surface(x=xs, y=ys, z=np.zeros_like(v), surfacecolor=v + 0.5, cmin=0, cmax=n,
                      colorscale=scale, showscale=False, name=name, text=text, opacity=1.0,
                      hovertemplate="x %{x:.0f} m, y %{y:.0f} m<br>%{text}<extra></extra>",
                      lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0), contours=dict(z=dict(show=False)))


def ground_by_grade(plant):
    step = 2   # 1 m tiles from 0.5 m cells
    g = plant.grades[::step, ::step].astype(int)
    free = plant.heights[::step, ::step] == 0
    g = np.where(free, g, 0)
    return _ground(plant, g, [GRADE_COL[q] for q in range(4)],
                   ["Ungraded", "Grade C fire area", "Grade B fire area", "Grade A fire area"], "Hazard grade")


def ground_by_coverage(ctx, design):
    p, plant = ctx["problem"], ctx["plant"]
    cf, cg = counts(p, design)
    v = np.zeros(p.fire.shape, int)
    v[p.gas.cells[:, 0], p.gas.cells[:, 1]] = np.where(cg > 0, 0, 4) if len(cg) else 0
    v[p.fire.cells[:, 0], p.fire.cells[:, 1]] = np.where(cf >= 2, 3, np.where(cf == 1, 2, 1))
    return _ground(plant, v, [COV_COL[k] for k in ("none", "unseen", "one", "two", "gas_miss")],
                   ["Not a hazard area", "Fire area NOT seen", "Seen by 1 flame detector",
                    "Seen by 2+ flame detectors", "Gas zone NOT covered"], "Coverage")


def _cone_mesh(x, y, z, az_deg, fm: FlameModel, n=18):
    wx, wy = flame_wedge(x, y, az_deg, fm, n=n)
    ax_, ay_ = wx[1:-1], wy[1:-1]
    X = [x] + list(ax_)
    Y = [y] + list(ay_)
    Z = [z] + [fm.fire_height_m] * len(ax_)
    I = [0] * (len(ax_) - 1)
    J = list(range(1, len(ax_)))
    K = list(range(2, len(ax_) + 1))
    return X, Y, Z, I, J, K


def detector_traces(ctx, rows, selected=None, show_cones=True, show_gas_radius=True, show_labels=False,
                    fm: FlameModel = FlameModel(), gm: GasModel = GasModel()):
    tr = []
    flames = [r for r in rows if r["type"].startswith("Flame")]
    gases = [r for r in rows if not r["type"].startswith("Flame")]
    if show_cones:
        X, Y, Z, I, J, K = [], [], [], [], [], []
        for r in flames:
            if selected and r["tag"] != selected:
                continue
            xs, ys, zs, i, j, k = _cone_mesh(r["x"], r["y"], r["mount_m"], r["azimuth_deg"], fm)
            n = len(X)
            X += xs; Y += ys; Z += zs
            I += [v + n for v in i]; J += [v + n for v in j]; K += [v + n for v in k]
        if X:
            tr.append(go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color="#3b82f6", opacity=0.13 if not selected else 0.22,
                                name="Flame detector cone of vision", hoverinfo="skip", showlegend=True))
    # poles
    px, py, pz = [], [], []
    for r in rows:
        px += [r["x"], r["x"], None]; py += [r["y"], r["y"], None]; pz += [0, r["mount_m"], None]
    tr.append(go.Scatter3d(x=px, y=py, z=pz, mode="lines", line=dict(color="#475569", width=3),
                           hoverinfo="skip", showlegend=False))
    for group, colour, symbol, label in ((flames, "#1d4ed8", "diamond", "Flame detector"),
                                         (gases, "#7c3aed", "circle", "Gas detector")):
        if not group:
            continue
        size = [12 if r["tag"] == selected else 7 for r in group]
        tr.append(go.Scatter3d(
            x=[r["x"] for r in group], y=[r["y"] for r in group], z=[r["mount_m"] for r in group],
            mode="markers+text" if show_labels else "markers", text=[r["tag"] for r in group],
            textposition="top center",
            textfont=dict(size=10, color=colour), marker=dict(size=size, color=colour, symbol=symbol,
                                                               line=dict(color="white", width=1)),
            name=label, customdata=[[r["mount_m"], r["azimuth_deg"] if r["azimuth_deg"] is not None else -1,
                                     r["unique_points"], r["cost"], r.get("why", "")] for r in group],
            hovertemplate="<b>%{text}</b> " + label + "<br>mounted at %{customdata[0]:.0f} m"
                          "<br>sole coverage of %{customdata[2]} m²<br>£%{customdata[3]:,.0f}"
                          "<br>%{customdata[4]}<extra></extra>"))
    if show_gas_radius and gases:
        cx, cy, cz = [], [], []
        t = np.linspace(0, 2 * np.pi, 40)
        for r in gases:
            cx += list(r["x"] + gm.radius_m * np.cos(t)) + [None]
            cy += list(r["y"] + gm.radius_m * np.sin(t)) + [None]
            cz += [0.05] * 40 + [None]
        tr.append(go.Scatter3d(x=cx, y=cy, z=cz, mode="lines", line=dict(color="#7c3aed", width=2, dash="dot"),
                               name="Gas detector radius", hoverinfo="skip"))
    return tr


def selected_view_trace(ctx, row):
    """Ground points the selected detector actually sees (after obstruction)."""
    p = ctx["problem"]
    if row["type"].startswith("Flame"):
        seen = p.Mf[row["cand"]].astype(bool)
        xy = p.fire.xy[seen]
    else:
        seen = p.Mg[row["cand"]].astype(bool)
        xy = p.gas.xy[seen]
    if not len(xy):
        return None
    return go.Scatter3d(x=xy[:, 0], y=xy[:, 1], z=np.full(len(xy), 0.08), mode="markers",
                        marker=dict(size=4, color="#2563eb", opacity=0.8, symbol="square"),
                        name=f"Seen by {row['tag']} ({len(xy)} m²)", hoverinfo="skip")


def figure(ctx, design, rows, colour_by="coverage", selected=None, show_cones=True, show_gas_radius=True,
           show_leaks=True, show_labels=False, height=680, z_scale=1.3):
    plant = ctx["plant"]
    fig = go.Figure()
    fig.add_trace(ground_by_coverage(ctx, design) if colour_by == "coverage" else ground_by_grade(plant))
    for t in equipment_traces(plant):
        fig.add_trace(t)
    for t in detector_traces(ctx, rows, selected, show_cones, show_gas_radius, show_labels):
        fig.add_trace(t)
    if selected:
        row = next((r for r in rows if r["tag"] == selected), None)
        if row is not None:
            t = selected_view_trace(ctx, row)
            if t is not None:
                fig.add_trace(t)
    if show_leaks and len(plant.leaks):
        fig.add_trace(go.Scatter3d(x=plant.leaks[:, 0], y=plant.leaks[:, 1], z=np.full(len(plant.leaks), 0.3),
                                   mode="markers", marker=dict(size=5, color="#dc2626", symbol="x"),
                                   name="Gas leak source", hovertemplate="Leak source<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[plant.panel[0]], y=[plant.panel[1]], z=[1.0], mode="markers+text",
                               text=["F&G panel"], textposition="top center", marker=dict(size=6, color="#111827",
                                                                                          symbol="square"),
                               name="F&G panel", hovertemplate="F&G panel (cables routed here)<extra></extra>"))
    zmax = max(10.0, float(plant.heights.max()) + 2)
    span = max(plant.width, plant.depth)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=0.0, x=0.0, bgcolor="rgba(255,255,255,0.75)",
                    font=dict(size=11)),
        scene=dict(
            xaxis=dict(title="x (m)", range=[0, plant.width], backgroundcolor="#f8fafc", showspikes=False),
            yaxis=dict(title="y (m)", range=[plant.depth, 0], backgroundcolor="#f8fafc", showspikes=False),
            zaxis=dict(title="height (m)", range=[-0.2, zmax], backgroundcolor="#f1f5f9", showspikes=False),
            aspectmode="manual",
            aspectratio=dict(x=plant.width / span * 1.6, y=plant.depth / span * 1.6, z=zmax / span * 1.6 * z_scale),
            camera=dict(eye=dict(x=0.3, y=-1.35, z=1.6), up=dict(x=0, y=0, z=1))),
        uirevision="keep-camera")
    return fig
