"""DetectorPlace dashboard:  streamlit run app.py

A guided, step-by-step demo: upload a drawing, let the AI read it, check its confidence, see what one
detector can see, build the 3D model, watch the genetic algorithm design the layout live, then click
any detector in 3D to ask about it. Each step has a Next button, so it can be presented in a minute or two.
"""
from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from detectorplace import confidence, explain, pipeline, report, scene, segnet, view3d
from detectorplace.assistant import DesignAssistant
from detectorplace.coverage import _cone_coverage, flame_wedge
from detectorplace.detectors import CostModel, FlameModel, GasModel, catalogue_dict
from detectorplace.evaluate import DEFAULT_REQS, Design, cost as design_cost, counts
from detectorplace.plant import GRADES, RES, demo_plant
from detectorplace.plot3d import clickable_chart

st.set_page_config(page_title="DetectorPlace", page_icon="🔥", layout="wide", initial_sidebar_state="collapsed")
CACHE = Path(__file__).parent / "results" / "ctx.pkl"
SAMPLES = Path(__file__).parent / "samples"
GREEN, AMBER, RED, BLUE, SLATE = "#16a34a", "#d97706", "#dc2626", "#2563eb", "#64748b"

STEPS = [("📄", "Upload"), ("🧠", "AI reads it"), ("✅", "Confidence"), ("👁️", "Line of sight"),
         ("🏗️", "3D model"), ("🧬", "Optimise"), ("💬", "Ask"), ("📑", "Report")]
LABELS = [f"{i + 1} {ic} {name}" for i, (ic, name) in enumerate(STEPS)]
NOTES = [
    "Any plot plan: a CAD file is read directly, a scan or PDF is read by the AI. One click to start.",
    "The AI finds every piece of equipment, its height, the danger zones and the leak points, and turns the "
    "drawing into an engineering model. Drag the slider to compare the two.",
    "Every item gets a confidence score. Anything the AI isn't sure about is flagged red, and the engineer checks it. "
    "The engineer stays in charge.",
    "Click anywhere to place a detector. Tall equipment blocks its view, so a leak that looks close may not be seen.",
    "The full 3D model: this is exactly what the coverage checks run through.",
    "The genetic algorithm tries thousands of layouts, keeps the best and improves them, like natural selection. "
    "Watch the cost come down while every safety target is still met.",
    "Click any detector and ask about it. Every answer comes from re-running the real calculations.",
    "One click gives the full design report, for an engineer to check and sign.",
]
DIRS = {"↑": 270, "↗": 315, "→": 0, "↘": 45, "↓": 90, "↙": 135, "←": 180, "↖": 225}


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


# Editable prices: (session key, label, model, field, step). Only cost fields; detector performance is untouched.
_DEF = {"flame": FlameModel(), "gas": GasModel(), "cost": CostModel()}
RATES = [("rate_f_supply", "Flame detector: supply (£)", "flame", "supply", 100.0),
         ("rate_f_install", "Flame detector: install (£)", "flame", "install", 100.0),
         ("rate_f_high", "Flame detector: extra for high mounting (£)", "flame", "high_mount_extra", 50.0),
         ("rate_g_supply", "Gas detector: supply (£)", "gas", "supply", 100.0),
         ("rate_g_install", "Gas detector: install (£)", "gas", "install", 50.0),
         ("rate_cable", "Cable cost per metre (£/m)", "cost", "cable_per_m", 1.0),
         ("rate_tail", "Tail allowance per device (m)", "cost", "tail_m", 1.0),
         ("rate_io", "Panel I/O and commissioning per device (£)", "cost", "io_per_device", 50.0),
         ("rate_fixed", "Fixed cost: panel, engineering, FAT (£)", "cost", "fixed", 1000.0)]


def reset_rates():
    for key, _, model, field, _ in RATES:
        st.session_state[f"_{key}"] = float(getattr(_DEF[model], field))


def rate_models():
    vals = {m: {} for m in _DEF}
    for key, _, model, field, _ in RATES:
        vals[model][field] = float(st.session_state[f"_{key}"])
    return {m: replace(_DEF[m], **vals[m]) for m in _DEF}


def new_result(ctx):
    """A new drawing clears the GA state, the detector chat and the engineer's confidence checks."""
    st.session_state.ctx = ctx
    st.session_state.ga_done = False
    st.session_state.dlg_chat = {}
    st.session_state.checked = set()


def go_to(i):
    st.session_state.nav = LABELS[i]


ss = st.session_state
if "ctx" not in ss:
    new_result(baseline_ctx())
    ss.input = None
if "_rate_fixed" not in ss:
    reset_rates()
ss.setdefault("_conf_thr", int(confidence.REVIEW_BELOW * 100))
ss["_conf_thr"] = ss["_conf_thr"]           # keep the threshold when step 3 is not on screen
ss.setdefault("checked", set())
ss.setdefault("dlg_chat", {})
ss.setdefault("los", (40.0, 23.0))
ss.setdefault("nav", LABELS[0])
if "_goto" in ss:                        # a jump requested by code (not a button callback) happens here
    ss.nav = LABELS[ss.pop("_goto")]
for _k, *_ in RATES:                     # keep entered prices when step 1 is not on screen
    ss[f"_{_k}"] = ss[f"_{_k}"]

ctx = ss.ctx
p, plant = ctx["problem"], ctx["plant"]
has_opt = "opt" in ctx and "explain" in ctx
models = rate_models()

st.markdown("""
<style>
.block-container {padding-top: 3rem; max-width: 1500px;}
[data-testid="stSidebarCollapsedControl"] {display: none;}
.dp-title {font-size: 1.7rem; font-weight: 700; margin: 0;}
.dp-sub {color: #64748b; font-size: 0.9rem; margin: 0 0 0.6rem 0;}
.dp-head {font-size: 1.45rem; font-weight: 700; margin: 0.6rem 0 0.1rem 0;}
.dp-lead {font-size: 1.02rem; color: #475569; margin: 0 0 0.8rem 0;}
.dp-cards {display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0.8rem; margin: 0.6rem 0;}
@media (max-width: 1100px) {.dp-cards {grid-template-columns: repeat(2, minmax(0, 1fr));}}
.dp-card {border: 1px solid rgba(100,116,139,0.25); border-left: 6px solid var(--c); border-radius: 10px;
          padding: 0.75rem 0.9rem; background: rgba(148,163,184,0.06); animation: pop .5s ease-out both;}
.dp-card:nth-child(2) {animation-delay: .1s} .dp-card:nth-child(3) {animation-delay: .2s}
.dp-card:nth-child(4) {animation-delay: .3s} .dp-card:nth-child(5) {animation-delay: .4s}
@keyframes pop {from {opacity: 0; transform: translateY(8px)} to {opacity: 1; transform: none}}
.dp-card .k {font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: #64748b; font-weight: 600;}
.dp-card .v {font-size: 1.75rem; font-weight: 700; line-height: 1.15; margin: 0.15rem 0;}
.dp-card .s {font-size: 0.82rem; color: #64748b;}
.dp-step {border-radius: 10px; padding: 0.7rem 0.9rem; background: rgba(148,163,184,0.08); height: 100%;}
.dp-step b {font-size: 0.95rem;} .dp-step div {font-size: 0.85rem; color: #64748b; margin-top: 0.2rem;}
.dp-note {border-left: 4px solid #d97706; background: rgba(217,119,6,0.08); padding: 0.5rem 0.8rem;
          border-radius: 6px; font-size: 0.95rem; margin-bottom: 0.6rem;}
.dp-big {font-size: 2.2rem; font-weight: 700; line-height: 1.1;}
</style>
""", unsafe_allow_html=True)


def card(label, value, sub, colour):
    return (f'<div class="dp-card" style="--c:{colour}"><div class="k">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def boxes(items):
    for c, (h, t) in zip(st.columns(len(items)), items):
        c.markdown(f'<div class="dp-step"><b>{h}</b><div>{t}</div></div>', unsafe_allow_html=True)
    st.write("")


def conf_items(ctx):
    """Per-item reader confidence (equipment and leak points), computed once per result."""
    if "conf_items" not in ctx:
        rows = confidence.items(ctx["plant"], ctx["img"], ctx.get("px_per_m", 10.0), ctx.get("nn_debug"))
        if "opt" in ctx and "explain" in ctx:
            for r in rows:
                r["relies"] = confidence.detectors_relying_on(ctx["problem"], ctx["opt"]["chosen"]["design"],
                                                              ctx["explain"]["schedule"], r, ctx["plant"])
        ctx["conf_items"] = rows
    return ctx["conf_items"]


citems = conf_items(ctx)
cthr = ss._conf_thr / 100


def open_flags():
    flags = list(ctx["explain"].get("flags", [])) if has_opt else list(ctx.get("input_flags", []))
    todo = [r for r in citems if r["confidence"] < cthr and r["tag"] not in ss.checked]
    if todo:
        flags.append({"severity": "warning", "source": "Reader confidence",
                      "message": f"{len(todo)} item(s) read with under {cthr:.0%} confidence are not checked yet: "
                                 + ", ".join(f"{r['tag']} ({r['confidence']:.0%})" for r in todo)
                                 + ". Tick them off in step 3 once an engineer has checked the drawing."})
    return flags


def plant_fig(show_grades=True, height=520):
    fig = go.Figure()
    if show_grades:
        g = np.where(plant.grades > 0, plant.grades, np.nan).astype(float)
        fig.add_heatmap(z=g, x0=RES / 2, dx=RES, y0=RES / 2, dy=RES, zmin=1, zmax=3, showscale=False,
                        colorscale=[[0, "#fff4c2"], [0.5, "#ffd9b0"], [1, "#ffc4c4"]], hoverinfo="skip")
    h = np.where(plant.heights > 0, plant.heights, np.nan)
    fig.add_heatmap(z=h, x0=RES / 2, dx=RES, y0=RES / 2, dy=RES, colorscale="Greys", zmin=-4, zmax=10,
                    showscale=False, hovertemplate="x %{x:.1f} m, y %{y:.1f} m<br>top %{z:.0f} m<extra></extra>")
    fig.update_yaxes(scaleanchor="x", title="y (m)", range=[plant.depth, 0], constrain="domain")
    fig.update_xaxes(title="x (m)", range=[0, plant.width], constrain="domain")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", showlegend=False)
    return fig


def model_image(shape, ppm):
    """The AI's reading drawn at the drawing's own size, for the before/after slider."""
    rgb = np.full(plant.shape + (3,), 248, np.uint8)
    for g, c in {1: (255, 240, 160), 2: (255, 205, 150), 3: (250, 160, 160)}.items():
        rgb[plant.grades == g] = c
    hh = plant.heights
    grey = np.clip(205 - 15 * hh, 50, 200).astype(np.uint8)
    rgb[hh > 0] = np.stack([grey] * 3, -1)[hh > 0]
    big = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    for x, y in plant.leaks:
        cv2.circle(big, (int(x * ppm), int(y * ppm)), max(5, int(0.7 * ppm)), (30, 80, 220), -1, cv2.LINE_AA)
    return big


COV_LEGEND = ('<span style="color:#2a9d8f">■</span> seen by 2+ flame detectors &nbsp; '
              '<span style="color:#f5a524">■</span> seen by 1 &nbsp; <span style="color:#e5484d">■</span> fire area not '
              'seen &nbsp; <span style="color:#8e4ec6">■</span> gas zone not covered &nbsp; '
              '<span style="color:#94a3b8">■</span> equipment (darker = taller)')
EMPTY = Design(np.array([], int), np.array([], int))


def header(i, title, lead):
    st.markdown(f'<p class="dp-head">{STEPS[i][0]} Step {i + 1} · {title}</p><p class="dp-lead">{lead}</p>',
                unsafe_allow_html=True)
    if ss.get("notes"):
        st.markdown(f'<div class="dp-note">🎤 <b>Say:</b> {NOTES[i]}</div>', unsafe_allow_html=True)


def footer(i, next_label=None, disabled=False):
    st.write("")
    a, _, c = st.columns([1, 3, 1.4])
    if i > 0:
        a.button("◀ Back", on_click=go_to, args=(i - 1,), width="stretch", key=f"back{i}")
    if i < len(STEPS) - 1:
        c.button(next_label or f"Next: {STEPS[i + 1][1]} ▶", on_click=go_to, args=(i + 1,), type="primary",
                 width="stretch", key=f"next{i}", disabled=disabled)


# ------------------------------------------------------------------------ header + stepper
h1, h2 = st.columns([5, 1], vertical_alignment="bottom")
h1.markdown(f'<p class="dp-title">🔥 DetectorPlace</p><p class="dp-sub">From a plant drawing to a checked, costed '
            f'fire &amp; gas detector layout · <b>{plant.name}</b> · all data and rates illustrative</p>',
            unsafe_allow_html=True)
h2.toggle("🎤 Presenter notes", key="notes")
nav = st.segmented_control("Step", LABELS, key="nav", label_visibility="collapsed", width="stretch")
if nav is None:                          # clicking the selected pill again would clear it
    nav = LABELS[0]
step = LABELS.index(nav)
st.progress((step + 1) / len(STEPS))

# ======================================================================== 1 · Upload
if step == 0:
    header(0, "Upload a plant drawing", "A CAD file is read directly from its layers. A scan or image is read by "
                                        "the AI. The demo drawing is ready to go.")
    a, b = st.columns([2, 3], gap="large")
    with a:
        src = st.radio("Plant drawing", ["Demo plot plan", "Demo plant, different drawing style",
                                         "Sample CAD file (DXF)", "Upload an image", "Upload a CAD file (DXF)"])
        args, preview = None, None
        if src == "Demo plot plan":
            preview = baseline_ctx()["img"]
        elif src == "Demo plant, different drawing style":
            st.caption("The same plant drawn in a style the OpenCV rules were never tuned for.")
            reader = st.radio("Reader", ["Learned model", "OpenCV rules"], horizontal=True)
            preview = scene.render(scene.demo_spec(), scene.hard_style())[0]
            args = dict(source=preview, kind="image", px_per_m=12.0, truth=demo_plant(),
                        reader="learned" if reader == "Learned model" else "opencv",
                        name="Synthetic gas compression station")
        elif src == "Sample CAD file (DXF)":
            st.caption("samples/demo_plant.dxf: layers EQUIP (thickness = height), HAZ_A/B/C, LEAK, FG_PANEL.")
            args = dict(source=str(SAMPLES / "demo_plant.dxf"), kind="dxf",
                        name="Synthetic gas compression station (CAD)")
        elif src == "Upload an image":
            up = st.file_uploader("Plot plan (PNG/JPG)", type=["png", "jpg", "jpeg"])
            ppm = st.number_input("Scale (pixels per metre)", 2.0, 50.0, 10.0, 0.5,
                                  help="For files in samples/training_drawings, use the number in the file name.")
            reader = st.radio("Reader", ["Learned model", "OpenCV rules"], horizontal=True)
            if up is not None:
                preview = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)[..., ::-1].copy()
                args = dict(source=preview, kind="image", px_per_m=float(ppm), name=up.name,
                            reader="learned" if reader == "Learned model" else "opencv")
        else:
            up = st.file_uploader("Plot plan (DXF)", type=["dxf"])
            st.caption("Layers: EQUIP* (closed outlines; height from thickness or an _H9 suffix), HAZ_A/B/C, LEAK*, "
                       "FG_PANEL, optional PLOT_LIMIT.")
            if up is not None:
                import io
                args = dict(source=io.StringIO(up.getvalue().decode("utf-8", errors="replace")), kind="dxf",
                            name=up.name)
        with st.expander("Prices and rates (optional)"):
            for key, label, _, field, stp in RATES:
                st.number_input(label, min_value=0.0, step=stp, format="%.1f" if field == "tail_m" else "%.0f",
                                key=f"_{key}")
            st.button("Reset to defaults", on_click=reset_rates, width="stretch")
            st.caption("Default rates are placeholders. Enter supplier and company rates for a real estimate.")
        if args is not None and args.get("reader") == "learned" and not segnet.available():
            st.error("The learned reader needs torch and models/segnet.pt.")
    with b:
        if preview is not None:
            st.image(preview, width="stretch")
        elif src.endswith("(DXF)"):
            st.info("CAD drawings are read layer by layer: equipment outlines with heights, hazard zones and leak "
                    "points. No recognition is needed, so there is nothing to guess.")
    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    if mid.button("🧠 Read this drawing ▶", type="primary", width="stretch",
                  disabled=(args is None and src != "Demo plot plan")):
        try:
            if args is None:
                new_result(baseline_ctx())
                ss.input = None
            else:
                with st.spinner("Reading the drawing and building the coverage model..."):
                    new_result(pipeline.build_from(**args, **models))
                ss.input = args
            ss._goto = 1
            st.rerun()
        except Exception as e:  # show input problems to the user, not a stack trace
            st.error(f"Could not process this input: {e}")

# ======================================================================== 2 · AI reads it
elif step == 1:
    header(1, "The AI turns the drawing into an engineering model",
           f"It found <b>{len(plant.obstacles)} pieces of equipment</b> with their heights, "
           f"<b>{int((plant.grades > 0).sum() * RES * RES):,} m²</b> of hazard area and "
           f"<b>{len(plant.leaks)} leak points</b>.")
    if ctx["img"] is not None:
        a, b = st.columns([3, 1], gap="large")
        with b:
            mix = st.slider("Drag ▶ to see what the AI read", 0, 100, 0, 5,
                            help="Drag to fade between the original drawing and what the AI read from it.")
            st.markdown('<span style="color:#6b7280">■</span> equipment (darker = taller)<br>'
                        '<span style="color:#f3a3a3">■</span> Grade A &nbsp; <span style="color:#f9c98f">■</span> '
                        'Grade B &nbsp; <span style="color:#f5e08f">■</span> Grade C<br>'
                        '<span style="color:#1e50dc">●</span> leak point', unsafe_allow_html=True)
            vs = ctx["vision_score"]
            if vs:
                st.metric("Equipment outline match", f"{vs['equipment_iou'] * 100:.1f}%")
                st.metric("Heights correct", f"{vs['height_band_accuracy'] * 100:.0f}%")
                st.metric("Leak points found", f"{vs['leaks_found']} of {vs['leaks_true']}")
        img = ctx["img"]
        mimg = model_image(img.shape[:2], ctx.get("px_per_m", 10.0))
        a.image((img * (1 - mix / 100) + mimg * (mix / 100)).astype(np.uint8), width="stretch")
    else:
        a, b = st.columns(2)
        rep = ctx["vision_debug"]["cad_report"]
        a.markdown("**Input: CAD drawing (DXF)**, read by layer, no recognition needed")
        a.dataframe(pd.DataFrame([{"Item": k.replace("_", " "), "Count": v} for k, v in rep.counts.items()] +
                                 [{"Item": "units", "Count": rep.units}]).astype(str), hide_index=True)
        fig = plant_fig()
        fig.add_scatter(x=plant.leaks[:, 0], y=plant.leaks[:, 1], mode="markers",
                        marker=dict(color="#1d4ed8", size=9, symbol="diamond"))
        b.plotly_chart(fig, width="stretch")
    with st.expander("How the drawing is read"):
        if ctx.get("reader", "").startswith("Learned"):
            st.write("Learned reader: a small U-Net (0.39 M parameters, CPU) trained on 2,000 randomised synthetic "
                     "plot plans with real P&ID line art as clutter. It outputs per-cell elevation band, hazard grade "
                     "and a leak-source heatmap, with a confidence for every cell.")
        else:
            st.write("OpenCV pipeline: HSV segmentation → morphology → connected components → elevation from fill "
                     "shade → resample to the analysis grid. The learned U-Net reads the same drawing as a second "
                     "opinion and supplies the confidence scores in step 3.")
        st.dataframe(pd.DataFrame([{"Hazard grade": GRADES[g], "Open area (m²)": int(((plant.grades == g) &
                                    (plant.heights == 0)).sum() * RES * RES)} for g in (3, 2, 1)]), hide_index=True)
    footer(1)

# ======================================================================== 3 · Confidence
elif step == 2:
    header(2, "How sure is the AI? The engineer checks the low scores",
           "Every piece of equipment and every leak point gets a confidence score. Red items go to an engineer. "
           "Danger zones are always set by engineers, not by the AI.")
    if not citems:
        st.info("CAD input is read exactly from its layers, with no recognition step, so there is no confidence "
                "score to check." if ctx["img"] is None else
                "Confidence scores need the learned reader (torch and models/segnet.pt).")
    else:
        low = [r for r in citems if r["confidence"] < cthr]
        ca, cb = st.columns([1, 1], gap="large")
        ca.image(confidence.overlay(ctx["img"], citems, ctx.get("px_per_m", 10.0), cthr), width="stretch",
                 caption="Green = confident · Red = engineer to check")
        with cb:
            st.slider("Flag anything under this confidence (%)", 50, 99, key="_conf_thr")
            done = len([r for r in low if r["tag"] in ss.checked])
            if low:
                st.markdown(f'<div class="dp-big">{done} / {len(low)}</div>flagged items checked by the engineer',
                            unsafe_allow_html=True)
                st.progress(done / len(low))
                b1, b2 = st.columns(2)
                b1.button("✅ Mark all as checked", width="stretch", type="primary" if done < len(low) else "secondary",
                          on_click=lambda: ss.update(checked=ss.checked | {r["tag"] for r in low}))
                b2.button("↺ Reset checks", width="stretch", on_click=lambda: ss.update(checked=set()))
            else:
                st.success(f"All {len(citems)} items are above {cthr:.0%} confidence.")
            low_sorted = sorted(low, key=lambda r: r["confidence"])
            tbl = pd.DataFrame([{"Item": r["tag"], "AI reads it as": r["type"],
                                 "Confidence": round(r["confidence"] * 100),
                                 "Used by": ", ".join(r.get("relies", [])) or "-",
                                 "Checked": r["tag"] in ss.checked} for r in low_sorted],
                               columns=["Item", "AI reads it as", "Confidence", "Used by", "Checked"])
            ed_key = f"conf_ed_{id(ctx)}_{len(ss.checked)}_{cthr}"

            def on_tick(key=ed_key, order=[r["tag"] for r in low_sorted]):
                chk = set(ss.checked)
                for row, change in ss[key]["edited_rows"].items():
                    if "Checked" in change:
                        (chk.add if change["Checked"] else chk.discard)(order[int(row)])
                ss.checked = chk

            if low:
                st.data_editor(tbl, hide_index=True, key=ed_key, on_change=on_tick,
                               disabled=["Item", "AI reads it as", "Confidence", "Used by"],
                               column_config={"Confidence": st.column_config.ProgressColumn(
                                   "Confidence", format="%d%%", min_value=0, max_value=100),
                                   "Checked": st.column_config.CheckboxColumn("Checked ✓")})
            if not has_opt:
                st.caption("'Used by' fills in after the layout is optimised in step 6.")
        with st.expander(f"All {len(citems)} items"):
            st.dataframe(pd.DataFrame([{"Item": r["tag"], "AI reads it as": r["type"],
                                        "Confidence": f"{r['confidence']:.0%}"} for r in citems]), hide_index=True)
    footer(2)

# ======================================================================== 4 · Line of sight
elif step == 3:
    header(3, "What can one detector actually see?",
           "<b>Click anywhere on the plan</b> to place a flame detector. Tall equipment blocks its view.")
    fm = FlameModel()
    a, b = st.columns([2.5, 1.5], gap="large")
    with b:
        d = st.segmented_control("Facing", list(DIRS), default="→", key="los_dir")
        z = st.segmented_control("Mounted at", ["3 m", "6 m"], default="3 m", key="los_z")
    az = DIRS.get(d or "→", 0)
    zm = 6.0 if z == "6 m" else 3.0
    x, y = ss.los
    ny, nx = plant.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    free = plant.heights == 0
    tx, ty = ((xx[free] + 0.5) * RES).astype(np.float64), ((yy[free] + 0.5) * RES).astype(np.float64)
    tilt = np.deg2rad(15.0 if zm == 3.0 else 30.0)
    vis = _cone_coverage(plant.heights.astype(np.float32), RES, np.array([x]), np.array([y]), np.array([zm]),
                         np.array([np.deg2rad(az)]), np.array([tilt]), tx, ty, np.full(len(tx), fm.fire_height_m),
                         float(fm.range_m), np.deg2rad(fm.hfov_deg / 2), np.deg2rad(fm.vfov_deg / 2),
                         fm.offaxis_loss)[0]
    ras = np.full(plant.shape, np.nan)
    ras[free] = np.where(vis == 1, 1.0, np.nan)
    wx, wy = flame_wedge(x, y, az, fm)
    inside = np.zeros(plant.shape, bool)                       # in the cone but blocked = shadow
    poly = np.stack([wx / RES, wy / RES], -1).astype(np.int32)
    cv2.fillPoly(inside.view(np.uint8), [poly], 1)
    shadow = np.where(inside & free & np.isnan(ras), 1.0, np.nan)
    fig = plant_fig(height=560)
    fig.add_heatmap(z=ras, x0=RES / 2, dx=RES, y0=RES / 2, dy=RES, colorscale=[[0, "#2563eb"], [1, "#2563eb"]],
                    opacity=0.45, showscale=False, hoverinfo="skip")
    fig.add_heatmap(z=shadow, x0=RES / 2, dx=RES, y0=RES / 2, dy=RES, colorscale=[[0, "#111827"], [1, "#111827"]],
                    opacity=0.25, showscale=False, hoverinfo="skip")
    fig.add_scatter(x=wx, y=wy, mode="lines", line=dict(color="#1d4ed8", width=1.5, dash="dot"), hoverinfo="skip")
    fig.add_scatter(x=plant.leaks[:, 0], y=plant.leaks[:, 1], mode="markers", hoverinfo="skip",
                    marker=dict(color="#dc2626", size=10, symbol="x"))
    fig.add_scatter(x=[x], y=[y], mode="markers", marker=dict(symbol="diamond", size=16, color="#1d4ed8",
                                                              line=dict(color="white", width=2)), hoverinfo="skip")
    fig.add_heatmap(z=np.zeros(plant.shape), x0=RES / 2, dx=RES, y0=RES / 2, dy=RES, opacity=0, showscale=False,
                    colorscale=[[0, "white"], [1, "white"]], hovertemplate="click to place the detector here<extra></extra>")
    with a:
        click = clickable_chart(fig, height=560, key="los_click")
    if click and click.get("nonce") != ss.get("los_nonce") and click.get("x") is not None:
        ss.los_nonce = click["nonce"]
        cx, cy = float(click["x"]), float(click["y"])
        if 0 < cx < plant.width and 0 < cy < plant.depth and plant.heights[int(cy / RES), int(cx / RES)] == 0:
            ss.los = (cx, cy)
            st.rerun()
        else:
            st.toast("That spot is inside a piece of equipment. Click on open ground.")
    seen_leaks = sum(1 for lx, ly in plant.leaks if ras[min(int(ly / RES), ny - 1), min(int(lx / RES), nx - 1)] == 1)
    with b:
        st.metric("Area it can see", f"{int(vis.sum() * RES * RES)} m²")
        st.metric("Hidden behind equipment", f"{int(np.nansum(shadow) * RES * RES)} m²")
        st.metric("Leak points in view", f"{seen_leaks} of {len(plant.leaks)}")
        st.markdown('<span style="color:#2563eb">■</span> can see &nbsp; <span style="color:#6b7280">■</span> '
                    'blocked (shadow)<br><span style="color:#dc2626">✕</span> leak point', unsafe_allow_html=True)
        st.caption("Try 6 m: it sees over the low pump skid. The 9 m vessels still cast shadows.")
    footer(3)

# ======================================================================== 5 · 3D model
elif step == 4:
    header(4, "The 3D plant model",
           "Every piece of equipment at its real height, with the hazard zones and leak points. Drag to rotate.")
    zones = st.toggle("Show hazard zones", True)
    st.plotly_chart(view3d.figure(ctx, EMPTY, [], colour_by="grade" if zones else "plain", show_cones=False,
                                  show_gas_radius=False, show_leaks=True, height=600),
                    width="stretch", config={"displaylogo": False}, key="model3d")
    st.markdown(('<span style="color:#f3a3a3">■</span> Grade A &nbsp; <span style="color:#f9c98f">■</span> Grade B '
                 '&nbsp; <span style="color:#fbe9a6">■</span> Grade C &nbsp; ' if zones else "")
                + '<span style="color:#94a3b8">■</span> equipment (darker = taller) &nbsp; '
                  '<span style="color:#dc2626">✕</span> gas leak point', unsafe_allow_html=True)
    footer(4, "Next: place the detectors ▶")

# ======================================================================== 6 · Optimise
elif step == 5:
    header(5, "The computer evolves the best layout",
           "A genetic algorithm tries thousands of layouts, keeps the best and improves them, like natural "
           "selection. Every layout is checked in 3D.")
    c1, c2, c3 = st.columns([2, 1.3, 1.3], vertical_alignment="bottom")
    gens = c1.select_slider("Generations", [10, 15, 20, 30, 40, 60, 80], 20,
                            help="About 1 second each. 20 is plenty for a demo.")
    run_live = c2.button("▶ Run the genetic algorithm", type="primary", width="stretch")
    saved_ok = ss.input is None and "opt" in baseline_ctx()
    use_saved = c3.button("⏩ Skip: use saved result", width="stretch", disabled=not saved_ok)
    left, right = st.columns([3, 1], gap="medium")
    live_area = left.empty()
    with right:
        cost_area, chart_area, status_area = st.empty(), st.empty(), st.empty()

    if use_saved:
        ss.ctx, ss.ga_done, ss.dlg_chat = baseline_ctx(), True, {}
        st.rerun()

    if run_live:
        run_ctx = ctx
        if ctx.get("catalogue") != catalogue_dict(**models):     # prices changed in step 1: rebuild costs
            with st.spinner("Applying the new prices..."):
                inp = ss.input
                if inp is None:
                    run_ctx = pipeline.build(**models)
                else:
                    if hasattr(inp.get("source"), "seek"):
                        inp["source"].seek(0)
                    run_ctx = pipeline.build_from(**inp, **models)
        rp, rcand = run_ctx["problem"], run_ctx["cand"]
        greedy_cost = None
        every = max(1, gens // 15)
        hist = []

        def on_step(g, n, best, d):
            hist.append(best)
            status_area.progress(g / n, f"Generation {g} of {n}")
            if best == best:
                cost_area.markdown(f'<div style="color:#64748b;font-size:.85rem">CHEAPEST SAFE LAYOUT SO FAR</div>'
                                   f'<div class="dp-big">£{best:,.0f}</div>'
                                   f'<div style="color:#64748b">{len(d.flame)} flame · {len(d.gas)} gas detectors</div>',
                                   unsafe_allow_html=True)
            ys = [h for h in hist if h == h]
            if ys:
                f = go.Figure(go.Scatter(x=[i + 1 for i, h in enumerate(hist) if h == h], y=[v / 1e3 for v in ys],
                                         mode="lines+markers", line=dict(color=GREEN, width=3), marker=dict(size=5)))
                f.update_layout(height=230, margin=dict(l=0, r=0, t=26, b=0), title=dict(text="Cost by generation",
                                                                                        font=dict(size=13)),
                                xaxis=dict(title=None, range=[0, n + 1]), yaxis=dict(title="£k"),
                                plot_bgcolor="rgba(0,0,0,0)")
                chart_area.plotly_chart(f, width="stretch", key=f"ga_cost_{g}", config={"displayModeBar": False})
            if g == 1 or g % every == 0 or g == n:
                fig = view3d.figure(run_ctx, d, explain.schedule(rp, rcand, d), colour_by="coverage", show_cones=True,
                                    show_gas_radius=False, show_leaks=True, height=560)
                fig.update_layout(showlegend=False)
                fig.add_annotation(text=f"<b>Generation {g} / {n}</b> · {len(d.flame)} flame + {len(d.gas)} gas",
                                   xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False, align="left",
                                   font=dict(size=15), bgcolor="rgba(255,255,255,0.85)")
                live_area.plotly_chart(fig, width="stretch", key=f"ga_live_{g}", config={"displaylogo": False})

        new_opt = pipeline.optimise(run_ctx, generations=gens, on_step=on_step)
        new = {k: v for k, v in run_ctx.items() if k != "conf_items"}
        new["opt"] = new_opt
        new["explain"] = pipeline.explain_design(new, new_opt["chosen"]["design"])
        ss.ctx, ss.ga_done, ss.dlg_chat = new, True, {}
        st.rerun()

    if ss.ga_done and has_opt:
        opt, ex = ctx["opt"], ctx["explain"]
        chosen = opt["chosen"]
        fig = view3d.figure(ctx, chosen["design"], ex["schedule"], colour_by="coverage", show_cones=True,
                            show_gas_radius=False, show_leaks=True, height=560)
        fig.update_layout(showlegend=False)
        fig.add_annotation(text=f"<b>Final layout</b> after {len(opt['history'])} generations", xref="paper",
                           yref="paper", x=0.01, y=0.99, showarrow=False, font=dict(size=15),
                           bgcolor="rgba(255,255,255,0.85)")
        live_area.plotly_chart(fig, width="stretch", key="ga_final", config={"displaylogo": False})
        hist = pd.DataFrame(opt["history"])
        f = go.Figure(go.Scatter(x=hist["generation"], y=hist["cheapest_feasible"] / 1e3, mode="lines",
                                 line=dict(color=GREEN, width=3), name="genetic algorithm"))
        f.add_hline(y=opt["greedy"]["cost"] / 1e3, line_dash="dash", line_color=SLATE,
                    annotation_text="rule of thumb", annotation_position="top right")
        f.update_layout(height=230, margin=dict(l=0, r=0, t=26, b=0), showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                        title=dict(text="Cost by generation", font=dict(size=13)), yaxis=dict(title="£k"))
        chart_area.plotly_chart(f, width="stretch", key="ga_cost_final", config={"displayModeBar": False})
        cost_area.markdown(f'<div style="color:#64748b;font-size:.85rem">FINAL LAYOUT</div>'
                           f'<div class="dp-big">£{chosen["cost"]:,.0f}</div><div style="color:#64748b">detectors '
                           f'&amp; cabling</div>', unsafe_allow_html=True)
        st.markdown(COV_LEGEND, unsafe_allow_html=True)

        flags = open_flags()
        nflag = {k: sum(x["severity"] == k for x in flags) for k in ("error", "warning", "info")}
        tcov = ex["truth_coverage"]
        met = sum(tcov[q.key] >= q.target for q in p.reqs)
        fixed = ctx["catalogue"]["cost"]["fixed"]
        saving = opt["greedy"]["cost"] - chosen["cost"]
        nfl, ngs = len(chosen["design"].flame), len(chosen["design"].gas)
        st.markdown('<div class="dp-cards">' + "".join([
            card("Total installed cost", f"£{(chosen['cost'] + fixed) / 1000:,.0f}k",
                 f"£{chosen['cost']:,.0f} detectors &amp; cabling + £{fixed / 1000:,.0f}k fixed", BLUE),
            card("Detectors", f"{nfl + ngs}", f"{nfl} flame · {ngs} gas", SLATE),
            card("Saving vs simple design", f"{saving / opt['greedy']['cost'] * 100:.0f}%",
                 f"£{saving:,.0f} cheaper than the greedy rule of thumb", GREEN if saving > 0 else AMBER),
            card("Coverage targets", f"{met} of {len(p.reqs)} met",
                 f"re-checked on {ctx.get('check_name', 'the true geometry')}", GREEN if met == len(p.reqs) else RED),
            card("Quality checks", "OK" if nflag["error"] == 0 and nflag["warning"] <= 1 else
                 (f"{nflag['error']} error{'s' if nflag['error'] != 1 else ''}" if nflag["error"]
                  else f"{nflag['warning']} warnings"),
                 f"{nflag['error']} errors · {nflag['warning']} warnings · {nflag['info']} notes",
                 RED if nflag["error"] else (AMBER if nflag["warning"] > 1 else GREEN)),
        ]) + "</div>", unsafe_allow_html=True)

        with st.expander("Coverage per target, cost breakdown and the cost-vs-risk trade-off"):
            a, b = st.columns([3, 2])
            reqs = list(p.reqs)[::-1]
            fig = go.Figure()
            fig.add_bar(y=[q.label for q in reqs], x=[tcov[q.key] * 100 for q in reqs], orientation="h",
                        marker_color=[GREEN if tcov[q.key] >= q.target else RED for q in reqs],
                        text=[f"<b>{tcov[q.key] * 100:.1f}%</b>  (target {q.target * 100:.0f}%)" for q in reqs],
                        textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
            for i, q in enumerate(reqs):
                fig.add_shape(type="line", x0=q.target * 100, x1=q.target * 100, y0=i - 0.42, y1=i + 0.42,
                              line=dict(color="#111827", width=3))
            fig.update_xaxes(range=[0, 125], title="% of area covered", tickvals=[0, 25, 50, 75, 100])
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                              plot_bgcolor="rgba(0,0,0,0)", bargap=0.35)
            a.plotly_chart(fig, width="stretch")
            tb = report.tables(ctx)
            rows_c = [(r[0], float(str(r[2]).replace(",", ""))) for r in tb["cost"][1:-1]
                      if float(str(r[2]).replace(",", "")) > 0]
            rows_c.sort(key=lambda r: r[1])
            fig = go.Figure(go.Bar(y=[r[0] for r in rows_c], x=[r[1] / 1000 for r in rows_c], orientation="h",
                                   marker_color=BLUE, text=[f"£{r[1] / 1000:,.1f}k" for r in rows_c],
                                   textposition="outside", cliponaxis=False))
            fig.update_xaxes(title="£k", range=[0, max(r[1] for r in rows_c) / 1000 * 1.35])
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
            b.plotly_chart(fig, width="stretch")
            fr = opt["front"]
            fig = go.Figure()
            fig.add_scatter(x=[q["cost"] / 1e3 for q in fr], y=[q["shortfall"] * 100 for q in fr],
                            mode="lines+markers", name="best trade-offs (Pareto front)",
                            marker=dict(size=6, color=["#16a34a" if q["feasible"] else "#94a3b8" for q in fr]))
            fig.add_scatter(x=[chosen["cost"] / 1e3], y=[chosen["shortfall"] * 100], mode="markers", name="chosen",
                            marker=dict(symbol="star", size=18, color="#d97706"))
            fig.add_scatter(x=[opt["greedy"]["cost"] / 1e3], y=[opt["greedy"]["shortfall"] * 100], mode="markers",
                            name="rule of thumb", marker=dict(symbol="square", size=11, color="#475569"))
            fig.update_yaxes(type="log", title="risk left uncovered (%)", tickvals=[0.1, 0.3, 1, 3, 10, 30, 100],
                             ticktext=["0.1", "0.3", "1", "3", "10", "30", "100"])
            fig.update_xaxes(title="installed cost (£k)")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")
            st.caption("Green points meet every target. The engineer picks a point on this front; the star is the "
                       "cheapest one that meets every target.")
            st.markdown("**Change the targets and re-optimise**")
            cols = st.columns(len(DEFAULT_REQS))
            new_reqs = [replace(q, target=cols[i].slider(q.label, 0.5, 0.99, q.target, 0.01, key=f"t{q.key}"))
                        for i, q in enumerate(p.reqs)]
            g1, g2 = st.columns(2)
            re_gens = g1.slider("Generations", 10, 150, 40, 10, key="re_gens")
            margin = g2.slider("Design margin on every target", 0.0, 0.05, p.margin, 0.01)
            if st.button("Re-optimise"):
                bar = st.progress(0.0, "Building coverage model...")
                inp = ss.input
                if inp is None:
                    new = pipeline.build(reqs=new_reqs, margin=margin, **models)
                else:
                    if hasattr(inp.get("source"), "seek"):
                        inp["source"].seek(0)
                    new = pipeline.build_from(**inp, reqs=new_reqs, margin=margin, **models)
                cb = lambda g, n, best: bar.progress(g / n, f"Generation {g}/{n}")
                new["opt"] = pipeline.optimise(new, generations=re_gens, callback=cb)
                new["explain"] = pipeline.explain_design(new, new["opt"]["chosen"]["design"])
                ss.ctx, ss.ga_done, ss.dlg_chat = new, True, {}
                st.rerun()
    elif not run_live:
        live_area.plotly_chart(view3d.figure(ctx, EMPTY, [], colour_by="grade", show_cones=False,
                                             show_gas_radius=False, show_leaks=True, height=560)
                               .update_layout(showlegend=False), width="stretch", key="ga_empty")
        cost_area.info("Press **▶ Run** to watch the layout being designed live.")
    footer(5, "Next: ask about a detector ▶", disabled=not ss.ga_done)

# ======================================================================== 7 · Ask
elif step == 6:
    header(6, "Click any detector and ask about it",
           "Every answer comes from re-running the real coverage calculations. The engineer decides.")
    if not (ss.ga_done and has_opt):
        st.info("Run the genetic algorithm first.")
        st.button("◀ Go to step 6", on_click=go_to, args=(5,), type="primary")
    else:
        assist = DesignAssistant(ctx, citems, cthr)
        ex, chosen = ctx["explain"], ctx["opt"]["chosen"]
        sched = ex["schedule"]
        cf_all, cg_all = counts(p, chosen["design"])

        @st.dialog("Ask about this detector", width="large")
        def detector_dialog(tag):
            r = assist.by_tag[tag]
            flame = r["type"].startswith("Flame")
            left, right = st.columns([3, 2], gap="medium")
            with left:
                st.markdown(f"### {tag} · {r['type']}")
                d1, d2, d3 = st.columns(3)
                d1.metric("Mounted at", f"{r['mount_m']:.0f} m")
                d2.metric("Only it sees", f"{r['unique_points']} m²")
                d3.metric("Installed cost", f"£{r['cost']:,.0f}")
                level = "technical" if st.toggle("Technical answer", key="tech_answers",
                                                 help="Off: short plain-English answers. On: full figures.") \
                    else "simple"
            with right:
                seen = (p.Mf if flame else p.Mg)[r["cand"]].astype(bool)
                xy = (p.fire if flame else p.gas).xy
                only = seen & ((cf_all if flame else cg_all) == 1)
                mf = plant_fig(height=230)
                mf.add_scatter(x=xy[seen & ~only, 0], y=xy[seen & ~only, 1], mode="markers", hoverinfo="skip",
                               marker=dict(color="#2563eb", size=3, symbol="square", opacity=0.6))
                mf.add_scatter(x=xy[only, 0], y=xy[only, 1], mode="markers", hoverinfo="skip",
                               marker=dict(color="#dc2626", size=3, symbol="square"))
                mf.add_scatter(x=[r["x"]], y=[r["y"]], mode="markers", hoverinfo="skip",
                               marker=dict(symbol="diamond" if flame else "circle", size=13, color="#1d4ed8",
                                           line=dict(color="white", width=2)))
                mf.update_xaxes(visible=False)
                mf.update_yaxes(visible=False)
                st.plotly_chart(mf, width="stretch", config={"displayModeBar": False, "staticPlot": True})
                st.markdown('<span style="color:#2563eb">■</span> it sees &nbsp; <span style="color:#dc2626">■</span> '
                            'only it sees', unsafe_allow_html=True)
            chat = ss.dlg_chat.setdefault(tag, [])
            q = None
            b1, b2, b3, b4 = st.columns([3, 3, 2, 3], vertical_alignment="bottom")
            if b1.button("❓ Why is it here?", width="stretch"):
                q = f"Why is {tag} here?"
            if b2.button("⚠️ What if it fails?", width="stretch"):
                q = f"What if {tag} fails?"
            mv_dir = b3.selectbox("Direction", ["left", "right", "up", "down"], label_visibility="collapsed")
            if b4.button(f"↔️ Move it 2 m {mv_dir}?", width="stretch"):
                q = f"Can I move {tag} 2 m {mv_dir}?"
            with st.form(f"ask_{tag}", clear_on_submit=True, border=False):
                f1, f2 = st.columns([5, 1])
                typed = f1.text_input("Your question", placeholder=f"Ask anything about {tag}",
                                      label_visibility="collapsed")
                if f2.form_submit_button("Ask", width="stretch") and typed.strip():
                    q = typed.strip()
            if q:
                if DesignAssistant.claude_available():
                    text, _ = assist.answer(q, [], tag, True, level)
                    chat.append({"q": q, "text": text})
                else:
                    name, res = assist.route(q, tag)
                    chat.append({"q": q, "name": name, "res": res})
            for m in reversed(chat):
                with st.chat_message("user", avatar="🧑‍🔧"):
                    st.markdown(m["q"])
                with st.chat_message("assistant", avatar="🔥"):
                    st.markdown(m["text"] if "text" in m else
                                assist.HELP if m["name"] is None else assist.describe(m["name"], m["res"], level))
            if not chat:
                st.caption("Pick a question above. Answers re-run the real coverage calculations.")

        o1, o2, o3, o4 = st.columns(4)
        show_cov = o1.toggle("Show coverage", True)
        cones = o2.toggle("Cones", True, disabled=not show_cov)
        labels_on = o3.toggle("Labels", True)
        leaks_on = o4.toggle("Leaks", True)
        focus = ss.get("dlg_tag") if ss.get("dlg_tag") in assist.by_tag else None
        fig3 = view3d.figure(ctx, chosen["design"], sched, colour_by="coverage" if show_cov else "plain",
                             selected=focus if show_cov else None, show_cones=cones and show_cov,
                             show_gas_radius=False, show_leaks=leaks_on, show_labels=labels_on, height=620)
        fig3.update_layout(showlegend=False)
        click = clickable_chart(fig3, height=620, key=f"ask3d_{id(ctx)}")
        l1, l2 = st.columns([3, 2], vertical_alignment="center")
        l1.markdown((COV_LEGEND + "<br>" if show_cov else "") + '<b>Click</b> a <span style="color:#1d4ed8">◆ flame</span>'
                    ' or <span style="color:#7c3aed">● gas</span> detector to ask about it.', unsafe_allow_html=True)
        with l2:
            p1, p2 = st.columns([2, 1])
            pick = p1.selectbox("Or pick one", list(assist.by_tag), label_visibility="collapsed")
            if p2.button("Ask", width="stretch", key="pick_ask"):
                ss.dlg_tag = pick
                detector_dialog(pick)
        if click and click.get("nonce") != ss.get("last_click"):
            ss.last_click = click.get("nonce")
            if click.get("tag") in assist.by_tag:
                ss.dlg_tag = click["tag"]
                detector_dialog(click["tag"])
            else:
                st.toast("Click on a detector marker: a blue diamond (flame) or purple circle (gas).")
        with st.expander("Technical detail: detector schedule, single-failure check and gaps"):
            sdf = pd.DataFrame(sched).drop(columns=["cand"])
            st.dataframe(sdf[["tag", "type", "x", "y", "mount_m", "azimuth_deg", "unique_points", "cost", "why"]],
                         hide_index=True, height=300)
            n1 = pd.DataFrame(ex["n_minus_1"])
            n1["targets_missed"] = n1["targets_missed"].apply(lambda v: ", ".join(v) or "none")
            st.markdown("**If one detector fails (N-1)**")
            st.dataframe(n1, hide_index=True, height=220)
            st.markdown("**Residual gaps in Grade A/B areas**")
            st.dataframe(pd.DataFrame(ex["gaps"]), hide_index=True)
    footer(6)

# ======================================================================== 8 · Report
else:
    header(7, "One click: the design report",
           "Scope, method, targets against achieved, detector schedule, coverage map, single-failure check, cost "
           "estimate, limitations and sign-off, filled from a standard template.")
    if not has_opt:
        st.info("Run the genetic algorithm in step 6 first.")
        st.button("◀ Go to step 6", on_click=go_to, args=(5,), type="primary")
    else:
        flags = open_flags()
        a, b = st.columns([1, 2], gap="large")
        with a:
            st.download_button("📑 Download the PDF report", report.build_pdf(ctx), "DetectorPlace_FG_design_report.pdf",
                               "application/pdf", type="primary", width="stretch")
            st.warning("Advisory output. A competent F&G engineer must verify and sign off the layout before use.")
            tb = report.tables(ctx)
            st.dataframe(pd.DataFrame(tb["cost"][1:], columns=tb["cost"][0]).astype(str), hide_index=True)
        with b:
            st.markdown("**What could make this design wrong?** The coverage maths is exact for the model it is "
                        "given. These checks flag where the model itself may be wrong.")
            for x in sorted(flags, key=lambda x: ("error", "warning", "info").index(x["severity"])):
                {"error": st.error, "warning": st.warning, "info": st.info}[x["severity"]](
                    f"**{x['source']}** · {x['message']}")
            st.dataframe(pd.DataFrame(tb["coverage"][1:], columns=tb["coverage"][0]), hide_index=True)
    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    mid.button("↺ Start again from step 1", on_click=go_to, args=(0,), width="stretch")
