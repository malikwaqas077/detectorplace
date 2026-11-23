"""Read a CAD plot plan (DXF) straight into the 2.5D plant model, and write one.

A CAD file carries exact geometry, so there is nothing to recognise: the reader maps layers to
meaning. The layer convention below is an assumption for the demo; on a real project it is
agreed once with the drafting team (or mapped from their existing layer standard).

  EQUIP*         closed polylines / circles. Height = entity thickness (extrusion), or a layer
                 suffix such as EQUIP_H9, else a flagged default.
  HAZ_A/B/C      closed polylines: fire hazard Grade A / B / C areas (also HAZARD_GRADE_A etc.).
  LEAK*          points, circles or block inserts: credible gas leak sources.
  FG_PANEL       a point: F&G panel location (cable routing origin).
  PLOT_LIMIT     optional closed polyline: the analysis boundary.
  anything else  ignored (grid, text, dimensions, title block) and listed in the read report.

Units come from $INSUNITS (mm, cm, m, in, ft). CAD y points up; the plant model's y points down
the drawing, so y is flipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .plant import RES, Obstacle, Plant

try:
    import ezdxf
    from ezdxf import path as ezpath
except ImportError:
    ezdxf = None

UNITS_TO_M = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0}
UNIT_NAMES = {0: "unitless", 1: "inches", 2: "feet", 4: "mm", 5: "cm", 6: "m"}
DEFAULT_HEIGHT = 5.0
SS = 4   # supersampling when rasterising polygons onto the 0.5 m grid


@dataclass
class CadReport:
    units: str = "m"
    scale_to_m: float = 1.0
    counts: dict = field(default_factory=dict)
    ignored_layers: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)       # (severity, message)


def _grade_of(layer: str):
    L = layer.upper()
    m = re.match(r"^HAZ(?:ARD)?[_-]?(?:GRADE[_-]?)?([ABC])\b", L)
    return {"A": 3, "B": 2, "C": 1}[m.group(1)] if m else None


def _layer_height(layer: str):
    m = re.search(r"_H(\d+(?:\.\d+)?)", layer.upper())
    return float(m.group(1)) if m else None


def _outline(e, tol):
    """Flattened outline of a polyline / circle / ellipse as an (n, 2) array, and whether it is closed."""
    t = e.dxftype()
    if t == "CIRCLE":
        p = ezpath.make_path(e)
        return np.array([(v.x, v.y) for v in p.flattening(tol)]), True
    if t in ("LWPOLYLINE", "POLYLINE", "ELLIPSE", "SPLINE", "HATCH"):
        if t == "HATCH":
            paths = list(ezpath.from_hatch(e))
            if not paths:
                return None, False
            p = paths[0]
        else:
            p = ezpath.make_path(e)
        pts = np.array([(v.x, v.y) for v in p.flattening(tol)])
        closed = t in ("ELLIPSE", "HATCH") or bool(getattr(e, "closed", False) or getattr(e, "is_closed", False)) \
            or (len(pts) > 2 and np.allclose(pts[0], pts[-1]))
        return pts, closed
    return None, False


def read_dxf(path_or_stream, name: str | None = None):
    """Return (Plant, CadReport)."""
    if ezdxf is None:
        raise ImportError("pip install ezdxf to read CAD files")
    doc = ezdxf.read(path_or_stream) if hasattr(path_or_stream, "read") else ezdxf.readfile(path_or_stream)
    rep = CadReport()
    code = int(doc.header.get("$INSUNITS", 0))
    msp = doc.modelspace()
    equip, zones, leaks, panel, limit = [], [], [], None, None
    tol = 0.02   # in drawing units; rescaled below once units are known
    for e in msp:
        layer = e.dxf.layer
        L = layer.upper()
        t = e.dxftype()
        if L.startswith("EQUIP"):
            pts, closed = _outline(e, 1e-3)
            if pts is None:
                rep.ignored_layers.setdefault(f"{layer} ({t})", 0)
                rep.ignored_layers[f"{layer} ({t})"] += 1
                continue
            if not closed:
                rep.issues.append(("warning", f"Open {t} on {layer} ignored: equipment outlines must be closed."))
                continue
            h = float(getattr(e.dxf, "thickness", 0) or 0)
            equip.append([pts, h, layer, e.dxf.handle])
        elif _grade_of(L) is not None:
            pts, closed = _outline(e, 1e-3)
            if pts is not None and closed:
                zones.append((_grade_of(L), pts))
            else:
                rep.issues.append(("warning", f"Open or unsupported {t} on {layer} ignored."))
        elif L.startswith("LEAK"):
            if t == "POINT":
                leaks.append(tuple(e.dxf.location)[:2])
            elif t in ("CIRCLE", "ARC"):
                leaks.append(tuple(e.dxf.center)[:2])
            elif t == "INSERT":
                leaks.append(tuple(e.dxf.insert)[:2])
        elif L in ("FG_PANEL", "F&G_PANEL", "PANEL"):
            if t == "POINT":
                panel = tuple(e.dxf.location)[:2]
            elif t == "INSERT":
                panel = tuple(e.dxf.insert)[:2]
            elif t == "CIRCLE":
                panel = tuple(e.dxf.center)[:2]
        elif L == "PLOT_LIMIT":
            pts, closed = _outline(e, 1e-3)
            if pts is not None:
                limit = pts
        else:
            key = f"{layer} ({t})"
            rep.ignored_layers[key] = rep.ignored_layers.get(key, 0) + 1

    # ---- units ----
    if code in UNITS_TO_M:
        k = UNITS_TO_M[code]
        rep.units = UNIT_NAMES[code]
    else:
        allpts = np.vstack([p for p, *_ in equip] + [p for _, p in zones]) if (equip or zones) else np.zeros((1, 2))
        span = float(np.ptp(allpts, axis=0).max())
        k = 0.001 if span > 2000 else 1.0
        rep.units = UNIT_NAMES.get(code, "unknown")
        rep.issues.append(("warning", f"Drawing units not set ($INSUNITS={code}). Assumed "
                                      f"{'millimetres' if k == 0.001 else 'metres'} from the drawing extent "
                                      f"({span:,.0f} units). Confirm the scale."))
    rep.scale_to_m = k

    # ---- extents and transform (flip y) ----
    if limit is not None:
        box = np.vstack([limit.min(0), limit.max(0)]) * k
    else:
        allpts = [p for p, *_ in equip] + [p for _, p in zones] + ([np.array(leaks)] if leaks else [])
        if not allpts:
            raise ValueError("No EQUIP, HAZ_* or LEAK entities found. Check the layer names.")
        allpts = np.vstack(allpts) * k
        box = np.vstack([allpts.min(0) - 3.0, allpts.max(0) + 3.0])
        rep.issues.append(("info", "No PLOT_LIMIT layer: the analysis area is the drawing extent plus 3 m."))
    x0, y_top = box[0, 0], box[1, 1]
    width = float(np.ceil((box[1, 0] - box[0, 0]) / RES) * RES)
    depth = float(np.ceil((box[1, 1] - box[0, 1]) / RES) * RES)
    T = lambda pts: np.column_stack([pts[:, 0] * k - x0, y_top - pts[:, 1] * k])
    shape = (int(round(depth / RES)), int(round(width / RES)))

    def raster(pts):
        m = np.zeros((shape[0] * SS, shape[1] * SS), np.uint8)
        cv2.fillPoly(m, [np.round(T(pts) / RES * SS).astype(np.int32)], 1)
        return cv2.resize(m.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_AREA) >= 0.5

    heights = np.zeros(shape, np.float32)
    obstacles, no_h = [], []
    for pts, h, layer, handle in equip:
        if h <= 0:
            h = _layer_height(layer) or 0.0
        else:
            h *= k
        if h <= 0:
            h = DEFAULT_HEIGHT
            no_h.append(handle)
        m = raster(pts)
        if not m.any():
            rep.issues.append(("warning", f"Equipment {handle} on {layer} is smaller than one 0.5 m cell; dropped."))
            continue
        heights[m] = np.maximum(heights[m], h)
        q = T(pts)
        obstacles.append(Obstacle(f"{layer}:{handle}", "rect", (*q.min(0), *q.max(0)), h))
    if no_h:
        rep.issues.append(("warning", f"{len(no_h)} equipment outline(s) have no height (no thickness and no _H "
                                      f"layer suffix); {DEFAULT_HEIGHT:.0f} m assumed."))
    grades = np.zeros(shape, np.int8)
    for g, pts in zones:
        m = raster(pts)
        grades[m] = np.maximum(grades[m], g)
    leaks_m = T(np.array(leaks, float).reshape(-1, 2)) if leaks else np.zeros((0, 2))
    if panel is not None:
        panel_m = tuple(float(v) for v in T(np.array([panel], float))[0])
    else:
        panel_m = (width - 2.0, depth - 2.0)
        rep.issues.append(("info", "No FG_PANEL point: cable lengths are measured to the bottom-right corner."))
    if not zones:
        rep.issues.append(("error", "No hazard grade areas (HAZ_A/B/C) found, so there is nothing to cover."))
    if not leaks:
        rep.issues.append(("warning", "No LEAK points found: gas detection will not be designed."))
    rep.counts = {"equipment": len(obstacles), "hazard_areas": len(zones), "leak_sources": len(leaks_m),
                  "ignored_entities": int(sum(rep.ignored_layers.values()))}
    plant = Plant(width, depth, heights, grades, leaks_m, panel=panel_m, obstacles=obstacles,
                  name=name or "Plant (from CAD)")
    return plant, rep


def write_dxf(spec, path, units="mm", panel=(69.0, 39.0)):
    """Write a scene.Spec (e.g. the demo plant) as a layered DXF, the way a CAD export would look.
    Heights are written as entity thickness; y is flipped back to CAD convention."""
    if ezdxf is None:
        raise ImportError("pip install ezdxf to write CAD files")
    inv = {v: k for k, v in UNIT_NAMES.items()}
    k = 1.0 / UNITS_TO_M[inv[units]]
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = inv[units]
    doc.units = inv[units]
    msp = doc.modelspace()
    for name, col in (("EQUIP", 8), ("HAZ_A", 1), ("HAZ_B", 30), ("HAZ_C", 2), ("LEAK", 5), ("FG_PANEL", 3),
                      ("PLOT_LIMIT", 7), ("GRID", 9), ("ANNO", 7)):
        doc.layers.add(name, color=col)
    P = lambda x, y: (x * k, (spec.depth - y) * k)
    rect = lambda x0, y0, x1, y1: [P(x0, y0), P(x1, y0), P(x1, y1), P(x0, y1)]
    msp.add_lwpolyline(rect(0, 0, spec.width, spec.depth), close=True, dxfattribs={"layer": "PLOT_LIMIT"})
    for x in np.arange(0, spec.width + 0.1, 10):
        msp.add_line(P(x, 0), P(x, spec.depth), dxfattribs={"layer": "GRID"})
    for y in np.arange(0, spec.depth + 0.1, 10):
        msp.add_line(P(0, y), P(spec.width, y), dxfattribs={"layer": "GRID"})
    for ob in spec.obstacles:
        attrs = {"layer": "EQUIP", "thickness": ob.height * k}
        if ob.kind == "rect":
            msp.add_lwpolyline(rect(*ob.geom), close=True, dxfattribs=attrs)
            tx, ty = ob.geom[0], ob.geom[1] - 0.8
        else:
            cx, cy, r = ob.geom
            msp.add_circle(P(cx, cy), r * k, dxfattribs=attrs)
            tx, ty = cx - r, cy - r - 0.8
        msp.add_text(ob.name, height=0.6 * k, dxfattribs={"layer": "ANNO", "insert": P(tx, ty)})
    for g, *r in spec.zones:
        msp.add_lwpolyline(rect(*r), close=True, dxfattribs={"layer": {3: "HAZ_A", 2: "HAZ_B", 1: "HAZ_C"}[g]})
    for x, y in spec.leaks:
        msp.add_point(P(x, y), dxfattribs={"layer": "LEAK"})
    msp.add_point(P(*panel), dxfattribs={"layer": "FG_PANEL"})
    doc.saveas(path)
    return path
