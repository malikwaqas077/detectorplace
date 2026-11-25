"""Per-item confidence for a drawing read from an image.

The learned reader (segnet.py) gives a probability for every cell. This module turns that into
one score per item the engineer cares about: each piece of equipment ("94% sure this is a 9 m
vessel") and each gas leak point. Items below the review threshold are flagged so an engineer
checks them by eye before the design is used.

If the plant was read by the OpenCV rules, the learned model is run on the same drawing as a
second opinion, so every image input gets a score. CAD (DXF) input is read exactly from layers
and has no recognition step, so it gets no score.
"""
from __future__ import annotations

import numpy as np

from .plant import RES, obstacle_mask
from .scene import BAND_HEIGHTS

REVIEW_BELOW = 0.85   # default: items under 85% confidence go to the engineer for a manual check
TYPE_BY_HEIGHT = {9.0: "Vessel / tank", 5.0: "Equipment package", 2.0: "Low skid"}


def _learned_maps(img, px_per_m, nn_debug=None):
    from . import segnet
    if nn_debug is None:
        if img is None or not segnet.available():
            return None
        _, nn_debug = segnet.extract_plant(img, px_per_m)
    pr_leak = nn_debug["confidence"]["leak_heat"]
    return nn_debug["eq_prob"], pr_leak


def items(plant, img, px_per_m=10.0, nn_debug=None):
    """One row per equipment item and leak point: tag, type, position, confidence (0-1), bbox (m).
    Returns [] when no learned model is available or the input has no image."""
    maps = _learned_maps(img, px_per_m, nn_debug)
    if maps is None:
        return []
    eq_prob, leak_heat = maps
    from .segnet import IN_PX_PER_M
    out = []
    for i, ob in enumerate(plant.obstacles):
        m = obstacle_mask(plant.shape, ob)
        if not m.any():
            continue
        band = min(BAND_HEIGHTS, key=lambda b: abs(b - ob.height))
        cls = BAND_HEIGHTS.index(band) + 1
        # Probability the model gives to "equipment of this height" over the item's footprint.
        conf = float(eq_prob[cls][m[:eq_prob.shape[1], :eq_prob.shape[2]]].mean())
        if ob.kind == "rect":
            x0, y0, x1, y1 = ob.geom
        else:
            cx, cy, r = ob.geom
            x0, y0, x1, y1 = cx - r, cy - r, cx + r, cy + r
        out.append({"tag": f"EQ-{i + 1:02d}", "type": f"{TYPE_BY_HEIGHT.get(band, 'Equipment')}, {band:.0f} m",
                    "x": (x0 + x1) / 2, "y": (y0 + y1) / 2, "bbox": (x0, y0, x1, y1), "confidence": conf,
                    "kind": "equipment"})
    H, W = leak_heat.shape
    rad = int(round(1.0 * IN_PX_PER_M))    # look within 1 m of the leak point
    for j, (x, y) in enumerate(plant.leaks):
        px, py = int(x * IN_PX_PER_M), int(y * IN_PX_PER_M)
        win = leak_heat[max(py - rad, 0):min(py + rad + 1, H), max(px - rad, 0):min(px + rad + 1, W)]
        conf = float(win.max()) if win.size else 0.0
        out.append({"tag": f"L{j + 1}", "type": "Gas leak point", "x": float(x), "y": float(y),
                    "bbox": (x - 1, y - 1, x + 1, y + 1), "confidence": conf, "kind": "leak"})
    return out


def overlay(img, rows, px_per_m=10.0, threshold=REVIEW_BELOW):
    """Draw the slide-style view: a dashed box and % score on each item, leak points circled.
    Green = confident, red = needs an engineer's check."""
    import cv2
    out = img.copy()
    s = max(1, int(round(img.shape[1] / 800)))
    for r in rows:
        ok = r["confidence"] >= threshold
        col = (22, 163, 74) if ok else (220, 38, 38)
        x0, y0, x1, y1 = (int(round(v * px_per_m)) for v in r["bbox"])
        if r["kind"] == "leak":
            cv2.circle(out, (int(r["x"] * px_per_m), int(r["y"] * px_per_m)), int(1.2 * px_per_m), col, 2 * s,
                       cv2.LINE_AA)
            tx, ty = x1 + 2, y0 + 4
        else:
            x0, y0, x1, y1 = x0 - 3, y0 - 3, x1 + 3, y1 + 3
            for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
                n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) // 8), 1)
                for k in range(0, n, 2):
                    p = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
                    q = (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n))
                    cv2.line(out, p, q, col, 2 * s, cv2.LINE_AA)
            tx, ty = x0 + 2, y1 + 13 * s
        label = f"{r['tag']} {r['confidence'] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s, 1)
        cv2.rectangle(out, (tx - 1, ty - th - 2), (tx + tw + 1, ty + 2), (255, 255, 255), -1)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s, col, 1, cv2.LINE_AA)
    return out


def detectors_relying_on(problem, design, schedule, row, plant):
    """Detectors whose coverage would be affected if this item was misread: for a leak point, the gas
    detectors covering it; for equipment, the flame detectors that see the hazard area next to it."""
    tags = []
    if row["kind"] == "leak":
        gxy = problem.gas.xy
        near = np.hypot(gxy[:, 0] - row["x"], gxy[:, 1] - row["y"]) <= 1.5
        if near.any():
            for r in schedule:
                if not r["type"].startswith("Flame") and problem.Mg[r["cand"]][near].any():
                    tags.append(r["tag"])
    else:
        x0, y0, x1, y1 = row["bbox"]
        fxy = problem.fire.xy
        ring = ((fxy[:, 0] >= x0 - 3) & (fxy[:, 0] <= x1 + 3) & (fxy[:, 1] >= y0 - 3) & (fxy[:, 1] <= y1 + 3))
        if ring.any():
            for r in schedule:
                if r["type"].startswith("Flame") and problem.Mf[r["cand"]][ring].any():
                    tags.append(r["tag"])
    return tags
