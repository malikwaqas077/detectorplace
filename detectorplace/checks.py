"""Input and result quality flags.

The coverage maths is exact for the model it is given. What can go wrong is the model: a
misread drawing, a missing layer, a wrong scale, placeholder detector data, or a design that
only just meets its targets. These checks make those risks visible to the engineer instead of
hiding them behind a confident number.
"""
from __future__ import annotations

import numpy as np

from .plant import RES

LOW_CONF = 0.6   # learned-reader cells below this confidence count as uncertain


def flag(severity, source, message):
    return {"severity": severity, "source": source, "message": message}


def data_flags():
    return [flag("warning", "Detector data", "Detector ranges, cones of vision and costs are placeholders. Replace "
                 "them with certified performance data (e.g. FM 3260) and company rates before relying on the design."),
            flag("info", "Gas model", "Gas detection uses a fixed cloud radius around each leak source, not "
                 "dispersion modelling or CFD.")]


def reader_flags(plant, debug, alt_plant=None, alt_name="the other reader"):
    """Flags for a model read from an image. `debug` comes from segnet.extract_plant (has confidence)
    or vision.extract_plant (no confidence). `alt_plant` is the same drawing read the other way."""
    out = []
    area = plant.width * plant.depth
    if area < 200 or area > 1_000_000:
        out.append(flag("warning", "Scale", f"The plant reads as {plant.width:.0f} m × {plant.depth:.0f} m. "
                                            "Check the drawing scale (pixels per metre)."))
    graded = plant.grades > 0
    if not graded.any():
        out.append(flag("error", "Drawing", "No hazard grade areas were found, so there is nothing to cover. "
                                            "Is this a plot plan in the expected convention?"))
    if len(plant.leaks) == 0:
        out.append(flag("warning", "Drawing", "No leak sources were found: gas detection will not be designed."))
    conf = (debug or {}).get("confidence")
    if conf is not None:
        unsure = (conf["equipment"] < LOW_CONF) | (conf["zone"] < LOW_CONF)
        frac = float(unsure.mean())
        sev = "error" if frac > 0.15 else "warning" if frac > 0.03 else "info"
        out.append(flag(sev, "Learned reader", f"{frac * 100:.1f}% of the drawing ({unsure.sum() * RES * RES:,.0f} m²) "
                        f"was read with low confidence." + (" The drawing may not follow the trained convention; "
                                                            "prefer the CAD file or check the scale." if sev == "error"
                                                            else " Low-confidence areas are shown on the confidence map.")))
    if alt_plant is not None and alt_plant.heights.shape == plant.heights.shape:
        a, b = plant.heights > 0, alt_plant.heights > 0
        iou = (a & b).sum() / max((a | b).sum(), 1)
        free = ~(a | b)
        zone_ag = float(np.mean(plant.grades[free] == alt_plant.grades[free])) if free.any() else 1.0
        sev = "warning" if iou < 0.9 or zone_ag < 0.95 else "info"
        out.append(flag(sev, "Cross-check", f"Agreement with {alt_name}: equipment IoU {iou * 100:.1f}%, hazard grades "
                                             f"{zone_ag * 100:.1f}%, leak sources {len(plant.leaks)} vs {len(alt_plant.leaks)}."
                        + (" Review the areas where the two readings differ." if sev == "warning" else "")))
    return out


def cad_flags(report):
    out = [flag(sev, "CAD", msg) for sev, msg in report.issues]
    if report.ignored_layers:
        layers = ", ".join(f"{k} ×{v}" for k, v in list(report.ignored_layers.items())[:6])
        out.append(flag("info", "CAD", f"Ignored layers (not part of the convention): {layers}."))
    out.append(flag("info", "CAD", f"Units: {report.units}. Read {report.counts.get('equipment', 0)} equipment items, "
                                   f"{report.counts.get('hazard_areas', 0)} hazard areas, "
                                   f"{report.counts.get('leak_sources', 0)} leak sources."))
    return out


def design_flags(problem, cov, check_cov=None, check_name="the cross-check model", margin=0.02):
    out = []
    for q in problem.reqs:
        got = cov[q.key]
        if got < q.target:
            out.append(flag("error", "Design", f"{q.label}: {got * 100:.1f}% is below the {q.target * 100:.0f}% target."))
        elif got < q.target + margin:
            out.append(flag("warning", "Design", f"{q.label}: {got * 100:.1f}% is within {margin * 100:.0f} points of the "
                                                 f"{q.target * 100:.0f}% target, so a small error in the plant model "
                                                 "could take it below."))
        if check_cov is not None and check_cov[q.key] < q.target:
            out.append(flag("warning", "Cross-check", f"{q.label} drops to {check_cov[q.key] * 100:.1f}% when the layout "
                                                       f"is re-checked on {check_name}."))
    return out


def summary(flags):
    n = {s: sum(f["severity"] == s for f in flags) for s in ("error", "warning", "info")}
    return n
