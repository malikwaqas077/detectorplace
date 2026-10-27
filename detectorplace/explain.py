"""Explainability for a proposed layout: why each detector is there, what happens if one
fails, and why any remaining gaps are left uncovered."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .detectors import Candidates
from .evaluate import Design, Problem, coverage, counts


def schedule(p: Problem, cand: Candidates, d: Design) -> list[dict]:
    cf, cg = counts(p, d)
    a_mask = p.fire.grade == 3
    rows = []
    for n, k in enumerate(sorted(d.flame, key=lambda k: (cand.f_y[k], cand.f_x[k])), 1):
        seen = p.Mf[k].astype(bool)
        z, az = cand.flame_option(k)
        rows.append({"tag": f"FD-{n:02d}", "type": "Flame (IR3)", "x": float(cand.f_x[k]), "y": float(cand.f_y[k]),
                     "mount_m": z, "azimuth_deg": az, "tilt_deg": float(np.rad2deg(cand.f_tilt[k])),
                     "points_seen": int(seen.sum()), "unique_points": int((seen & (cf == 1)).sum()),
                     "grade_a_points": int((seen & a_mask).sum()), "cost": float(p.f_cost[k]), "cand": int(k)})
    for n, k in enumerate(sorted(d.gas, key=lambda k: (cand.g_xy[k, 1], cand.g_xy[k, 0])), 1):
        seen = p.Mg[k].astype(bool)
        rows.append({"tag": f"GD-{n:02d}", "type": "Point gas (IR)", "x": float(cand.g_xy[k, 0]),
                     "y": float(cand.g_xy[k, 1]), "mount_m": 0.5, "azimuth_deg": None, "tilt_deg": None,
                     "points_seen": int(seen.sum()), "unique_points": int((seen & (cg == 1)).sum()),
                     "grade_a_points": None, "cost": float(p.g_cost[k]), "cand": int(k)})
    for r in rows:
        r["why"] = _why(r)
    return rows


def _why(r):
    if r["unique_points"] == 0:
        if r.get("grade_a_points"):
            return (f"Second detector on {r['grade_a_points']} m² of Grade A, giving the 2ooN voting needed for "
                    "executive action. No area depends on it alone.")
        return "Adds redundancy only. Every point it sees is also seen by another detector."
    area = "fire" if r["type"].startswith("Flame") else "gas release"
    s = f"Sole coverage of {r['unique_points']} m² of graded {area} area"
    if r.get("grade_a_points"):
        s += f". Also views {r['grade_a_points']} m² of Grade A, supporting 2ooN voting"
    return s + "."


def n_minus_1(p: Problem, d: Design, rows: list[dict]) -> list[dict]:
    """Coverage with each detector out of service (fault or maintenance)."""
    out = []
    for r in rows:
        if r["type"].startswith("Flame"):
            dd = Design(d.flame[d.flame != r["cand"]], d.gas)
        else:
            dd = Design(d.flame, d.gas[d.gas != r["cand"]])
        cov = coverage(p, dd)
        below = [q.key for q in p.reqs if cov[q.key] < q.target]
        out.append({"tag": r["tag"], **{k: round(v * 100, 1) for k, v in cov.items()}, "targets_missed": below})
    return out


def gaps(p: Problem, d: Design, min_area=4) -> list[dict]:
    """Connected uncovered regions in Grade A/B areas, with the reason they are uncovered."""
    cf, _ = counts(p, d)
    grid = np.zeros(p.fire.shape, int)
    idx = -np.ones(p.fire.shape, int)
    for i, (r, c) in enumerate(p.fire.cells):
        idx[r, c] = i
        if p.fire.grade[i] >= 2 and cf[i] == 0:
            grid[r, c] = 1
    lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
    seen_by_any = p.Mf.any(0)
    out = []
    for k in range(1, n + 1):
        pts = idx[lab == k]
        if len(pts) < min_area:
            continue
        xy = p.fire.xy[pts].mean(0)
        visible = seen_by_any[pts].mean()
        reason = ("Shadowed: no candidate mounting position can see this area past the equipment. "
                  "Consider a detector mounted on the structure itself or a different technology."
                  if visible < 0.5 else
                  "Visible from available positions but left uncovered because the grade target is already "
                  "met; covering it would cost an extra detector.")
        out.append({"x": round(float(xy[0]), 1), "y": round(float(xy[1]), 1), "area_m2": int(len(pts)),
                    "grade": "A" if p.fire.grade[pts].max() == 3 else "B", "reason": reason})
    return sorted(out, key=lambda g: (-g["area_m2"]))
