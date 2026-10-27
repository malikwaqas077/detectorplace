"""Static figures (matplotlib) used by the generated design report."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy import ndimage  # noqa: E402

from .coverage import flame_wedge  # noqa: E402
from .evaluate import counts  # noqa: E402
from .plant import RES  # noqa: E402

COV_COLOURS = ["#d7263d", "#f4a259", "#2a9d8f"]   # unseen, 1 detector, 2+ detectors


def coverage_map(ctx, design, rows, path=None, dpi=160):
    p, plant = ctx["problem"], ctx["plant"]
    cf, cg = counts(p, design)
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ext = (0, plant.width, plant.depth, 0)
    ras = np.full(p.fire.shape, np.nan)
    lvl = np.clip(cf, 0, 2)
    ras[p.fire.cells[:, 0], p.fire.cells[:, 1]] = lvl
    ax.imshow(np.ones_like(plant.heights), extent=ext, cmap=ListedColormap(["white"]))
    ax.imshow(ras, extent=ext, cmap=ListedColormap(COV_COLOURS), vmin=0, vmax=2, alpha=0.55,
              interpolation="nearest")
    gas_r = np.full(p.gas.shape, np.nan)
    gas_r[p.gas.cells[:, 0], p.gas.cells[:, 1]] = (cg == 0).astype(float)
    gas_r[gas_r == 0] = np.nan
    ax.imshow(gas_r, extent=ext, cmap=ListedColormap(["#6c2bd9"]), alpha=0.45, interpolation="nearest")
    for g, col in ((1, "#c9a227"), (2, "#e07b00"), (3, "#c0392b")):
        ax.contour(np.arange(plant.shape[1]) * RES + RES / 2, np.arange(plant.shape[0]) * RES + RES / 2,
                   ndimage.binary_closing(plant.grades >= g, np.ones((7, 7))).astype(float), levels=[0.5], colors=col, linewidths=1.1, linestyles="--")
    h = np.where(plant.heights > 0, plant.heights, np.nan)
    ax.imshow(h, extent=ext, cmap="Greys", vmin=-4, vmax=10, interpolation="nearest")
    for r in rows:
        if r["type"].startswith("Flame"):
            wx, wy = flame_wedge(r["x"], r["y"], r["azimuth_deg"])
            ax.plot(wx, wy, color="#1d4ed8", lw=0.5, alpha=0.35)
            ax.plot(r["x"], r["y"], marker="^", ms=9, mfc="#1d4ed8", mec="white")
        else:
            ax.add_patch(plt.Circle((r["x"], r["y"]), 5.0, fill=False, ec="#6c2bd9", lw=0.6, alpha=0.5))
            ax.plot(r["x"], r["y"], marker="o", ms=7, mfc="#6c2bd9", mec="white")
        ax.annotate(r["tag"], (r["x"], r["y"]), xytext=(4, 4), textcoords="offset points", fontsize=7,
                    fontweight="bold")
    ax.plot(*plant.panel, marker="s", ms=8, c="black")
    ax.annotate("F&G panel", plant.panel, xytext=(5, -8), textcoords="offset points", fontsize=7)
    ax.set_xlim(0, plant.width); ax.set_ylim(plant.depth, 0)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], ls="", marker="s", c=c, label=l) for c, l in
                       zip(COV_COLOURS, ["fire area not seen", "seen by 1", "seen by 2+"])] +
              [Line2D([], [], ls="", marker="^", c="#1d4ed8", label="flame detector"),
               Line2D([], [], ls="", marker="o", c="#6c2bd9", label="gas detector"),
               Line2D([], [], ls="", marker="s", c="#b79be8", label="gas zone uncovered")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=7.5, ncol=6, frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(path or buf, dpi=dpi)
    plt.close(fig)
    return path or buf.getvalue()


def pareto_plot(opt, path=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    fr = opt["front"]
    ax.plot([r["cost"] / 1e3 for r in fr], [r["shortfall"] * 100 for r in fr], "o-", ms=3, label="NSGA-II front")
    c = opt["chosen"]
    ax.plot(c["cost"] / 1e3, c["shortfall"] * 100, "*", ms=14, c="#d97706", label="chosen (cheapest compliant)")
    g = opt["greedy"]
    ax.plot(g["cost"] / 1e3, g["shortfall"] * 100, "s", ms=7, c="#6b7280", label="greedy heuristic")
    ax.set_xlabel("installed cost (£k)"); ax.set_ylabel("risk-weighted shortfall (%)")
    ax.set_yscale("symlog", linthresh=1)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(path or buf, dpi=160)
    plt.close(fig)
    return path or buf.getvalue()
