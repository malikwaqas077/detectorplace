"""Compare the OpenCV rules and the learned reader on drawings neither was tuned or trained on.

    python scripts/eval_readers.py [n_heldout]

Writes docs/reader_eval.json and docs/readers.png.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detectorplace import checks, scene, segnet, vision  # noqa: E402
from detectorplace.plant import demo_plant, render_plot_plan  # noqa: E402
from detectorplace.vision import score_extraction  # noqa: E402

PID = Path("/home/claude/mgupta70/pid_symbol_detection/data/raw/images")


def score(truth, got):
    s = score_extraction(truth, got)
    d = np.linalg.norm(got.leaks[:, None] - truth.leaks[None], axis=2) if len(got.leaks) and len(truth.leaks) else None
    s["leak_recall"] = float(np.mean(d.min(0) <= 1.0)) if d is not None else 0.0
    s["false_leaks"] = int(np.sum(d.min(1) > 1.0)) if d is not None else len(got.leaks)
    return s


def both(img, ppm, truth):
    a, _ = vision.extract_plant(img, ppm)
    b, dbg = segnet.extract_plant(img, ppm)
    return score(truth, a), score(truth, b), a, b, dbg


def mean(rows):
    keys = ["equipment_iou", "height_band_accuracy", "zone_accuracy", "leak_recall"]
    out = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    out["false_leaks_per_drawing"] = round(float(np.mean([r["false_leaks"] for r in rows])), 2)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    res = {}
    cv_rows, nn_rows = [], []
    for i in range(n):
        rng = np.random.default_rng(900_000 + i)          # disjoint from the training seeds
        spec, style = scene.random_spec(rng), scene.random_style(rng)
        img, _ = scene.render(spec, style, rng)
        a, b, *_ = both(img, style.px_per_m, scene.to_plant(spec))
        cv_rows.append(a); nn_rows.append(b)
    res["heldout_random_styles"] = {"n": n, "opencv": mean(cv_rows), "learned": mean(nn_rows)}

    truth = demo_plant()
    for key, img, ppm in (("demo_original_style", render_plot_plan(truth), 10.0),
                          ("demo_hard_style", scene.render(scene.demo_spec(), scene.hard_style())[0], 12.0)):
        a, b, pa, pb, dbg = both(img, ppm, truth)
        res[key] = {"opencv": {k: v for k, v in a.items() if k != "leak_max_error_m"},
                    "learned": {k: v for k, v in b.items() if k != "leak_max_error_m"}}
        if key == "demo_hard_style":
            hard = (img, pa, pb, dbg)

    ood = []
    for f in sorted(PID.glob("*.jpg"))[:3]:
        im = cv2.imread(str(f))[..., ::-1]
        im = cv2.resize(im, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
        p, dbg = segnet.extract_plant(im, 10.0)
        fl = checks.reader_flags(p, dbg)
        unsure = float(((dbg["confidence"]["equipment"] < checks.LOW_CONF) |
                        (dbg["confidence"]["zone"] < checks.LOW_CONF)).mean())
        ood.append({"sheet": f.name, "low_confidence_fraction": round(unsure, 4),
                    "graded_cells": int((p.grades > 0).sum()), "flags": [x["severity"] + ": " + x["message"] for x in fl]})
    res["out_of_distribution_pid_sheets"] = ood
    (ROOT / "docs" / "reader_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    img, pa, pb, dbg = hard
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.6))
    ax[0, 0].imshow(img); ax[0, 0].set_title("Input: demo plant drawn in a different style")
    for a_, p_, t_ in ((ax[0, 1], pa, "OpenCV rules (tuned for the original style)"),
                       (ax[1, 0], pb, "Learned reader (U-Net, never saw this plant)")):
        rgb = np.full(p_.heights.shape + (3,), 255, np.uint8)
        for g, c in ((1, (255, 245, 170)), (2, (255, 215, 170)), (3, (255, 190, 190))):
            rgb[p_.grades == g] = c
        for h, c in ((2.0, 170), (5.0, 115), (9.0, 60)):
            rgb[p_.heights == h] = c
        a_.imshow(rgb, extent=(0, p_.width, p_.depth, 0))
        if len(p_.leaks):
            a_.plot(p_.leaks[:, 0], p_.leaks[:, 1], "D", color="#1d4ed8", ms=5)
        a_.set_title(t_)
    c = np.minimum(dbg["confidence"]["equipment"], dbg["confidence"]["zone"])
    im_ = ax[1, 1].imshow(c, cmap="RdYlGn", vmin=0.3, vmax=1, extent=(0, pb.width, pb.depth, 0))
    ax[1, 1].set_title("Learned reader confidence (red = check this area)")
    fig.colorbar(im_, ax=ax[1, 1], fraction=0.03)
    for a_ in ax.flat:
        a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "readers.png", dpi=110)
