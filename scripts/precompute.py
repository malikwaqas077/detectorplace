"""Run the full pipeline once and cache it for the dashboard (also writes docs figures and the PDF).

    python scripts/precompute.py [generations] [milp_seconds]
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from detectorplace import pipeline, report, viz  # noqa: E402

gens = int(sys.argv[1]) if len(sys.argv) > 1 else 120
milp_s = float(sys.argv[2]) if len(sys.argv) > 2 else 300
ctx = pipeline.build()
ctx["opt"] = pipeline.optimise(ctx, generations=gens, milp_seconds=milp_s)
ctx["explain"] = pipeline.explain_design(ctx, ctx["opt"]["chosen"]["design"])
(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results" / "ctx.pkl", "wb") as f:
    pickle.dump(ctx, f)
cv2.imwrite(str(ROOT / "docs" / "plot_plan.png"), ctx["img"][..., ::-1])
viz.coverage_map(ctx, ctx["opt"]["chosen"]["design"], ctx["explain"]["schedule"], ROOT / "docs" / "coverage_map.png")
viz.pareto_plot(ctx["opt"], ROOT / "docs" / "pareto.png")
report.build_pdf(ctx, str(ROOT / "docs" / "sample_report.pdf"))
o = ctx["opt"]
summary = {"vision": ctx["vision_score"], "timing": {**ctx["timing"], **o["timing"]}, "rays": ctx["rays"],
           "chosen": {"cost": o["chosen"]["cost"], "flame": len(o["chosen"]["design"].flame),
                      "gas": len(o["chosen"]["design"].gas), "coverage": o["chosen"]["coverage"]},
           "truth_coverage": ctx["explain"]["truth_coverage"],
           "greedy": {"cost": o["greedy"]["cost"], "n": len(o["greedy"]["design"].flame) + len(o["greedy"]["design"].gas)},
           "milp": {k: v for k, v in o.get("milp", {}).items() if k not in ("design", "coverage")},
           "n1_failures": sum(bool(r["targets_missed"]) for r in ctx["explain"]["n_minus_1"]),
           "gaps": ctx["explain"]["gaps"]}
(ROOT / "docs" / "results.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
