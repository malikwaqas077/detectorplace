"""Record a captioned walkthrough of the DetectorPlace dashboard (Playwright + Edge).

    streamlit run app.py --server.port 8620 --server.headless true
    python scripts/record_demo.py demo.mp4
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = "http://localhost:8620"
OUT = Path(sys.argv[1]).resolve()
TMP = OUT.parent / "_video_tmp"
W, H = 1440, 900
R = json.loads((ROOT / "docs" / "results.json").read_text())

OVERLAY_JS = """
([kind, title, sub]) => {
  let el = document.getElementById('dp-cap');
  if (!el) { el = document.createElement('div'); el.id = 'dp-cap'; document.body.appendChild(el); }
  if (kind === 'none') { el.style.display = 'none'; return; }
  if (kind === 'card') {
    el.style.cssText = 'display:flex;position:fixed;inset:0;z-index:2147483647;background:linear-gradient(135deg,#1c1917,#7c2d12);' +
      'color:#fff;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:60px;' +
      'font-family:Segoe UI,Arial,sans-serif;';
    el.innerHTML = '<div style="font-size:52px;font-weight:700;margin-bottom:22px">' + title + '</div>' +
                   '<div style="font-size:26px;line-height:1.5;max-width:1150px;opacity:.93">' + sub + '</div>';
  } else {
    el.style.cssText = 'display:block;position:fixed;left:50%;top:18px;transform:translateX(-50%);z-index:2147483647;width:1150px;' +
      'background:rgba(28,25,23,.92);color:#fff;padding:16px 26px;border-radius:12px;max-width:1150px;' +
      'font-family:Segoe UI,Arial,sans-serif;box-shadow:0 8px 30px rgba(0,0,0,.35);';
    el.innerHTML = '<div style="font-size:15px;letter-spacing:.08em;text-transform:uppercase;color:#fdba74;margin-bottom:6px">' +
                   title + '</div><div style="font-size:22px;line-height:1.45">' + sub + '</div>';
  }
}
"""


def main():
    ch, g, m, v = R["chosen"], R["greedy"], R["milp"], R["vision"]
    saving = (g["cost"] - ch["cost"]) / g["cost"] * 100
    TMP.mkdir(exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(TMP),
                            record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        t_start = time.monotonic()
        page.goto(URL)
        page.get_by_text("From a drawing to a 2.5D engineering model").wait_for(timeout=180_000)
        page.wait_for_timeout(3000)

        def cap(kind, title="", sub="", hold=0):
            page.evaluate(OVERLAY_JS, [kind, title, sub])
            if hold:
                page.wait_for_timeout(hold)

        def tab(name):
            page.get_by_role("tab", name=name).click()
            page.wait_for_timeout(2500)

        t_card = time.monotonic() - t_start
        cap("card", "DetectorPlace",
            "AI-assisted fire &amp; gas detector placement, coverage validation and cost estimation"
            "<br><br><span style='font-size:22px'>A working demonstrator by <b>Waqas Ahmad</b><br>"
            "for the Teesside University × Protex (UK) Systems KTP (T03269)"
            "<br><br>Synthetic gas compression station · illustrative detector data and rates</span>", 7000)

        cap("bar", "Step 1 · Read the drawing",
            "The input is a plot-plan export. Computer vision pulls out equipment footprints, elevation bands, "
            "hazard grades and leak sources, ignoring text, grid lines and noise.", 8500)
        cap("bar", "Checked against the source geometry",
            f"Equipment footprint IoU {v['equipment_iou'] * 100:.1f}%, every elevation band correct, "
            f"hazard grades {v['zone_accuracy'] * 100:.1f}% in agreement, {v['leaks_found']} of "
            f"{v['leaks_true']} leak sources found.", 7500)

        tab("2 · Line of sight")
        cap("bar", "Step 2 · What can one detector see?",
            "3D ray marching from the detector to every point at fire height, with range, horizontal and "
            "vertical cones and off-axis range loss. Tall vessels cast shadows.", 8000)
        s = page.get_by_role("slider").nth(2)
        s.focus(); s.press("ArrowRight")
        page.wait_for_timeout(3500)
        cap("bar", "Mount it higher",
            "At 6 m it sees over the 2 m pump skid in front of it and covers more ground, but costs more to "
            "mount and access. The optimiser weighs that trade-off for every position.", 8000)
        az = page.get_by_role("slider").nth(3)
        az.focus()
        for _ in range(2):
            az.press("ArrowRight"); page.wait_for_timeout(1500)
        page.wait_for_timeout(1500)

        tab("3 · Optimiser")
        cap("bar", "Step 3 · Cost against risk",
            f"{R['rays'] / 1e6:.1f} million line-of-sight checks feed a multi-objective genetic algorithm "
            "(NSGA-II with repair and redundancy pruning). Each point on the front is a complete layout.", 9000)
        cap("bar", "Validated against two baselines",
            f"Cheapest compliant layout £{ch['cost']:,.0f}: {saving:.1f}% below the greedy heuristic. "
            + (f"An exact MILP given {m['seconds']:.0f} s proves no layout can cost less than £{m['bound']:,.0f}."
               if m.get("bound") else ""), 9500)

        tab("4 · Design review")
        cap("bar", "Step 4 · The proposed layout",
            f"{ch['flame']} flame and {ch['gas']} gas detectors. Green is seen by two or more (voted executive "
            "action), amber by one, red not seen.", 8500)
        cap("bar", "Explainable and re-checked",
            "Every detector has a reason. Coverage is re-checked on the true geometry, with a design margin "
            "so extraction errors cannot push the real plant below target.", 8500)
        page.mouse.wheel(0, 700)
        page.wait_for_timeout(1200)
        cap("bar", "What if a detector fails?",
            "The N-1 table shows which targets slip if any one detector is out of service, and every "
            "remaining gap is listed with the reason it is uncovered.", 8500)
        page.mouse.wheel(0, -700)

        tab("5 · Report")
        cap("bar", "Step 5 · Report from a standard template",
            "Scope, method, targets against achieved coverage, detector schedule, map, N-1 check, gaps, "
            "Class 4 cost estimate and sign-off, generated as a PDF.", 8500)

        cap("card", "github.com/malikwaqas077/detectorplace",
            "Python · OpenCV · Numba · NSGA-II · SciPy/HiGHS MILP · Streamlit · ReportLab · tests · CI<br><br>"
            "<span style='font-size:22px'>Next for a real project: certified detector performance data, "
            "full 3D geometry, CFD gas clouds and an engineer-in-the-loop design workflow"
            "<br><br>Waqas Ahmad · malikwaqas077@gmail.com</span>", 8000)
        video = page.video
        ctx.close()
        src = Path(video.path())
        b.close()

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{max(t_card - 0.3, 0):.2f}", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "22", "-preset", "medium", "-movflags", "+faststart", str(OUT)], check=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
