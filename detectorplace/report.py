"""Fill the standard report template from the analysis results and render it to PDF."""
from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import viz

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "fg_design_report.json"


def context_fields(ctx, project_code="DEMO"):
    p, ex, opt = ctx["problem"], ctx["explain"], ctx["opt"]
    sched = ex["schedule"]
    n1_fail = [r for r in ex["n_minus_1"] if r["targets_missed"]]
    if n1_fail:
        n1 = (f"Losing any one of {len(n1_fail)} detectors ({', '.join(r['tag'] for r in n1_fail[:8])}"
              f"{'...' if len(n1_fail) > 8 else ''}) drops at least one requirement below target. This is normal "
              "for a cost-minimal layout; the project should decide whether N-1 compliance is required for "
              "Grade A areas, which would add detectors.")
    else:
        n1 = "Every target is still met with any single detector out of service."
    gaps = ex["gaps"]
    gap_txt = (f"{len(gaps)} uncovered regions of 4 m² or more remain in Grade A/B areas "
               f"(largest {gaps[0]['area_m2']} m²). Each is listed with the reason it is uncovered."
               if gaps else "No uncovered Grade A/B regions of 4 m² or more remain.")
    tc = ex["truth_coverage"]
    truth = ", ".join(f"{q.key} {tc[q.key] * 100:.1f}%" for q in p.reqs)
    return {"project_code": project_code, "plant_name": ctx["truth"].name.lower(),
            "n_fire_points": len(p.fire.xy), "n_gas_points": len(p.gas.xy), "n_leaks": len(ctx["plant"].leaks),
            "vision_iou": f"{ctx['vision_score']['equipment_iou'] * 100:.1f}",
            "n_candidates": f"{p.Mf.shape[0]:,}", "n_gas_candidates": p.Mg.shape[0], "rays": f"{ctx['rays']:,}",
            "n_flame": sum(r["type"].startswith("Flame") for r in sched),
            "n_gas": sum(not r["type"].startswith("Flame") for r in sched),
            "n1_summary": n1, "gap_summary": gap_txt, "truth_check": truth,
            "cost_total": opt["chosen"]["cost"] + ctx["catalogue"]["cost"]["fixed"]}


def tables(ctx):
    p, ex = ctx["problem"], ctx["explain"]
    cov = ex["coverage"]
    t = {"coverage": [["Requirement", "Target", "Achieved", "Status"]] +
         [[q.label, f"{q.target * 100:.0f}%", f"{cov[q.key] * 100:.1f}%",
           "Met" if cov[q.key] >= q.target else "NOT MET"] for q in p.reqs]}
    t["schedule"] = [["Tag", "Type", "x (m)", "y (m)", "Mount (m)", "Azimuth", "Unique m²", "Cost (£)"]] + \
        [[r["tag"], r["type"], f"{r['x']:.1f}", f"{r['y']:.1f}", f"{r['mount_m']:.1f}",
          "-" if r["azimuth_deg"] is None else f"{r['azimuth_deg']:.0f}°", r["unique_points"],
          f"{r['cost']:,.0f}"] for r in ex["schedule"]]
    keys = [q.key for q in p.reqs]
    t["n_minus_1"] = [["Out of service"] + keys + ["Below target"]] + \
        [[r["tag"]] + [f"{r[k]:.1f}" for k in keys] + [", ".join(r["targets_missed"]) or "-"]
         for r in ex["n_minus_1"]]
    t["gaps"] = [["Location (x, y)", "Grade", "Area m²", "Reason"]] + \
        [[f"({g['x']}, {g['y']})", g["grade"], g["area_m2"], g["reason"]] for g in ex["gaps"][:12]] \
        if ex["gaps"] else [["-", "-", "-", "No residual gaps"]]
    sched = ex["schedule"]
    cat = ctx["catalogue"]
    nf = sum(r["type"].startswith("Flame") for r in sched)
    ng = len(sched) - nf
    hi = sum(r["type"].startswith("Flame") and r["mount_m"] > 3 for r in sched)
    fm, gm, cm = cat["flame"], cat["gas"], cat["cost"]
    devices = sum(r["cost"] for r in sched)
    det = nf * (fm["supply"] + fm["install"]) + hi * fm["high_mount_extra"] + ng * (gm["supply"] + gm["install"])
    io_ = len(sched) * cm["io_per_device"]
    t["cost"] = [["Item", "Qty", "Amount (£)"],
                 ["Flame detectors, supplied and installed", nf, f"{nf * (fm['supply'] + fm['install']):,.0f}"],
                 ["Upper-level mounting brackets / access", hi, f"{hi * fm['high_mount_extra']:,.0f}"],
                 ["Point gas detectors, supplied and installed", ng, f"{ng * (gm['supply'] + gm['install']):,.0f}"],
                 ["Field cabling to F&G panel (routed)", "-", f"{devices - det - io_:,.0f}"],
                 ["Panel I/O, termination, commissioning", len(sched), f"{io_:,.0f}"],
                 ["Panel, engineering, cause & effect, FAT (fixed)", 1, f"{cm['fixed']:,.0f}"],
                 ["Total", "", f"{devices + cm['fixed']:,.0f}"]]
    return t


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_pdf(ctx, path=None, project_code="DEMO"):
    tpl = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    f = context_fields(ctx, project_code)
    tb = tables(ctx)
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9, leading=12, spaceAfter=5)
    h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4,
                       textColor=colors.HexColor("#0f3d5e"))
    cell = ParagraphStyle("c", parent=body, fontSize=7.4, leading=9, spaceAfter=0)
    buf = path or io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title=tpl["document_title"], author="DetectorPlace")
    st = [Paragraph(_esc(tpl["document_title"]), ss["Title"]),
          Paragraph(f"Document {tpl['document_number'].format(**f)} · Revision {tpl['revision']} · "
                    f"{date.today():%d %b %Y} · Synthetic demonstration data", body), Spacer(1, 4)]
    for sec in tpl["sections"]:
        st.append(Paragraph(_esc(sec["title"]), h))
        if sec.get("text"):
            st.append(Paragraph(_esc(sec["text"].format(**f)), body))
        if sec.get("table"):
            data = [[Paragraph(_esc(c), cell) for c in row] for row in tb[sec["table"]]]
            t = Table(data, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eef5")),
                                   ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c4ce")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            st.append(t)
        if sec.get("figure") == "coverage_map":
            png = viz.coverage_map(ctx, ctx["opt"]["chosen"]["design"], ctx["explain"]["schedule"])
            st.append(Image(io.BytesIO(png), width=178 * mm, height=117 * mm))
    st.append(Paragraph("Sign-off", h))
    st.append(Table([[s, "Name:", "Date:"] for s in tpl["sign_off"]], colWidths=[75 * mm, 50 * mm, 40 * mm],
                    style=[("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    doc.build(st)
    return path or buf.getvalue()
