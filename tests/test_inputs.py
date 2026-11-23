"""v0.2 inputs: CAD (DXF) reader, randomised drawings, learned reader and quality flags."""
import io

import numpy as np
import pytest

from detectorplace import cad, checks, pipeline, scene, segnet
from detectorplace.evaluate import DEFAULT_REQS
from detectorplace.plant import demo_plant

ezdxf = pytest.importorskip("ezdxf")


def test_dxf_round_trip_matches_demo_plant(tmp_path):
    path = tmp_path / "demo.dxf"
    cad.write_dxf(scene.demo_spec(), str(path), units="mm")
    p, rep = cad.read_dxf(str(path))
    t = demo_plant()
    assert rep.units == "mm" and rep.scale_to_m == pytest.approx(0.001)
    assert p.heights.shape == t.heights.shape
    assert (p.heights == t.heights).mean() > 0.998          # circles rasterise a few cells differently
    assert (p.grades == t.grades).mean() == 1.0
    assert np.allclose(np.sort(p.leaks, 0), np.sort(t.leaks, 0))
    assert p.panel == pytest.approx((69.0, 39.0))
    assert "GRID (LINE)" in rep.ignored_layers and "ANNO (TEXT)" in rep.ignored_layers


def test_dxf_layer_height_units_and_missing_data_are_flagged():
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (4, 0), (4, 3), (0, 3)], close=True, dxfattribs={"layer": "EQUIP_H9"})
    msp.add_lwpolyline([(10, 0), (12, 0), (12, 2), (10, 2)], close=True, dxfattribs={"layer": "EQUIP"})
    msp.add_lwpolyline([(20, 0), (22, 0), (22, 2)], dxfattribs={"layer": "EQUIP"})            # open
    msp.add_lwpolyline([(-5, -5), (30, -5), (30, 10), (-5, 10)], close=True, dxfattribs={"layer": "HAZ_B"})
    buf = io.StringIO()
    doc.write(buf)
    buf.seek(0)
    p, rep = cad.read_dxf(buf)
    assert sorted(o.height for o in p.obstacles) == [cad.DEFAULT_HEIGHT, 9.0]
    msgs = " ".join(m for _, m in rep.issues)
    assert "no height" in msgs and "Open LWPOLYLINE" in msgs and "No LEAK points" in msgs
    assert (p.grades == 2).sum() > 0


def test_random_scene_labels_match_geometry():
    rng = np.random.default_rng(5)
    spec = scene.random_spec(rng)
    style = scene.random_style(rng)
    img, lab = scene.render(spec, style, rng)
    assert img.shape[:2] == lab["equip"].shape == lab["zone"].shape
    plant = scene.to_plant(spec)
    assert set(np.unique(lab["equip"])) <= {0, 1, 2, 3}
    # every leak is on open ground
    for x, y in plant.leaks:
        assert plant.heights[int(y / 0.5), int(x / 0.5)] == 0


def test_design_flags_margin_and_failure():
    class P:
        reqs = DEFAULT_REQS
    cov = {q.key: q.target + 0.05 for q in DEFAULT_REQS}
    assert checks.design_flags(P, cov) == []
    cov[DEFAULT_REQS[0].key] = DEFAULT_REQS[0].target + 0.005
    cov[DEFAULT_REQS[1].key] = DEFAULT_REQS[1].target - 0.01
    sev = sorted(f["severity"] for f in checks.design_flags(P, cov))
    assert sev == ["error", "warning"]


def test_build_from_dxf_runs_end_to_end(tmp_path):
    path = tmp_path / "demo.dxf"
    cad.write_dxf(scene.demo_spec(), str(path))
    ctx = pipeline.build_from(str(path), kind="dxf")
    assert ctx["problem"].Mf.shape[0] > 1000 and ctx["source"].startswith("CAD")
    assert any(f["source"] == "CAD" for f in ctx["input_flags"])


@pytest.mark.skipif(not segnet.available(), reason="learned reader weights not present")
def test_learned_reader_reads_unseen_style_of_demo_plant():
    img = scene.render(scene.demo_spec(), scene.hard_style())[0]
    p, dbg = segnet.extract_plant(img, 12.0)
    t = demo_plant()
    a, b = t.heights > 0, p.heights > 0
    assert (a & b).sum() / (a | b).sum() > 0.85
    assert dbg["confidence"]["equipment"].shape == p.heights.shape
    d = np.linalg.norm(p.leaks[:, None] - t.leaks[None], axis=2)
    assert np.mean(d.min(0) <= 1.0) >= 0.75
