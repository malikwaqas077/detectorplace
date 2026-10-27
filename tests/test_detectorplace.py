import numpy as np
import pytest

from detectorplace import baselines, explain, nsga2
from detectorplace.coverage import _cone_coverage, fire_targets, flame_matrix, gas_matrix, gas_targets
from detectorplace.detectors import make_candidates
from detectorplace.evaluate import Design, Problem, coverage, objectives
from detectorplace.plant import RES, Plant, demo_plant, render_plot_plan
from detectorplace.vision import extract_plant, score_extraction


def open_plant(w=30, d=20):
    shape = (int(d / RES), int(w / RES))
    return Plant(w, d, np.zeros(shape, np.float32), np.ones(shape, np.int8), np.array([[15.0, 10.0]]))


def see(plant, x, y, z, az, tx, ty, tz=0.5, rng=20.0, hfov=90, vfov=90, tilt=15, loss=0.0):
    return _cone_coverage(plant.heights, RES, np.array([x]), np.array([y]), np.array([z]),
                          np.array([np.deg2rad(az)]), np.array([np.deg2rad(tilt)]), np.array([tx]),
                          np.array([ty]), np.array([tz]), rng, np.deg2rad(hfov / 2), np.deg2rad(vfov / 2), loss)[0, 0]


# ---- line of sight -----------------------------------------------------------------------

def test_clear_line_of_sight():
    assert see(open_plant(), 2, 10, 3, 0, 15, 10) == 1


def test_tall_obstacle_blocks():
    p = open_plant()
    p.heights[16:24, 16:20] = 9.0          # 8-10 m in x, 8-12 m in y
    assert see(p, 2, 10, 3, 0, 15, 10) == 0


def test_high_mount_sees_over_low_obstacle():
    p = open_plant()
    p.heights[16:24, 16:20] = 2.0
    assert see(p, 2, 10, 3, 0, 15, 10) == 1 or see(p, 2, 10, 6, 0, 15, 10, tilt=30) == 1
    assert see(p, 2, 10, 6, 0, 15, 10, tilt=30) == 1


def test_out_of_range_and_behind():
    p = open_plant()
    assert see(p, 2, 10, 3, 0, 25, 10, rng=20) == 0
    assert see(p, 15, 10, 3, 0, 5, 10) == 0       # target behind the detector


def test_offaxis_range_loss():
    p = open_plant()
    # 18 m away at 40 deg off-axis: in range on axis, out of range once the cone edge loses 45 %
    x, y = 2 + 18 * np.cos(np.deg2rad(40)), 2 + 18 * np.sin(np.deg2rad(40))
    assert see(p, 2, 2, 3, 0, x, y, loss=0.0) == 1
    assert see(p, 2, 2, 3, 0, x, y, loss=0.45) == 0


def test_vertical_cone_misses_point_directly_below():
    assert see(open_plant(), 10, 10, 6, 0, 10.3, 10, tilt=0, vfov=40) == 0


# ---- computer vision ---------------------------------------------------------------------

def test_drawing_round_trip():
    truth = demo_plant()
    got, _ = extract_plant(render_plot_plan(truth))
    s = score_extraction(truth, got)
    assert s["equipment_iou"] > 0.97
    assert s["height_band_accuracy"] > 0.97
    assert s["zone_accuracy"] > 0.95
    assert s["leaks_found"] == s["leaks_true"] and s["leak_max_error_m"] < 0.5


def test_vision_ignores_noise_and_text():
    truth = demo_plant()
    got, _ = extract_plant(render_plot_plan(truth, noise=10, seed=3))
    assert score_extraction(truth, got)["equipment_iou"] > 0.95


# ---- optimisation ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def problem():
    plant, _ = extract_plant(render_plot_plan(demo_plant()))
    cand = make_candidates(plant)
    ft, gt = fire_targets(plant), gas_targets(plant)
    return Problem(flame_matrix(plant, cand, ft), gas_matrix(plant, cand, gt), cand.f_cost, cand.g_cost,
                   cand.n_opt, ft, gt), cand


def test_non_dominated_sort_and_crowding():
    F = np.array([[1, 5], [2, 3], [3, 1], [3, 4], [4, 4]], float)
    fronts = nsga2.non_dominated_sort(F)
    assert sorted(fronts[0]) == [0, 1, 2]
    assert sorted(fronts[1]) == [3]
    cd = nsga2.crowding(F[fronts[0]])
    assert np.isinf(cd).sum() == 2 and np.isfinite(cd).sum() == 1


def test_encode_decode_round_trip(problem):
    p, _ = problem
    d = Design(np.array([5, 3 * p.n_opt + 7]), np.array([0, 4]))
    d2 = nsga2.decode(p, *nsga2.encode(p, d))
    assert sorted(d2.flame) == sorted(d.flame) and sorted(d2.gas) == sorted(d.gas)


def test_greedy_is_feasible_and_minimal(problem):
    p, _ = problem
    d = baselines.greedy(p)
    assert objectives(p, d)[2]
    for k in d.flame:   # every detector is needed
        assert not objectives(p, Design(d.flame[d.flame != k], d.gas))[2]


def test_ga_meets_targets_and_beats_or_matches_greedy(problem):
    p, _ = problem
    front, hist = nsga2.run(p, pop_size=40, generations=12, seed=1, seeds=(baselines.greedy(p),))
    feas = [r for r in front if r["feasible"]]
    assert feas
    assert feas[0]["cost"] <= objectives(p, baselines.greedy(p))[0] + 1e-6
    costs = [r["cost"] for r in front]
    assert costs == sorted(costs)
    assert np.isfinite(hist[-1]["cheapest_feasible"])


def test_one_option_per_position(problem):
    p, _ = problem
    front, _ = nsga2.run(p, pop_size=30, generations=5, seed=2)
    for r in front:
        pos = r["design"].flame // p.n_opt
        assert len(pos) == len(set(pos.tolist()))


def test_explanations(problem):
    p, cand = problem
    d = baselines.greedy(p)
    rows = explain.schedule(p, cand, d)
    assert len(rows) == len(d.flame) + len(d.gas)
    assert all(r["why"] for r in rows)
    n1 = explain.n_minus_1(p, d, rows)
    cov = coverage(p, d)
    for r in n1:   # losing a detector can never increase coverage
        assert all(r[k] <= cov[k] * 100 + 1e-6 for k in cov)
