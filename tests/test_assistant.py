"""Per-item reader confidence and the design assistant (offline engine)."""
import pickle
from pathlib import Path

import pytest

from detectorplace import confidence, segnet
from detectorplace.assistant import DesignAssistant

CTX = Path(__file__).resolve().parent.parent / "results" / "ctx.pkl"
pytestmark = pytest.mark.skipif(not CTX.exists(), reason="needs the cached demo run (results/ctx.pkl)")


@pytest.fixture(scope="module")
def ctx():
    with open(CTX, "rb") as f:
        return pickle.load(f)


@pytest.mark.skipif(not segnet.available(), reason="needs torch and models/segnet.pt")
def test_every_item_gets_a_confidence_and_low_ones_are_flagged(ctx):
    rows = confidence.items(ctx["plant"], ctx["img"], 10.0)
    assert len(rows) == len(ctx["plant"].obstacles) + len(ctx["plant"].leaks)
    assert all(0.0 <= r["confidence"] <= 1.0 for r in rows)
    assert any(r["confidence"] < confidence.REVIEW_BELOW for r in rows)      # something to review on the demo
    assert sum(r["kind"] == "leak" and r["confidence"] > 0.9 for r in rows) == len(ctx["plant"].leaks)


def test_assistant_answers_from_real_calculations(ctx):
    a = DesignAssistant(ctx)
    tag = a.rows[0]["tag"]
    fail = a.detector_fails(tag)
    assert len(fail["coverage_if_it_fails"]) == len(ctx["problem"].reqs)
    # Removing a detector can never increase coverage.
    for c in fail["coverage_if_it_fails"]:
        assert int(c["after"].rstrip("%")) <= int(c["before"].rstrip("%"))
    # A zero move reproduces the current coverage exactly.
    same = a.move_detector(tag, 0, 0)
    assert all(c["before"] == c["after"] for c in same["coverage_after_move"])
    assert not a.move_detector(tag, -1000, 0)["possible"]
    assert tag in a.answer_offline(f"Why is {tag} here?")
    assert "No detector called FD-99" in a.answer_offline("Why is FD-99 here?")
