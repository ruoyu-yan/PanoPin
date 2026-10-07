"""Recorded seed-stage score matrices of a real scene: the joint assignment must place every
pano in the segment its station stands in. Fixtures come from eval/record_score_matrix.py
(the truth label uses GT only to name the segment; the scores read panos + clouds)."""
import json
from pathlib import Path

import pytest
from panopin import coverage

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "score_matrix").glob("*.json"))
FAILED_FIXTURES = sorted((Path(__file__).parent / "fixtures" / "score_matrix_failed").glob("*.json"))


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_joint_assignment_places_every_pano_in_its_own_segment(path):
    d = json.loads(path.read_text())
    assigned = coverage.assign_rooms(d["score_matrix"], d["room_order"])
    assert set(assigned) == set(d["score_matrix"])
    wrong = {p: (r, d["truth_segment"][p]) for p, r in assigned.items() if r != d["truth_segment"][p]}
    assert not wrong, f"{path.name}: pano -> (assigned, truth) {wrong}"


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_fixture_schema(path):
    d = json.loads(path.read_text())
    assert set(d) >= {"scene", "recorded", "room_order", "score_matrix", "truth_segment"}
    for pano, row in d["score_matrix"].items():
        assert set(row) == set(d["room_order"]), pano
        assert d["truth_segment"][pano] in d["room_order"]


def test_at_least_three_draws_are_recorded():
    assert len(FIXTURES) >= 3, "run eval/record_score_matrix.py three times (Task 3 of the plan)"


@pytest.mark.parametrize("path", FAILED_FIXTURES, ids=[p.stem for p in FAILED_FIXTURES])
def test_known_failing_draw_is_still_failing(path):
    """A recorded draw on which the joint assignment misplaces panos, kept as evidence. On it
    both corridor panos score better in each other's corridor, so no rule on the score matrix
    alone can separate them (spec section 12). When a later change makes this test fail, the
    fixture is placed correctly now and should move back to fixtures/score_matrix/."""
    d = json.loads(path.read_text())
    assigned = coverage.assign_rooms(d["score_matrix"], d["room_order"])
    wrong = {p: (r, d["truth_segment"][p]) for p, r in assigned.items() if r != d["truth_segment"][p]}
    assert wrong, (f"{path.name}: now places every pano correctly {assigned}; "
                   "move it to fixtures/score_matrix/")
