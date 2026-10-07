from panopin import coverage

# Score matrix modeling the D31 loss-sink: hallway_3 is a loss-sink (low-ish for everyone);
# office_4's true pano p_o4b is a WEAK lock (higher at its own room than the sink) -> under
# per-pano argmin it is captured by the sink, but each room still has one STRONG pano.
SM = {
    "p_h3a":  {"hallway_3": 0.06, "office_4": 0.40, "office_5": 0.40},  # strong hallway_3
    "p_h3b":  {"hallway_3": 0.07, "office_4": 0.40, "office_5": 0.40},  # strong hallway_3
    "p_o4a":  {"hallway_3": 0.30, "office_4": 0.07, "office_5": 0.40},  # strong office_4
    "p_o4b":  {"hallway_3": 0.12, "office_4": 0.22, "office_5": 0.40},  # WEAK office_4 (sink-captured)
    "p_o5a":  {"hallway_3": 0.30, "office_4": 0.40, "office_5": 0.06},  # strong office_5
    "p_o5b":  {"hallway_3": 0.30, "office_4": 0.40, "office_5": 0.08},  # strong office_5
}
TRUE = {"p_h3a": "hallway_3", "p_h3b": "hallway_3", "p_o4a": "office_4",
        "p_o4b": "office_4", "p_o5a": "office_5", "p_o5b": "office_5"}


def test_room_anchored_seeds_cover_every_room_correctly():
    seeds = coverage.room_anchored_seeds(SM)
    assert set(seeds) == {"hallway_3", "office_4", "office_5"}
    for room, (pano, score) in seeds.items():
        assert TRUE[pano] == room                       # every room's best pano is a TRUE pano
        assert score == min(SM[p][room] for p in SM)    # returned score is the room's minimum


def test_room_anchored_immune_to_loss_sink():
    # per-pano argmin sends the weak p_o4b to the hallway_3 loss-sink...
    conf = coverage.pano_confidence(SM)
    assert conf["p_o4b"][0] == "hallway_3"              # captured under per-pano assignment
    # ...but room-anchored still seeds hallway_3 with a genuine hallway pano, not the impostor.
    assert coverage.room_anchored_seeds(SM)["hallway_3"][0] in ("p_h3a", "p_h3b")


def test_pano_confidence_ranks_weak_lock_last():
    conf = coverage.pano_confidence(SM)
    order = sorted(conf, key=lambda p: -conf[p][1])     # most-confident first
    # the weak-lock impostor p_o4b (winner score 0.12) must rank LAST.
    assert order[-1] == "p_o4b"
    # a confidence gate admitting the strong panos covers all 3 rooms with only correct picks.
    admitted = [p for p in order if conf[p][1] >= -0.10]   # winner_score <= 0.10
    assert all(conf[p][0] == TRUE[p] for p in admitted)
    assert {conf[p][0] for p in admitted} == {"hallway_3", "office_4", "office_5"}


def test_empty_matrix():
    assert coverage.room_anchored_seeds({}) == {}
    assert coverage.pano_confidence({}) == {}


# Two corridor panos whose scores nearly tie across the two corridors — the Area_2_manhattan7
# pattern measured on 2026-10-06. Per-pano argmin sends BOTH to seg_00; the joint assignment
# separates them because p_h2->seg_03 + p_h3->seg_00 (0.167) beats the swap (0.177).
CORRIDORS = {
    "p_h2": {"seg_00": 0.082, "seg_03": 0.084, "seg_02": 0.30},
    "p_h3": {"seg_00": 0.083, "seg_03": 0.095, "seg_02": 0.31},
    "p_o4": {"seg_00": 0.25,  "seg_03": 0.26,  "seg_02": 0.055},
}


def test_assign_rooms_separates_tied_corridor_panos():
    assert coverage.pano_confidence(CORRIDORS)["p_h2"][0] == "seg_00"   # argmin: both ...
    assert coverage.pano_confidence(CORRIDORS)["p_h3"][0] == "seg_00"   # ... in seg_00
    a = coverage.assign_rooms(CORRIDORS, ["seg_00", "seg_02", "seg_03"])
    assert a == {"p_h2": "seg_03", "p_h3": "seg_00", "p_o4": "seg_02"}


def test_assign_rooms_surplus_panos_are_left_out():
    sm = {"a": {"X": 0.05, "Y": 0.40}, "b": {"X": 0.42, "Y": 0.06}, "c": {"X": 0.30, "Y": 0.35}}
    assert coverage.assign_rooms(sm, ["X", "Y"]) == {"a": "X", "b": "Y"}


def test_assign_rooms_more_rooms_than_panos_leaves_rooms_unassigned():
    sm = {"a": {"X": 0.05, "Y": 0.40, "Z": 0.50}}
    assert coverage.assign_rooms(sm, ["X", "Y", "Z"]) == {"a": "X"}


def test_assign_rooms_empty():
    assert coverage.assign_rooms({}, ["X"]) == {}


def test_assign_rooms_never_uses_a_room_twice():
    a = coverage.assign_rooms(SM, ["hallway_3", "office_4", "office_5"])
    assert len(set(a.values())) == len(a) == 3
