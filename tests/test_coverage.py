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
