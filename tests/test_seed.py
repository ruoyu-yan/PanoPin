from panopin import seed

# Reuse the D31 loss-sink model: hallway_3 is a loss-sink; office_4's p_o4b is a weak lock that
# per-pano argmin sends to the sink, but each room still has a strong pano. Poses are opaque
# markers here (seeds_from_scores just plumbs them through, no GPU).
SM = {
    "p_h3a": {"hallway_3": 0.06, "office_4": 0.40, "office_5": 0.40},
    "p_h3b": {"hallway_3": 0.07, "office_4": 0.40, "office_5": 0.40},
    "p_o4a": {"hallway_3": 0.30, "office_4": 0.07, "office_5": 0.40},
    "p_o4b": {"hallway_3": 0.12, "office_4": 0.22, "office_5": 0.40},   # weak, sink-captured
    "p_o5a": {"hallway_3": 0.30, "office_4": 0.40, "office_5": 0.06},
    "p_o5b": {"hallway_3": 0.30, "office_4": 0.40, "office_5": 0.08},
}
TRUE = {"p_h3a": "hallway_3", "p_h3b": "hallway_3", "p_o4a": "office_4",
        "p_o4b": "office_4", "p_o5a": "office_5", "p_o5b": "office_5"}
# poses[pano][room] = (t, R); use a marker so we can assert the RIGHT pose is plumbed through.
POSES = {p: {r: ([f"t.{p}.{r}"], [[f"R.{p}.{r}"]]) for r in rooms} for p, rooms in SM.items()}


def test_seeds_from_scores_one_correct_seed_per_room():
    seeds = seed.seeds_from_scores(SM, POSES)
    assert set(seeds) == {"hallway_3", "office_4", "office_5"}
    for room, rs in seeds.items():
        assert rs.room == room
        assert TRUE[rs.pano] == room                       # each room seeded by a TRUE pano
        assert rs.t == [f"t.{rs.pano}.{room}"]             # the seed's OWN (pano,room) pose plumbed
        assert rs.R == [[f"R.{rs.pano}.{room}"]]
        assert rs.score == min(SM[p][room] for p in SM)


def test_seed_never_uses_the_sink_impostor():
    # p_o4b (weak) is captured by the sink under per-pano argmin, but must NOT seed any room.
    seeds = seed.seeds_from_scores(SM, POSES)
    assert "p_o4b" not in {rs.pano for rs in seeds.values()}


def test_empty():
    assert seed.seeds_from_scores({}, {}) == {}
