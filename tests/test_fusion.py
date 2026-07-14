from panopin import fusion


def test_verify_select_picks_min_color():
    cands = [{"geom": 100, "color": 0.5}, {"geom": 90, "color": 0.2}, {"geom": 80, "color": 0.4}]
    assert fusion.verify_select(cands) == 1


def test_verify_select_single():
    assert fusion.verify_select([{"color": 0.9}]) == 0
