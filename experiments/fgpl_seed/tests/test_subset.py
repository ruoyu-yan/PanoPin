import os, pytest
from experiments.fgpl_seed import subset

def test_rooms_are_the_six_same_shape_subset():
    assert subset.ROOMS == ["office_1", "office_4", "office_5", "office_6", "office_7", "hallway_3"]

@pytest.mark.skipif(not os.path.exists("/mnt/d"), reason="needs S3DIS GT on /mnt/d")
def test_build_subset_resolves_clouds_and_panos():
    rows = subset.build_subset()
    rooms = {r["room"] for r in rows}
    assert rooms == set(subset.ROOMS)                       # every room contributes >=1 pano
    for r in rows:
        assert os.path.exists(r["cloud_txt"]), r["cloud_txt"]
        assert os.path.exists(r["pano_jpg"]), r["pano_jpg"]
        assert r["pano_name"] == r["uuid"]
    per_room = {}
    for r in rows:
        per_room[r["room"]] = per_room.get(r["room"], 0) + 1
    assert all(v <= subset.MAX_PANOS_PER_ROOM for v in per_room.values())
