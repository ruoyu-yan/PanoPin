"""Rebuild the Manhattan scene's FGPL inputs from the raw room clouds.

Deliberately NOT the build_* __main__ blocks: those call subset.build_subset(),
which resolves to the 6-room ablation subset, not the 5-room Manhattan pool.
"""
from experiments.fgpl_seed import build_features, build_linemap, build_ply, manhattan

if __name__ == "__main__":
    rows = manhattan.build_pool()
    rooms = sorted({r["room"] for r in rows})
    print("rooms:", rooms)
    assert rooms == sorted(["office_5", "hallway_1", "lounge_1",
                            "conferenceRoom_1", "WC_1"]), rooms
    print(f"panos: {len(rows)}")
    assert len(rows) == 22, len(rows)

    ply = build_ply.build_combined_ply(rows, scene=manhattan.SCENE)
    print("ply:", ply)
    lm = build_linemap.build_linemap(ply, scene=manhattan.SCENE,
                                     out_subdir=manhattan.LINEMAP_SUBDIR)
    print("line map:", lm)
    feats = build_features.stage_and_build(rows)
    print(f"features for {len(feats)} panos")
