"""Build the FGPL map for the strictly-Manhattan scene (2026-07-16 demo). Expensive; run first.

Confines the FGPL search space to Manhattan world by building the line map from ONLY the 5
Manhattan room clouds. Writes to an ISOLATED namespace (`manhattan.SCENE` /
`manhattan.LINEMAP_SUBDIR`) so the cached 6-room ablation map — which the oracle/p1/D35 arms were
produced against — is left untouched.

Pano features are per-image and map-independent, so they share work/features/ (a pano appearing in
both pools is simply reused).

    conda run -n panopin python -m experiments.fgpl_seed.manhattan_map
Baker/cluster/features internally shell out to scan_env (paths.SCAN_ENV).
"""
import os
import sys
import time

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from experiments.fgpl_seed import manhattan, build_ply, build_linemap, build_features, paths


def main():
    t0 = time.time()
    rows = manhattan.build_pool()
    rooms = sorted({r["room"] for r in rows})
    print(f"[map] Manhattan scene '{manhattan.SCENE}': {len(rows)} panos, {len(rooms)} rooms "
          f"{rooms}", flush=True)

    ply = build_ply.build_combined_ply(rows, scene=manhattan.SCENE)
    import open3d as o3d
    import numpy as np
    n = np.asarray(o3d.io.read_point_cloud(str(ply)).points).shape[0]
    print(f"[map] ply {ply} points={n}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    lm = build_linemap.build_linemap(ply, scene=manhattan.SCENE,
                                     out_subdir=manhattan.LINEMAP_SUBDIR)
    import pickle
    with open(lm, "rb") as f:
        d = pickle.load(f)
    print(f"[map] line map {lm}: {len(d['dense_starts'])} dense lines, "
          f"{len(d['inter_3d'])} intersections  ({(time.time()-t0)/60:.1f} min)", flush=True)

    feats = build_features.stage_and_build(rows)
    print(f"[map] features for {len(feats)} panos  ({(time.time()-t0)/60:.1f} min)", flush=True)

    # The 6-room ablation map must be intact for the cached arms to remain reproducible.
    assert (paths.WORK / "linemap" / "3d_line_map.pkl").exists(), "clobbered the ablation map!"
    print(f"[map] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
