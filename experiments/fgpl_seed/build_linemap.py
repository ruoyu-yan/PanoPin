"""Baker (.ply -> room_geometry.pkl) then cluster (-> 3d_line_map.pkl)."""
from experiments.fgpl_seed import paths, fgpl_tool, build_ply, subset

def build_linemap(ply_path, scene=None, out_subdir="linemap"):
    """Scene/out_subdir default to the original 6-room ablation map; pass both to build an
    isolated map (e.g. the Manhattan scene) without clobbering a cached one."""
    scene = scene or paths.SCENE
    out = paths.subdir(out_subdir)
    fgpl_tool.run_tool(paths.BAKER, {
        "point_cloud_name": scene,
        "point_cloud_path": str(ply_path),
        "output_dir": str(out),
    }, tag=f"baker_{scene}")
    room_geom = out / "room_geometry.pkl"
    assert room_geom.exists(), room_geom
    fgpl_tool.run_tool(paths.CLUSTER, {
        "point_cloud_name": scene,
        "input_pkl": str(room_geom),
        "output_dir": str(out),
    }, tag=f"cluster_{scene}")
    line_map = out / "3d_line_map.pkl"
    assert line_map.exists(), line_map
    return line_map

if __name__ == "__main__":
    rows = subset.build_subset()
    ply = build_ply.build_combined_ply(rows)
    lm = build_linemap(ply)
    import pickle
    with open(lm, "rb") as f:
        d = pickle.load(f)
    print("3d_line_map keys:", sorted(d.keys()))
    print("n dense lines:", len(d["dense_starts"]), " n intersections:", len(d["inter_3d"]))
