"""Baker (.ply -> room_geometry.pkl) then cluster (-> 3d_line_map.pkl)."""
from experiments.fgpl_seed import paths, fgpl_tool, build_ply, subset

def build_linemap(ply_path):
    out = paths.subdir("linemap")
    fgpl_tool.run_tool(paths.BAKER, {
        "point_cloud_name": paths.SCENE,
        "point_cloud_path": str(ply_path),
        "output_dir": str(out),
    }, tag="baker")
    room_geom = out / "room_geometry.pkl"
    assert room_geom.exists(), room_geom
    fgpl_tool.run_tool(paths.CLUSTER, {
        "point_cloud_name": paths.SCENE,
        "input_pkl": str(room_geom),
        "output_dir": str(out),
    }, tag="cluster")
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
