"""Concatenate the subset's unique room .txt clouds (X Y Z R G B) into one .ply
for the FGPL baker (Open3D-readable). Color is kept but the baker ignores it."""
import numpy as np, open3d as o3d
from experiments.fgpl_seed import paths, subset


def build_combined_ply(rows):
    clouds = sorted({r["cloud_txt"] for r in rows})
    xyz_all, rgb_all = [], []
    for c in clouds:
        arr = np.loadtxt(c)                      # (N,6): X Y Z R G B
        xyz_all.append(arr[:, :3])
        rgb_all.append(arr[:, 3:6])
    xyz = np.concatenate(xyz_all, axis=0)
    rgb = np.concatenate(rgb_all, axis=0)
    if rgb.max() > 1.5:                          # S3DIS Original RGB may be 0-255
        rgb = rgb / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1))
    out = paths.subdir("clouds") / f"{paths.SCENE}.ply"
    o3d.io.write_point_cloud(str(out), pcd)
    return out


if __name__ == "__main__":
    rows = subset.build_subset()
    p = build_combined_ply(rows)
    import numpy as np
    print("wrote", p, "points=", np.asarray(o3d.io.read_point_cloud(str(p)).points).shape[0])
