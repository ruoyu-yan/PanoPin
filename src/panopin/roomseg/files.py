"""Read a merged cloud; write the `segment` command's outputs (spec §3.1)."""
import json
from pathlib import Path

import cv2
import numpy as np

from .core import segment
from .errors import SegmentationError
from .params import DEFAULTS


def read_cloud(path):
    """(xyz float64[N,3], rgb uint8[N,3]) in file order, from .ply or an S3DIS-style .txt."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ply":
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        if not pcd.has_colors():
            raise SegmentationError(f"{path} has no colours; PanoPin's colour registration needs RGB")
        xyz = np.asarray(pcd.points, dtype=np.float64)
        rgb = np.rint(np.asarray(pcd.colors) * 255.0).astype(np.uint8)
    elif suffix == ".txt":
        import pandas as pd
        a = pd.read_csv(path, sep=r"\s+", header=None).values
        if a.ndim != 2 or a.shape[1] < 6:
            raise SegmentationError(f"{path}: expected columns X Y Z R G B")
        xyz = a[:, :3].astype(np.float64)
        rgb = np.clip(np.rint(a[:, 3:6]), 0, 255).astype(np.uint8)
    else:
        raise SegmentationError(f"unsupported cloud format '{suffix}' (use .ply or .txt)")
    if len(xyz) == 0:
        raise SegmentationError(f"{path} holds no points")
    return xyz, rgb


def write_outputs(out_dir, xyz, rgb, labels, image, report):
    """Write seg_XX.txt, clouds.json, labels.npy, segmentation.json, rooms.png; return clouds."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("seg_*.txt"):
        stale.unlink()
    clouds = {}
    for k, s in enumerate(report["segments"]):
        f = out / f"{s['name']}.txt"
        m = labels == k
        np.savetxt(f, np.column_stack([xyz[m], rgb[m]]), fmt="%.6f %.6f %.6f %d %d %d")
        clouds[s["name"]] = str(f.resolve())
    (out / "clouds.json").write_text(json.dumps(clouds, indent=1))
    np.save(out / "labels.npy", labels.astype(np.int32))
    (out / "segmentation.json").write_text(json.dumps(report, indent=1))
    cv2.imwrite(str(out / "rooms.png"), _colour(image))
    return clouds


def segment_cloud(cloud, out_dir, n_panos=None, p=DEFAULTS):
    """Read, segment, then write. Nothing is written if reading or segmenting raises."""
    xyz, rgb = read_cloud(cloud)
    labels, image, report = segment(xyz, p, n_panos)
    report["input"] = str(Path(cloud).resolve())
    write_outputs(out_dir, xyz, rgb, labels, image, report)
    return report


def _colour(image):
    lut = np.random.RandomState(7).randint(60, 256, size=(int(image.max()) + 1, 3)).astype(np.uint8)
    lut[0] = 0
    return np.flipud(lut[image])   # north up
