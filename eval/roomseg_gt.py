"""Per-point ground-truth room labels for a merged S3DIS scene (HARNESS ONLY; spec §6.2).

The merged scene .ply is the exact union of its rooms' .txt files (measured 2026-10-01), so every
merged point is matched to a room point within 1 mm, or this module refuses.
Run: conda run -n panopin python eval/roomseg_gt.py --scene Area_3_manhattan4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from panopin.roomseg.files import read_cloud  # noqa: E402

CACHE = REPO / "runs" / "roomseg_gt"


def load_scenes(path=REPO / "config" / "roomseg_scenes.json"):
    return json.loads(Path(path).read_text())["scenes"]


def gt_labels_for(xyz, rooms_dir, rooms, tol=1e-3):
    """Index into `rooms` of every xyz point, by nearest match to the rooms' .txt files."""
    parts, ids = [], []
    for i, r in enumerate(rooms):
        a = pd.read_csv(Path(rooms_dir) / r / f"{r}.txt", sep=r"\s+", header=None,
                        usecols=[0, 1, 2]).values
        parts.append(a)
        ids.append(np.full(len(a), i, np.int32))
    ref, rid = np.concatenate(parts), np.concatenate(ids)
    if len(ref) != len(xyz):
        raise ValueError(f"merged cloud has {len(xyz)} points, the room files hold {len(ref)} points")
    d, j = cKDTree(ref).query(xyz, k=1)
    if d.max() > tol:
        raise ValueError(f"{int((d > tol).sum())} points have no room point within {tol} m "
                         f"(max {d.max():.4f} m)")
    return rid[j]


def cached_gt(scene):
    """(xyz, gt) for a configured scene; GT labels cached in runs/roomseg_gt/<scene>.npy."""
    cfg = load_scenes()[scene]
    xyz, _ = read_cloud(cfg["cloud"])
    f = CACHE / f"{scene}.npy"
    if f.exists():
        gt = np.load(f)
        if len(gt) == len(xyz):
            return xyz, gt
    gt = gt_labels_for(xyz, cfg["rooms_dir"], cfg["rooms"])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, gt)
    return xyz, gt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    a = ap.parse_args()
    xyz, gt = cached_gt(a.scene)
    rooms = load_scenes()[a.scene]["rooms"]
    print(a.scene, len(xyz), "points", {r: int((gt == i).sum()) for i, r in enumerate(rooms)})
