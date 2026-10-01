"""Score a room segmentation against the S3DIS partition (HARNESS ONLY; spec §6.2).

5 cm voxels (Matterport density varies with range), majority labels per voxel, voxels within
0.2 m (XY) of another GT room ignored. Then one-to-one Hungarian matching on IoU, F1 at 0.5 / 0.7,
PQ, named splits / merges, unassigned share, and pano containment.
Run: conda run -n panopin python eval/roomseg_score.py --scene S --seg-dir runs/roomseg/S --out runs/roomseg_eval/S
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "eval"))
from panopin.roomseg.raster import disk  # noqa: E402
import roomseg_gt  # noqa: E402

VOXEL = 0.05
BAND_CELLS = 4          # 0.2 m ignore band
SPLIT_SHARE = 0.80      # a GT room is split if no segment holds >= 80 % of it
MERGE_SHARE = 0.10      # a segment merges rooms if it holds >= 10 % of each of >= 2 rooms


def _majority(vox, lab):
    lab = lab.astype(np.int64)
    base = lab.min()
    off = lab - base
    span = int(off.max()) + 1
    u, c = np.unique(vox.astype(np.int64) * span + off, return_counts=True)
    pv, pl = u // span, u % span
    order = np.lexsort((pl, -c, pv))            # by voxel, then count desc, then label asc
    _, first = np.unique(pv[order], return_index=True)
    return pl[order][first] + base


def voxels(xyz, gt, pred, v=VOXEL):
    keys = np.floor(np.asarray(xyz) / v).astype(np.int64)
    vkeys, vox = np.unique(keys, axis=0, return_inverse=True)
    vox = vox.ravel()
    return vkeys, _majority(vox, gt), _majority(vox, pred)


def near_other_room(vkeys, vgt, radius_cells=BAND_CELLS):
    xy = vkeys[:, :2] - vkeys[:, :2].min(axis=0)
    shape = tuple(xy.max(axis=0) + 1)
    k = disk(radius_cells)
    occ = {}
    for r in np.unique(vgt):
        m = np.zeros(shape, bool)
        m[xy[vgt == r, 0], xy[vgt == r, 1]] = True
        occ[r] = m
    near = np.zeros(len(vgt), bool)
    for r in occ:
        others = np.zeros(shape, bool)
        for s, m in occ.items():
            if s != r:
                others |= m
        if not others.any():
            continue
        dil = ndimage.binary_dilation(others, k)
        sel = vgt == r
        near[sel] = dil[xy[sel, 0], xy[sel, 1]]
    return near


def metrics(vgt, vpred, rooms, seg_names):
    G, S = len(rooms), len(seg_names)
    inter = np.zeros((G, S))
    ok = vpred >= 0
    np.add.at(inter, (vgt[ok], vpred[ok]), 1)
    gsize = np.bincount(vgt, minlength=G).astype(float)
    ssize = np.bincount(vpred[ok], minlength=S).astype(float)
    union = gsize[:, None] + ssize[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    rr, cc = linear_sum_assignment(-iou)
    matching = {r: None for r in rooms}
    tp_ious = []
    for g, s in zip(rr, cc):
        if iou[g, s] > 0:
            matching[rooms[g]] = {"segment": seg_names[s], "iou": round(float(iou[g, s]), 4)}
            tp_ious.append(float(iou[g, s]))

    def f1(t):
        tp = sum(i > t for i in tp_ious)
        return 0.0 if tp == 0 else 2 * tp / (G + S)

    tp5 = [i for i in tp_ious if i > 0.5]
    rq = 0.0 if not tp5 else len(tp5) / (len(tp5) + 0.5 * (S - len(tp5)) + 0.5 * (G - len(tp5)))
    sq = float(np.mean(tp5)) if tp5 else 0.0
    share = np.divide(inter, gsize[:, None], out=np.zeros_like(inter), where=gsize[:, None] > 0)
    splits = [rooms[g] for g in range(G) if share[g].max() < SPLIT_SHARE]
    merges = {seg_names[s]: [rooms[g] for g in range(G) if share[g, s] >= MERGE_SHARE]
              for s in range(S) if (share[:, s] >= MERGE_SHARE).sum() >= 2}
    return {"matching": matching, "f1_05": f1(0.5), "f1_07": f1(0.7), "pq": sq * rq, "sq": sq,
            "rq": rq, "splits": splits, "merges": merges,
            "unassigned_frac": float((~ok).mean()), "K": S, "G": G}


def containment(xyz, gt, pred, panos, rooms, seg_names, matching, r=0.15):
    """Is each pano's GT camera XY inside its room's matched segment? (majority of points within r)"""
    tree = cKDTree(np.asarray(xyz)[:, :2])
    rows = {}
    for key, t in panos.items():
        room = key.split("_", 2)[2]
        if room not in rooms:
            continue
        idx = tree.query_ball_point(np.asarray(t[:2], float), r)
        row = {"room": room, "n_points": len(idx), "gt_inside": False, "pred_segment": None,
               "pred_inside": False}
        if idx:
            row["gt_inside"] = rooms[int(np.bincount(gt[idx]).argmax())] == room
            pv = pred[idx]
            pv = pv[pv >= 0]
            if len(pv):
                row["pred_segment"] = seg_names[int(np.bincount(pv).argmax())]
            m = matching.get(room)
            row["pred_inside"] = bool(m) and row["pred_segment"] == m["segment"]
        rows[key] = row
    return rows


def score_scene(scene, seg_dir):
    cfg = roomseg_gt.load_scenes()[scene]
    xyz, gt = roomseg_gt.cached_gt(scene)
    pred = np.load(Path(seg_dir) / "labels.npy")
    seg_names = list(json.loads((Path(seg_dir) / "clouds.json").read_text()))
    if len(pred) != len(xyz):
        raise ValueError(f"labels.npy has {len(pred)} rows, the scene cloud {len(xyz)} points")
    vkeys, vgt, vpred = voxels(xyz, gt, pred)
    keep = ~near_other_room(vkeys, vgt)
    m = metrics(vgt[keep], vpred[keep], cfg["rooms"], seg_names)
    panos = {k: v["t"] for k, v in json.loads(Path(cfg["pose_json"]).read_text()).items()}
    m["containment"] = containment(xyz, gt, pred, panos, cfg["rooms"], seg_names, m["matching"])
    m["scene"], m["split"], m["seg_dir"] = scene, cfg["split"], str(seg_dir)
    m["voxels_scored"], m["voxels_ignored"] = int(keep.sum()), int((~keep).sum())
    return m


def to_markdown(m):
    lines = [f"### {m['scene']} ({m['split']}) — {m['seg_dir']}", "",
             f"K={m['K']} vs G={m['G']} · F1@0.5 {m['f1_05']:.2f} · F1@0.7 {m['f1_07']:.2f} · "
             f"PQ {m['pq']:.3f} · unassigned {m['unassigned_frac']:.2%} · "
             f"splits {m['splits'] or 'none'} · merges {m['merges'] or 'none'}", "",
             "| GT room | segment | IoU |", "|---|---|---|"]
    for room, x in m["matching"].items():
        lines.append(f"| {room} | {x['segment'] if x else '—'} | {x['iou'] if x else 0:.3f} |")
    c = m["containment"]
    lines += ["", f"containment: pred {sum(r['pred_inside'] for r in c.values())}/{len(c)}, "
              f"GT partition itself {sum(r['gt_inside'] for r in c.values())}/{len(c)}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seg-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    m = score_scene(a.scene, a.seg_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "score.json").write_text(json.dumps(m, indent=1))
    md = to_markdown(m)
    (a.out / "score.md").write_text(md)
    print(md)
