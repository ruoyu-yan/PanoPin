"""Score one arm's FGPL poses vs S3DIS GT using the PanoPin harness metrics."""
import math
from eval import metrics

def _nearest_room(xyz, centroids):
    best, bd = None, float("inf")
    for room, c in centroids.items():
        d = sum((xyz[i] - c[i]) ** 2 for i in range(3))
        if d < bd:
            bd, best = d, room
    return best

def score_arm(poses, gt, rows, centroids):
    true_room = {r["pano_name"]: r["room"] for r in rows}
    pred_t = {u: p["translation"] for u, p in poses.items() if p is not None}
    pred_R = {u: p["rotation"] for u, p in poses.items() if p is not None}
    gt_loc = {u: gt[u]["location"] for u in gt}
    gt_R = {u: gt[u]["R_cw"] for u in gt}
    wrong = 0
    for u, p in poses.items():
        if p is None:
            continue
        if _nearest_room(p["translation"], centroids) != true_room[u]:
            wrong += 1
    n_loc = len(pred_t)
    return {
        "n_localized": n_loc,
        "translation": metrics.translation_errors(pred_t, gt_loc),
        "rotation": metrics.rotation_errors(pred_R, gt_R),
        "wrong_room_rate": (wrong / n_loc) if n_loc else None,
    }
