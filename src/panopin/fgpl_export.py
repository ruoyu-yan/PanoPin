"""PanoPin -> FGPL alignment export (2026-07-15): turn per-pano low-percentile color
scores + coarse poses into the demo6_alignment.json FGPL consumes.

Pure assembly + serialization (NO GPU, NO CPO import). Consumes the outputs of the shipped
GPU stage `seed.localize_and_score` (score_matrix {pano:{room: low-pct score}}, poses
{pano:{room:(t,R)}}). FGPL reads seeds PER PANO and positional-only (only pano_name +
camera_position); each pano may appear at most once in matches. Gate & omit weak-lock panos
with a room-anchored coverage backstop; convert PanoPin's raw-frame t to FGPL's aligned-frame
camera_position. Fair: reads only scores/poses/metadata, never GT (D5).

Deployment contract: the caller MUST set FGPL cfg["pano_names"] = the returned admitted list
(an unseeded name in pano_names -> KeyError in FGPL's loader)."""
import json
import numpy as np

from panopin import coverage


def raw_t_to_camera_position(t_raw, R_meta, tol=1e-6):
    """Raw-frame CPO translation -> FGPL aligned-frame camera_position [ax, ay].

    FGPL converts back via raw_3d = R.T @ [ax, ay, 0] (aligned_meters_to_raw_3d), so the
    inverse is camera_position = (R @ t_raw)[:2]. Exact for a yaw R (Manhattan alignment);
    a non-yaw / malformed R fails the round-trip guard and raises rather than emit a silently
    wrong seed."""
    R = np.asarray(R_meta, dtype=float)
    t = np.asarray(t_raw, dtype=float)
    cam = (R @ t)[:2]
    back = (R.T @ np.array([cam[0], cam[1], 0.0]))[:2]
    if not np.allclose(back, t[:2], atol=tol):
        raise ValueError(
            f"frame round-trip failed (metadata rotation not yaw-like?): {back} vs {t[:2]}")
    return [float(cam[0]), float(cam[1])]


def build_matches(score_matrix, poses, room_order, R_meta, tau=0.10, guarantee_coverage=True):
    """Per-pano gate + coverage backstop -> (matches, admitted_pano_names).

    Gate: admit each pano at its winner (argmin) room iff winner_score <= tau; weak-lock panos
    are omitted. Backstop (guarantee_coverage): any room in room_order with no admitted pano is
    seeded by its best UNASSIGNED pano (keeps per-pano uniqueness; equals the D32 room-anchored
    pick in the common all-covered case). Each pano appears at most once."""
    if not score_matrix:
        return [], []
    assigned = {}   # pano -> room, at most one room per pano
    for pano, (room, neg_score) in coverage.pano_confidence(score_matrix).items():
        if -neg_score <= tau:
            assigned[pano] = room
    covered = set(assigned.values())
    if guarantee_coverage:
        for room in room_order:
            if room in covered:
                continue
            free = [p for p in score_matrix if p not in assigned]
            if not free:
                continue  # cannot cover without a duplicate emission; leave uncovered
            best = min(free, key=lambda p: score_matrix[p][room])
            assigned[best] = room
            covered.add(room)
    matches = []
    for pano, room in assigned.items():
        t, _R = poses[pano][room]
        cam = raw_t_to_camera_position(t, R_meta)
        matches.append({
            "pano_name": pano,
            "room_idx": room_order.index(room),
            "room_label": room,
            "score": float(score_matrix[pano][room]),
            "rotation_deg": 0.0,
            "camera_position": cam,
        })
    admitted_pano_names = [m["pano_name"] for m in matches]
    return matches, admitted_pano_names


def write_alignment_json(matches, admitted_pano_names, out_path, extra_meta=None):
    """Serialize the exact demo6_alignment.json schema FGPL reads (align_polygons_demo6.py)."""
    meta = {"pipeline": "PanoPin color seed (gated per-pano)",
            "pano_names": list(admitted_pano_names),
            "source": "fgpl_export"}
    if extra_meta:
        meta.update(extra_meta)
    with open(out_path, "w") as f:
        json.dump({"metadata": meta, "matches": matches}, f, indent=4)


def export_alignment(score_matrix, poses, room_order, metadata_path, out_path,
                     tau=0.10, guarantee_coverage=True):
    """Read metadata.json rotation, build + write demo6_alignment.json, return admitted panos.
    The caller MUST set FGPL cfg["pano_names"] to the returned list."""
    with open(metadata_path) as f:
        R_meta = json.load(f)["rotation_matrix"]
    matches, admitted = build_matches(score_matrix, poses, room_order, R_meta,
                                      tau=tau, guarantee_coverage=guarantee_coverage)
    write_alignment_json(matches, admitted, out_path, extra_meta={"tau": tau})
    return admitted
