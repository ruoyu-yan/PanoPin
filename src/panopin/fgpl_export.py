"""PanoPin -> FGPL alignment export (2026-07-15): turn per-pano low-percentile color
scores + coarse poses into the demo6_alignment.json FGPL consumes.

Pure assembly + serialization (NO GPU, NO CPO import). Consumes the outputs of the shipped
GPU stage `seed.localize_and_score` (score_matrix {pano:{room: low-pct score}}, poses
{pano:{room:(t,R)}}). FGPL reads seeds PER PANO and positional-only (only pano_name +
camera_position); each pano may appear at most once in matches. Assign panos to rooms jointly
(one per room, minimum total score), gate only the surplus; convert PanoPin's raw-frame t to
FGPL's aligned-frame camera_position. Fair: reads only scores/poses/metadata, never GT (D5).

Deployment contract: the caller MUST set FGPL cfg["pano_names"] = the returned admitted list
(an unseeded name in pano_names -> KeyError in FGPL's loader)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from panopin import coverage
from panopin.roomseg import shape as shape_mod


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


def build_matches(score_matrix, poses, room_order, R_meta, tau=0.10, guarantee_coverage=True,
                  shapes=None):
    """Joint assignment (panos <= rooms) or the legacy per-pano gate -> (matches, admitted).

    Joint (guarantee_coverage=True and len(score_matrix) <= len(room_order), the CLI's usual
    case): panos and rooms are matched one-to-one by `coverage.assign_rooms` (minimum total
    score); every pano is in that assignment and admitted unconditionally, even above tau.
    Rooms left over stay unseeded. Tagged "assignment": "joint".

    Legacy (panos outnumber rooms, or guarantee_coverage=False): admit each pano at its argmin
    room iff its score <= tau; with guarantee_coverage, any room with no admitted pano is then
    seeded by its best UNASSIGNED pano. Tagged "assignment": "argmin". The joint rule is not
    used here because with several panos per room a one-to-one assignment forces a confident
    pano out of its room into a pano-less segment.

    shapes {room: roomseg.shape.SegmentShape} (optional): a pano whose room is a corridor
    (is_corridor) is seeded at the room's plan centroid instead of its CPO position; records
    carry "seed_basis" and "extent_ratio". Only a corridor holding exactly ONE admitted pano
    moves: when two or more share it (legacy branch), each keeps its CPO position, because
    identical seeds would make FGPL's Voronoi cells degenerate and the arbitration could not
    tell those panos apart (the 0.9 m centroid figure was measured with one pano per corridor).

    Each pano appears at most once. Records are emitted in score-matrix order."""
    if not score_matrix:
        return [], []
    assigned = {}   # pano -> room, at most one room per pano
    if guarantee_coverage and len(score_matrix) <= len(room_order):
        assigned = coverage.assign_rooms(score_matrix, room_order)
        basis = "joint"
    else:
        basis = "argmin"
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
    panos_in_room = Counter(assigned.values())
    matches = []
    for pano in score_matrix:              # matrix order: deterministic, independent of the solver
        if pano not in assigned:
            continue
        room = assigned[pano]
        t, _R = poses[pano][room]
        seed_basis, ratio = "cpo", None
        if shapes is not None and room in shapes:
            ratio = float(shapes[room].extent_ratio)
            if shape_mod.is_corridor(shapes[room]) and panos_in_room[room] == 1:
                cx, cy = shapes[room].centroid_xy
                t = [cx, cy, float(t[2])]
                seed_basis = "centroid"
        cam = raw_t_to_camera_position(t, R_meta)
        matches.append({
            "pano_name": pano,
            "room_idx": room_order.index(room),
            "room_label": room,
            "score": float(score_matrix[pano][room]),
            "rotation_deg": 0.0,
            "camera_position": cam,
            "assignment": basis,
            "seed_basis": seed_basis,
            "extent_ratio": ratio,
        })
    admitted_pano_names = [m["pano_name"] for m in matches]
    return matches, admitted_pano_names


def write_alignment_json(matches, admitted_pano_names, out_path, extra_meta=None):
    """Serialize the exact demo6_alignment.json schema FGPL reads (align_polygons_demo6.py)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    meta = {"pipeline": ("PanoPin color seed (joint one-pano-per-room assignment; "
                         "per-pano gate when panos outnumber rooms)"),
            "pano_names": list(admitted_pano_names),
            "source": "fgpl_export"}
    if extra_meta:
        meta.update(extra_meta)
    with open(out_path, "w") as f:
        json.dump({"metadata": meta, "matches": matches}, f, indent=4)


def export_alignment(score_matrix, poses, room_order, metadata_path, out_path,
                     tau=0.10, guarantee_coverage=True, shapes=None):
    """Read metadata.json rotation, build + write demo6_alignment.json, return admitted panos.
    The caller MUST set FGPL cfg["pano_names"] to the returned list."""
    with open(metadata_path) as f:
        R_meta = json.load(f)["rotation_matrix"]
    matches, admitted = build_matches(score_matrix, poses, room_order, R_meta,
                                      tau=tau, guarantee_coverage=guarantee_coverage,
                                      shapes=shapes)
    write_alignment_json(matches, admitted, out_path, extra_meta={"tau": tau})
    return admitted
