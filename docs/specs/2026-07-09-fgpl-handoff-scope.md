# T4 — FGPL hand-off scope (what PanoPin's coarse seed must deliver)

**Date:** 2026-07-09 · **Status:** scoping · grounds project task T4.
Based on a recon of `/home/ruoyu/scan2measure-webframework/` (FGPL = the fine localizer PanoPin feeds).

## The contract FGPL actually requires
Per pano, the production fine step (`src/pose_estimation/multiroom_pose_estimation.py`) consumes **only two
fields** (the `matches[]` subset of `demo6_alignment.json`, written today by the jigsaw
`src/floorplan/align_polygons_demo6.py`):

```json
{ "pano_name": "...", "room_label": "room_03", "camera_position": [x, y] }
```

- **`room_label`** — a SINGLE (top-1) room assignment per pano.
- **`camera_position`** — an approximate **2D [x, y]** position in the aligned/Manhattan-meters floorplan
  frame (converted to raw-3D XY via `metadata.json`'s `rotation_matrix`).

**Not required / not consumed by FGPL:** rotation (it re-derives orientation from scratch via 24 Manhattan
vanishing-direction candidates), Z/height, scale, or any full 6-DoF pose. `rotation_deg` is even written by
the jigsaw but read nowhere downstream.

## How rough the seed can be
The position only needs to be **~3 m accurate** — it is used solely to build a Voronoi partition (2 m overlap
margin) that selects which local slice of the shared 3D line map FGPL searches. FGPL then does its own full
translation-grid + rotation search over that slice. Documented (`docs/superpowers/specs/2026-03-18-local-
linefilter-design.md:136`): positions "only need to be roughly correct (within ~3m)". **The failure mode FGPL
guards against is WRONG ROOM, not imprecise position.**

## Top-1 vs top-k
The **production** pipeline is strictly **top-1**: the jigsaw writes one best room per pano; there is no
shortlist mechanism. A native **top-k room-selection** (`infer_room` over a `map_room_list`) exists ONLY in
the deprecated original FGPL (`archive/legacy_repos/panoramic-localization/fgpl/`) and was NOT ported. So to
hand off top-k, that selection logic would have to be re-added downstream.

## What PanoPin already produces vs the gap
PanoPin's `localize_pair` returns `(t, R, loss)` per (pano, room); the assigned room = min-loss room, and
`t` gives a 3D position → project to XY for `camera_position`. So PanoPin already produces **both required
fields** — modulo a frame conversion (S3DIS point-cloud frame → the app's aligned-meters floorplan frame;
only relevant in the real app, not the S3DIS benchmark). The one hard requirement is getting **`room_label`
right (top-1)**.

## The decision this forces
FGPL (production) needs **top-1 correct room**. PanoPin's measured **recall@1 = 78%** (prototype, 3 rooms)
therefore means ~22% of panos would seed FGPL with the WRONG room → FGPL's catastrophic failure mode. So
**78% top-1 is NOT clearly "good enough"** for the current pipeline. Two ways forward:

- **(A) Raise PanoPin's top-1 accuracy** on the failing cases (same-shape office pairs; degenerate "loss-sink"
  rooms like WC_1) so top-1 is reliable. Keeps the hand-off simple (matches the current jigsaw interface).
- **(B) Re-enable FGPL's native top-k room selection** (`infer_room`) so PanoPin hands off a top-k shortlist
  and FGPL disambiguates. PanoPin's misses were rank 2–3, so recall@k is much higher than 78% — this likely
  covers most misses without improving PanoPin itself, at the cost of downstream work in FGPL and running
  FGPL on k rooms (k× the fine-step cost).

**Recommendation:** measure PanoPin's **recall@k** (not just @1) at prototype scale first — it's the number
that decides A vs B. If recall@2/@3 is high (likely, given misses are rank 2–3), **(B)** is cheaper and more
robust than chasing a hard same-shape-office discrimination in PanoPin. Also decide the deliverable format:
PanoPin should emit the `matches[].{room_label|room_shortlist, camera_position}` schema so it is a drop-in
replacement for `align_polygons_demo6.py`.
