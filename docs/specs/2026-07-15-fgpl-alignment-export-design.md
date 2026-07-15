# Design — PanoPin → FGPL alignment export (gated per-pano seeds)

**Date:** 2026-07-15 · **Branch:** `feat/deploy-regime` (off `main`, continues the D30–D33 lineage) ·
**Builds on:** D30 (`robust_score.low_percentile_scores` — the per-room color score), D32
(`coverage.room_anchored_seeds` / `pano_confidence` — threshold-free, loss-sink-immune room seeding),
D33 (`seed.seed_rooms` + the `largeval_*` GPU caches; the deployable localize→score→seed assembly).
**Consumes / wires into:** FGPL `scan2measure-webframework/src/pose_estimation/multiroom_pose_estimation.py`.
**Deliverable:** a new module `src/panopin/fgpl_export.py` + tests; results/notes appended to
`docs/PROGRESS.md` + `docs/DECISIONS.md`.

## 1. Goal

Turn PanoPin's per-pano color scores + coarse poses into the exact `demo6_alignment.json` FGPL reads,
so the real Scan2BIM pipeline can consume PanoPin in place of the SAM3 jigsaw's coarse role. Two jobs:

1. **Gate & omit** weak-lock panos (user decision, 2026-07-15): emit a seed only for panos PanoPin can
   trust; the ~15–22% weak-lock (window/occlusion, D23/D31) panos are excluded rather than seeded with a
   guessed room.
2. **Frame-convert** PanoPin's raw-S3DIS-frame pose into FGPL's aligned-frame `camera_position`.

**Non-goal (this spec):** re-running localization (that is `seed.localize_and_score`, shipped D33);
tuning the gate threshold across areas; the Electron/pipeline wiring; and the live FGPL GPU round-trip
(that is the explicit fast-follow in §8).

## 2. What FGPL consumes (verified against the code, 2026-07-15)

Read from `scan2measure-webframework/src/pose_estimation/multiroom_pose_estimation.py` and the producer
`src/floorplan/align_polygons_demo6.py`:

- **The seed is PER-PANO, positional only.** `load_panorama_positions` (line 104) iterates
  `alignment['matches']` and reads **only** `pano_name` + `camera_position`. `room_label` appears once
  inside a `print` (line 121); `room_idx`, `rotation_deg`, `score` are never read. Confirms the memory's
  "only `camera_position` is load-bearing".
- **FGPL localizes each pano independently.** Phase B loops `for pano_idx, pano_name in enumerate(pano_names)`
  (line 291); each pano's own `camera_position` carves *its* Voronoi cell (`get_local_mask`, line 134) and
  the geometric search runs inside that local region. Anchoring a room does **not** propagate to other panos
  in it — so "one seed per room" would localize only one pano per room. To pose N panos you must emit N
  matches.
- **`pano_names` comes from FGPL's config**, not the JSON (`cfg.get("pano_names", ...)`, line 152), and
  `load_panorama_positions` filters matches to that list. A name in `pano_names` with no match entry →
  `KeyError` at line 129/234. ⇒ the deployment must set `cfg["pano_names"]` to exactly the emitted set.
- **Output schema** (`align_polygons_demo6.py:572-595`), one entry per pano:
  ```json
  { "metadata": { "pano_names": [...], "scale": ..., ... },
    "matches": [ { "pano_name": str, "room_idx": int, "room_label": str,
                   "score": float, "rotation_deg": float, "camera_position": [x, y] } ] }
  ```

## 3. Frame conversion (verified math)

FGPL converts the stored (aligned) `camera_position` back to raw world XY via `aligned_meters_to_raw_3d`
(line 93): `raw_3d = R.T @ [ax, ay, 0]`, `raw_xy = raw_3d[:2]`, where `R = metadata['rotation_matrix']`
(raw→aligned, a Manhattan-alignment yaw). PanoPin's CPO `t` is already raw. Invert:

```
camera_position = (R @ t_raw)[:2]
```

For a yaw `R` (rotation about world-Z, which the Manhattan alignment is) the z-coupling cancels exactly, so
`aligned_meters_to_raw_3d((R @ t_raw)[:2], meta) == t_raw[:2]`. **This identity is asserted, not assumed**
(`raw_t_to_camera_position` fails loud if the round-trip error exceeds a tolerance — the guard against a
non-yaw or malformed metadata). In the D25 ablation `metadata.json` was identity, so this was a silent
no-op; in real deployment the density-image alignment makes `R ≠ I`, which is exactly the bug this guards.

## 4. Component design

New `src/panopin/fgpl_export.py` — **pure assembly + serialization, no GPU, no CPO import.** It consumes the
outputs of the existing GPU stage `seed.localize_and_score` (`score_matrix {pano:{room: low-pct score}}`,
`poses {pano:{room:(t, R)}}`), mirroring seed.py's GPU/pure split so the module is fully unit-testable
without a GPU.

```
raw_t_to_camera_position(t_raw, R_meta) -> [ax, ay]
    Frame inversion (§3) + round-trip assertion via FGPL's own aligned_meters_to_raw_3d.

build_matches(score_matrix, poses, room_order, R_meta, tau=0.10, guarantee_coverage=True)
        -> (matches: list[dict], admitted_pano_names: list[str])
    Gate (§5) + coverage backstop (§5) + frame-convert each admitted (pano, room).
    room_order: the candidate-room list fixing room_idx; also the room set scanned for the backstop.
    Pure: no file IO, no GPU. This is the unit under test.

write_alignment_json(matches, admitted_pano_names, out_path, extra_meta=None) -> None
    Serialize the exact §2 schema (metadata.pano_names = admitted_pano_names).

export_alignment(score_matrix, poses, room_order, metadata_path, out_path,
                 tau=0.10, guarantee_coverage=True) -> admitted_pano_names
    Convenience wrapper: read metadata.json['rotation_matrix'], call the above, write the file,
    return admitted_pano_names (so the caller can set FGPL cfg["pano_names"]).
```

`room_order` is the caller's `list(candidate_clouds)` (same object passed to `seed.localize_and_score`),
keeping room→index deterministic and decoupling this module from how clouds are loaded.

## 5. Gating + coverage backstop

Both computed from the `score_matrix` alone (fair — no GT, D5). "Score" = `low_percentile_scores` (D30),
lower = better; genuine locks ~0.06–0.08, weak-lock ~0.12+.

- **Per-pano gate (gate & omit).** For each pano: `room = argmin_r score[p][r]`; `winner = score[p][room]`.
  Admit iff `winner <= tau` (default **0.10** — sits in the empirical gap between genuine and weak locks,
  D30/D32). Emit one match per admitted pano. Weak panos are absent from both `matches` and
  `admitted_pano_names`; FGPL never sees them.
- **Coverage backstop (`guarantee_coverage=True`, user-approved 2026-07-15).** After gating, any room in
  `room_order` with zero admitted panos gets its **room-anchored** best pano added:
  `argmin_p score[p][room]` (D32). That pano is a genuine pano of the room by construction — the
  room-anchored contest is 100% correct and loss-sink-immune (D32) — so **precision is not sacrificed**;
  the backstop only prevents silently dropping a room whose every pano fell below the gate. Empirically
  rare (every room had a confident pano at n≤32) but keeps the "every room covered" deployment promise
  literally true.
- **Per-pano uniqueness (FGPL constraint).** FGPL's `load_panorama_positions` builds `positions[pano_name]`
  = one XY per pano (a duplicate `pano_name` silently overwrites, last-wins), so **each pano may appear at
  most once in `matches`.** The gate assigns each pano its winner room. The backstop, for an uncovered
  room, therefore picks the best *unassigned* pano for that room (`argmin_{p not yet assigned} score[p][room]`),
  not simply `argmin_p` — which equals the D32 room-anchored pick in the common all-covered case (a room's
  genuine best pano is abstained-at-that-room, hence free; it is only "taken" when it matches a foreign
  room even better, i.e. the impostor case, where the free-pano fallback is the correct degradation). If no
  free pano remains for a room, it is left uncovered and logged (never a duplicate emission).

Edge cases: empty `score_matrix` → `([], [])`. A room absent from a pano's score dict is treated as
missing/skipped for that pano (defensive; localize_and_score fills all pairs).

## 6. Output contract

- `matches[i]`: `room_idx` = `room_order.index(room)`; `room_label` = picked room; `score` = `winner`
  (or the backstop's room-anchored score); `rotation_deg` = `0.0` (FGPL ignores it and re-searches
  rotation — we do **not** emit CPO rotation, which D25 found convention-unresolved/aliased); `camera_position`
  = `raw_t_to_camera_position(t, R_meta)` where `(t, R) = poses[pano][room]`. `build_matches` expects the
  live tuple form `poses[pano][room] == (t, R)` produced by `seed.localize_and_score`.
- `metadata`: `{ "pipeline": "PanoPin color seed (gated per-pano)", "pano_names": admitted_pano_names,
  "tau": tau, "source": "fgpl_export" }` (plus any `extra_meta`).
- **Deployment contract (documented in the module docstring + PROGRESS):** the orchestrator MUST set FGPL
  `cfg["pano_names"] = admitted_pano_names`. `export_alignment` returns it for exactly this.

## 7. Testing (every task ships a runnable check; show output)

Both suites are **GPU-free** and must be green before "done" (`conda run -n panopin python -m pytest tests/test_fgpl_export.py`):

1. **Pure unit** (`tests/test_fgpl_export.py`, synthetic score_matrix/poses):
   - gating: weak panos (winner > tau) omitted, genuine kept; `admitted_pano_names` matches emitted set.
   - coverage backstop: a room whose only panos are weak still appears in `matches` (and the added pano is
     that room's argmin_p); with `guarantee_coverage=False` it is dropped.
   - schema: every match has the 6 keys with correct types; `room_idx == room_order.index(room_label)`;
     `metadata.pano_names == admitted_pano_names`.
   - **frame round-trip:** for a known yaw `R` (e.g. 37°), assert
     `aligned_meters_to_raw_3d(camera_position, {"rotation_matrix": R}) ≈ t_raw[:2]` within 1e-6; and that a
     non-yaw `R` trips the assertion (fails loud).
2. **Integration smoke** (`tests/test_fgpl_export_smoke.py` or a `smoke/` script, uses cached D33 data, no
   GPU): load `experiments/fgpl_seed/work/seeds/largeval_residuals.json` (→ `low_percentile_scores`) and
   `.../largeval_cache.json` (→ per-(pano,room) `t,R` under `cache[pano]["poses"][room]`, stored as
   `{"t","R"}` dicts — the test adapts them to the `(t, R)` tuple form `build_matches` expects), build a
   real `demo6_alignment.json` over an 8-room `room_order`, then **load it back through FGPL's own
   `load_panorama_positions`** (import the function; identity metadata → raw==aligned) and assert the
   recovered positions equal the cached `t[:2]` for every admitted pano, and that every room is covered.
   This exercises FGPL's real entry point without its GPU search.

## 8. Fast-follow (explicitly out of this spec)

**Live FGPL GPU round-trip.** Run FGPL `multiroom_pose_estimation` on a PanoPin-seeded `demo6_alignment.json`
for a couple of Area_3 rooms (needs `3d_line_map.pkl` + `panopin-gpu`), compare the refined pose to S3DIS
GT. D25 already showed a correct-room color seed → oracle-quality FGPL pose, so this confirms *plumbing*
across the two repos, not the method. It is the natural predecessor to the Electron wiring and gets its own
task once the builder + §7 checks are green.

## 9. Success criterion

`fgpl_export` produces a schema-exact `demo6_alignment.json` in which (a) every admitted pano's
`camera_position` round-trips to its raw `t[:2]` through FGPL's own converter, (b) only confident panos are
admitted (gate) yet (c) every candidate room is covered (backstop), (d) `admitted_pano_names` is returned
for `cfg["pano_names"]`. Verified by the two GPU-free suites in §7 on synthetic data and real cached D33
data. This is the "wire `seed.seed_rooms` into the FGPL runner input" NEXT step from D33, correctly
re-scoped from per-room to per-pano after the 2026-07-15 consumer trace.
