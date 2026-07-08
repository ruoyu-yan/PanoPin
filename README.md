# PanoPin

**Coarse panorama → room registration to bootstrap FGPL on multi-room scenes.**

A short, deterministic, image-based coarse step that tells FGPL *which room each panorama
belongs to and roughly where* — so FGPL's fine localization works on large buildings with
many similar rooms, the case where the jigsaw polygon matching from the scan2measure thesis
breaks down.

> **Developing PanoPin (human or agent)?** Start with [`CLAUDE.md`](CLAUDE.md) (the operating
> charter), then [`docs/PROGRESS.md`](docs/PROGRESS.md) for live state and
> [`docs/tasks.json`](docs/tasks.json) for the backlog. The evaluation harness is in
> [`eval/`](eval/); best-practice sources in [`docs/best-practices-references.md`](docs/best-practices-references.md).

---

## Status

- **2026-07-08:** project seeded. Direction agreed in the Feyzullah meeting; **method design
  not started.** Next step is a brainstorming pass (`superpowers:brainstorming`) → spec → plan.
- No code yet.

## Why it exists

The parent project **Point_360 / Scan2BIM** (`/home/ruoyu/Point_360/`) ships as an **app**:
point cloud + panoramas → BIM model. That app must know each panorama's camera pose.

- The thesis (**scan2measure**, `/home/ruoyu/scan2measure-webframework/`) estimates pose with
  **jigsaw polygon matching → FGPL** (Fully Geometric Panoramic Localization). Jigsaw
  enumerates pano→room assignments and optimizes IoU — it degrades badly on large, repetitive
  layouts (e.g. S3DIS Area 3: many near-identical offices → ambiguous and combinatorial).
- FGPL itself is strong at *fine* localization once it knows the rough room + pose. The missing
  piece is a robust, deterministic **coarse** step to seed it. **That is PanoPin.**

## The idea (to be refined in design)

Given the panoramas + the multi-room point cloud / floor plan:

1. Coarsely assign each panorama to a room and a rough pose — image-based and deterministic,
   with no combinatorial jigsaw search.
2. Hand that seed to **FGPL** for fine refinement.

Target: reliable multi-room localization end-to-end, replacing jigsaw's coarse role.

## Fallback (if PanoPin doesn't pan out)

For the Scan2BIM demo, fall back to a **small scene with distinct room shapes** where the
thesis' jigsaw matching already works, plus the **manual-checkpoint** (a human confirms each
pano is assigned to the correct room). PanoPin is the "make it scale" upgrade over that path.

## Relationship to other repos

- **Point_360** — `/home/ruoyu/Point_360/` — parent app + evaluation; consumes the pose unit.
  See its `roadmap.md` §"Scope update — 2026-07-08" for how PanoPin fits the product.
- **scan2measure-webframework** — `/home/ruoyu/scan2measure-webframework/` — thesis code;
  FGPL + jigsaw live here (`src/pose_estimation/`, `src/floorplan/align_polygons_demo6.py`).
  PanoPin replaces jigsaw's coarse role and feeds FGPL.

## Next step

Brainstorm the coarse-registration method: inputs, image features / cues, the room-assignment
strategy, how the coarse seed is handed to FGPL, and how success is measured — then write a
design spec and an implementation plan.
