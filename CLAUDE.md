# PanoPin — CLAUDE.md

**Mission.** Given a set of 360° panoramas + a multi-room point cloud, decide **which room each
panorama is in** and (optionally) a **rough camera pose** — a fast, deterministic COARSE seed
handed to **FGPL** for fine localization. This unblocks pose estimation on large, repetitive
multi-room buildings where the scan2measure thesis' jigsaw matching fails. PanoPin is one small,
fast unit inside the larger Scan2BIM app (`/home/ruoyu/Point_360/`).

**Success = one number:** pano→room assignment accuracy on S3DIS (secondary: coarse-pose error).
The random baseline scores ~6% (chance = 1/23 ≈ 4.3%). Beat it, then push toward ~100% with a
simple, quick-to-run method.

## Your mandate (read first)
- **You own the architecture and the method route.** Nothing here dictates HOW to solve it.
- **Deterministic / rule-based is strongly preferred** over learned methods — this is a small,
  fast cog in a bigger machine. Reach for learning only if a deterministic route demonstrably
  can't hit the bar, and record why in `docs/DECISIONS.md`.
- **Research before you build.** Survey several approaches (image↔cloud feature matching,
  geometric / floor-plan reasoning, retrieval, projection consistency, …). Feature matching using
  the thesis' pano and point-cloud feature extractors (`/home/ruoyu/scan2measure-webframework/`)
  is ONE option and a ready reference — not a mandate. Don't over-fit to a single idea.

## Non-negotiable operating protocol
Grounded in `docs/best-practices-references.md` (verified Anthropic sources).

1. **Every session starts with the ritual:** read `docs/PROGRESS.md` + `docs/DECISIONS.md` +
   `docs/tasks.json`, run `git log --oneline -15`, run `python tests/test_harness.py`, then pick
   the next open task. Only then write code.
2. **Design before implementing a method.** `superpowers:brainstorming` → write the spec to
   `docs/specs/YYYY-MM-DD-<topic>-design.md` → `superpowers:writing-plans` → `docs/plans/`.
   Do NOT implement a new method before its design is written and committed.
3. **Document as you go (context dies between sessions; artifacts don't):** update
   `docs/PROGRESS.md` at the END of every session (what changed, what's next, where you stopped);
   append to `docs/DECISIONS.md` for any non-trivial choice (with the why); keep
   `docs/tasks.json` truthful — flip a task's `"passes"` to `true` ONLY after its `verify`
   command runs green.
4. **Every task ships a runnable check; show the output.** Nothing is "done" without evidence.
   Score methods with the harness (below). Don't assert success — prove it.
5. **The harness and GT are sacred.** Never edit `eval/` metrics or S3DIS GT to make a number
   look better. Fix the method, not the ruler.
6. **Prefer sub-agents** (`superpowers:dispatching-parallel-agents`, the read-only `Explore`
   agent) for research, search, and verbose work — brief them self-contained, pull back a short
   summary. Run an adversarial review of your diff against the task's requirements before "done".
7. **Use any skills/tools that fit** (brainstorming, writing-plans, systematic-debugging, …).
   Package repeatable procedures as project skills in `.claude/skills/`, not as CLAUDE.md bloat.
8. **Commit often** on a feature branch, with descriptive messages.

## The fixed target — the harness (`eval/`)
Method-agnostic scorer. Stdlib-only (runs in any Python 3; `python` here = miniconda 3.13).
```
python eval/make_manifest.py --area Area_3            # anonymized solver INPUT -> runs/manifest_Area_3/
python eval/baselines/random_room.py --area Area_3    # -> runs/random_room.json (the floor)
python eval/score.py --predictions runs/random_room.json --area Area_3
python tests/test_harness.py                          # harness sanity (no data needed)
```
- **Solver contract:** read only the manifest (anonymized panos + candidate room clouds); emit a
  predictions JSON per `eval/PREDICTIONS_SCHEMA.md`; score it with `eval/score.py`.
- **FAIRNESS (critical):** the room name is embedded in the raw pano filename and pose JSON — that
  is GT. A solver must NEVER read room/pose from filenames, `camera_to_room.json`, or the pose
  files. Work only from the anonymized manifest; `make_manifest.py` enforces this.

## Data (see `config/datasets.json`)
- **S3DIS Area_3** — 85 panos / 21 rooms (of 23) with pose + room GT (dev + eval set).
  - panos + pose GT: `/mnt/d/.../scan2measure-webframework/data/area_3/pano/{rgb,pose}` (`/mnt/d`
    is the slow 9p bridge — stage a working subset to ext4 for tight loops).
  - room point clouds: `…/Point_360/data/s3dis/Stanford3dDataset_v1.2/Area_3/<room>/<room>.txt`
    (fast ext4). Format `X Y Z R G B`; RGB is float in the Original version.
- **Frame:** Original S3DIS cloud ↔ pano/pose frame = identity (~5 mm), per Point_360.
- **Gotchas:** `pointcloud.mat` is v7.3 HDF5 (slow/flaky on /mnt/d); hallway_5 & storage_2 have no
  pano; a room's trailing `_<area>` is stripped (`office_3_3` → `office_3`).

## Layout
- `CLAUDE.md` (this) · `README.md` (human overview)
- `docs/` — `PROGRESS.md`, `DECISIONS.md`, `tasks.json`, `best-practices-references.md`,
  `specs/`, `plans/`
- `eval/` — the harness · `config/datasets.json` — data paths
- `src/panopin/` — the package you build · `tests/` · `runs/` (gitignored outputs)

## Related repos
- `/home/ruoyu/Point_360/` — parent Scan2BIM app + eval (consumes this pose unit; see its
  `roadmap.md` §"Scope update — 2026-07-08").
- `/home/ruoyu/scan2measure-webframework/` — thesis code: **FGPL** (`src/pose_estimation/`) that
  PanoPin feeds, and the pano / point-cloud feature extractors (optional feature-matching reference).
