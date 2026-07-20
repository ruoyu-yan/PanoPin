# Scan2BIM localization unit — design

**Date:** 2026-07-20
**Status:** design approved, plan not yet written
**Supersedes the framing of:** T7 "Option B — production/Electron pipeline wiring", which named the wrong target repo

## 1. Context

PanoPin is developed **for the Scan2BIM pipeline (`/home/ruoyu/Point_360`)**, not for
scan2measure. Earlier PanoPin documents (D35/D36 "NEXT" lines, T7's note in `docs/tasks.json`)
describe the deliverable as wiring into scan2measure's Electron app. That target is wrong and is
corrected here.

The goal, in the user's words: **FGPL + upright rotation prior + PanoPin wired together as one
bigger unit inside the Scan2BIM pipeline.**

### 1.1 What already works

- **The method is proven.** D36 (strictly-Manhattan, 5 rooms / 22 panos, Area_3): same map, same
  panos, same estimator, only seeding changes. FGPL alone = 1/5 rooms, 82% wrong-room, 13.27 m.
  FGPL + PanoPin + prior = 5/5 rooms, 18%, **0.084 m**. Control: the prior *without* PanoPin =
  13.32 m, i.e. no help — the prior fixes orientation, PanoPin fixes placement.
- **The interface contract is proven.** `src/panopin/fgpl_export.py` writes a schema-exact
  `demo6_alignment.json`, verified through FGPL's own `load_panorama_positions` to 1e-6
  (`tests/test_fgpl_export_smoke.py`, passed not skipped), plus two end-to-end estimator runs
  (D35 round-trip 10/10; D36 Manhattan).

What is missing is **packaging, not correctness**.

### 1.2 Findings that shape this design

Four facts established by inspection on 2026-07-20:

1. **Pose estimation *is* part of Scan2BIM.** `DavidThesis/ruoyu_thesis.txt:1388-1500` (Ch. 5)
   documents the algorithm — 26 perspective views + LSD, sphere back-projection, principal-direction
   clustering, 24 rotation candidates, translation grid, LDF+PDF cost, two-phase spherical ICP,
   ranked by tight inlier count below 0.1 rad with ties on average sphere distance (`n_tight` /
   `avg_dist`). Line 1467: *"an independent reimplementation of the FGPL algorithm."* This is
   Feyzullah's own work, part of this project — **there is nothing third-party to vendor.**

2. **But no pose-estimation code is in the Point_360 tree.** Verified across all five branches
   (`main`, `s3dis-eval`, `feat/s3dis-pose-adapter`, `feat/segbackend-instances`,
   `smoke-test-tmb`): FGPL appears only in prose (`ruoyu_thesis.txt`, `paper/references.bib`,
   `paper/sections/Related_Work.tex`). `local_filter_results.json` is only ever *read*, never
   written. The code lives in scan2measure, declared as a submodule at
   `DavidThesis/scan2measure-webframework` — **which is uninitialized, so the directory is empty.**

3. **The submodule points at the wrong remote.** `.gitmodules` declares
   `github.com/ugurfeyzullah/scan2measure-webframework` (upstream). The working clone at
   `/home/ruoyu/scan2measure-webframework` has origin `github.com/ruoyu-yan/scan2measure-webframework`
   — **the user already has a fork.** The pinned commit `acdc812` does not exist in that fork
   (`git cat-file`: could not get object info). So `git submodule update --init` today would fetch
   upstream code containing neither the upright prior nor the PanoPin narrowing work.

4. **Env boundaries are irreducible.** `point360` is py3.12 / torch 2.7; `panopin-gpu` is py3.8 /
   torch 2.0.1+cu118 (pinned by CPO); `scan_env` is torch 1.12+cu116 / open3d 0.19. No shared
   process is possible. Point_360 already solves this: `run_scan2measure_3d_lines.py:186` shells
   into another env via `conda run`. Relocating code does not remove this boundary — it changes
   ownership and reproducibility only.

### 1.3 The pose contract

Stages 2 and 4 consume one file:

```json
{ "<pano_name>": { "R": [[3x3]], "t": [x, y, z] } }
```

`R` is world→camera, `t` is the camera centre in world coordinates
(`project_newdata_semantic_masks.py:74-81`). **Only `R` and `t` are ever read**; FGPL's
`n_tight`, `avg_dist`, and `n_matched` are discarded — there is no confidence gating anywhere in
the pipeline. The pose JSON's keys also serve as the pano manifest
(`pano_mask_config.py:20-31`). Reference producer: `s3dis/pose_adapter.py:35-54`.

## 2. Goals and non-goals

### Goals

- A **Stage 0** in Point_360 that produces the pose contract from PanoPin + FGPL + prior.
- FGPL-with-prior present **in the repo as a pinned dependency**, not as a property of one
  machine's checkout state.
- PanoPin cited by Point_360 as a **pinned submodule**.
- A `--pose-source gt|panopin` seam, so swapping GT for estimated poses is a flag.

### Non-goals

- **Changing the paper's evaluation.** S3DIS dev/eval continues to feed GT poses; pose accuracy is
  not a metric of this paper (`roadmap.md:16-25`). The seam exists so a GT-vs-PanoPin A/B is
  *possible* later, not so it happens now.
- **T10 (rotation-margin confidence signal).** Deferred; §6.3 captures most of its practical value
  for near-zero cost.
- **Cross-area validation.** Everything remains Area_3. Tracked separately.
- **Building line maps from a raw product cloud.** Supported but not exercised; see §7.1.

## 3. Architecture

```
Point_360  (main project, branch feat/localization-unit off s3dis-eval)
├── .gitmodules
│   ├── DavidThesis/scan2measure-webframework → ruoyu-yan/… @ pin containing prior   [FGPL unit]
│   └── external/PanoPin                      → ruoyu-yan/PanoPin @ pin             [helper]
├── run_localization.py        ← Stage 0 CLI, peer of the four existing stage scripts
└── localization/
    ├── preflight.py           ← capability + pin verification
    └── fgpl_pose_adapter.py   ← local_filter_results.json → {pano:{R,t}}, pure
```

Poses feed Stages 2 and 4, so the unit is **Stage 0**. Everything downstream is unchanged and
unaware of which pose source ran.

Both dependencies become **pinned submodules** so a fresh clone reproduces the whole unit. FGPL is
a re-point plus a pin (§4.1); PanoPin is new. `external/PanoPin` is proposed rather than
`DavidThesis/` because the latter is thesis-specific.

### 3.1 Components

| Component | Env | Purpose |
|---|---|---|
| `run_localization.py` | point360 | Orchestration only: preflight, dispatch, `conda run`, postflight. Deliberately thin — subprocess code resists testing. |
| `localization/preflight.py` | point360 | Verifies submodules initialized, FGPL pin contains the prior, prior enabled in config, envs exist, inputs resolve. Pure logic over inspected state. |
| `localization/fgpl_pose_adapter.py` | point360 | `local_filter_results.json` → `{pano:{R,t}}`. **Pure function** — no subprocess, no torch. Where the coordinate convention lives. Sibling of `s3dis/pose_adapter.py:35-54`. |

The thin-CLI / pure-adapter split exists because the error-prone part is the convention math, and
it must be testable without spinning up two conda envs.

### 3.2 PanoPin's side

Stage 0 must **not** call `experiments/fgpl_seed/*` — that is experiment code with hardcoded paths
and cached-grid assumptions. PanoPin's shipped, tested API is `seed.seed_rooms` +
`fgpl_export.export_alignment`. This design adds a thin documented CLI wrapper over them, which is
PanoPin's public interface and is overdue independently of this work.

## 4. Data flow

```
scene work dir (room .ply clouds + rgb/ panoramas)
   │
   ├─ preflight ─────────────────────────── GATE 1 (fail closed)
   │
   ├─ [--pose-source gt]  → pose_adapter.convert_room(axis_fix=C) ─┐
   │                                                               │
   └─ [--pose-source panopin]                                      │
        ├─ (a) line maps + sphere features        [scan_env]       │
        │       …or reuse cached artifacts                         │
        ├─ (b) PanoPin seeding                    [panopin-gpu]    │
        │       low-pct scores → room-anchored seeds               │
        │       → demo6_alignment.json + admitted pano list        │
        ├─ (c) FGPL estimator, prior ON           [panopin-gpu]    │
        │       cfg["pano_names"] = admitted → local_filter_results.json
        └─ (d) fgpl_pose_adapter                  [point360, pure] │
                                                                   │
   ┌───────────────────────────────────────────────────────────────┘
   ├─ postflight ──────────────────────────  GATE 2 (fail closed)
   └─ {pano: {"R": 3x3, "t": [3]}}  →  Stages 2 and 4
```

Both arms converge on the identical contract file; downstream cannot tell them apart.

### 4.1 Settling the FGPL source of truth

1. Merge `feat/upright-rotation-prior` (4 commits, incl. `b12ec86`, the load-bearing canonical-frame
   fix) into `ruoyu-yan/scan2measure-webframework`'s `main`, so the pin lands on a stable commit.
2. Re-point `.gitmodules` URL from `ugurfeyzullah/…` to `ruoyu-yan/…`.
3. Pin to the merged commit and initialize the submodule.

This makes "FGPL + prior, inside Point_360" true in the repo rather than true only on one machine.

## 5. Gates and error handling

Fail-closed and loud throughout. This is consistent with fixing a known weakness: `roadmap.md:469-490`
already records `--no-require-cloud-support` failing closed **silently** as an open defect.

### 5.1 Gate 1 — prior presence (preflight)

Checks **two** things, because either alone fails silently:

1. The pinned FGPL commit contains the prior implementation.
2. The run config **enables** it.

The second matters because the prior ships **opt-in, default off** — deliberately, to leave the
single-room pipeline and TMB untouched. A correct pin with a default config yields FGPL enumerating
all 24 rotation candidates instead of 4, poses at ~0.96 m or flipped instead of 0.084 m, and a
valid-looking BIM. Gate 1 makes that state unreachable by accident.

### 5.2 Gate 2 — reprojection (postflight)

Runs the `s3dis/validate_projection.py` check on the unit's output before Stage 2 consumes it.

This catches the one untested combination. Two paths are each proven separately — FGPL poses →
Point_360 (TMB frame, raw `R`/`t`, no `axis_fix`) and S3DIS poses → Point_360 (GT via
`convert_room(axis_fix=C)`, frame gate PASS) — but **FGPL-estimated poses in the S3DIS frame have
never been fed to Point_360.** The concrete reason they might differ: D25/D36 ran with an identity
`metadata.json` (raw S3DIS frame, no density-image transform), whereas TMB's poses came through
real metadata with `R≠I`. `fgpl_export.raw_t_to_camera_position` already carries a fail-loud guard
for exactly that case.

`roadmap.md:464-467` flags this as risk hotspot 1.3: get the convention wrong and masks land on the
wrong 3D points **with no error**.

**Stated limit:** Gate 2 needs GT (`global_xyz.exr`), so it runs on S3DIS and *not* on product
scenes like TMB. For the no-GT case the fallback is the weaker round-trip guard inside
`fgpl_export.raw_t_to_camera_position`. This is written down rather than left implicit so the gate
is not mistaken for being general.

### 5.3 Quality sidecar

The adapter preserves FGPL's `n_tight`, `avg_dist`, and `n_matched` into a **sidecar** file, keeping
the pose JSON byte-compatible with `load_pose`. Weak poses become visible. This captures most of
T10's practical value at near-zero cost.

## 6. Testing

Point_360 convention: plain-assert scripts run directly with `python`, tests beside their subjects,
ending in an `if __name__ == "__main__":` block printing `ALL PASS`. No pytest.

- **`test_localization_preflight.py`** (no GPU) — four cases: uninitialized submodule detected;
  FGPL pin lacking the prior detected; prior present but config disabled detected; clean state passes.
- **`test_fgpl_pose_adapter.py`** (pure, no GPU) — fixture is a **real cached
  `local_filter_results.json`** from the D36 runs, with assertions on actual numeric values.

The fixture choice is deliberate. The upright-prior unit test used `principal_3d=eye(3)`, which
collapsed the canonical and world frames into one, passed 5/5, and hid the bug entirely until a real
run went 10/22 → 0/22. **A convention adapter tested against identity matrices proves nothing.**

## 7. Acceptance criteria

1. **The unit reproduces D36.** Compare Stage 0's output poses against S3DIS GT on the Manhattan
   pool; PanoPin's harness achieved 5/5 rooms and 0.084 m median. Materially different poses from
   the same inputs mean the wiring is wrong. This isolates integration error from method error
   before any BIM is built.
2. **Gate 1 fails closed, demonstrably.** Pointed at a commit without the prior, the unit refuses
   to run.
3. **Gate 2 passes** on the Manhattan pool output.
4. **TMB smoke test stays green** — `conda run -n point360 python smoke_test/run_smoke.py`,
   unchanged. Stage 0 is purely additive, so this is the no-regression proof.
5. **End-to-end on conferenceRoom_1.** Run Stages 2–4 with estimated poses instead of GT and diff
   the Dynamo artifact against the GT-pose baseline (wall 13 / door 16 / floor 15 / ceiling 15 /
   beam 24 / window 8). Hard requirements: schema intact, no crash. Count and geometry deltas are
   **reported, not thresholded** — fixing a numeric tolerance before the first run would either
   rubber-stamp a regression or cry wolf. A human judges the delta; the spec guarantees it is
   measured and visible.

### 7.1 Scope boundary

FGPL needs line maps and sphere features built from the cloud before the estimator runs (in PanoPin:
`build_linemap.py` / `build_features.py` / `build_ply.py`, under `scan_env`). For the S3DIS Manhattan
acceptance target these artifacts are cached, so Stage 0 **supports** building them but the
acceptance run reuses the cache. Building from a raw product cloud is a known, separately-testable
extension rather than an untested assumption baked into version one.

## 8. Concurrency and branching

Another session is working in `/home/ruoyu/Point_360` on `feat/segbackend-instances` (4A.2a), with
uncommitted changes. Verified state at time of writing:

```
s3dis-eval                548728b   quiescent, == origin/s3dis-eval
feat/segbackend-instances 742e26c   other session, 3 ahead, dirty tree
```

A branch alone is **not** isolation: the repo has one worktree (`git worktree list`), and one
worktree holds one checked-out branch. Checking out a new branch there would yank the other
session's tree out from under uncommitted work.

```bash
cd /home/ruoyu/Point_360
git worktree add ../Point_360-localization -b feat/localization-unit s3dis-eval
```

Branch from `s3dis-eval`, not from `feat/segbackend-instances`, to avoid inheriting in-flight
4A.2a commits.

**Merge back into `s3dis-eval` when complete.** Conflict risk is near zero — file surfaces are
disjoint (this work: `run_localization.py`, `localization/`, `.gitmodules`; theirs:
`seg_backends.py`, `seg_types.py`, `index.html`, `vendor/`, `smoke_test/`).

Note: submodules inside a worktree work but are a known git rough edge — submodule working
directories are per-worktree, so `git submodule update --init` runs inside ours.

Unrelated but worth recording: local `main` (85f35a0) has diverged from `origin/main` (b191399).

## 9. Task order

1. Create the worktree and branch (§8).
2. Merge the prior into the scan2measure fork's `main`; re-point and pin the FGPL submodule (§4.1).
3. Add the PanoPin submodule; add PanoPin's CLI wrapper (§3.2).
4. `localization/fgpl_pose_adapter.py` + its test (§6) — pure, testable first.
5. `localization/preflight.py` + its test, including Gate 1 (§5.1).
6. `run_localization.py` orchestration, including Gate 2 wiring (§5.2).
7. Acceptance runs 1–5 (§7).
8. Merge back into `s3dis-eval` (§8).

## 10. Risks

| Risk | Mitigation |
|---|---|
| Convention mismatch, FGPL-estimated poses in S3DIS frame (silent) | Gate 2; non-identity test fixture |
| Prior silently absent or disabled (11x degradation, no error) | Gate 1, checking both presence and enablement |
| Submodule left uninitialized (today's actual state) | Preflight check fails closed |
| Acceptance target is Area_3, Manhattan-filtered, n=22 | Acknowledged; cross-area tracked separately as T5 |
| Other session's work collides | Worktree isolation; disjoint file surfaces |

## 11. Open question

`external/PanoPin` is proposed as the submodule path. `DavidThesis/` is thesis-specific and
Point_360 has no existing `third_party/` convention. Not blocking — a rename before implementation
is trivial.
