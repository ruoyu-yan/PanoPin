# PanoPin — Handover

_Last written: 2026-07-20, end of the Scan2BIM-localization-unit (T7) session. This is current —
not the stale 2026-07-10 version that used to live here._

## DO THIS FIRST: merge `feat/localization-unit` into `s3dis-eval`

T7 (Scan2BIM localization unit — Stage 0: PanoPin-seeded, prior-enabled FGPL, inside Point_360) is
**built, fully accepted (5/5 criteria), and whole-branch-reviewed READY TO MERGE = YES.** It has not
been merged or pushed. That is the only thing left. Do it before starting any new work.

The work lives in a **separate git worktree**, `/home/ruoyu/Point_360-localization`
(branch `feat/localization-unit`), left in place on purpose so this merge can happen from the main
checkout without juggling branches in one working tree:

```bash
cd /home/ruoyu/Point_360          # main checkout, currently on s3dis-eval @ ccd6510
git merge feat/localization-unit
git push origin s3dis-eval
```

**Expect a clean fast-forward, not a real merge.** `s3dis-eval` has not moved since it was last
pushed (`ccd6510`), and `feat/localization-unit`'s tip already contains `ccd6510` as an ancestor —
verified directly: `git merge-base --is-ancestor ccd6510 <feat/localization-unit tip>` → true. The
reason it contains it: partway through T7, a **sibling session** finished its own work
(`feat/segbackend-instances`, 4A.2a per-instance segmentation) and merged it into `s3dis-eval`
(`ccd6510`); this session then merged `s3dis-eval` into `feat/localization-unit` (merge commit
`90f7b3d`) to pick that up — zero conflicts, zero file overlap (`comm -12` on the two branches'
changed-file lists was empty; two sessions worked the same repo for hours via worktree isolation
without interfering). So `git merge feat/localization-unit` from `s3dis-eval` today has nothing left
to reconcile — it should just move the ref forward.

If it is *not* a fast-forward, or `git status` shows anything unexpected, **stop and diagnose before
forcing anything** — that would mean an assumption above is wrong. One known, harmless bit of noise:
`/home/ruoyu/Point_360/data/localization/` (Stage-0 acceptance-run outputs) is **untracked and not
gitignored** in the worktree — it will not affect the merge (untracked files are per-worktree), but
decide whether to `.gitignore` it or commit a trimmed version before it accumulates further.

After the merge lands and is confirmed on GitHub: `feat/localization-unit` can be deleted (same
convention `feat/segbackend-instances` followed after its merge). Do not delete it before confirming
the push. The worktree `/home/ruoyu/Point_360-localization` itself can then be removed with
`git worktree remove` — not required immediately, no rush.

## What T7 actually built, in one paragraph

Point_360 gained a **Stage 0**: `run_localization.py` + `localization/` (`preflight.py` = Gate 1 —
fails closed unless the FGPL submodule is initialized, the upright prior is present *and enabled*,
and PanoPin's determinism fix is present; `gate_reprojection.py` = Gate 2 — a reprojection-error
blunder detector, PASS/FAIL, not an accuracy metric; `fgpl_pose_adapter.py` — pure FGPL-output →
`{pano:{R,t}}` pose-contract conversion, `axis_fix = identity`, measured as the global optimum over
all 24 signed-permutation rotations). PanoPin gained **its first CLI**, `src/panopin/cli.py`
(`seed_from_clouds`), the entry point Stage 0 calls. Two submodules, pinned and tagged (below).

## Acceptance results (measured, `localization/ACCEPTANCE.md` in the Point_360 repo — read that file
for full numbers and caveats; do not requote figures from memory)

Area_3, strictly-Manhattan 5-room pool (`office_5`, `hallway_1`, `lounge_1`, `conferenceRoom_1`,
`WC_1`), 22 panoramas, FGPL inputs rebuilt fresh from the raw S3DIS `.txt` clouds (not cached).

| # | Criterion | Result |
|---|---|---|
| 1 | Reproduce D36 | PASS — trans median 0.089 m vs 0.084 m target |
| 2 | Gate 1 fails closed | PASS |
| 3 | Gate 2 reprojection | PASS — 59.0 px (threshold 100) |
| 4 | TMB smoke test unchanged | PASS — exact baseline match |
| 5 | Stages 2–4 end-to-end | PASS — schema intact, no crashes |

**Read this as "reproduces D36", not "beats D36."** Rotation median (0.76° vs 1.92°) and room count
(19/22 vs ~18/22) look favorable but rest on single medians at n=22 with no dispersion — one good
draw is not evidence of superiority, especially given the next section. Mean translation is 3.49 m
against the 0.089 m median: bimodal, dominated by 3 wrong-room panos (`1a557181`, `f0e54fcd`,
`da0bb9ad`), not a uniform improvement.

## The determinism finding — the real headline of this session (now **D37** in `docs/DECISIONS.md`)

The *first* acceptance run FAILED criterion 1 at **0.639 m** (7.6x worse than target) and Gate 2
(**261.6 px**, threshold 100). Root cause, bisected not guessed: `data_utils.read_txt_pcd` draws an
`np.random` permutation whenever `sample_rate>1` (the deployed config uses 30), and multi-threaded
float reduction order adds a second, larger wobble. **`src/panopin/determinism.py::pin()` already
fixed both, and every `experiments/fgpl_seed/*` script called it — but nothing under
`src/panopin/` did.** The shipped path (`cli.py` → `seed.localize_and_score` → `cpo_adapter`) ran
completely unpinned; the experiment path that produced every prior D30-D36 headline was pinned all
along. **Fixed in PanoPin commit `be674b9`** (`pin()` at the top of `seed.localize_and_score`).
Re-running acceptance with the fix: 0.639 m → 0.089 m, Gate 2 FAIL → PASS.

This corrects **D1** (2026-07-08), which recorded CPO as inherently non-reproducible on CPU and
deferred the problem — that characterization was inaccurate; the fix already existed and simply
never reached the deployable unit. D1's *preference* (deterministic-first) still stands.
**Cost:** `pin()` sets `torch.set_num_threads(1)`, so seeding is now single-threaded — determinism
is bought with wall-clock, not free.

## Whole-branch review found 2 Criticals — both closed

A per-task review structurally cannot see cross-task integration bugs; the final whole-branch review
(before the "READY TO MERGE = YES" verdict) found two:
- **C1:** `external/PanoPin` was pinned at `9a89ba8` — *before* the determinism fix — and `be674b9`
  had not been pushed yet. A fresh clone would have silently reproduced the 0.639 m / Gate-2-FAIL
  numbers while `ACCEPTANCE.md` claimed 0.089 m. **Closed:** PanoPin pushed, re-pinned to `be674b9`.
- **C2:** re-pinning the FGPL submodule (`acdc812` → `528061b`) was not a fast-forward — it silently
  reverted a "preserve Point_360 native setup patches" whitespace fix applied via `git apply`.
  `acdc812` was on no branch at all (reachable only via `.gitmodules`, one `git gc` from being lost).
  **Closed:** FGPL re-pinned to `cff1cab` (= `528061b` with `acdc812`'s patch cherry-picked back in),
  pushed as `feat/upright-prior-p360` on the fork (non-destructive to the original branch).
- Also disclosed (Important, not Critical): `ACCEPTANCE.md` §3's BIM-reconstruction table was
  produced from a **hand-remapped pose file**, not the shipped CLI's direct output (keys differ —
  bare-uuid vs `camera_<uuid>_<room>` — `run_localization.py`'s own docstring says it does not rewrite
  keys on either arm). §3 now discloses this manual step explicitly; it is not yet a one-command
  reproduction.

## Current state of both repos

**Point_360** (worktree `/home/ruoyu/Point_360-localization`, remote
`github.com/ugurfeyzullah/Point_360`):
- `feat/localization-unit` @ `d93d704`, 13 commits ahead of `s3dis-eval`, **not merged, not pushed**
  (no upstream tracking branch set).
- `s3dis-eval` @ `ccd6510` (main checkout `/home/ruoyu/Point_360`), in sync with
  `origin/s3dis-eval`, **untouched by T7** — will fast-forward on merge (see above).
- `origin/main` @ `b191399`, untouched, unrelated to this work.
- Two pinned submodules on `feat/localization-unit`:
  - `DavidThesis/scan2measure-webframework` @ `cff1cab` (fork `ruoyu-yan/scan2measure-webframework`),
    tagged `p360-pin-2026-07-20` — tag pushed to the fork.
  - `external/PanoPin` @ `be674b9` (fork `ruoyu-yan/PanoPin`), tagged `scan2bim-pin-2026-07-20` —
    tag pushed to the fork.
  - Both tags exist specifically so deleting `feat/localization-unit` later cannot orphan either pin.

**PanoPin** (this repo, `/home/ruoyu/PanoPin`):
- Branch `docs/scan2bim-localization-spec` @ `be674b9`, in sync with
  `origin/docs/scan2bim-localization-spec` (pushed). `main` untouched by this work.
- Whether/when to merge this branch into PanoPin's own `main` is an **open decision**, separate from
  the Point_360 merge above — not addressed this session; the Point_360 pin already points at the
  exact commit (`be674b9`) regardless of PanoPin's own branch topology.

## What is verified vs still open

**Verified this session:** Gate 1 fails closed (tested against a real pre-fix commit from git
history); Gate 2's absolute floor (`n_scored>=3`) and real controls (transposed R, t-rotated,
yaw-sweep) actually trip; `axis_fix=identity` is the brute-forced global optimum over all 24 signed
permutations (88° clear of runner-up); the determinism fix is exercised by a real reproducibility
test (`tests/test_determinism.py`), not a synthetic toy; the TMB smoke test byte-matches the
pre-existing baseline; Stages 2–4 run to completion on estimated poses with schema intact.

**Still open, honestly, per `ACCEPTANCE.md` §4:**
- Area_3 only, Manhattan-filtered, n=22. **Cross-area generalization is untested.**
- The density-image front-end (`R != I`) is **never exercised** — `fgpl_export.raw_t_to_camera_position`'s
  round-trip guard exists for that case and has no test coverage of it actually firing.
- **Gate 2 is a blunder detector, not an accuracy metric** — its median-of-medians absorbs outliers
  by design; a PASS means "no convention error", not "these poses are good." Criterion 1 measures
  quality.
- `docs/tasks.json` T9 (room-scoped line filtering — keep the per-pano seed's position pin while
  giving FGPL whole-room geometry) and T10 (ship the rotation-margin confidence signal) remain open,
  unrelated to T7 and not blocking the merge.
- A **doc defect** was found in `docs/point360_env.md`: its literal Stage-4 command passes
  `--no-require-cloud-support`, which fails closed on wall/door for *any* poses, GT included. Not
  fixed this session (out of scope for T7); tracked in Point_360's own `roadmap.md` §8.

## Read order for a cold-start next session

1. This file (you're reading it).
2. `docs/PROGRESS.md` top block (this session's entry).
3. `docs/DECISIONS.md` **D37** (this session), then **D36** for the Manhattan-demo context it builds on.
4. `docs/tasks.json` — **T7** (do the merge; then it can flip to `done`), **T9**/**T10** (next real work).
5. `/home/ruoyu/Point_360-localization/localization/ACCEPTANCE.md` (read-only source of every number
   quoted above).
6. `.superpowers/sdd/progress.md` — the full task-by-task ledger, if you need more detail than the
   summaries above.
