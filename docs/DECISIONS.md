# PanoPin — Decision log (ADR-style)

_Append-only. One entry per non-trivial choice: the decision + the why. Newest at bottom._

## D1 — Deterministic / simplicity-first (2026-07-08)
Prefer rule-based / geometric methods; use learning only if a deterministic route demonstrably
can't hit the bar. **Why:** PanoPin is a small, fast unit in a larger app — speed, ease of
execution, and maintainability matter more than squeezing the last point via a trained model.

## D2 — Fixed target = pano→room accuracy (primary) + coarse-pose error (secondary) (2026-07-08)
Scored against S3DIS GT via the `eval/` harness. **Why:** directly answers "are panos assigned to
the right place?"; room accuracy is method-agnostic and unambiguous; coarse pose is the FGPL seed.

## D3 — PanoPin outputs a COARSE seed; FGPL does fine localization (2026-07-08)
Scope boundary: PanoPin = room assignment (+ optional coarse pose), handed to FGPL
(`scan2measure-webframework/src/pose_estimation/`). **Why:** keeps PanoPin small; FGPL already
solves fine pose once seeded with the right room + rough position.

## D4 — Dev / eval dataset = S3DIS Area_3 (2026-07-08)
85 panos / 21 rooms with room + pose GT already on disk; Original cloud ↔ pano frame = identity.
**Why:** a real multi-room building carrying the exact GT PanoPin needs; matches Point_360.

## D5 — Fairness: solvers see only anonymized manifests (2026-07-08)
The room name leaks via pano filename / pose JSON / `camera_to_room.json` — all GT.
`eval/make_manifest.py` exposes UUID-only panos + candidate room clouds; solvers must not read GT
paths. **Why:** otherwise a solver "wins" trivially by string-parsing, and the metric is meaningless.

## D6 — Harness is stdlib-only (2026-07-08)
No numpy / yaml in `eval/`. **Why:** portable, fast, runs in any Python 3 regardless of whatever
env the solver ends up needing.

## D7 — Method route is OPEN (2026-07-08)
Feature matching with the thesis extractors is one documented option, not a mandate; the agent
researches and chooses. **Why:** user's explicit instruction — avoid premature commitment.

## D8 — Decider = appearance/content (CPO render-and-compare), not shape (2026-07-08)
Room selection is decided by spatially-resolved color-consistency (CPO), not room shape/geometry.
**Why:** shape/line geometry cannot disambiguate rooms of identical shape — the exact case PanoPin
exists to solve and where the thesis' jigsaw fails; the Area_3 EDA confirmed ~7 identical-shape pairs.
Content (posters, furniture, windows), carried in the candidate clouds' RGB, is the discriminating
signal; global color is degenerate but per-bearing appearance matching is not. Supersedes the
short-lived geometry-first proposal. See `specs/2026-07-08-coarse-pano-to-room-options.md`.

## D9 — Build upon CPO by composing its primitives (2026-07-08)
Reuse CPO from `82magnolia/panoramic-localization` (Apache-2.0) by calling its building-block functions
(`color_utils`, `utils`, `sampling_loss`, `data_utils`) from our own orchestrator that RETURNS
`(t, R, loss)`; do not fork/patch its entry point (`localize_single.localize` returns `None`).
**Why:** minimal, self-contained, reproducible; avoids maintaining a fork; the primitives are the
validated core. Cost for room ranking = min `refine_pose_sampling_loss` loss.

## D10 — Dedicated `panopin` CPU env (2026-07-08)
New conda env: python 3.8, torch 1.10 (CPU), numpy~1.23, opencv, pandas, scipy, scikit-learn, Pillow;
drop tensorflow-cpu / open3d / pylsd-nova / einops. **Why:** CPO needs torch (CPU ok, training-free);
a fresh env keeps PanoPin self-contained and reproducible (matches the parent project's fresh-env
pattern), rather than coupling to the thesis' `scan_env` with its unspecified torch.

## D11 — Two-tier funnel for CPU tractability (2026-07-08)
Rank all candidate rooms by a cheap spatial score (coarse histogram/score-map, no Adam), then run CPO's
Adam refinement only on the top-k rooms. **Why:** full CPO per room (score-maps + 6×100 Adam) across
~21 rooms on CPU is expensive; the cheap rank preserves same-shape candidates while bounding refine cost.
