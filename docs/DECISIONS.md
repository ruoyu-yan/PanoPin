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

## D12 — `open3d` cannot be dropped from the `panopin` env (2026-07-08) — corrects D10
D10 assumed `open3d` could be dropped; it cannot. The vendored `third_party/cpo/data_utils.py` imports
`open3d` at module load (line 12), so importing any CPO primitive requires it. The `panopin` env
therefore includes `open3d==0.19.0` (matching upstream CPO's `requirements.txt`). **Why recorded:**
D10 is now factually corrected here (DECISIONS is append-only); `tensorflow-cpu`, `pylsd-nova`, and
`einops` remain dropped as D10 stated (not needed by the CPO color path). Also: `opencv-python` is
unpinned (D10 gave no version; resolved 5.0.0.93) and `pytest` is the test runner.

## D13 — Synthetic box-room test can't validate CPO discrimination; validate on real data (2026-07-09)
The Task-3 synthetic discrimination test (same-shape rooms, spatial-only colour difference) is marked
`xfail`. Empirically (2026-07-08, four fixture configs — 2-wall swap, 4-wall cyclic, 4-wall adjacent-pair
— with `match_color` on AND off) CPO scored the WRONG room lower **every time**, and even the self-match
loss stayed ~0.22 (not ~0): CPO's colour-histogram pose search needs real texture to lock onto a pose,
which a flat-walled box lacks, so room-identity signal sits below the noise floor. Two insights carried
forward: (a) a CYCLIC colour permutation of the walls is geometrically a 90° room rotation, which CPO's
yaw search aligns away — so **rooms related by a rotation are a genuine failure mode on real data too**;
(b) `localize_pair` itself is correct (reviewed; returns finite, sensible values) — this is a
fixture-fidelity limit, not a bug. **Decision:** don't chase a synthetic margin; validate the "content
disambiguates same-shape rooms" claim on REAL S3DIS data at Task 4 (M0 smoke — does CPO localize a real
pano to its real room with low loss + near-GT pose?) and Task 7. **THE top open risk for the project:**
whether CPO's real-data discrimination margin is adequate on Area_3's ~7 near-identical room pairs.

## D14 — D13 validated on real in-frame data: CPO self-localizes AND discriminates (2026-07-09)
Task 4 M0 + discrimination probes on real Area_3 data (`smoke/reproduce_cpo_one_room.py`,
`smoke/discriminate_one_pano.py`) resolve the D13 core claim **for in-frame panos**:
- **Self-localization:** the office_3 pano localizes to its office_3 cloud at **0.039 m** (also
  0.086/0.135 m across reruns) — centimetre-level, frame identity confirmed. What the flat synthetic
  box couldn't do, real texture does.
- **Discrimination:** office_3 pano vs {office_1,2,6,9, hallway_1} → correct room wins, loss **0.1185
  vs 0.21–0.26**, runner-up margin **+0.089 (+75%)**, and it is the ONLY room with a correct recovered
  pose (0.095 m vs 5–17 m). Content, not shape, picks the right room among near-identical offices.
- **A no-match floor exists:** when CPO can't lock on, loss floors ~0.16–0.26 (matches the D13 flat-box
  ~0.22). A genuine match breaks well below (~0.12). This hints at a natural match/no-match threshold,
  usable later for confidence / FGPL hand-off.
**Caveat (n=2, margin is pano-dependent):** a second in-frame query, office_5 vs {office_1,4,7,8,
hallway_2}, PASSES but by only **+0.0056 loss (+3.5%)** over runner-up hallway_2 — office_5 self-
localizes cleanly (0.058 m) but its correct-room loss (0.161) sits near the no-match floor and hallway_2
coincidentally scored 0.167. So across n=2 in-frame: office_3 robust (+75%), office_5 within-noise
(+3.5%). **The method works but the discrimination margin varies a lot per pano; thin-margin panos are
not reliable** under the nondeterminism below.
**Caveat (determinism — now correctness-relevant, not just polish):** CPO is NOT bit-reproducible on CPU
even with `np.random.seed` — torch score-map scatter / Adam vary run-to-run (~±0.02 loss). Since some
correct rooms win by <0.02 (office_5), this noise **can flip the selected room**. Fixing determinism
(torch.manual_seed / use_deterministic_algorithms) and/or using full CPO settings (top_k=6, num_iter=100
→ more converged, lower correct-room loss → wider margin) is now a Task-7 priority, not a deferred item.
**Broader confirmation = Task 7** (all 76 in-frame panos); the split of robust vs thin-margin vs
out-of-frame panos will set the achievable accuracy.

## D15 — Task-5 test strategy: mock the funnel logic, validate discrimination on real data (2026-07-09)
The plan's Task-5 acceptance test reused the synthetic same-shape box fixture, which D13 proved CPO
cannot pass. **Decision:** split the two concerns. (a) The funnel *control flow* (rank all rooms cheap →
refine only top-k → return min-loss) is unit-tested in `tests/test_select_room.py` by MOCKING
`localize_pair` — deterministic, data-free, always runs. (b) *Real discrimination* is validated
end-to-end on real Area_3 rooms via `smoke/select_room_real.py`. **Why:** never gate CI on a test whose
premise is known-false; keep the honest validation where it belongs (real data). Supersedes the plan's
Task-5 Step-1 synthetic test (a callout was added to the plan).

## D16 — Two-tier funnel is correct but not yet cheap; Tier-1 needs a lighter scorer (2026-07-09)
Measured: with stock CPO cfg the per-room cost is dominated by the fixed inlier score-map detection
(make_score_map_2d/3d over ~9360 inlier poses, **~20 s/room on CPU at sample_rate=30**, ~scales with
point count), which runs in BOTH tiers. `num_iter`/`top_k_candidate` add little on top. So the D11 funnel
(Tier-1 cheap rank → Tier-2 refine top-k) currently saves only the small refine cost, NOT the bulk. Full
Area_3 (85 panos × 21 rooms) at ~20 s/pair ≈ **~10 h** — an overnight job as-is. **Decision:** keep the
funnel (correct, and the structure is right) but flag that Task 7 tractability needs a genuinely light
Tier-1 scorer (skip inlier detection; small pose pool / histogram-only) before scaling. Do NOT prematurely
optimize; get the accuracy signal first, then make Tier-1 cheap. Tracked for Task 7.

## D17 — 9/85 Area_3 panos are OUT OF FRAME (camera outside its room cloud) (2026-07-09)
`smoke/check_frame_alignment.py` checked every pano's GT `camera_location` against its room cloud's bbox:
**76/85 (89%) are inside (max offset 0.000 m — identity holds); 9 panos across 4 rooms are OUTSIDE** —
lounge_2 (2.04 m), office_10 (1.54 m), office_9 (0.87 m), office_2 (0.68 m). CPO seeds candidate
translations from the cloud's quantile extent, so a camera outside the cloud **can never be localized**
(office_9 self-error 5.8 m even at full budget top_k=6/num_iter=100 — the failure that first flagged this).
Offsets are sub-metre-to-2 m and partly axis-aligned (office_9 sits inside on Y, just outside on X/Z),
consistent with a small pose↔cloud misalignment or a clipped room cloud — NOT a wrong-room mislabel.
**Implications:** (a) the pose-error metric is meaningless for these 9; (b) room-assignment likely fails
for them (no low-loss match anywhere — probe #2 office_9 confirmed a wrong room won by noise); (c) this
is a DATA/GT limitation, not a CPO method failure, and it **caps achievable accuracy at ~89%** unless the
out-of-frame panos are fixed (recover a per-room offset) or excluded/flagged. **Decision:** do NOT edit
the harness now (charter #5); document it, report Task-7 accuracy split as in-frame (76) vs all (85), and
leave a per-room frame-offset fix as an open investigation. This partly explains why the thesis' jigsaw
needed a manual per-pano checkpoint.

**Follow-up (2026-07-09, prompted by user): are these cameras off the WHOLE cloud or just their room
segment?** `smoke/check_out_of_frame_panos.py` tested each of the 9 against ALL Area_3 room bboxes:
**8/9 fall outside EVERY room's bbox** (nearest room = their OWN labeled room, 0.7–2.0 m away — several at
normal ~1.4 m camera height but beyond the room walls); **1/9** (an office_2 pano) sits inside hallway_2's
bbox (a doorway/boundary case). So for the 8 it is genuinely "camera off the mapped point cloud" — the
user's point: expected, unmatchable, NOT a method failure; the sensible label is still their own room
(it's the nearest), so this reads as a GT pose error or clipped room segment, not a mislabel. **Reframe:**
these 9 are a coverage/GT fact, not a "blocker" — exclude-and-report-separately is the right handling
(the office_2/hallway_2 boundary pano is the only arguably-recoverable one). The METHOD's job is only the
76 in-frame panos.

## D18 — Cheap Tier-1 refuted by the recall gate; full CPO pipeline is required (2026-07-09)
The cheap-Tier-1 optimization (spec 2026-07-09-cheap-tier1-design) FAILED its validation gate (plan
Task 3). Measured on real Area_3, ranking each of several in-frame panos against all 21 rooms:
- `score_room_cheap` (small-pool histogram_pose_search + single-forward `sampling_loss`, NO inlier score
  maps, NO Adam): **recall@5 = 0/5**, 1.4 s/room. Degenerate — tiny/corridor clouds (hallway_5/6, WC) win;
  the true room ranks 7–21. The recon's "uniform weights give a clean cross-room loss" is EMPIRICALLY
  FALSE: unweighted mean colour residual favours small/low-coverage clouds.
- reduced-inlier `localize_pair` (4/4/4 pool, light Adam): **recall@5 = 1/3**, 3.3 s/room. Shrinking the
  pool also loses discrimination.
- FULL `localize_pair` (full inlier detection + top_k=6/num_iter=100), office_3 vs all 21 rooms:
  **rank 1**, loss 0.081 vs runner-up hallway_6 0.124 (**+53% margin**), 41 s/room. The method DOES
  recall against the full room set — but only with the expensive inlier weighting AND Adam refine, which
  together suppress the degenerate-small-cloud loss floor (~0.12–0.15).
**Conclusion:** there is NO cheap shortcut that preserves discrimination; the ~20 s inlier detection and
the Adam refine are both load-bearing. A full 76-pano in-frame eval is ~9–18 h. Plan 2 is HALTED at Task 3.
**Options going forward:** (a) accept the cost, run the full pipeline overnight for the real number;
(b) research a coverage-normalized cheap scorer to remove the small-cloud bias (uncertain); (c) treat the
method as correct-but-slow, document the office_3 rank-1/21 evidence, defer speed to future work.
Diagnostics: smoke/tier1_recall*.py.
