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

## D19 — "Fix the speed": no cheap algorithmic path; the real fix is GPU (needs a modern env) (2026-07-09)
Per D18 the cheap Tier-1 failed. User chose "fix the speed (research)". Findings:
- Adam-no-inlier scorer (skip ~20 s inlier detection, keep unweighted Adam refine): recall@5 = 2/3, 18 s/room
  (smoke/exp_adam_scorer.py). Adam helps (0/5 → 2/3) but inlier weighting still needed for clean recall.
- COVERAGE de-bias REFUTED (smoke/exp_coverage.py): the degenerate hallway/WC winners are NOT low-coverage;
  true rooms have HIGH coverage (0.84–0.93) comparable to the winners, and loss/coverage ranks the true room
  WORSE. So it is not a size/coverage artifact.
- ROOT CAUSE: at the coarse (histogram-search) pose the TRUE room's residual is genuinely HIGH (0.38–0.47),
  worse than corridors/hallways (0.24–0.30) whose repetitive texture matches many poses; the true room's low
  loss only emerges AFTER Adam refines the pose. => **discrimination REQUIRES per-room Adam pose refinement;
  there is no cheap coarse proxy.** This is intrinsic to CPO colour-matching on this data.
- SPEED FIX = GPU, not algorithm. CPO is GPU-native; `panopin` is CPU-only (D10). Machine has an RTX 4060
  (8 GB, Ada sm_89). Tried `scan_env` (torch 1.12.1+cu116, cuda True, has all CPO deps once torch_scatter was
  added) → **segfault**: cu116 predates Ada, its CUDA kernels crash on sm_89. (torch_scatter reverted from
  scan_env afterward.) **A working GPU run needs a fresh env: torch 2.x / cu118+ + matching torch_scatter +
  CPO deps (open3d, opencv, pandas, sklearn).** CPO's torch-1.10 code likely needs only minor tweaks for torch 2.x.
**Decision/OPTIONS for next session:** (a) build a modern-GPU env and run the CORRECT full pipeline on the
4060 (~40–60 min for 76×21, the real accuracy number); (b) accept a ~10–18 h CPU overnight run; (c) stop with
the finding of record (method works — office_3 rank 1/21 +53% — but is CPU-bound). Plan-2 (cheap Tier-1) is
ABANDONED; score_room_cheap / determinism helper remain as artifacts but are not the path.

## D20 — GPU gives only ~2x; CPO's bottleneck is iterative Python loops, not GPU-able math (2026-07-09)
Built `panopin-gpu` (torch 2.0.1+cu118 + torch_scatter 2.1.2+pt20cu118 + CPO deps) — runs correctly on the
RTX 4060 (Ada). One-pano timing, full `localize_pair`, office_3 vs all 21 rooms:
- **rank 1** (loss 0.0807, matches CPU 0.0814 — recall preserved on GPU), but **mean 22.4 s/room** vs CPU
  ~41 s — only ~**1.8x**, NOT the hoped ~20x.
- **Why:** CPO's costly parts are ITERATIVE loops — inlier detection iterates ~9360 poses in Python, and the
  Adam refine runs top_k=6 × num_iter=100 = 600 small optimizer steps. Both are kernel-launch/Python-overhead
  bound; the GPU can't accelerate them much. The GPU only helps the tensor-heavy score-map math.
- **Reducing ONLY the inlier pool** (keep full search + refine) did NOT speed it up either (~20 s/room, killed
  after 12 min) — the Adam refine is also a bottleneck, not just inlier detection.
**Conclusion:** the accurate CPO pipeline is ~20–22 s/room regardless of CPU/GPU or these config tweaks →
~7–8 min to assign ONE pano against 21 rooms. Genuine speed (~1–2 s/room) would require VECTORISING CPO's
score-map + refine loops (batch the pose pool as tensor ops) — real dev work on vendored code — or a lighter
method for the coarse-seed role. Hardware alone is not the fix. `panopin-gpu` env kept (works; ~2x faster;
useful if the loops are later vectorised). Smoke: smoke/exp_reduced_inlier_gpu.py, smoke/tier1_recall_full.py.

## D21 — Whole-area recall@1 collapses to ~17%; a FIXED set of "loss-sink" rooms dominates → calibrate (2026-07-09)
Sizing run for the FGPL top-k hand-off (B): full localize_pair, 6 in-frame panos vs ALL 23 Area-3 rooms on
panopin-gpu (smoke/tier1_recall_full.py). True-room ranks: office_3=1, office_4=2, office_5=3, office_7=22,
WC_2=5, conferenceRoom_1=4 → **recall@1=1/6 (17%), @2=33%, @3=50%, @5=83%**, ~23 s/room.
- **Whole-area recall@1 (~17%) is FAR below the 3-room prototype (78%)** — the prototype looked good only
  because it excluded the degenerate rooms. Against 23 rooms the true room is usually rank 2–5.
- **Root pattern:** the SAME small/corridor clouds — hallway_5, hallway_6, hallway_4, WC_1/WC_2, storage_2 —
  occupy the top loss ranks for EVERY query (loss floor ~0.11–0.16 regardless of the pano). They are a fixed,
  identifiable set of "loss sinks"; the true room (~0.13–0.16 when a thin match) gets buried under them.
  office_7 (rank 22) shows a genuine hard miss even so.
**Implications:** (a) B's top-k shortlist needs k≈5 for 83% coverage but is polluted by the fixed loss-sinks —
FGPL's geometry can reject shape-mismatched hallways but not same-shape offices, and rank-22 misses won't make
the list; (b) THE promising lever = **per-room loss CALIBRATION** (subtract each room's baseline/typical loss
before ranking) to demote the loss-sinks — cheap (re-ranks losses already computed), should lift recall@1 AND
clean the shortlist. **Recommendation:** test calibration before building B's plumbing. Needs the full
pano×room loss matrix (this run printed only top-8/pano) — re-run capturing all 23 losses per pano.

## D22 — Percentile calibration ~doubles recall@1 (33%→58%); ~1/3 of panos are hard-unmatchable (2026-07-09)
Tested per-room loss calibration on a 12-pano × 23-room loss matrix (full localize_pair on panopin-gpu,
saved to runs/calib_matrix.json; formula sweep in smoke/analyze_calibration.py — instant, no GPU).
Recall@1/3/5 over 12 in-frame query panos:
- **raw:** 4/12 (33%) / 5/12 / 8/12 (67%).
- **sub_median** (subtract each room's median baseline): 6/12 / 6/12 / **6/12** — lifts @1 but WRECKS the WCs
  (WC_1 5→16, WC_2 5→13): the WCs are themselves loss-sinks (low baseline), so subtraction demotes the true
  room too. Net @5 DROPS to 50%.
- **percentile** (WINNER): for each candidate room, score = fraction of OTHER panos whose loss to that room is
  <= this pano's loss (scale/spread-free). **recall@1 7/12 (58%), @3 7/12, @5 8/12 (67%)** — nearly doubles
  @1 vs raw AND keeps @5 (fixes office_5/WC_2/conf→1, rescues office_7 21→4, doesn't wreck the WCs). z-score
  and minmax also reach 7/12 @1 but lose @5.
**Adopt percentile calibration in the solver.** Caveat: it needs a reference set of panos per building to build
each room's loss distribution (bootstrap for a fixed scene). **Hard ceiling:** recall@5 stays ~67% even
calibrated — office_8 (rank ~19-23 every formula) and storage_1 are genuine CPO color-match FAILURES (true room
not in top-5 at all), so ~1/3 of panos (n=12 sample) are unrescuable by calibration OR a shortlist. This caps
B's shortlist coverage. n=12 is small — validate on more in-frame panos. Matrix cached in runs/ for re-analysis.

## D23 — The ~1/3 hard floor = panos that fail to SELF-localize (window/occlusion-dominated), a confidence signal (2026-07-09)
Diagnosed the D22 recall ceiling from the saved loss matrix + the actual pano images. The hard-miss panos
don't fail at DISCRIMINATION — they fail to self-localize to their OWN room: own-room loss ~0.25-0.26 (the
CPO "no-match floor"), vs ~0.08 for a clean lock (office_3). Ranks: office_3 own-loss 0.084 (rank 1),
WC_2 0.138 (rank 5), office_7 0.247 (rank 21), office_8 0.264 (rank 20), storage_1 0.258 (rank 23).
**Cause (from the images):** office_3 (locks) is evenly textured with distinctive warm content (leather
chairs, accent wall, photos), small window. office_8 (fails) is ~40% dominated by a large backlit WINDOW +
blinds, big blank walls, and a foreground chair OCCLUSION — regions with poor/absent colored-geometry
correspondence and window-driven exposure/white-balance mismatch vs the scanner's cloud RGB. So the failure
mode is a SCENE/VIEWPOINT property (window/blank/occlusion-dominated views), NOT texture poverty and NOT a
ranking problem — hence uncatchable by calibration or a shortlist.
**Two levers:** (a) a ROBUST loss (trimmed/median color residual, or properly applying CPO's inlier
confidence weighting to the cross-room ranking) could downweight the un-matchable regions and rescue
panos where they're a MINORITY — but not window-DOMINATED ones. (b) The own-room loss is a natural
**CONFIDENCE** signal (~0.08 lock vs ~0.25 no-lock): PanoPin can flag low-confidence panos rather than
mis-assign them — hand FGPL only the confident seeds (~2/3), flag the rest for manual/other handling. This
fits the coarse-seed role honestly (never feed FGPL a confidently-wrong room). n=12 sample; the ~1/3 rate
should be validated on more panos.

## D24 — CORRECTION: D22/D23 calibration numbers leaked GT; fair calibration = minmax, gate is a tradeoff (2026-07-09)
The D22/D23 "percentile calibration -> recall@1 58%, clean 7/7 confidence separation" used a baseline that
excluded rows BY TRUE ROOM LABEL — a **ground-truth leak**. It asymmetrically excluded the scored pano only
for its OWN room (pushing that room's percentile to 0) while including it for wrong rooms — manufacturing the
separation. A real solver has no GT (D5), so this is invalid. Re-ran the sweep FAIRLY (leave-one-out on the
scored pano only; smoke/analyze_calibration.py rewritten):
- **percentile fairly = 4/12 (33%) — no better than raw.** Its apparent win was entirely the leak.
- **minmax is the fair winner: recall@1 7/12 (58%) vs raw 4/12 (33%)** (zscore 6/12). minmax normalizes each
  room's loss to its own leave-one-out [min,max]; a genuine match sits at/below the room's min -> score <=0.
- **Confidence gate is NOT clean (D23's 100%/58% was the same leak).** Fair minmax winner-scores: CORRECT
  [-1.66..-0.14] vs WRONG [-0.30..+0.29] — they OVERLAP ~[-0.3,-0.14]. It's a precision/coverage tradeoff:
  threshold -0.3 -> 4/4 (100%) at 33% coverage; -0.19 -> 6/7 (86%) at 58%. It DOES reliably flag the D23
  hard-fail (self-localization-failure) panos, which score high/positive.
**Honest v1 (n=12, whole-area):** minmax calibration, overall recall@1 = 58%; confident subset 100% precision
at 33% coverage (tunable). Modest and NOISY (n=12) — validate on more panos. **D23's core observation stands**
(hard-fail panos fail to self-localize, own-room loss ~0.25, window/occlusion cause) — that used raw own-room
losses, no leak; only the calibration/confidence NUMBERS in D22/D23 were inflated. Implemented in
src/panopin/calibrate.py (minmax + gate); scored via smoke/score_v1.py; unit-tested tests/test_calibrate.py.


## D25 — FGPL seed ablation: color POSITION seed ~= oracle when room is right; bottleneck = room recall; P2/P3 add nothing; FGPL rotation = signed-perm C (2026-07-10)
Ran the 5-arm ablation (Oracle/P1/P2/P3/Wrong-room) feeding PanoPin's coarse seed to the modified FGPL
(multiroom_pose_estimation), all in the RAW S3DIS frame, 6-room same-shape subset, 12 panos. Spec+plan
docs/{specs,plans}/2026-07-10-panopin-fgpl-seed-ablation*.
- **P1 (color position seed) ~= oracle where room is right:** median trans over the 8 correct-room panos
  = 0.762m vs oracle 0.725m (identical). Full-set P1 median 0.946m vs oracle 0.789m; the gap is ENTIRELY
  the 4 wrong-room misses (~22m each, all -> hallway_3 loss-sink). => the POSITION seed works; the only
  lever is room-assignment recall (8/12 = 67% RAW min-loss). Directly motivates wiring v1 calibration
  (lifts recall 33->58%) into cpo_seeds (which currently uses raw min-loss, no calibrate).
- **P2 (translation-grid narrowing +-2m) = NO effect** (identical to P1): the Voronoi partition already
  constrains enough. Drop it.
- **P3 (CPO-yaw rotation prior) HURTS** (median 1.807m, wrong-room 42%, rot 179deg): CPO's rotation is
  convention-unresolved (~120deg off GT in BOTH camera->world and world->camera), so the yaw prior is
  garbage and forcing it degrades FGPL. Don't use CPO rotation as a prior until its convention is solved.
- **FGPL rotation convention:** FGPL outputs Rp = C @ R_wc with C=[[0,0,1],[-1,0,0],[0,-1,0]] (equirect
  signed-perm, same as Point_360's fix). Rp.T @ C = camera->world; 0.4-0.5deg on precisely-localized
  oracle panos -> FGPL rotation is ACCURATE for those. But it is FREQUENTLY ~90deg Manhattan-aliased even
  with a GT position seed (oracle rot median 90deg): a GOOD rotation prior could help, CPO can't give one.
- Method notes: gate bar 0.5m was miscalibrated for a COARSE seed (FGPL needs ~3m) -> oracle substantively
  PASSES. Runtime ~8min/arm on panopin-gpu; the XDF search is CPU-bound numpy (GPU only ~1.2x, D20-like).
  A latent Task-5 gap (estimator needs a density PNG, viz-only) was found+fixed via the smoke.


## D26 — Wiring v1 minmax calibration into the seed did NOT boost FGPL on the n=12 subset (it slightly hurt); the D25 "next step" is closed NEGATIVE (2026-07-13)
Executed the D25/HANDOVER next step: modified `cpo_seeds.py` to cache EVERY room's pose (not just min-loss)
+ compute the v1 minmax-calibrated room (`panopin.calibrate`, D24); added a `p1_cal` arm seeding FGPL from
the calibrated room; dropped the D25-dead p2/p3; re-ran (cpo_seeds 25min panopin-gpu + 4-arm ablation).
**Result — calibration made FGPL WORSE, not better, on this 6-room n=12 subset:**
- Full-set trans median: oracle 0.789 / **p1 (raw min-loss) 1.171 / p1_cal (calibrated) 1.839** / wrong_room
  7.004. Wrong-room: p1 33% (4/12) -> **p1_cal 42% (5/12)**. Right-room slice: p1 0.921 (8) vs p1_cal 1.158 (8).
- **Room recall unchanged: raw 8/12 = calibrated 8/12** (fresh self-consistent losses). Calibration TRADES
  misses, not removes them: per-pano it FIXED 2 loss-sink misses (870 office_4 22.0->2.3m; 0e3 office_7
  24.2->3.5m) but BROKE 2 correct panos into siblings (7e48 office_1->office_5 1.0->14.9m; decc hallway_3->
  office_5 0.9->22.1m). The sibling-confusion errors it INTRODUCES are as large as the loss-sink errors it
  removes. The shared multi-pano Voronoi also means changed seeds perturb UNCHANGED panos (614/995/d6f
  nudged ~+1m), because the partition is global, not per-pano.
- **The offline projection (10/12) that motivated this was a NOISE ARTIFACT.** It ran calibrate on the OLD
  committed loss cache; a fresh cache (CPO non-reproducible, D1) gave 8/12. Cross-run noise is real and
  large at n=12: p1's OWN median drifted 0.946 (2026-07-10 run) -> 1.171 (this run) with NO code change.
  So the p1->p1_cal delta (+0.67m) is only ~3x the bare p1 run-to-run noise (~0.2m) -> under-powered.
- **Salvageable piece = the confidence GATE, not the assignment.** Confident subset 3/3 correct (25%
  coverage); it correctly WITHHOLDS both breaks (7e48 conf -0.229, decc +0.187 -> not confident) — but also
  withholds the 2 fixes. Precision-preserving, low coverage (same tradeoff as D24). So the promising use is
  ABSTAIN-not-mis-seed (hand FGPL nothing on flagged panos), NOT swapping in the calibrated room wholesale.
- **Conclusion:** minmax calibration is too noisy at n=12 to move the room-recall bottleneck, and used as a
  seed selector it net-hurts FGPL here. Options going forward (none run yet): (a) gate-based ABSTENTION arm
  (seed only confident panos, fall back otherwise); (b) SCALE to all ~76 in-frame panos to beat the noise
  before concluding either way; (c) a better assignment lever than minmax (the robust/masked-loss
  ceiling-push thread, feat/robust-loss-ceiling) to fix loss-sinks WITHOUT breaking genuine corridors.
- Code: `cpo_seeds.py` (per-room poses + calibrate), `seed_and_config.py` (p1_cal), `run_all.py` (4-arm),
  `tests/test_seed_and_config.py` (+p1_cal test, 10/10 green). Numbers in RESULTS.md "Calibration re-run".


## D27 — Robust-loss does NOT beat the mean gate; but a RAW-unweighted residual mean beats CPO's match_color+weighted loss on gate precision (2026-07-13)
Precision/coverage comparison (spec+plan docs/{specs,plans}/2026-07-13-robust-gate-comparison*; code
`experiments/fgpl_seed/{robust_capture,robust_analysis}.py` + `src/panopin/robust_score.py` + adapter
`residuals_at_pose`). Rule = **seed only when confident** (user) → metric = **prefix_correct** = panos the
minmax gate can confidently AND correctly commit before the FIRST wrong room (= coverage at 100% precision).
Both methods share `calibrate.minmax_scores`; only the per-room score differs. Executed subagent-driven
(5 tasks, all reviews clean; the Task-4 analysis was independently re-run byte-for-byte by the reviewer).
- **Results (n=12):** a = deployed CPO loss (match_color+weighted mean) **3/12**; a′ = raw *unweighted* mean
  of the same per-point residuals at the same cached poses **9/12**; all robust variants (median / low-pct
  @10,20,25 / trim-top @10,20,30) **8–9/12** (best 9, median 8). recall@1(all): a 8/12, a′/c 9/12.
- **(1) ROBUSTNESS IS NOT THE LEVER:** best robust (c) ties a′ (9 vs 9). The spec §2 hypothesis — a robust
  statistic separates true room from loss-sink better than the mean — is **REFUTED at n=12** on this subset.
  Robust-loss is parked.
- **(2) UNEXPECTED LEAD:** the plain **raw unweighted residual mean lifts the confidently-correct prefix
  3→9** vs CPO's deployed match_color+weighted loss. Independently reviewed as a REAL effect (same pano, same
  cached pose per (pano,room); the divergence is match_color+weighting distorting the minmax gate's per-room
  normalization — consistent with the D21 loss-sink degeneracy). **The a′ baseline (added after the Task-2
  match_color finding) was load-bearing:** without it the naive read is "robust beats deployed 9 vs 3," which
  is WRONG — the gain is raw-vs-matchcolor+weighting, not mean-vs-robust.
- **Caveats:** n=12; prefix_correct is a tail-sensitive metric; the a-vs-a′ gap conflates THREE differences
  (match_color, weighting, fixed-seed-0 subsample) so the causal factor is NOT yet isolated. Treat as a LEAD,
  not a deployed decision. residuals are raw/unweighted; the fidelity gate validates them vs `sampling_loss`
  (exact, |diff|=0), NOT the weighted cache loss. (Precision note: a′ and every c-variant are computed on the
  stored 101-pt percentile GRID, so a′ = mean-of-grid ≈ the per-point residual mean — a close estimate, and
  identical treatment across a′/c keeps the mean-vs-robust comparison fair.)
- Whole-branch review (opus, 37cdabc..d2aa0a5) = SOUND: D26+D27 numbers reproduced byte-for-byte; fairness/
  no-third_party/comparison-validity/fidelity all hold; the never-task-reviewed D26 p1_cal wiring holds up
  (uses `poses[room_cal]`, test-covered). NOTE for next session: 2 PRE-EXISTING `test_narrowing.py` reds
  (scan2measure on `main` lacks the parked P2/P3 narrowing params) — not a regression from this work.


## D28 — Driver isolation: the raw-mean gate advantage is BOTH match_color AND weighting (~half each), NOT subsample (2026-07-13)
Followed up D27's a(3/12)→a′(9/12) prefix_correct lead by toggling the three confounded factors at the SAME
cached poses. `residuals_at_pose` gained `match_color`/`seed` flags (branch `feat/driver-isolation`;
fidelity-gated BOTH paths vs `cpo.sampling_loss`'s own scalar, |diff|=0). Code
`experiments/fgpl_seed/{driver_capture,driver_analysis}.py`; numbers `experiments/fgpl_seed/DRIVER_ISOLATION.md`.
- **Results (n=12, prefix_correct):** a (matchcolor+weighted+CPOsub) **3/12**; a′ (raw+unweighted+seed0)
  **9/12**; a′_mc (matchcolor+unweighted+seed0) **6/12**; a′_seed1/seed2 (raw, alt subsample) **9/9**.
- **match_color = MAJOR driver:** turning it ON drops a′ 9→6 (half the Δ=6 gap). CPO's histogram-matching
  homogenizes colors across rooms → less cross-room separability for the gate.
- **subsample = NOT a driver:** a′_seed1=a′_seed2=a′=9 (zero effect) — confirms the fixed seed-0 subsample in
  `residuals_at_pose` is not an artifact.
- **weighting = MAJOR driver (by elimination, hedged):** a′_mc(6, matchcolor+unweighted) → a(3,
  matchcolor+WEIGHTED); the remaining Δ3 is attributed to weighting (the D21 loss-sink is an inlier-weighting
  artifact). CAVEAT (review): a′_mc and a differ in weighting AND subsample; subsample is 0-effect in the RAW
  regime (a′_seed1=a′_seed2=a′), and this is EXTRAPOLATED to the weighted regime — plausible but not airtight,
  since weighting reshapes the score-maps where subsample could have more leverage. So weighting is NOT
  directly measured (a weighted variant needs the expensive inlier score-maps, ~24min); it is the parsimonious
  inference for the Δ3, flagged as a lead not a measurement.
- **Conclusion:** BOTH of CPO's fine-localization refinements — match_color AND inlier weighting — HURT the
  coarse room-discrimination gate, roughly additively (~3 each). The best gate score is a **RAW UNWEIGHTED
  residual mean at the localized pose** (cheap: `residuals_at_pose`, no score-maps). Nuance: match_color-
  unweighted raises early-confidence (prefix 3→6) but lowers full recall (8→7); the weighted `a` maximizes
  total recall (8) yet wrecks confidence ordering (prefix 3) — for seed-only-when-confident, prefix wins,
  so raw-unweighted (9) dominates.
- **Caveats:** n=12 (prefix tail-sensitive); weighting elimination-inferred not measured. LEAD, validate at
  larger n before deploying.
- **NEXT:** (i) [optional] directly measure weighting with a weighted-unmatchcolor variant to confirm the
  elimination; (ii) **wire the raw-unweighted residual-mean score into the gate's room assignment** (replace
  `cache per_room` in `calibrate.assign`) + validate at larger n (the D26 (b) scaling lever, now with the
  RIGHT score); robust-loss stays parked (D27).
- **⚠️ REFINED by D29:** at whole-area scale (23 rooms) the raw-mean SCORE still wins (8/12 vs CPO 6/12), but
  the prefix_correct=9/12 *gate* advantage here is 6-ROOM-SPECIFIC — the minmax confidence gate breaks at scale
  (both →2/12). Keep the raw-mean score; the open problem moves to the confidence gate. Read D29 before acting.
- **NEXT:** (i) ISOLATE the driver — ablate match_color alone / weighting alone / subsample (cheap, reuses
  `residuals.json`); (ii) if raw-mean holds, wire a raw-residual room score into the gate and validate at
  larger n (the D26 (b) scaling lever, now with a better score to scale); (iii) robust-loss stays parked.
  Numbers: `experiments/fgpl_seed/ROBUST_RESULTS.md`.


## D29 — Whole-area validation: the raw-mean gate advantage does NOT generalize (6-room artifact); the recall bottleneck at scale is unsolved (2026-07-13)
User-approved D28 follow-up: re-scored the SAME 12 panos against ALL 23 Area_3 candidate rooms (the hard
loss-sink regime, v1 D21) — deployed CPO-loss gate vs raw-mean gate, both through the same `calibrate` gate.
Localize 12×23 = 104 min GPU (incremental-save); residuals at those poses; code
`experiments/fgpl_seed/wholearea_{localize,capture,analysis}.py`; numbers `WHOLEAREA_RESULTS.md`.
- **Results (12 panos × 23 rooms) — SCORE vs GATE (corrected after review):**
  | room score | uncalibrated recall@1 | +minmax gate prefix_correct | +minmax gate recall@1 |
  | CPO-loss | 6/12 | 2/12 | 5/12 |
  | raw-mean | **8/12** | 2/12 | 6/12 |
- **The raw-mean SCORE DOES generalize:** uncalibrated, raw-mean picks the true room 8/12 vs CPO-loss 6/12 at
  23 rooms — the D28 *score* advantage HOLDS at scale (raw-mean is the strongest raw room-classifier tried;
  CPO min-loss misses ALL → hallway loss-sinks 2/5/6, D21 confirmed).
- **What does NOT survive is the confidence GATE:** the D28 6-room *prefix* advantage (3→9) collapses to
  **2/12 for BOTH** at 23 rooms, because the minmax calibration — tuned on the easy 6-room case — HURTS both
  scores at scale (CPO 6→5, raw-mean 8→6 recall) and can't order confidence well over 23 rooms. So the weak
  link at scale is the GATE over raw-mean, not the raw-mean score.
- **Mechanism:** adding 17 loss-sink candidate rooms (small/empty corridors, WCs, storage) gives EVERY score
  — raw-mean included — degenerate clouds to be fooled by (wrong landings scatter across storage_1/2, WC_1,
  conferenceRoom_1, office_2/8, lounge_2). Raw-mean's win was specific to the easy 6-room set (only hallway_3
  as loss-sink). Dropping match_color+weighting (D28) helps separate 6 similar offices but does NOT cure the
  fundamental loss-sink degeneracy at scale.
- **Implication (the real state of PanoPin's room recall):** GOOD NEWS = the raw-mean SCORE is a validated
  improvement that HOLDS at scale (8/12 vs CPO 6/12 at 23 rooms) — keep it as the room score. The OPEN problem
  RELOCATES to the CONFIDENCE GATE: minmax calibration (D24), tuned on the easy 6-room case, breaks at 23 rooms
  — it hurts both scores (raw-mean 8→6) and gives only ~2/12 confident-correct coverage. So the next lever is
  a BETTER confidence mechanism over the raw-mean score (drop/replace minmax at scale; a scale-aware gate; or
  candidate PRE-FILTERING to drop loss-sink rooms before scoring so the gate has an easier job). Robust-loss
  (D27) stays parked. The color POSITION seed (D25) still works WHEN the room is right.
- **Deployable now:** `calibrate.assign(robust_score.raw_mean_scores(grids))` is wired + tested (`raw_mean_scores`
  in `src/panopin/robust_score.py`) and IS the best room-assignment available (raw-mean 8/12 > CPO 6/12), but
  its minmax gate under-commits at scale — usable for the ASSIGNMENT, not yet for high-precision abstention.
- **Discipline note:** whole-area validation CAUGHT a false lead (the 6-room D28 result) before it was
  deployed — exactly why "validate at scale before wiring in" mattered. Caveat: n=12 (prefix tail-sensitive),
  but the 9→2 collapse is far larger than the run-to-run noise.

## D30 — Deployment-regime characterization: recall 85–92% at 3–5 covered rooms; low-percentile (q=20) is the deployable room score (2026-07-14)
**Context.** User reframed deployment (2026-07-14): real input = a 3–5 room cloud, EVERY room covered by ≥1 pano
— NOT 12 panos vs all 23 rooms. Rescored the cached whole-area residual grids (12×23, 101 percentiles/pair)
restricted to subsets of the 6 covered rooms — fully OFFLINE, no GPU. Fidelity check reproduces WHOLEAREA
(k=23: CPO 6/12, raw-mean 8/12). Scripts: `experiments/fgpl_seed/deploy_regime{,_xcheck,_qpin,_gate}.py`;
synthesis in `DEPLOY_REGIME.md`.
- **The deployment regime is far easier than whole-area:** raw-mean recall at k=3–5 covered rooms = **85–88%**
  (vs 67% at 23 rooms). So the reframing materially improves the real target metric.
- **Low-percentile is the best per-room score AND it cross-validates.** On the whole-area cache median /
  trimmed-mean / low-pct all hit 92% (k≤6), but on the INDEPENDENT 6-room cache (`residuals.json`) only
  **low-percentile generalizes**: 75% vs raw-mean 67% at k=6; median/trimmed revert to baseline (cache-specific
  — NOT adopted). low-pct's advantage GROWS with room count (loss-sink signature). q-pin = a WIDE plateau
  q∈[5,25] (identical recall; q≥30 reverts) → not an n=12 knife-edge. Pinned **q=20**.
- **Mechanism:** low-pct scores "how well the best-matching q% of points align". The true room has a strong
  low core; a loss-sink matches mediocrely everywhere so its best q% is still worse. It rescues office_5 from
  the hallway_3 loss-sink (6-room cache 8/12→9/12).
- **Remaining bottleneck = hallway_3 loss-sink capture** (3 deeper captures survive low-pct) — the next lever.
- **low-pct needs NO calibration (Q4):** minmax calibration lifts the loss-sink-degenerate CPO-loss a lot
  (whole-area 83→92%) and raw-mean modestly, but adds ~nothing on top of low-pct. **low-pct q20 + plain argmin**
  reaches the calibration ceiling scoring each pano INDEPENDENTLY (no cross-pano stats) — fits the fast
  per-pano coarse-seed mandate and works with few panos. This partly reopens D27/D29: robustness IS a lever for
  RECALL (D27 tested median/trimmed on the gate-prefix metric and missed the low-percentile-for-recall win).
- **Shipped:** `robust_score.low_percentile_scores(grids, q=20)` (+ test) — the recommended deployable room
  score, superseding `raw_mean_scores` for assignment. Caveat: **n=12** (1–2-pano flips); a larger-pano GPU run
  should confirm before the deployable `select_room`/`cpo_seeds` swap.
- **NEXT:** (a) larger-n GPU validation of low-pct; (b) rewire `select_room`/`cpo_seeds` room ranking to
  low-percentile of `residuals_at_pose`; (c) attack the hallway_3 loss-sink directly (pre-filter / detector).

## D31 — Loss-sink attack REFUTED: the sink is a symptom of weak-lock (D23) panos, not the cause; per-pano vs joint assignment both no-op (2026-07-14)
**Context.** D30 localized hallway_3 as the dominant miss. User delegated the architecture choice
("try both and compare"). Probed separability, then compared per-pano-relative (minmax, rel_median,
rel_q25) vs JOINT (per-room rank) assignment over the low-pct q20 score matrix, offline on both
caches. Scripts `experiments/fgpl_seed/loss_sink_probe.py` + `deploy_regime_sink.py`; LOSS_SINK_RESULTS.md.
- **loss_sink_probe:** the sink IS separable in isolation — hallway_3's true panos score 0.064-0.071
  vs impostors 0.081-0.120 at the sink. BUT the whole-area cache has NO sink (low-pct already solved
  it) — the sink is partly CPO run-to-run noise (±0.02).
- **Comparison result — NEITHER architecture net-improves recall:** minmax / rel_median / rel_q25
  (per-pano) AND rank (joint) all land at plain low-pct **argmin** (75% 6-room / 92% whole-area).
- **Decisive diagnosis (full k=6 assignment):** removing the sink's pull moves the 3 impostors OFF
  hallway_3 but into OTHER WRONG rooms (870532d7 office_4->office_7; 0e30c45e office_7->office_6/
  hallway_3) — NEVER their true room. **The sink is a SYMPTOM of the miss, not its cause.**
- **Root cause = weak ABSOLUTE color lock:** the 3 impostors match their own true room at low-pct
  0.15-0.25 vs ~0.06-0.08 for clean panos — the **D23 window/occlusion hard floor** (~1/3 of panos
  can't self-localize). No score-matrix normalization/assignment recovers a room the color signal
  does not support. (Reproduces D26/D29 "calibration trades misses" at the mechanism level.)
- **Implication:** **low-pct argmin (D30) is the CEILING for color-only room assignment.** Shipped
  nothing (no sink method works). Further recall needs either (a) a NON-color cue (geometry/rotation)
  for weak-lock panos, or (b) a CONFIDENCE GATE that ABSTAINS on weak-lock panos — which, in the
  all-covered deployment (>=1 pano/room), still yields full per-ROOM coverage via each room's strong
  panos. **Per-pano recall is not the deployment metric; per-room COVERAGE is.**
- **NEXT (reframe):** stop optimizing per-pano recall; test the gate/coverage criterion — with a
  confidence signal (low-pct score / margin), is there a confident subset that is (a) high-precision
  and (b) covers every room? That is the real all-covered-deployment success metric. Offline-testable.

## D32 — Reframe SOLVES deployment: per-room COVERAGE = 100% at precision 1.0 (confidence gate / room-anchored seeding), n=12 (2026-07-14)
**Context.** D31 showed per-pano recall is capped by weak-lock panos (D23) and the loss-sink is a
symptom. Reframed to the real deployment metric: per-ROOM coverage (one correct seed/room for FGPL)
in the all-covered regime. Tested offline on the low-pct q20 scores, both caches.
`experiments/fgpl_seed/deploy_regime_coverage.py`; COVERAGE_RESULTS.md.
- **Result: per-pano recall 75-92% becomes per-room coverage 100% at precision 1.0** on BOTH caches,
  every k in {3,4,5,6}. Weak-lock panos carry the HIGHEST absolute low-pct score (~0.12+ vs genuine
  ~0.06-0.08) → they rank LAST, so a confidence gate on the winner-score admits all correct panos
  first and hands FGPL ZERO wrong seeds while every room is still covered by its strong pano(s).
- **winner-score >> margin** as the confidence signal (6-room 100% vs 83-90%): absolute low-pct value
  measures lock quality directly; margin can be spuriously large for an impostor.
- **Room-anchored assignment (`argmin_p score[p][r]`) = 100% correct seeds, threshold-free,
  loss-sink-immune:** a room's genuine panos match it best, so each room self-seeds with its own pano
  (the loss-sink attracts panos under per-pano argmin but never wins the room-anchored contest).
- **Shipped:** `src/panopin/coverage.py` (`room_anchored_seeds`, `pano_confidence`) + tests (4/4).
  This is the deployable PanoPin→FGPL hand-off: pair with `robust_score.low_percentile_scores`.
- **Net story for "improve performance":** the win is NOT a higher per-pano number — it is realizing
  per-ROOM coverage is the metric and that it is already ~100% at precision 1.0 via threshold-free
  seeding. Ceiling on per-pano recall (D31) is real; it does not bind the deployment.
- **Caveat:** n=12. **NEXT:** larger-pano GPU validation of the coverage/gate claim; then wire
  `coverage.room_anchored_seeds(low_percentile_scores(...))` into the deployable seed path
  (`cpo_seeds`/`select_room`) for the FGPL hand-off.

## D33 — Larger-n validation (n=32, 8 diverse rooms, real loss-sink): D30+D32 CONFIRMED; deployable seeder shipped (2026-07-14)
**Context.** n=12 was under-powered. Built a diverse 8-room pool (office/hallway/lounge/conference/WC)
with all in-frame panos = 32 (2.7x n), localized each pano vs all 8 rooms (~99 min GPU), captured
residual grids, and re-ran the D30/D32 metrics over 3..8-room all-covered subsets. Code
`experiments/fgpl_seed/largeval{,_localize,_capture,_analysis}.py`; LARGEVAL_RESULTS.md.
- **Real stress test:** raw CPO min-loss recall = 19/32; **hallway_1 is the new loss-sink** (captures
  ~3 impostors) — the pool is not benign.
- **Per-pano (D30 HOLDS):** low-pct 78-85% beats CPO-loss 59-74% at EVERY k (by ~11-19 pts); the
  low-pct advantage widens at larger n.
- **Per-ROOM coverage (D32 CONFIRMED):** **room-anchored 100% AND winner-gate@precision1 100% at ALL
  k in {3,4,5,6,7,8}** — the deployment claim reproduces beyond n=12, on new diverse rooms, WITH a
  loss-sink present. room-anchored self-seeds each room with its genuine best-matching pano; the
  loss-sink attracts panos under per-pano argmin but never wins the room-anchored contest.
- **SHIPPED the deployable hand-off:** `src/panopin/seed.py` `seed_rooms(panos, candidate_clouds)` ->
  `{room: RoomSeed(room, pano, t, R, score)}` + per-pano confidence, composing localize_pair ->
  residuals_at_pose -> low_percentile_scores (D30) -> room_anchored_seeds (D32). Pure assembly
  (`seeds_from_scores`) unit-tested without GPU (tests/test_seed.py, 3/3). This is the validated
  PanoPin->FGPL seed: one correct coarse seed per room.
- **Net:** PanoPin's deployment performance = **~100% per-room coverage at precision 1.0**; per-pano
  recall (78-85%) is the color-only ceiling (D31) and does NOT bind deployment. Caveat: still one area
  (Area_3); cross-area validation is the remaining generalization question.
- **NEXT:** wire `seed.seed_rooms` into the FGPL runner input (demo6_alignment.json per-room seeds) so
  the real Scan2BIM pipeline consumes it; optional cross-area check.

## D34 — PanoPin->FGPL alignment export: per-pano re-scope, gate & omit + coverage backstop, frame guard (2026-07-15)
**Context.** Before wiring the D33 seed hand-off, traced FGPL's actual consumer
(`multiroom_pose_estimation.load_panorama_positions`) per spec
`docs/specs/2026-07-15-fgpl-alignment-export-design.md`. Shipped `src/panopin/fgpl_export.py` (4 pure
functions, no GPU, no CPO import) + `tests/test_fgpl_export.py` (10 unit) + `tests/test_fgpl_export_smoke.py`
(2 integration-smoke); full suite **12 passed, 2 warnings in 0.20s**.
- **The consumer trace re-scopes the hand-off from per-room to per-pano.** `load_panorama_positions` reads
  `alignment['matches']` per-pano, positional-only: only `pano_name` + `camera_position` are load-bearing
  (`room_label`/`room_idx`/`rotation_deg`/`score` are ignored — `room_label` appears only in a `print`). Each
  pano is localized independently from its own seed via its own Voronoi cell, so anchoring a room
  (`coverage.room_anchored_seeds`, D32) does NOT localize that room's other panos. D33's "wire
  `seed.seed_rooms`" hand-off therefore means **one match per pano**, not one per room.
- **Gate & omit (user decision) + room-anchored coverage backstop.**
  `build_matches(score_matrix, poses, room_order, R_meta, tau=0.10, guarantee_coverage=True)` admits a pano
  at its argmin room iff the low-pct winner score <= `tau=0.10` (the empirical gap between genuine ~0.06-0.08
  and weak-lock ~0.12+ locks, D30/D32); weak-lock panos (~15-22%, the D23/D31 color hard floor) are omitted
  rather than guess-seeded, so FGPL gets zero wrong seeds. Any room left uncovered by the gate gets a
  room-anchored backstop pick restricted to FREE (unassigned) panos, since FGPL's `load_panorama_positions`
  keys positions by `pano_name` (a duplicate silently last-wins) — per-pano uniqueness is enforced by
  construction, not by luck.
- **Frame conversion, guarded, not assumed.** PanoPin's CPO `t` is raw-S3DIS-frame; FGPL's `camera_position`
  is aligned-frame (`raw = R.T @ [ax,ay,0]`). `raw_t_to_camera_position(t_raw, R_meta, tol=1e-6)` inverts
  exactly for a yaw `R`: `camera_position = (R @ t_raw)[:2]`, and asserts the round-trip through FGPL's own
  converter rather than trusting the math — fails loud on non-yaw/malformed metadata. In the D25 ablation
  `metadata.json` was identity (silent no-op); real deployment has `R != I`, which is exactly what this
  guards against.
- **Real-loader validation, not just schema inspection.** The integration smoke builds an actual
  `demo6_alignment.json` from the cached D33 largeval grids/poses and loads it back through FGPL's own
  `load_panorama_positions` (not skipped — FGPL is importable in the `panopin` env): positions recover to
  1e-6, confirming the schema is exactly right end-to-end, not just by inspection of the source.
- **Fast-follow (out of scope, spec §8):** the live FGPL GPU round-trip — run `multiroom_pose_estimation` on
  a PanoPin-seeded `demo6_alignment.json` for a couple of Area_3 rooms and compare the refined pose to S3DIS
  GT (D25 already showed a correct-room color seed -> oracle-quality FGPL pose, so this confirms plumbing,
  not the method). Then Electron/pipeline wiring.
- Commits (Tasks 1-5): 45b5389, d2463ab, 9c449a0, fe66aa5, 06dcdd2. All LOCAL on `feat/deploy-regime`,
  nothing pushed.

## D35 — PanoPin<->FGPL validation round-trip (§8): plumbing proven, per-room coverage 6/6, right-room trans ~1.05m, not oracle-tight (2026-07-15)
**Context.** Spec §8 fast-follow to D34: does the `fgpl_export` seed actually drive FGPL's real
estimator end-to-end, and does the D34 `tau=0.10` gate hold up THROUGH FGPL (not just on the score
matrix)? Method: build the `fgpl_export` seed (gated at `tau=0.10`) on the `area3_seed_ablation`
6-room subset (office_1/4/5/6/7 + hallway_3, 12 in-frame panos), run FGPL's real
`multiroom_pose_estimation` estimator on the admitted panos, score the refined poses vs S3DIS GT
alongside the cached `oracle`/`p1` reference arms. Code `experiments/fgpl_seed/roundtrip.py` + unit
test `tests/test_roundtrip.py` (3/3); results `experiments/fgpl_seed/ROUNDTRIP_RESULTS.md`.
- **The gate admitted 10/12 panos** (2 weak-lock omitted per D30/D34's tau=0.10); FGPL ran on those
  10 with **zero errors, 10/10 localized**.
- **Per-room coverage 6/6** (every room has >=1 admitted pano that refines into it) — hallway_3,
  office_1/4/5/6/7 all OK.
- **Accuracy (fgpl_export gated vs cached refs):** trans median **1.082 m** (right-room slice
  **1.048 m, n=9**), mean 3.044 m, max 21.838 m, rotation median 105.2°, wrong-room 0.20 — vs
  cached `oracle` (0.696 m median, 0.647 mean, 1.579 max, rot 89.9°, wrong 0.00) and cached `p1`
  raw min-loss (0.998 m median, 5.205 mean, 24.426 max, rot 105.0°, wrong 0.20).
- **(1) Plumbing PROVEN:** the D34 `fgpl_export` seed drives FGPL's real estimator to a
  `camera_pose.json` for all 10 admitted panos with zero errors — spec §8's end-to-end question is
  answered, PanoPin and FGPL run together.
- **(2) Deployment claim HOLDS through refinement:** per-room coverage stays 6/6 even with 2/10
  wrong-room panos, because each room still has another correct pano — exactly the D32 argument for
  why per-ROOM coverage, not per-pano recall, is the deployment metric.
- **(3) Accuracy is decent but NOT oracle-tight:** right-room translation (~1.05 m) is well short of
  oracle (0.70 m); and the gated seed's overall median (1.082 m) is basically TIED with the old p1
  raw-min-loss seed (0.998 m) at n=10 — better tail/mean (3.0 vs 5.2 m) but not a clear median win.
  Consistent with the D31/D23 per-pano color ceiling — gating removes the worst weak-lock panos but
  doesn't lift the survivors to oracle quality.
- **(4) Rotation (~105° median) is an FGPL limitation, not a PanoPin regression:** the oracle itself
  is ~90° off GT (FGPL Manhattan-aliasing, D25) — rotation was never claimed accurate by this seed.
- **(5) This run doubles as the deferred D34 `tau=0.10` precision check, now measured THROUGH FGPL**
  (not just on the offline score matrix): the gate's wrong-room rate (0.20) matches `p1`'s, so
  gating at this tau does not visibly change wrong-room rate on this small sample, but it does drop
  2 weak-lock panos before they reach FGPL at all.
- **Caveats:** oracle/p1 cached reference poses ran with a 12-pano Voronoi vs this arm's 10-pano
  Voronoi (a reference context, not a controlled ablation); Area_3 subset only (6 rooms, 12 panos).
- **NEXT:** Option B production/Electron pipeline wiring (real `demo6_alignment.json` seeds feeding
  the real Scan2BIM FGPL run); optional larger-n / cross-area validation of this round-trip.
- Commits: a42f452, 4b0fedd, e6dec74, c9794bf, 76d2654, e2397fc. All LOCAL on `feat/fgpl-roundtrip`,
  nothing pushed.

## D36 — Strictly-Manhattan demo: PanoPin is what makes multi-room FGPL work (13.3m → 0.08m); the 180° flip diagnosed, an upright prior shipped, two follow-on fixes REFUTED (2026-07-16)
**Context.** User asked for a co-worker demo confined to Manhattan-world rooms (Point_360 `roadmap.md`
§5 classification), using single-room `.ply` clouds so the search space is Manhattan too: run PanoPin
+ the scan2measure pose unit on the panos in those rooms, score vs S3DIS GT. Branch
`feat/manhattan-demo`; scan2measure work on `feat/upright-rotation-prior`.

- **NEITHER previously-validated pool is strictly Manhattan.** The D33 largeval pool contains 3 of
  Area_3's 4 non-Manhattan rooms (office_3/7/8); the D25/D35 subset contains 2 (office_4/7). So the
  D35 headline numbers CANNOT be quoted as Manhattan results. New pool = largeval ∩ Manhattan =
  **5 rooms / 22 panos** (office_5, hallway_1, lounge_1, conferenceRoom_1, WC_1), which keeps the
  cached D33 grids usable (per-(pano,room) localizations are independent → restricting the candidate
  set offline == only ever searching those rooms, the D30 rescoring argument). Keeps the real
  loss-sink (hallway_1) and stays diverse. `manhattan.py`.
- **The Manhattan restriction is visibly correct:** the 5-room map (5.16M pts) yielded 3 **exactly
  axis-aligned** principal directions with **0% unclassified** sparse lines — the Stage-3
  3-orthogonal-direction assumption fits perfectly, which office_3/7/8's diagonal walls would break.
- **Room assignment (offline, cached, no GPU):** room-anchored **5/5 correct**, per-room coverage
  100%, winner-gate@precision-1 100%; per-pano low-pct **82%** vs raw CPO **64%**. All 4 per-pano
  misses rank in the bottom 5 by confidence (D32 reproduces under strict Manhattan).
  `MANHATTAN_RESULTS.md`.
- **THE HEADLINE — FGPL cannot do multi-room alone.** Same 22 panos, same map, same estimator; only
  the seeding changes: **FGPL alone (global mode, seed never read) = 1/5 rooms, 82% wrong-room,
  13.27 m median.** +PanoPin = **5/5 rooms, 18%, 0.96 m** (0.084 m with the prior, below). This is
  the "cross-room false minima" the Voronoi exists to prevent, and it quantifies PanoPin's value.
  It also **refutes "give FGPL more search area"**: global mode IS that idea's maximum.
- **The 5-seed room-anchored config** (1 seed/room) = 5/5 rooms, 0% wrong-room, **0.045 m, 0 flips** —
  but it poses **5 panos, not 22**. D34 established the hand-off is per-PANO (FGPL localizes each
  pano from its own seed), so this is NOT a substitute; it answers the narrower "one good entry
  point per room" question.
- **180° flip DIAGNOSED (scan2measure).** The fix (QUERY_SPHERE_LEVEL=3, TOP_K=10, n_tight selection)
  IS active in multiroom — it was starved, not bypassed. Of 12 aliased poses: **8 had the camera ON
  ITS SIDE, 2 UPSIDE-DOWN, only 2 true yaw flips.** `build_rotation_candidates` enumerates the full
  octahedral group (24), so 20 candidates tip/invert the camera — impossible for tripod capture.
- **SHIPPED: opt-in upright prior** (scan2measure `feat/upright-rotation-prior`, spec
  `docs/superpowers/specs/2026-07-16-upright-rotation-prior-design.md`). 22-seed arm: trans median
  **0.960 → 0.084 m**, rot median 89.9 → 1.8°, locked 10/22 → 15/22, **zero regressions**, XDF search
  **~10x faster** (24 → 4 candidates). Default off; single-room pipeline and TMB untouched.
- **The prior ALONE does not rescue FGPL (control):** FGPL+prior WITHOUT PanoPin = **13.32 m** (vs
  13.27 m as-is), rooms 1/5 → 3/5. The two contributions fix different failures — the prior picks a
  better ORIENTATION only once you are in the right PLACE, and placement was what broke.
- **It does NOT solve the flip.** 7/22 still alias (3 are wrong-ROOM panos = PanoPin errors; among
  right-room panos 9/19 → 4/19). The 4 surviving candidates differ only by yaw. **Read the medians as
  "which side of a bimodal split most panos are on", NOT as error magnitudes** — the flipped group
  shrank in COUNT, not in error (survivors still 88-180°).
- **REFUTED #1 — "multiple panos per room causes the flips" was RIGHT as a mechanism but my
  cross-room test was wrong.** hallway_1 has the MOST panos and the BEST lock rate; lounge_1 has
  fewer and fails 4/4. The intervention settles it: identical seeds, 22 competing panos → 4, **3/4
  aliased panos recover** (+127fc8df in the anchored arm). Cross-sectional correlation ≠ causation;
  rooms have different geometry budgets. I also wrongly called lounge_1 "intrinsically ambiguous"
  because the oracle failed there — the oracle ran with the SAME starved partition.
- **REFUTED #2 — per-room architecture (per-room line map + global mode, no Voronoi) is WORSE.** On
  the 18 correctly-assigned panos: locked 14/18 → **12/18**. `fafa0629` came back rotation-locked at
  0.7° but **8.8 m down the hallway**. **The seed does TWO jobs: it starves the rotation search of
  geometry AND it pins the position.** Removing it returns the geometry and discards the constraint.
  The 5-seed arm wins by landing on both. `PERROOM_RESULTS.md`.
- **`tau=0.10` does NOT transfer** to this pool (admits 21/22 including 3 wrong-room) — the score
  distribution sits lower than the pool it was tuned on. Threshold-free room-anchored seeding is the
  claim that holds. (Same trap as any absolute threshold.)
- **UNSHIPPED lever — the rotation margin.** best minus second-best `n_tight` across DISTINCT
  rotations separates good from aliased **42/44 with ZERO false positives** (good 3-56, aliased
  0-11; several aliased poses have margin **0** — an exact tie broken arbitrarily). Unlike absolute
  `n_tight` it is computed WITHIN a pano, so it avoids the tau calibration trap.
  `multiroom_pose_estimation.py:471` computes every ingredient and discards the runner-up;
  `n_tight`/`avg_dist`/line counts are already persisted to `local_filter_results.json`.
- **Process note — the falsifiable prediction earned its keep twice.** The first upright-prior
  implementation filtered CANONICAL-frame candidates against a WORLD-frame gravity vector
  (`canonical_rot = principal_3d`; the search emits `R_world = R_cand @ principal_3d`), selecting
  exactly the wrong 4 → locked 10/22 → **0/22**. The unit test used `principal_3d = eye(3)`, which
  collapses the two frames and hid the bug entirely (passed 5/5, proved nothing). The prediction
  caught it immediately, and later stopped a genuine improvement being sold as a solved problem.
- **NEXT.** (a) Room-scoped filtering — filter `get_local_mask` by `room_label` (which PanoPin
  already sends and FGPL only prints, D34) while KEEPING the per-pano seed: geometry from the whole
  room AND position still pinned. The untested middle; even odds. (b) Ship the rotation margin as a
  confidence output. (c) Cross-area validation (everything is Area_3).
- Caveats: Area_3 only, n=22, Manhattan rooms only. The nearest-centroid room metric misjudges
  boundary panos (`5c2959c3` is 2.9 cm from GT yet scored wrong-room), so 18% wrong-room is slightly
  pessimistic.

## D37 — CORRECTION: D1's "CPO is inherently non-reproducible" was inaccurate; the fix already existed, it just never reached the shipped path (2026-07-20)
**Context.** Stage 0 acceptance (T7, `localization/ACCEPTANCE.md` in the Point_360 repo) failed
criterion 1 on its first run: translation median 0.639 m vs the D36 target of 0.084 m (7.6x worse),
and Gate 2 at 261.6 px (threshold 100, FAIL). Bisected, not guessed: comparing a cached seed against
a fresh reseed of the identical inputs showed room labels differing on 5/22 panos and seed positions
drifting >1 m on 7/22 (max 27.46 m) — the drifted seeds mapped 1:1 onto the bad final poses; the
9/22 panos with identical seeds gave the good poses.
- **Root cause, isolated to two sources:** (a) `data_utils.read_txt_pcd` draws an `np.random`
  permutation whenever `sample_rate>1`; the deployed config uses `sample_rate=30`, so every unpinned
  run sampled a different point subset per cloud; (b) multi-threaded float reduction order (named as
  the larger source in `determinism.py`'s own docstring).
- **The fix already existed.** `src/panopin/determinism.py::pin()` sets
  `torch.set_num_threads(1)` and seeds `np.random`/`torch`, fixing both sources. **Every**
  `experiments/fgpl_seed/*` script called it (e.g. `largeval_localize.py:15`) — but **nothing under
  `src/panopin/` did**. The shipped path — `cli.py` → `seed.localize_and_score` →
  `cpo_adapter.localize_pair` — ran completely unpinned, while the experiment/validation path that
  produced every prior headline (D30–D36) was pinned all along. The library and the paper trail
  behind it were never actually running the same code path.
- **D1 (2026-07-08) is CORRECTED, not superseded.** D1 recorded CPO as inherently non-reproducible on
  CPU and deferred the problem as an accepted cost. That factual characterization was inaccurate —
  the project had already solved determinism; the solution simply never reached the deployable unit.
  D1's *preference* (deterministic/simplicity-first) stands unchanged; only its claim about CPO does not.
- **Fix:** `pin()` placed at the top of `seed.localize_and_score` — the chokepoint both entry points
  (`cli.seed_from_clouds` and `seed.seed_rooms`) pass through — PanoPin commit `be674b9`.
  `tests/test_determinism.py::test_localize_and_score_is_reproducible_across_runs` exercises the real
  path, not a toy: two back-to-back calls on identical synthetic inputs disagreed before the fix,
  agree after. Full suite: 54 passed, 1 xfailed, no regressions.
- **Measured consequence** (re-running Stage 0 acceptance with the fix): translation median
  **0.639 m → 0.089 m** (D36 target 0.084 m); Gate 2 **FAIL 261.6 px → PASS 59.0 px** (threshold
  100 px). Rotation and room-assignment also moved favorably, but rest on single medians at n=22 with
  no dispersion — the honest claim is that this reproduces D36, not that it beats it
  (`ACCEPTANCE.md` §1).
- **Cost:** `pin()` sets `torch.set_num_threads(1)`, so seeding is now single-threaded — determinism
  is bought with wall-clock, not free.
- **Cross-reference:** corrects **D1**; the reproduced number is **D36**'s (0.084 m, strictly-Manhattan
  5-room/22-pano pool). Full root-cause narrative in `localization/ACCEPTANCE.md` §2 (Point_360 repo,
  read-only from PanoPin's side).
