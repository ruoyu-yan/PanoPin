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
- **⚠️ SUPERSEDED by D29:** the raw-mean advantage in this decision is a 6-ROOM-SUBSET ARTIFACT — it does NOT
  survive the whole-area (23-room) regime. Do NOT act on the "wire it in" recommendation; read D29 first.
- **NEXT:** (i) ISOLATE the driver — ablate match_color alone / weighting alone / subsample (cheap, reuses
  `residuals.json`); (ii) if raw-mean holds, wire a raw-residual room score into the gate and validate at
  larger n (the D26 (b) scaling lever, now with a better score to scale); (iii) robust-loss stays parked.
  Numbers: `experiments/fgpl_seed/ROBUST_RESULTS.md`.


## D29 — Whole-area validation: the raw-mean gate advantage does NOT generalize (6-room artifact); the recall bottleneck at scale is unsolved (2026-07-13)
User-approved D28 follow-up: re-scored the SAME 12 panos against ALL 23 Area_3 candidate rooms (the hard
loss-sink regime, v1 D21) — deployed CPO-loss gate vs raw-mean gate, both through the same `calibrate` gate.
Localize 12×23 = 104 min GPU (incremental-save); residuals at those poses; code
`experiments/fgpl_seed/wholearea_{localize,capture,analysis}.py`; numbers `WHOLEAREA_RESULTS.md`.
- **Results (12 panos × 23 rooms):** raw min-loss (no calibration) recall@1 **6/12** (misses ALL → hallway
  loss-sinks 2/5/6 — D21 confirmed at scale). **CPO-loss gate: prefix_correct 2/12, recall 5/12. raw-mean
  gate: prefix_correct 2/12, recall 6/12.**
- **The D28 6-room advantage (raw-mean prefix 3→9) does NOT survive:** at 23 rooms both gates collapse to
  prefix **2/12**; raw-mean only marginally better on recall (6 vs 5, within n=12 noise). The two gates TRADE
  which panos they get right (different sets), netting ~0. Sub-finding: calibration adds nothing for raw-mean
  at scale (gate recall 6 == raw min-loss 6) and slightly HURTS CPO-loss (gate 5 < raw 6).
- **Mechanism:** adding 17 loss-sink candidate rooms (small/empty corridors, WCs, storage) gives EVERY score
  — raw-mean included — degenerate clouds to be fooled by (wrong landings scatter across storage_1/2, WC_1,
  conferenceRoom_1, office_2/8, lounge_2). Raw-mean's win was specific to the easy 6-room set (only hallway_3
  as loss-sink). Dropping match_color+weighting (D28) helps separate 6 similar offices but does NOT cure the
  fundamental loss-sink degeneracy at scale.
- **Implication (the real state of PanoPin's room recall):** the whole-area room-recall bottleneck — the
  problem since D26 — is NOT solved by ANY lever tried: v1 minmax calibration, robust-loss (D27), and raw-mean
  (D28) ALL collapse to ~2/12 confident-correct / ~50% raw recall on 23 rooms. A genuinely different approach
  is needed — e.g. candidate PRE-FILTERING to drop loss-sink rooms before scoring, a scale/coverage-aware
  score that tiny clouds can't game, or cross-pano consistency. The color POSITION seed (D25) still works
  WHEN the room is right; room selection at scale is the open hard problem.
- **Discipline note:** whole-area validation CAUGHT a false lead (the 6-room D28 result) before it was
  deployed — exactly why "validate at scale before wiring in" mattered. Caveat: n=12 (prefix tail-sensitive),
  but the 9→2 collapse is far larger than the run-to-run noise.
