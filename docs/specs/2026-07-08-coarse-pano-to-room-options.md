# Coarse pano→room registration — options note (T1)

**Date:** 2026-07-08 · **Status:** decided → feeds the design spec (T2) · **Task:** `tasks.json` T1

## Problem
Assign each 360° equirectangular RGB panorama to the correct room among a set of candidate
**colored** point clouds (`X Y Z R G B`), and emit a **coarse camera pose** (translation, optional
yaw) as a seed for the downstream line-based fine localizer **FGPL**. Dev/eval = S3DIS Area_3.

## Decision drivers
- **Deterministic / training-free strongly preferred** (D1, D7); small, fast, self-contained unit.
- **Repetitive multi-room buildings are the real target** — many rooms of near-identical shape/size.
  This is exactly where the thesis' jigsaw coarse stage fails and why PanoPin exists.
- Output is a **coarse seed for FGPL** (room label + rough camera XY), not a final 6-DoF pose (D3).
- **Fairness (D5):** the solver reads only the anonymized manifest (panos + candidate clouds), never GT.

## Key empirical finding (Area_3 EDA)
- **Room geometry** (footprint 33× range, hallway aspect 5–9×) is discriminative *on Area_3* — but
  optimizing for it **rebuilds the thesis' blind spot**: ~7 room pairs are near-identical in size
  (office_9≈office_10 at 0.02 m; WC_1≈WC_2; office_5≈6; office_4≈7; office_1≈2 …), and a repetitive
  building is degenerate under shape *by construction*.
- **Global color** (room mean / whole-room histogram / floor / ceiling) is **near-degenerate**
  (22/23 rooms within ΔRGB<10), and is further undermined by a ~35-level pano↔cloud exposure gap and
  tripod/vignette-corrupted equirectangular poles.
- **Implication:** the discriminative signal for same-shape rooms is **spatially-resolved
  appearance/content** — *where* the room-specific colored content sits (posters, furniture, windows),
  which the candidate clouds also carry in their RGB. Not shape, and not global color.

## Options considered

| # | Route | How | Deterministic | Coarse pose | Repetitive-room fit | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Room-shape geometry matching** | recover room footprint/aspect from pano layout; match to cloud footprints | yes | partial | **fails** — identical shapes collide | **Rejected** (reproduces thesis blind spot) |
| 2 | **Global color/geometry signature retrieval** | compact per-pano & per-room descriptors; NN match | yes | no | weak (degenerate signatures) | **Rejected as decider**; usable only as a cheap, conservative prefilter |
| 3 | **Render-and-compare / CPO-style appearance matching** | project each colored cloud to the sphere; score per-bearing color-consistency vs the pano over candidate poses; best (room, pose) wins | yes | **yes** | **designed for it** — keys on room-specific content | **Chosen** |
| 4 | **Cross-modal 2D↔3D feature matching → PnP** | joint image/cloud descriptors, match + geometric verification | learned (mostly) | yes | feature aliasing; heavy | **Rejected** — learned, trained on outdoor LiDAR (KITTI) domain, off-mandate |
| 5 | **Pano-depth → colored cloud → colored-ICP** | lift pano to cloud via 360 depth net; colored-ICP vs each candidate | semi (learned depth) | yes | color+geometry ICP can disambiguate | **Deferred fallback** — learned depth + scale ambiguity |

## Recommendation — build upon CPO
Adopt **Option 3 (render-and-compare)** and **build directly upon CPO** from the
`82magnolia/panoramic-localization` library (Apache-2.0) — the same repo that ships PICCOLO, LDL, and
our downstream **FGPL**. CPO is:
- **Training-free, deterministic, rendering-free** color-consistency scoring — projects a colored
  cloud to the sphere and scores color histograms over many candidate poses using spherical-projection
  equivariance (no per-pose rendering).
- **"Change robust"** — its 2D/3D score maps reweight colors for robustness to illumination/appearance
  change, i.e. the exposure-gap mitigation the EDA demands is *built in*.
- Explicitly avoids feature-point matching; keys on color/spatial context → matches the
  "content, not shape" thesis.

PanoPin becomes a thin, well-bounded wrapper: run CPO's color scoring **per candidate room**, pick the
room by best color-consistency cost, emit **room label + coarse camera XY (+ yaw)** as the FGPL seed.
Reusing CPO (rather than reimplementing) adds a **PyTorch (CPU)** dependency — accepted, in line with
the explicit instruction to build upon existing validated research.

## Accuracy bar & benchmark (same dataset family)
- **CPO** on Stanford2D3D (unchanged): **0.83** @ 0.05 m / 5°.
- **FGPL** on purpose-built **repetitive-office splits** (40 & 60 office rooms): 0.68 / 0.64 @ 0.1 m / 5°
  (LDL 0.21 / 0.19 on the same) — a repetitive-building benchmark we can align to later. These are
  full-pose numbers; PanoPin's primary metric (room assignment) is coarser and should sit higher.

## Pitfalls the design must handle
- **Exposure/illumination gap** (pano darker than cloud by ~35 levels) → use CPO's score-map reweighting.
- **Corrupted poles** (tripod/vignette) → mask top/bottom elevation bands.
- **Textureless beige walls** → weight high-variance/content regions where the signal lives.
- **True twins** identical in *both* size and content → accept best-effort; report honestly; escalate
  (feature/depth fallback) only if the deterministic route can't clear the bar, logged in DECISIONS.

## Rejected / fallback prior art
- Cross-modal learned registration (DeepI2P, CorrI2P, CMR-Agent): learning-based, outdoor-KITTI domain — poor fit.
- **FreeReg** (ICLR 2024): training-free image→cloud reg via pretrained diffusion+depth — far-future fallback only.
- Relevant to cite: **AirRoom** (object-level room re-ID — validates "objects disambiguate rooms");
  **Render-Rank-Refine** (J. Imaging 2026 — the exact coarse-to-fine indoor render-and-compare architecture).

## References
- CPO — arXiv 2207.05317 · PICCOLO — arXiv 2108.06545 · LDL — arXiv 2308.13989 · FGPL — arXiv 2403.19904
- Code: https://github.com/82magnolia/panoramic-localization (Apache-2.0)
- AirRoom — arXiv 2503.01130 · FreeReg — ICLR 2024
- Source: T1 survey (3 research agents, 2026-07-08) + deep-research run `wf_08df4c9d-eb3` (16 verified claims).
