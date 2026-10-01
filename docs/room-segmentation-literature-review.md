# Literature review — automatic room segmentation of indoor point clouds

**Date:** 2026-10-01 · **Purpose:** decide how PanoPin should produce its own `{room: cloud}` candidate
list from ONE merged multi-room cloud, replacing the S3DIS ground-truth room partition that every
PanoPin / Stage-0 result so far has used (Area_3 5-room acceptance, 5/5 rooms, 0.086 m median, D38).
**Review only — nothing implemented.** **Method:** five parallel literature threads (A density-image
floorplan nets + SAM; B direct 3D / 2.5D methods; C robotics occupancy-map + 3D-scene-graph rooms;
D metrics, benchmarks, S3DIS GT quirks; E open-vocab / foundation / panorama-assisted / joint
localisation) over arXiv, Semantic Scholar, OpenAlex, Crossref, CVF/ECVA, publisher pages and the
GitHub API. ~150 papers screened, ~110 retained. Each thread read key PDFs in full; licences come
from the repos' actual LICENSE / `package.xml` files, not READMEs. Three of the threads' claims were
corrected by direct measurement on our data (§8.3). Detailed per-thread notes (≈200 kB) were working
files and are not committed; every claim below carries its source.
**Verification marks:** unmarked = venue/year/DOI or URL confirmed live; **[abs]** = only abstract or
metadata read; **[UNVERIFIED]** = could not be confirmed and must not be quoted as fact.

---

## 0. Bottom line

1. **What goes into the density image matters more than the model that segments it.** Every method
   that works on real *furnished* scans with *open doors* rasterises a **horizontal band near the
   ceiling**, not the full-height cloud. A band above the door heads and below the ceiling sees the
   continuous wall over each doorway (the lintel), so open doors close by construction, and almost
   all furniture drops out. This was found independently in scan-to-BIM (Macher 2017; Jung 2017;
   Gourguechon 2023), SAM-based work (Albadri 2025: 65/67 rooms; FloorSAM 2025) and robotics
   (HOV-SG 2024; He 2021). The thesis front-end (`generate_density_image.py`) projects the **full
   height** of the cloud, so it hits both hard cases head-on.
2. **Learned floorplan networks are a poor bet on our data.** None was ever evaluated on a real
   *multi-room* scan: SceneCAD, the only real benchmark, was filtered to single rooms. Trained on
   Structured3D, they drop from about 92 to 52–86 room IoU on real SceneCAD. On S3DIS, Mask3D and
   RoomFormer trained on Structured3D score **mAP 0.0 / 0.2 and miss 67–68 of 68 rooms**
   (Brunklaus 2025). The thesis's "training-domain limitation" for RoomFormer is partly self-inflicted:
   RoomFormer's training images are unique-voxel counts stretched separately along each axis, while
   ours are raw counts at preserved aspect ratio (§2.2).
3. **SAM 3's "floor plan" text prompt is the wrong way to use SAM here.** SAM 3 is a concept segmenter:
   a noun phrase finds all instances of that concept, and "floor plan" names the whole drawing. Both
   published SAM room pipelines use **point prompts / automatic mask generation** on SAM 1/2 (Apache-2.0).
4. **Many visibility methods need scanner positions** (Ochmann 2014/2016, Mura 2014/2016, Turner &
   Zakhor 2014, Ikehata 2015, Cai & Fan 2021, and Yang 2021's *released code*). We don't have these. The
   methods that work without them synthesise viewpoints (Ambruș 2017), cast rays between wall
   patches (Ochmann 2019) or use distance fields (Bobkov 2017). **None of them has public code.**
5. **Doorless corridor junctions are unsolved by every geometric method**, by the authors' own
   admission (Ochmann 2019 on *S3DIS Area 3 itself*, Hydra IJRR 2024, ProClosure 2026). **And S3DIS cuts
   corridors with straight lines where there is no wall** (§8.3). The ground truth itself is therefore
   inconsistent at exactly that place, and the evaluation has to deal with it explicitly.
6. **The one-pano-per-room structure is the asset nobody else has.** K = N is a free stopping rule
   for merging, and needs no poses. After PanoPin runs, each segment's pano count (0, 1 or ≥2) shows
   where it under- or over-segmented. **No paper alternates room partitioning of a merged cloud with
   panorama localisation** (thread E searched specifically). That looks like a real gap.
7. **Recommendation (§9):** (1) a deterministic baseline — ceiling-band occupancy image → seeded
   watershed / region-adjacency merge to K = N; (2) the same band image into point-prompted SAM, which
   replaces the thesis's SAM3 text route; (3) a closed loop that re-seeds the partition from PanoPin's
   own pano positions. Score all three, plus the thesis SAM3 route as a control, against the S3DIS
   partition with the protocol in §8.4.

---

## 1. The problem, in PanoPin's terms

**Interface.** `panopin.cli seed --clouds {room: cloud_path}`. For each pano, CPO searches **inside each
candidate room's extent**; the best-scoring room wins; FGPL then refines inside that room's region.
Room segmentation must therefore produce a dict of per-room point clouds, keyed by arbitrary names
(the names carry no meaning, since the panos are anonymised).

**Hard constraints from the deployment setting.**
- **One merged cloud, no scan structure.** A real user gives one merged TLS cloud. S3DIS is the same:
  Matterport scans merged, with no station positions shipped. The local `Area_3/1.e57` holds four
  clouds named by room (`office_4…7 - Cloud`) with **no pose blocks**. It is a GT room export, not
  a source of station positions, and a solver must never read it.
- **Camera positions unknown at segmentation time** (the chicken-and-egg). Estimating them is
  PanoPin's job. Any method seeded by scanner or camera positions applies only after a first
  localisation pass (§7).
- **N panos ↔ N rooms**, one-to-one, in deployment. (The S3DIS test scenes have several panos per
  room. Sample one per room to mimic deployment; §8.4.)
- **Data characteristics:** Manhattan offices + corridors, furnished, doors often open, hallways that
  run into neighbouring spaces with no door.
- **PanoPin's mandate (CLAUDE.md, D1):** deterministic / rule-based strongly preferred; learning only
  if a deterministic route demonstrably can't hit the bar.

**What a segmentation error costs PanoPin.**
- **Merge** (two rooms in one segment): both panos compete for the same candidate. CPO's search
  area doubles, and FGPL refines over a two-room region. FGPL alone on a 5-room map got 1/5 rooms
  (PROGRESS 2026-07-27), so a merged region brings back exactly the ambiguity PanoPin removes.
  The damage is moderate, and it shows up as a segment that wins two panos.
- **Split** (one room in two segments): the camera's half is missing part of the geometry the pano sees.
  Its colour score weakens and may lose to another room. If the camera falls in neither half's
  extent, CPO cannot find it at all.
- **Camera outside its segment** is the fatal case. That makes **pano containment** — is each GT
  camera position inside the segment matched to its room? — the operational metric (§8.4). Note
  **D17**: 9/85 Area_3 panos already sit *outside their own GT room cloud*, so even the GT partition
  does not reach 100% on that metric.

**What the thesis front-end does today** (`scan2measure-webframework/src/preprocessing/generate_density_image.py`,
`src/segmentation/SAM3_room_segmentation.py`, `SAM3_mask_to_polygons.py`):
- RANSAC floor and Manhattan histogram alignment.
- Projection of **all points (full height)** onto a fixed **256×256** canvas, with a 10% pad and aspect
  ratio preserved. Pixel value = raw point count / max.
- CLAHE + invert, then SAM3 with the text prompt "floor plan" at confidence 0.1. Masks covering >70% of
  the image or >85% of the occupied pixels are dropped.
- Douglas–Peucker polygons.

On the test scenes, the fixed 256 px canvas gives 0.057–0.094 m/px (measured below). That is fine for a
0.9 m door, but the resolution falls in proportion to building size: about 0.23 m/px on a 50 m floor.

| Test scene | Points | XY extent (m) | m/px at 256² |
|---|---|---|---|
| `Area_3_manhattan4` (hallway_1, office_3/5/7) | 3 403 895 | 8.4 × 16.4 | 0.077 |
| `Area_3_manhattan6` (+ office_4/6) | 4 960 388 | 8.4 × 16.4 | 0.077 |
| `Area_2_manhattan4` (hallway_2, office_6/7/8) | 2 569 755 | 12.1 × 4.5 | 0.057 |
| `Area_2_manhattan7` (+ office_4/5, hallway_3) | 5 990 502 | 19.9 × 10.4 | 0.094 |

---

## 2. Density-image / floorplan-reconstruction methods (learned)

### 2.1 The family at a glance

| Method | Venue | Output | Struct3D room F1 | Real cross-domain room IoU (S3D→SceneCAD) | Code · licence |
|---|---|---|---|---|---|
| Floor-SP (Chen, Liu, Wu, Furukawa) | ICCV 2019 | room polygons (Mask R-CNN + shortest path) | 88 | – | MIT · 785 s/scene |
| MonteFloor (Stekovic et al.) | ICCV 2021 | polygons (MCTS over proposals) | 95.0 | – | **no official code** |
| HEAT (Chen, Qian, Furukawa) | CVPR 2022 | wall-centreline planar graph | 95.4 | **52.5** | **non-commercial**; GPL-3.0 for research |
| RoomFormer (Yue et al.) | CVPR 2023 | ≤20 polygons × ≤40 vertices | 97.3 | 74.0 | MIT |
| SLIBO-Net (Su et al.) | NeurIPS 2023 | slicing-box polygons | 98.4 | – | **no code** |
| PolyDiffuse (Chen, Deng, Furukawa) | NeurIPS 2023 | refiner on top of RoomFormer | 98.4 (RF+PD) | – | GPL-3.0 |
| PolyRoom (Liu et al.) | ECCV 2024 | polygons, seg-initialised | 98.3 | 85.2 | **no LICENSE** |
| FRI-Net (Xu et al.) | ECCV 2024 | room-wise implicit → polygon | 99.1 | 80.6 | **no LICENSE** |
| PolyGraph | IEEE TVCG 2025 | wall graph | 96.7 | – | **no LICENSE** (authors [UNVERIFIED]) |
| CAGE (Liu et al.) | arXiv 2509.15459 (no venue) | interior polygons | 99.1 | 85.6 | MIT + **Commons Clause** |
| Raster2Seq (Phung, Averbuch-Elor) | SIGGRAPH 2026 | polygon sequences; has a S3D-density checkpoint | 98.7 [table from arXiv HTML, not PDF] | – | MIT, ungated HF weights |
| FloorNet (Liu, Wu, Furukawa) | ECCV 2018 | corners + IP | – | – | MIT; **needs posed RGB-D stream** |

Room F1 uses the MonteFloor protocol (best-IoU match at IoU > 0.5, Structured3D test, 256² density map).
Sources: RoomFormer Tab. 1–3, FRI-Net / PolyRoom / CAGE tables. In-domain SceneCAD room IoU is
84.9–92.8 for the same models.

### 2.2 Why these numbers don't transfer to us
- **No multi-room real evaluation exists.** RoomFormer's preprocessing README: *"we filtered out
  multi-room scenes (which are rare) in SceneCAD"*. Structured3D is synthetic and residential. Room F1
  of 97–99 says nothing about corridors, open doors or TLS offices.
- **Measured collapse on S3DIS.** Brunklaus, Kellner & Reiterer, *Remote Sensing* 17(7):1124, 2025, treat
  each S3DIS Area as one scene with rooms as instances. Trained on Structured3D, Area 5 gives
  Mask3D mAP 0.0 / RoomFormer 0.2, with 67–68 of 68 rooms missed. After **4 days of fine-tuning on S3DIS
  Areas 1–4 and 6**, they reach mAP 17.9, mRec50 55.8, mmIoU 78.4. The authors blame offices vs.
  apartments, an order-of-magnitude larger extent, and inconsistent hallway labels. Code MIT.
- **Talotta et al.** (GeoIndustry '23, Amazon; doi 10.1145/3615888.3627810): *"training with synthetic data
  is useful but not sufficient … with RoomFormer on real indoor scans"*.
- **Preprocessing mismatch** (thread A read RoomFormer's `PointCloudReaderPanorama.py`). Training
  images:
  - quantise points to 10 mm × 10 mm × 100 mm voxels and count **unique voxels per column**, which
    roughly encodes vertical extent, so walls dominate;
  - stretch **each axis independently** to 256 px after a 10% margin, so the aspect ratio is not kept;
  - normalise by the max;
  - come from Structured3D's `full` (furnished) renders.

  Our image uses raw counts and keeps the aspect ratio, so it is out of distribution twice over. This
  is a cheap fix if a learned model is ever re-tried.
- **Polygon validity.** RoomFormer has no validity constraint, so self-intersecting or overlapping
  polygons happen, which matches the thesis's observation. PolyRoom and CAGE were designed to avoid
  this.

### 2.3 Mapping polygon output to `{room: cloud}`
1. Point-in-polygon on XY, lifted over all Z (split by storey first, since every method assumes one floor).
2. Interior-polygon methods (RoomFormer, PolyRoom, FRI-Net, CAGE, SAM masks) leave the **walls between
   rooms unassigned**. Either buffer each room outward by half a wall thickness (Albadri 2025) or assign
   leftovers to the nearest room with a distance cap (`distance_transform_edt(return_indices=True)`).
3. Centreline methods (HEAT, PolyGraph) split walls naturally but can leave rooms as open loops.
4. Self-intersections need `make_valid` before the containment test. Overlaps need a priority rule.

**Verdict:** a learned polygon net needs real multi-room fine-tuning data we don't have, and most of the
strong ones carry licence problems. Keep **RoomFormer or Raster2Seq (MIT) with matched preprocessing**
only as a *control* that measures the domain gap, not as a candidate.

---

## 3. Foundation-model segmentation of density images

| Model | Venue | Licence | Notes |
|---|---|---|---|
| SAM (Kirillov et al.) | ICCV 2023 | Apache-2.0 | point/box prompts + automatic mask generator. Albadri 2025 used it |
| HQ-SAM (Ke et al.) | NeurIPS 2023 | Apache-2.0 | sharper thin structures; room masks bleed through 1–3 px walls. No density-map paper |
| SAM 2 (Ravi et al.) | ICLR 2025 | Apache-2.0 | image mode ≈ SAM with a better backbone |
| SAM 3 (arXiv 2511.16719) | ICLR 2026 | custom "SAM License": **commercial use allowed**; acknowledgement required; ITAR/military excluded; terminates on IP litigation vs Meta; HF weights **gated (manual approval)** | 2D **concept** segmentation from noun phrases. No 3D ability. No paper uses it on density maps |
| SAM 3D (arXiv 2511.16624) | – | SAM License | single-image *object-to-3D generation*. **Irrelevant** to rooms despite the name |

**The two papers that apply SAM to top-down rasters of real LiDAR for rooms:**

- **Albadri, González-Cabaleiro, Túñez-Alcalde, Fernández, Díaz-Vilariño**, "A SAM-Based Approach for
  Automatic Indoor Point Cloud Segmentation", ISPRS Archives XLVIII-G-2025, 131–137 (CC BY 4.0).
  - **Pipeline:**
    - RANSAC ceiling, then a **0.3 m slice 0.5 m below the ceiling** ("above the furniture and below
      the beams"), rendered as a 1 cm/px binary occupancy image.
    - SAM automatic masks (points_per_side 11–30, IoU 0.85, stability 0.95).
    - Masks smaller than 1.5 m² are rejected.
    - Masks are lifted to the full cloud, and each room is **buffered outward by half a wall
      thickness** (0.07–0.12 m) to pick up wall points.
  - **Data:** 3 real single-storey buildings (5–24 M points, including ISPRS CS2) with clutter and open
    and closed doors.
  - **Results:** 65/67 rooms right (4 over-, 1 under-segmentation; match rule ≥75% overlap).
  - **Failure mode:** **SAM produced no corridor masks in 2 of 3 cases**, so a second pass on the
    leftover points was needed. Needs no scanner poses. No code.
- **FloorSAM** (Ye et al., arXiv 2509.15750, 2025; venue none).
  - **Pipeline:**
    - Keep only points within 0.1 m of each 0.1 m column's max Z (ceiling level).
    - Log + blur + CLAHE.
    - **Prompt points = local density peaks**, then multi-mask SAM.
    - Multi-stage IoU filtering, then contour regularisation; doorframes are fitted to restore topology.
  - **Data:** GibLayout (NavVis, 3–10 rooms) and ISPRS TUB2/UoM.
  - **Results:** 98/100 and 17/17 rooms. Beats pretrained Floor-SP / HEAT / RoomFormer / PolyRoom on
    boundary P/R, though those baselines were not fine-tuned.
  - **Failure mode:** missing ceiling points.
  - **Code: an empty placeholder repo, no licence.**

**Implication for the thesis route.** Keep the "SAM on a top-down image" idea, but change three things:
the **input** (ceiling band, not full height), the **prompt** (points / automatic, not "floor plan"), and the
**lift** (buffer + nearest-room for walls).

---

## 4. Direct 3D / 2.5D point-cloud room segmentation

### 4.1 Applicability matrix (the scanner-position column is the gate)

| Method | Scanner pos / per-scan needed? | Core idea | Open doors · doorless corridor | S3DIS result | Code |
|---|---|---|---|---|---|
| **Ochmann, Vock, Klein 2019**, ISPRS JPRS 151:251 | **No** (scan count "for reference only") | rays between 40 cm wall patches → visibility graph → Markov clustering (no K) → ILP cell complex | MCL cuts weak links; **on S3DIS Area 3 a hallway "ends without a terminating wall surface" — a virtual wall was hand-drawn** | Area 3, qualitative | none; Gurobi + OptiX; needs **oriented** normals |
| **Ambruș, Claici, Wendt 2017**, **IEEE RA-L 2(2):749** (*not CVPR*) | **No**: synthesises viewpoints along the free-space medial axis (3 m radius) | openings matched to a parametric door model → obstacles; cell complex + α-expansion; merge if <20% of separating edge is real wall | doors explicit; corridors over-split then merged | not evaluated (own 10 RGB-D clouds: P 0.95 / R 0.94 / IoU 0.89; 30–120 s) | none |
| **Bobkov et al. 2017**, ICME | **No** | 18 cm voxels; furniture-robust anisotropic potential field + ray-cast visibility; HDBSCAN | doors = PF minima; weak on long corridors | **wrong rooms 2/44, 10/40, 5/23, 7/68** (Areas 1/2/3/5) vs Armeni 8/12/7/13 | third-party reimplementation, **no licence** |
| **Jung, Stachniss, Kim 2017**, IJGI 6:206 | **No** | rasterise **ceiling band only**; window larger than the largest door finds room cores; skeleton linking closes doorways; connected components | closes doors narrower than the window; **does not split openings wider than it** | – (Bormann furnished maps: corr 84.8 / compl 67.8) | none |
| **Li, Shi, Sun 2025**, IEEE JSTARS 18:18358 | **No** | 1 m supervoxels → 2D; doorway removal by geometric features; anchor-pixel wall linking; wavefront | explicit | **Area 1 93.65%, Area 3 96.31%** accuracy; free-space height threshold raised 0.5 → 0.8 m for S3DIS | none |
| Armeni et al. 2016 (the S3DIS paper), CVPR | **No** | axis histograms, peak-gap-peak wall filters, merge graph; **Manhattan** | hallway bottlenecks over-segmented | **ARI 0.94/0.82/0.69/0.66, mean 0.77** | [UNVERIFIED] still downloadable |
| Gourguechon, Macher, Landes 2023, ISPRS Annals X-M-1 | No (MLS trajectory optional) | occupancy + density-discontinuity images, erosion kernels, merge | furnished, open + closed doors tested; 45 rooms, 7 over / 1 under | – | none |
| Macher, Landes, Grussenmeyer 2017, Appl. Sci. 7:1030 [abs] | No | 2D region growing on a **slice above door height** — origin of the slice trick | open doors bypassed | – | none |
| Frías et al. 2020, ISPRS Archives | No | 3D erosion with a cube of side = door width; connected components | erosion merges corridors with rooms | – | none |
| Hübner et al. 2021/2022 (VoxIR), ISPRS JPRS 181 / Annals V-4 | No (2022 adds point-cloud input) | voxel labels incl. "Wall Opening"; rooms + transition spaces | weak at corridor/adjacent separation (Cui 2026) | – | **MIT** (C#) — the only permissive geometric room code found |
| Martens & Blankenbach 2023 (VOX2BIM+), PFG 91 | No | wall voxels → DT seeds → bounded region growing | – | – | none |
| Tang 2022 (AutCon) / Wang 2026 (JSTARS) / Tu 2026 (IJGI) | No | 3D semantic seg → morphology / watershed + MRF / grid completion | semantic doors | S3DIS reported (Tang: IoU 93.5% over 13 rooms [abs]) — **semantic networks trained on S3DIS ⇒ leakage risk** | none / FloorSG repo **404** / on request |
| Chen et al. 2024, JAG 127 [abs] | No | cross-section arrangement; rooms from ipa (Bormann) room maps | – | 6 S3DIS scenes | repo, **no licence** |
| Drobnyi, Li, Brilakis 2024, JCCE [abs] | No | grow empty regions; doors = transitions between spaces | – | S3DIS + TUMCMS, numbers not extracted | none |
| Mask3D rooms (Brunklaus 2025) | No | learned 3D instance seg | learned | Area 5 mAP 17.9 *after* S3DIS fine-tune | MIT |
| Yang et al. 2021, IJGI 10:739 | **Paper says optional; released code requires per-scan viewpoints or MLS trajectory** | ray-cast occupancy → EDT → sphere-packing seeds → 3D growing | corridors about as narrow as doors get filtered out | – | GPL-3.0 |
| Ochmann 2014 / 2016; Mura 2014 / 2016; Turner & Zakhor 2014; Ikehata 2015; Cai & Fan 2021 | **YES** | visibility from scan stations, diffusion / k-medoids / graph cut | – | – | mostly none |

**Enabler worth knowing.** ASPIRE (Michailidis & Pajarola, *Vis. Comput.* 35, 2019 [abs]) recovers TLS
scanner positions from a merged cloud by point-pattern analysis. It could unlock the "YES" row. Whether the
pattern survives in S3DIS's subsampled Matterport data is untested, and there is no code.

### 4.2 Takeaways
- The three strongest pose-free 3D methods (Ochmann 2019, Ambruș 2017, Bobkov 2017) all lack public
  code. Reproducing any of them is a project in itself: CGAL arrangements, ray casting, MCL or
  α-expansion, and Gurobi for Ochmann.
- The S3DIS numbers that do exist come from simple raster or voxel pipelines (Li/Shi/Sun 2025: 96% on
  Area 3; Bobkov: 5/23 rooms wrong on Area 3). Raster methods are not handicapped on our data.

---

## 5. Robotics: 2D occupancy-map room segmentation and 3D scene-graph rooms

### 5.1 The comparative survey: Bormann et al., ICRA 2016 (doi 10.1109/ICRA.2016.7487234)
"Room segmentation: Survey, implementation, and analysis". It compares four methods on **20 office maps
at 0.05 m/cell, each with and without furniture**. Its own Voronoi random field (VRF) method is *not*
in the paper; it was added to the code later, and no published VRF numbers were found.

| Recall / Precision (%) | Morphological | Distance transform | Voronoi | Feature (AdaBoost) |
|---|---|---|---|---|
| no furniture | 98.1 / 88.5 | 96.9 / 88.4 | 95.0 / 94.8 | 89.2 / 90.4 |
| **furnished** | **84.6 / 90.5** | **76.1 / 88.4** | **86.6 / 94.5** | **85.1 / 87.1** |
| runtime (s) | 1.6 | 1.8 | 13.0 | 269 |

- Failure modes reported: morphological and distance-transform methods "grow into the corridor".
  Voronoi "often oversegments corridors".
- **Caveat:** the README admits most maps are "simply drawn using graphics software", and the furniture
  is artificially drawn clutter. The benchmark therefore *understates* what a real furnished scan does.
- **Code:** `ipa320/ipa_coverage_planning` (`ipa_room_segmentation`), ROS1 only (noetic_dev, pushed
  2025-11). There is **no LICENSE file**; `package.xml` says *"LGPL for academic and non-commercial use"*.
  The algorithms are standalone C++/OpenCV classes (`segmentMap()`), usable without ROS.
- **Metric-scale defaults to re-derive:** room-area bounds and `max_area_for_merging` = 12.5 m².

### 5.2 Beyond Bormann

| Method | Venue | Pose-free? | Evidence | Code · licence |
|---|---|---|---|---|
| **ROSE²** (Luperto, Kucner, Tassi, Magnusson, Amigoni) | **IEEE RA-L 7(3), 2022** (doi 10.1109/LRA.2022.3186495) | yes | FFT-based clutter removal → dominant-direction wall lines → face clustering. **10 real cluttered maps: IoU 73.3 vs Voronoi 28.65, morph 51.2, distance 54.7**; Bormann furnished P/R 93.5/91.0 | Python **MIT** (paper version, "needs adjustments"); ROS version GPL-3.0 |
| **Sharif, Mohan, Suvarna** (Neato) | arXiv 2303.13798 (no peer-reviewed venue found) | yes | ridge-filter declutter → down-sampling cores → watershed → merge (L0 = 3 m). **Best published furnished-Bormann result: R 94.1 / P 98.1** | Python **BSD-3-Clause** |
| MAORIS (Mielle et al.) | ICRA 2018 | yes | free-space "ripples"; MCC 0.98. **Unfurnished maps only, by the authors' own statement** | GPL-3.0 (C++); MIT rewrite on Codeberg |
| DuDe (Fermin-Leon et al.) | ICRA 2017 | yes | contour decomposition; P/R barely changes with furniture (86.3→85.8 R); always splits L-corridors | **no licence** |
| Area Graph (Hou et al.) | ICAR 2019 | yes | Voronoi + alpha shapes | GPL-3.0 |
| Hiller et al. | IROS 2019 | yes | CNN door detection + watershed; real maps qualitative only | BSD-3 (data generator) |
| Kleiner et al. (iRobot) | IROS 2017 | yes | clutter removal + watershed + merge; **numbers [UNVERIFIED]** (no open PDF) | none |
| **He, Sun, Hou, Ha, Schwertfeger** | Auton. Robots 45(5), 2021 | **yes — TLS cloud "without pose"** | 0.15 m voxels; **doors found as free columns with a lintel above**; 6 TLS datasets; <5 s | none |
| **HOV-SG room step** (Werby et al.) | RSS 2024 | system needs poses; **the room function needs only the cloud** | points in [floor + 1.5 m, ceiling − 0.3 m] → 5 cm BEV histogram → wall mask (0.25·max) → EDT → Otsu seeds → watershed. HM3DSem P/R 84.1/83.6; **but Point2Graph measured it at AP50 0.06 on MP3D** | MIT (README: "academic; contact for commercial") |
| Hydra (Hughes, Chang, Carlone) | RSS 2022 / IJRR 2024 | system needs poses; room step = places graph | dilation sweep + persistent homology to choose δ ∈ [0.5, 1.2] m; *"may fail … open floor-plans"* | BSD-2 |
| **Point2Graph** (Xu et al.) | ICRA 2025 | **yes, point cloud only** | z-slices → occupancy/border maps → RoomFormer fine-tuned on 100 MP3D scenes: AP50 0.53 | **no licence** |
| **ProClosure** (Muthuraj et al.) | arXiv 2609.12614 (Sep 2026) | uses camera trajectory as seeds, **but farthest-point seeds lose only 0.802→0.781 mIoU (p = 0.064)** | progressive boundary closure seals each opening at its own scale; door frames added as boundaries; HM3D F1@0.25 0.890 vs HOV-SG 0.741; worst floor is open-plan | **no licence** |
| OccuSG | arXiv 2606.13727 | needs poses | DuDe-based room nodes; *"wall-accurate room boundaries remain an open problem"* | MIT |
| SeLRoS (Kim & Min) | IROS 2024 | needs objects | LLM merges furniture-fragmented segments; classic methods ≈ 60 IoU on furnished ProcTHOR | MIT |
| Kimera, Hydra-Multi, Clio, Khronos, S-Graphs+ | various | **need trajectories** | S-Graphs+ fits 4-wall / 2-wall (corridor) room models | – |

### 5.3 Takeaways
- Narrow-passage cues (Voronoi critical points, erosion, distance-transform thresholds, Hydra dilation,
  ProClosure) **do** work for open doors: an open door is still a narrow gap. They **fail** where a doorway
  is as wide as the corridor, where a corridor opens without constriction, and where furniture creates
  false constrictions. The last case is why Voronoi drops to 28.65 IoU on real cluttered maps.
- Wall-line methods (ROSE², Liu & von Wichert 2014, S-Graphs+) can split along the *virtual continuation*
  of a wall line. That is the only family with a mechanism for constriction-free transitions, and only
  when a wall line exists on either side.
- With a 3D cloud, the declutter step that ROSE² does by FFT and Sharif does by ridge filter comes almost
  free from **choosing the height band** (HOV-SG, He 2021).

---

## 6. Frontier: open-vocabulary 3D, LLM layout, panorama-assisted

- **No open-vocabulary 3D method outputs rooms.** OpenScene, OpenMask3D, SAM3D-2023, SAI3D, Open3DIS,
  SAMPro3D, Segment3D, Point-SAM, SAM2Point and ConceptGraphs all work at the object or part level. Most
  also need **posed RGB-D frames** to lift 2D masks, which we don't have. "Room" queries return heat maps
  or surface classes, not partitions.
- **LLM layout models** take a z-up point cloud with no poses, but output walls, doors, windows and
  boxes, **not rooms**:
  - SpatialLM (arXiv 2506.07491; NeurIPS 2025 per repo title [UNVERIFIED venue]);
  - SceneScript (ECCV 2024).

  Their encoders are **CC-BY-NC-4.0**. They could feed a wall + door → room closure step, but are
  untested on TLS offices.
- **Scene-graph systems that do output rooms** (HOV-SG, Hydra, OccuSG, ProClosure, Prior-SG) all use a
  *geometric* room step; the foundation model only names the rooms (§5.2).
- **Panorama-assisted floor-plan assembly** (ZInD CVPR 2021; Extreme-SfM ICCV 2021; SALVe ECCV 2022;
  CoVisPose; Graph-CoVis; PSMNet; Floorplan-Jigsaw ICCV 2019) solves the *inverse* problem: it places
  rooms from about one pano per room using HorizonNet layouts and door detections, with **no point
  cloud**. The pattern is useful, the tools are not. Licences: SALVe CC BY-NC-ND; ZInD data
  non-commercial; Extreme-SfM no licence.

---

## 7. Using the one-pano-per-room structure, and the chicken-and-egg

Ordered by how much the method depends on pano poses.

1. **K = N as a stopping rule (no poses).** Over-segment (watershed, SAM masks, superpixels), then merge
   the region-adjacency graph down to exactly N. Merge order comes from wall support on the shared
   boundary, i.e. Ambruș's "<20% real wall" rule.
   - **This settles the hardest free parameter in every classical method:** when to stop merging corridor
     pieces and open-door rooms.
   - **Breaks when a space has no pano:** closets, stairwells, an uncaptured corridor stub. Use K ≥ N with
     an "unassigned / no-pano" label, and let the score in item 3 decide.
2. **Pano-layout ↔ region matching (no poses; the thesis idea).** LGT-Net / HorizonNet polygons are
   matched to region footprints, with Hungarian assignment (Kuhn 1955) over the N×N matrix. It is weak
   here: furniture truncates the layouts, open doors make them leak, and Manhattan offices are many
   near-identical rectangles. FGPL's 40/60-office splits exist precisely because similar rooms are hard.
3. **PanoPin as the verifier (after one pass).** Run PanoPin on candidate segments. Each pano is
   assigned to a segment and gets a coarse position, which yields per-segment pano counts:

   | Pano count | Meaning | Repair |
   |---|---|---|
   | 2 | merge | split the segment with 2 seeds |
   | 0 | split or pano-less space | merge it into its best-supported neighbour, or drop it |
   | 1 | consistent | keep |

   Re-seed a **marker-controlled watershed** (Meyer 1994) or **random walker** (Grady 2006, TPAMI
   28(11); `skimage.segmentation.random_walker`) with the N pano positions on the band image, then
   re-run PanoPin. This is the Ikehata / Ochmann visibility clustering with *real* viewpoints, which
   is what those methods originally assumed.
4. **Localise first, partition second.** FGPL (CVPR 2024) was evaluated on joined multi-room maps: all
   of 2D-3D-S Area 1, plus 40- and 60-office splits. CPO builds translation candidates from the free
   space of the whole cloud. So no partition is *needed* to localise. **But on our data this is what
   PanoPin exists to avoid:** FGPL alone got 1/5 rooms and 82% wrong-room on the 5-room map, and CPO
   across a whole cloud runs into the D19/D20 speed wall. Use it only as the second half of the loop
   in item 3, not as the first pass.

**Gap.** Thread E found **no paper that alternates room partitioning of a merged cloud with panorama
localisation**. Floorplan-Jigsaw and Extreme-SfM come closest, but they start from per-room partial
scans or panos only. Item 3 is a defensible contribution.

---

## 8. Evaluation

### 8.1 Metrics in the literature

| Metric | Definition | Used by |
|---|---|---|
| Bormann recall / precision | per GT room: max overlap with a segment / GT area. Per segment: max overlap / segment area. Under-segmentation inflates recall, over-segmentation inflates precision | Bormann 2016, ROSE², DuDe |
| Room P/R/F1 @ IoU > 0.5 | one-to-one best-IoU matching (`s3d_floorplan_eval` code) | MonteFloor, HEAT, RoomFormer and successors |
| Room / Room++ @ IoU > **0.7** | Room++ also requires correct adjacency (a topology metric) | Floor-SP, FloorNet |
| Adjusted Rand Index | chance-corrected pair counting (Hubert & Arabie 1985) | **Armeni 2016 on S3DIS: mean 0.77** |
| Panoptic Quality = SQ × RQ | IoU > 0.5 matching (Kirillov et al., CVPR 2019) | general |
| Variation of information | H(P\|G) + H(G\|P) (Meilă 2007); the two terms separate over- from under-segmentation | general |
| mAP + "Successfully Detected Rooms" | needs per-segment confidences; SDR formula [UNVERIFIED] | Brunklaus 2025 |
| ISPRS completeness / correctness / accuracy | **walls only**; spaces judged by an expert panel | Khoshelham et al. 2017–2021 |

### 8.2 Benchmarks with room ground truth
- **Real point cloud + room GT + panoramas: only S3DIS / 2D-3D-S and Matterport3D** (region polygons
  extruded to the ceiling; 90 buildings).
- Structured3D (synthetic, 3,500 scenes) and ZInD (real panos + floor plans, no cloud; non-commercial)
  have panos but no real cloud.
- SceneCAD is mostly single-room.
- ISPRS indoor benchmark: real, with reference BIMs, but rooms are not scored.
- Bormann's 20 maps: 2D, largely drawn.
- HM3D-Sem: rooms implied by object annotations.
- **Already on disk:** the Astacus GT BIM for S3DIS Area 1 + 2 (`Point_360/data/s3dis_bim_gt/`). If its
  IFC has `IfcSpace` entities [UNVERIFIED], it gives a **second, independent room GT for the Area_2
  scenes**, with a different hallway convention from S3DIS's.

### 8.3 S3DIS ground-truth facts that change the scoring (measured on our v1.2 copy)
- **Merged scene = exact union of room clouds.** Point counts match exactly (Area_3_manhattan4
  3 403 895; Area_2_manhattan4 2 569 755), so every point has one unambiguous GT room.
- **Rooms are disjoint; walls are not duplicated.** Points:
  - 0% exact duplicates between neighbouring rooms;
  - 0.27–0.60% of hallway points lie within 5 cm of an office cloud (this session);
  - 0 points within 1 mm across all tested pairs (thread D).

  Each room holds only the inner face of its own walls; office-to-office gaps of 0.11–0.22 m equal the
  wall thickness. (Thread C assumed shared walls are duplicated. The measurement shows otherwise.)
- **Some hallways are cut by straight lines where there is no wall.** **Correction (§10.1): the
  Area_2 cut below is a door across the corridor**, so it has a lintel; only the wall check was done
  here.
  - Area_2: `hallway_2` max x = 7.77 = `hallway_3` min x (an L-junction), **inside `Area_2_manhattan7`**
    (confirmed this session).
  - Area_3: `hallway_1` | `hallway_2` at x = 19.35. Our Area_3 scenes contain only `hallway_1`, so here
    the cut coincides with the scene crop.

  A segmenter that correctly returns one continuous corridor gets scored as a *merge*. The original
  authors also call the hallway convention inconsistent (Armeni 2016; Bobkov 2017; Brunklaus 2025).
- Area 1 / Area 3 corridor points sit above the ceilings of neighbouring rooms (Li/Shi/Sun 2025). This
  matters only for top-down methods that use full-height maxima.
- Never build multi-room scenes from `Stanford3dDataset_v1.2_Aligned_Version`: it rotates each room on
  its own. That copy also has a corrupted line in `Area_3/hallway_2/hallway_2.txt:926337`; the
  non-aligned v1.2 is clean.

### 8.4 Proposed evaluation protocol (for the follow-up spec — not implemented)

**Data.**

| Role | Scenes | Rooms | Why |
|---|---|---|---|
| Dev (tune here) | `Area_3_manhattan4`, `Area_3_manhattan6` | 4 / 6 | — |
| Holdout (score once) | `Area_2_manhattan4`, `Area_2_manhattan7` | 4 / 7 | different building area; `manhattan7` holds the wall-less hallway_2\|hallway_3 cut |
| Stretch | whole `Area_2.ply` | 40 | scale and non-Manhattan robustness |

- **GT labels:** S3DIS per-room membership.
- **Fairness:** a solver reads only the merged `.ply`, never the per-room `.txt`, `1.e57` or names. This
  mirrors D5.

**Scoring basis.** Compute every overlap on **5 cm occupied voxels**, not raw points. Matterport density
varies strongly with distance, so raw point counts over-weight areas near the scanner. Use an
**ignore band** of about 0.3 m around a virtual hallway cut and about 0.2 m around door thresholds.
Report against **two GT variants**:
- S3DIS as shipped;
- wall-less hallway chains merged into one room. The ISPRS benchmark convention ("an opening … does not
  constitute the subdivision of a space") is the precedent.

**Metrics.**
- **Primary:**
  - **pano containment**: the fraction of GT camera centres inside the segment matched to their room.
    Report it next to the GT partition's own containment, since D17 shows the GT is below 100%.
  - **room F1 at IoU 0.5 and 0.7**, with PQ / SQ / RQ.
- **Failure decomposition:**
  - #split GT rooms (no segment holds ≥ 80% of the room);
  - #merged segments (a segment holds ≥ 10% of each of ≥ 2 rooms), naming the rooms;
  - Bormann P/R;
  - VI split into its two terms.
- **Comparability:** ARI (Armeni 2016 precedent).
- **Per-scene tables.** With 4–7 rooms, one error moves F1 by 15–25 points; no pooled headline without
  per-scene rows.

**Downstream (the number that matters).** Feed the predicted `{room: cloud}` to `panopin.cli seed` in
two regimes:
- **one pano per room**, sampled across several random draws, matching deployment;
- **all panos**, matching the existing acceptance runs.

Compare against the same run with GT room clouds: room accuracy, wrong-room rate, and the 0.086 m median
of D38.

**Baselines in every table.**
- **B0** = the thesis route unchanged (full-height 256² image + SAM3 "floor plan"). It anchors every claim.
- **B1** = connected components of band free space with no doorway closing. This shows how much doorway
  handling buys.
- **B2** = ipa morphological / Voronoi on the band image.
- **Optional:** RoomFormer / Raster2Seq with matched preprocessing, as a domain-gap control.

---

## 9. Ranked short list for PanoPin

### ① Ceiling-band occupancy image → seeded watershed → merge to K = N (deterministic baseline; recommended first)

**Pipeline.**
1. **Align:** floor/ceiling from the z-histogram (thesis RANSAC + Manhattan alignment already exist).
2. **Rasterise:** points in a band [ceiling − ~0.6 m, ceiling − ~0.1 m], **above door heads**. Tune it
   against the Jung / HOV-SG / Albadri choices: Albadri uses ceiling − 0.5 m, 0.3 m thick; HOV-SG uses
   floor + 1.5 m to ceiling − 0.3 m. Use a **fixed metric resolution (5 cm/px)**, not a fixed 256 px.
   Occupied pixels in the band are walls / lintels; the footprint comes from a full-height or floor-band
   projection.
3. **Partition:** EDT on the free space → markers (Otsu or h-maxima, as in HOV-SG / Sharif) → watershed.
   This over-segments.
4. **Merge:** region-adjacency merge ordered by wall support on the shared boundary (Ambruș's 20% rule),
   **stopping at N regions**, with a floor on the boundary-support score so a real wall is never merged
   across just to hit N.
5. **Lift:** every point (full Z) labelled by its XY cell; wall and lintel cells by nearest label via EDT
   indices. Name the regions `seg_0..seg_{K-1}`.

**Why it's first.**
- Deterministic and CPU-only (D1).
- Every component has a published, measured precedent on real furnished scans: the band (Jung,
  Albadri, HOV-SG), DT + watershed (HOV-SG, Sharif: furnished R 94.1 / P 98.1), and wall-support merge
  (Ambruș).
- Permissively licensed references to read against (HOV-SG MIT, Sharif BSD-3, ROSE² MIT), though about
  200 lines of our own is cleaner than vendoring.
- Seconds per scene on 3–6 M points.
- It reuses the thesis alignment and metadata plumbing.

**Risks.**
- **(a)** The doorless corridor junction (`Area_2_manhattan7`) has no lintel, so the band does nothing
  there. Only K = N or a wall-line continuation (ROSE²-style) can split it, and S3DIS's cut there is
  arbitrary anyway (§8.3).
- **(b)** S3DIS Matterport ceilings may have holes. FloorSAM's failure mode was missing ceiling points,
  and Li/Shi/Sun had to raise a threshold for Areas 1 / 3.
- **(c)** Glass partitions and low-ceiling soffits.
- **(d)** K = N fails when a space has no pano; fall back to a support threshold plus K ≥ N.

**Fallback if (a) dominates:** add ROSE²-style dominant-direction wall-line extension as extra cut
candidates for the merge step.

### ② The same band image → point-prompted SAM (replaces the thesis's SAM3 text prompt)

**Pipeline.** The band image from ①, with FloorSAM-style prompts (density-peak points, or the DT maxima
from ①) or SAM's automatic mask generator, on **SAM / SAM 2 / HQ-SAM (Apache-2.0)**. Then Albadri's
filtering: area ≥ 1.5 m², IoU dedup, half-wall buffer lift, and a second pass on the leftovers for
corridors. Merge to K = N as in ①.

**Why.**
- It is the closest published match to the user's working assumption, and it reuses the thesis SAM code
  path.
- Albadri 65/67 and FloorSAM 98/100 + 17/17 are the best room counts on real multi-room LiDAR in this
  review.
- SAM is often better than watershed on irregular room shapes.

**Risks.**
- Corridors are the documented failure: no masks in 2 of 3 Albadri cases.
- GPU needed. PanoPin already pins single-threaded CPU for determinism (D37), so SAM inference must also
  be pinned and checked for reproducibility.
- Neither paper released code, so this is a re-implementation from parameters.
- If SAM 3 is kept, use its *visual* (point/box) prompts. Its licence is fine for research, but the
  weights are gated.

**Role.** Run ② head-to-head against ① on the same band image. That isolates the segmenter from the
image. If ① ≈ ②, prefer ① (D1).

### ③ Closed loop: segment → PanoPin → re-seed from pano positions → PanoPin again (the contribution)

**Pipeline.**
1. Take ①'s (or ②'s) K = N candidates.
2. Run `panopin seed`.
3. Read the per-segment pano counts and coarse positions.
4. Re-partition with a **marker-controlled watershed or random walker seeded by the N pano positions**
   on the band image. Optionally add pano-visibility affinities: the points each localised pano sees by
   ray casting.
5. Re-run PanoPin.
6. Iterate until the pano → segment assignment stops changing (expect 1–2 rounds).

**Why.**
- It turns the one-pano-per-room structure into both a stopping rule *and* an error detector (2 panos =
  merge, 0 = split).
- It converts the unsolved doorless-junction case into a seeded problem: two seeds on either side of a
  wall-less transition settle it.
- No published method does this (§7).

**Risks.**
- A wrong first-pass localisation can make the loop *confirm* its own mistake. Mitigate by re-seeding
  only from panos that pass PanoPin's confidence gate (D30 / D32 low-percentile score).
- Costs one extra PanoPin pass (CPO is the slow part, D19 / D20).
- It cannot fix a first pass whose segments don't contain the camera at all. That is why the
  containment metric is primary.
- Needs ① or ② to be decent first; it is an add-on, not a replacement.

### Not short-listed, and why
- **Learned polygon nets** (RoomFormer family): no real multi-room evaluation; near-zero on S3DIS without
  4 days of in-domain fine-tuning; most strong ones are unlicensed or restricted. Keep only as a control.
- **Ochmann 2019 / Ambruș 2017 / Bobkov 2017:** the strongest pose-free 3D methods, but no code. Each is a
  multi-week reproduction (Gurobi / OptiX / CGAL), and Ochmann already needed a hand-drawn wall on
  S3DIS Area 3.
- **Scanner-position methods:** inapplicable without ASPIRE-style pose recovery, which is itself untested
  on S3DIS.
- **Open-vocab 3D / LLM layout:** no room output, posed frames needed, or non-commercial encoders.

---

## 10. Open questions for the user (decide before the spec)

1. **Which GT hallway convention is the target:** S3DIS as shipped, or "wall-less corridor chains =
   one room"? It decides whether a correct continuous corridor counts as a merge.
2. **Is K = N a deployment guarantee** (every enclosed space, corridors included, gets exactly one
   pano)? Or can there be pano-less spaces (closets, an uncaptured corridor)? This decides between K = N
   and K ≥ N + unassigned.
3. Is a GPU acceptable inside PanoPin's seed step? This decides whether ② is in scope beyond a
   comparison.
4. Should the Astacus Area_2 IFC be checked for `IfcSpace`, as an independent second room GT?

### 10.1 Decisions (2026-10-01, Ruoyu)

1. **Corridor convention = A, S3DIS as shipped:** corridor pieces stay separate rooms even with no wall
   between them. B (the enclosure convention) is dropped as the target.
2. **K = N is not guaranteed.** Long corridors carry several panos, so the pano count N is at most an
   **upper bound** on the room count. In ③, a segment holding 2 panos is no longer automatically a
   merge. Tie-breaker: if the two pano positions see each other across open floor in the band image,
   keep one room; if a wall or narrowing lies between them, split.
3. **GPU is acceptable** in the seed step (② is in scope).
4. **Don't check the Astacus IFC** for `IfcSpace`.
5. **Section points:** one per room, chosen by the existing deterministic rule
   (`Point_360/s3dis/choose_stations.py::choose`: clearance; corridors by centroid when clearance
   spreads < 0.35 m), not at random. **One section point per corridor is enough for the cropped Area_3
   scenes.** In deployment, a room's candidate panos are the ones PanoPin assigned to that segment.

**What convention A means for the segmenter.** Measured on Area_2. This *corrects* both §8.3 and an
earlier draft of this section.
- The `hallway_2`|`hallway_3` cut at **x = 7.77 m is a door across the full corridor width.** S3DIS
  annotates it: `hallway_2/Annotations/door_1` sits at x 7.63–7.77, y 19.43–21.07, z 0.03–2.10, and
  `hallway_3/Annotations/door_4` at x 7.77–8.74, likely its opened leaves.
- The lintel shows in the ceiling band (points at z 2.06–2.51 m).
- So here A follows a door, not an arbitrary plane, and the ceiling-band image closes it the same way
  it closes office doors.
- Both pieces carry panos: 7 in `hallway_2` and 9 in `hallway_3` in `_area_2_manhattan7/pose.json`.

**Remaining risk:** S3DIS elsewhere labels some corridors with junctions as one room (Armeni 2016;
Bobkov 2017), and the Area_3 `hallway_1`|`hallway_2` cut was not checked for a door. The
whole-`Area_2.ply` stretch run (§8.4) is the check.

---

## 11. Limitations of this review
- A targeted design survey, not a PRISMA systematic review. Screening was by relevance to PanoPin's
  setting: merged cloud, no poses, furnished, open doors.
- Several numbers are reported but not reproduced:
  - Tang 2022 IoU 93.5% and Tu 2026 F1 0.99 are from abstracts.
  - The S3DIS-evaluated semantic methods use networks trained on S3DIS (leakage).
  - Raster2Seq's table was read from arXiv HTML.
  - Kleiner 2017's and Tang 2024's numbers were not accessible.
- Several 2026 items are arXiv-only and not peer-reviewed (ProClosure, OccuSG, Prior-SG, Z-FLoc), as are
  CAGE, FloorSAM and Sharif 2023.
- This review says *which* methods to test. It does not replace measuring them on our scenes (§8.4).

---

## 12. References (deduplicated; URLs/DOIs verified live on 2026-10-01 unless marked)

**Density-image floorplan reconstruction**
- Chen, Liu, Wu, Furukawa. *Floor-SP.* ICCV 2019. arxiv.org/abs/1908.06702 · github.com/woodfrog/floor-sp
- Stekovic, Rad, Fraundorfer, Lepetit. *MonteFloor.* ICCV 2021. arxiv.org/abs/2103.11161
- Chen, Qian, Furukawa. *HEAT.* CVPR 2022. arxiv.org/abs/2111.15143 · github.com/woodfrog/heat
- Yue, Kontogianni, Schindler, Engelmann. *Connecting the Dots (RoomFormer).* CVPR 2023. arxiv.org/abs/2211.15658 · github.com/ywyue/RoomFormer
- Su, Tung, Peng, Wonka, Chu. *SLIBO-Net.* NeurIPS 2023. openreview.net/forum?id=HYo2Ao3hP8
- Chen, Deng, Furukawa. *PolyDiffuse.* NeurIPS 2023. arxiv.org/abs/2306.01461
- Liu et al. *PolyRoom.* ECCV 2024. arxiv.org/abs/2407.10439 · github.com/3dv-casia/PolyRoom
- Xu et al. *FRI-Net.* ECCV 2024. arxiv.org/abs/2407.10687 · github.com/Daisy-1227/FRI-Net
- *PolyGraph.* IEEE TVCG 31(10), 2025. doi 10.1109/TVCG.2025.3544769 · github.com/Fern327/PolyGraph (authors [UNVERIFIED])
- Liu et al. *CAGE.* arXiv 2509.15459, 2025. github.com/ee-Liu/CAGE
- Phung, Averbuch-Elor. *Raster2Seq.* SIGGRAPH 2026. arxiv.org/abs/2602.09016
- Liu, Wu, Furukawa. *FloorNet.* ECCV 2018. arxiv.org/abs/1804.00090
- Talotta et al. *Floorplan generation from noisy point cloud.* GeoIndustry 2023. doi 10.1145/3615888.3627810

**Foundation models on top-down images**
- Kirillov et al. *Segment Anything.* ICCV 2023. github.com/facebookresearch/segment-anything
- Ke et al. *HQ-SAM.* NeurIPS 2023. arxiv.org/abs/2306.01567
- Ravi et al. *SAM 2.* ICLR 2025. arxiv.org/abs/2408.00714
- *SAM 3: Segment Anything with Concepts.* ICLR 2026. arxiv.org/abs/2511.16719 · *SAM 3D.* arxiv.org/abs/2511.16624
- Albadri et al. *A SAM-Based Approach for Automatic Indoor Point Cloud Segmentation.* ISPRS Archives XLVIII-G-2025:131. doi 10.5194/isprs-archives-XLVIII-G-2025-131-2025
- Ye et al. *FloorSAM.* arXiv 2509.15750, 2025.

**Direct 3D / 2.5D**
- Ochmann, Vock, Klein. ISPRS JPRS 151:251, 2019. doi 10.1016/j.isprsjprs.2019.03.017 · arxiv.org/abs/1907.00631
- Ochmann et al. GRAPP 2014, doi 10.5220/0004689601200127 [abs] · C&G 54:94, 2016, doi 10.1016/j.cag.2015.07.008 [abs]
- Mura et al. C&G 44:20, 2014, doi 10.1016/j.cag.2014.07.005 [abs] · Mura, Mattausch, Pajarola. CGF 35(7):179, 2016, doi 10.1111/cgf.13015 [abs]
- Ambruș, Claici, Wendt. *Automatic Room Segmentation From Unstructured 3-D Data of Indoor Environments.* IEEE RA-L 2(2):749, 2017. doi 10.1109/LRA.2017.2651939
- Bobkov et al. *Room segmentation in 3D point clouds using anisotropic potential fields.* ICME 2017. doi 10.1109/ICME.2017.8019484
- Armeni et al. *3D Semantic Parsing of Large-Scale Indoor Spaces.* CVPR 2016. doi 10.1109/CVPR.2016.170
- Jung, Stachniss, Kim. IJGI 6(7):206, 2017. doi 10.3390/ijgi6070206
- Li, Shi, Sun. IEEE JSTARS 18:18358, 2025. doi 10.1109/JSTARS.2025.3586490
- Gourguechon, Macher, Landes. ISPRS Annals X-M-1-2023:93. doi 10.5194/isprs-annals-X-M-1-2023-93-2023
- Macher, Landes, Grussenmeyer. Appl. Sci. 7:1030, 2017. doi 10.3390/app7101030 [abs]
- Frías et al. ISPRS Archives XLIV-4/W1-2020:49. doi 10.5194/isprs-archives-XLIV-4-W1-2020-49-2020
- Hübner et al. ISPRS JPRS 181:254, 2021, doi 10.1016/j.isprsjprs.2021.07.002 [abs] · ISPRS Annals V-4-2022:121, doi 10.5194/isprs-annals-V-4-2022-121-2022 · github.com/huepat/voxir
- Martens & Blankenbach. PFG 91:273, 2023. doi 10.1007/s41064-023-00243-1
- Yang et al. IJGI 10:739, 2021. doi 10.3390/ijgi10110739 · github.com/yhexie/AxVSPRoomSeg3D
- Michailidis & Pajarola. *ASPIRE.* Vis. Comput. 35:1209, 2019. doi 10.1007/s00371-019-01711-9 [abs]
- Ikehata, Yan, Furukawa. *Structured Indoor Modeling.* ICCV 2015. doi 10.1109/ICCV.2015.156
- Turner & Zakhor. GRAPP 2014. doi 10.5220/0004680300220033 [abs] · Cai & Fan. Remote Sens. 13:1947, 2021. doi 10.3390/rs13101947
- Tang et al. AutCon 141:104422, 2022, doi 10.1016/j.autcon.2022.104422 [abs] · Tang et al. JAG 135:104265, 2024, doi 10.1016/j.jag.2024.104265 [abs]
- Chen et al. JAG 127:103685, 2024, doi 10.1016/j.jag.2024.103685 [abs] · Wang et al. JSTARS 19:17021, 2026, doi 10.1109/JSTARS.2026.3691937
- Drobnyi, Li, Brilakis. JCCE 38(5), 2024. doi 10.1061/JCCEE5.CPENG-5853 [abs] · Tu, Shi, Sun. IJGI 15:188, 2026. doi 10.3390/ijgi15050188
- Brunklaus, Kellner, Reiterer. Remote Sens. 17(7):1124, 2025. doi 10.3390/rs17071124 · github.com/mvg-inatech/room-instance-segmentation-mask3d

**Robotics / scene graphs**
- Bormann, Jordan, Li, Hampp, Hägele. *Room segmentation: Survey, implementation, and analysis.* ICRA 2016. doi 10.1109/ICRA.2016.7487234 · github.com/ipa320/ipa_coverage_planning
- Luperto et al. *ROSE².* IEEE RA-L 7(3), 2022. doi 10.1109/LRA.2022.3186495 · github.com/goldleaf3i/declutter-reconstruct
- Sharif, Mohan, Suvarna. arXiv 2303.13798, 2023. github.com/sharif1093/py_floor_plan_segmenter
- Mielle, Magnusson, Lilienthal. *MAORIS.* ICRA 2018. doi 10.1109/ICRA.2018.8461128
- Fermin-Leon, Neira, Castellanos. *DuDe.* ICRA 2017. doi 10.1109/ICRA.2017.7989297
- Hou, Yuan, Schwertfeger. *Area Graph.* ICAR 2019. doi 10.1109/ICAR46387.2019.8981588
- Hiller et al. IROS 2019. doi 10.1109/IROS40897.2019.8968111 · Kleiner et al. IROS 2017. doi 10.1109/IROS.2017.8206429
- Thrun. Artif. Intell. 99(1), 1998. doi 10.1016/S0004-3702(97)00078-7 · Friedman, Pasula, Fox. *Voronoi Random Fields.* IJCAI 2007
- He, Sun, Hou, Ha, Schwertfeger. Auton. Robots 45(5):755, 2021. doi 10.1007/s10514-021-09991-8
- Werby et al. *HOV-SG.* RSS 2024. doi 10.15607/RSS.2024.XX.077 · github.com/hovsg/HOV-SG
- Hughes, Chang, Carlone. *Hydra.* RSS 2022, doi 10.15607/RSS.2022.XVIII.050 · IJRR 2024, doi 10.1177/02783649241229725
- Xu et al. *Point2Graph.* ICRA 2025. doi 10.1109/ICRA55743.2025.11127471
- Muthuraj et al. *ProClosure.* arXiv 2609.12614, 2026 · Cueto Zumaya et al. *OccuSG.* arXiv 2606.13727, 2026
- Kim & Min. *SeLRoS.* IROS 2024. doi 10.1109/IROS58592.2024.10801361

**Frontier / panorama-assisted / localisation**
- Peng et al. *OpenScene.* CVPR 2023 · Takmaz et al. *OpenMask3D.* NeurIPS 2023 · Yin et al. *SAI3D.* CVPR 2024 · Nguyen et al. *Open3DIS.* CVPR 2024 · Huang et al. *Segment3D.* ECCV 2024 · Zhou et al. *Point-SAM.* ICLR 2025 · Guo et al. *SAM2Point.* arXiv 2408.16768 · Gu et al. *ConceptGraphs.* ICRA 2024
- Mao et al. *SpatialLM.* arXiv 2506.07491 (NeurIPS 2025 [UNVERIFIED venue]) · Avetisyan et al. *SceneScript.* ECCV 2024
- Cruz et al. *ZInD.* CVPR 2021 · Shabani et al. *Extreme Structure from Motion for Indoor Panoramas.* ICCV 2021 · Lambert et al. *SALVe.* ECCV 2022 · Lin, Li, Wang. *Floorplan-Jigsaw.* ICCV 2019
- Kim, Jeong, Kim. *Fully Geometric Panoramic Localization (FGPL).* CVPR 2024. arxiv.org/abs/2403.19904 · Kim et al. *CPO.* ECCV 2022

**Partition primitives & metrics**
- Shi & Malik. *Normalized Cuts.* TPAMI 22(8), 2000. doi 10.1109/34.868688 · Grady. *Random Walks for Image Segmentation.* TPAMI 28(11), 2006. doi 10.1109/TPAMI.2006.233
- Meyer. *Topographic distance and watershed lines.* Signal Process. 38(1), 1994. doi 10.1016/0165-1684(94)90060-4 · Felzenszwalb & Huttenlocher. IJCV 59(2), 2004. doi 10.1023/B:VISI.0000022288.19776.77 · Kuhn. *Hungarian method.* NRLQ 1955. doi 10.1002/nav.3800020109
- Hubert & Arabie. *Comparing partitions.* J. Classif. 1985. doi 10.1007/BF01908075 · Meilă. *Comparing clusterings.* JMVA 98(5), 2007. doi 10.1016/j.jmva.2006.11.013 · Kirillov et al. *Panoptic Segmentation.* CVPR 2019. arxiv.org/abs/1801.00868
- Khoshelham et al. *ISPRS benchmark on indoor modelling.* ISPRS Archives XLII-2/W7:367, 2017. doi 10.5194/isprs-archives-XLII-2-W7-367-2017
- Armeni, Sax, Zamir, Savarese. *2D-3D-Semantic data.* arXiv 1702.01105, 2017 · Zheng et al. *Structured3D.* ECCV 2020 · Chang et al. *Matterport3D.* 3DV 2017
