# Cheap Tier-1 scorer + deterministic in-frame eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the two-tier funnel's Tier-1 genuinely cheap (skip the ~20 s inlier detection) and pin determinism, so a full S3DIS Area_3 in-frame evaluation runs in minutes and yields a trustworthy pano→room accuracy number.

**Architecture:** Add a stripped CPO scorer `score_room_cheap` (small-pool `histogram_pose_search` with `img_weight=None` → single-forward `sampling_loss`; no `make_score_map_2d/3d`, no Adam) as Tier-1; keep the full `localize_pair` as Tier-2 on the top-k survivors. Pin determinism with `torch.set_num_threads(1)` + `np.random.seed`. Run `solve.py` over the anonymized manifest, score with `eval/score.py`, and report accuracy **in-frame (76) vs all (85)**.

**Tech Stack:** Python 3.8, PyTorch 1.10 CPU, numpy/opencv, vendored CPO (`third_party/cpo`); harness stdlib-only. Run everything in the `panopin` conda env (`conda run -n panopin ...`).

## Global Constraints
- **Deterministic / training-free only** (D1, D7): no training, no learned weights.
- **Fairness (D5):** `src/panopin/*` reads ONLY the anonymized manifest (panos by uuid + candidate room clouds). GT (`eval/s3dis_gt.py`, room names, `camera_location`) may be used ONLY in `smoke/` dev scripts, never in `src/panopin/`.
- **Harness stays stdlib-only (D6):** never add deps to `eval/`. Never edit `eval/` metrics/GT to change a number (charter #5).
- **Compose CPO primitives; do NOT edit `third_party/cpo/`** (D9).
- **CPU only** (D14 determinism): force `device='cpu'`.
- **Point-cloud format:** whitespace `X Y Z R G B`, RGB 0–255 (CPO's `read_txt_pcd` /255). Pano equirectangular RGB, resized to 2048×1024, z-up.
- **CPO cfg:** built via `panopin.cpo_config.load_cfg(**overrides)`; `cfg.dataset == 'stanford'`.
- **Out-of-frame panos (D17):** 9/85 have GT camera outside their room cloud; they are excluded-and-reported, not method failures. Accuracy is reported in-frame (76) and all (85).

## File Structure
- Create `src/panopin/determinism.py` — `pin(seed=0)`: `torch.set_num_threads(1)`, `np.random.seed`, `torch.manual_seed`.
- Modify `src/panopin/cpo_adapter.py` — add `score_room_cheap(cfg, pano_path, cloud_path) -> (t, R, loss)`; add `sampling_loss` to the CPO imports.
- Modify `src/panopin/select_room.py` — Tier-1 uses `score_room_cheap`; Tier-2 keeps `localize_pair`.
- Create `src/panopin/solve.py` — CLI: manifest → predictions JSON; pins determinism.
- Modify `tests/test_select_room.py` — rework the mock to mock `score_room_cheap` (Tier-1) and `localize_pair` (Tier-2) separately.
- Create `tests/test_determinism.py`, `tests/test_cpo_adapter_cheap.py`, `tests/test_solve.py`.
- Create `smoke/tier1_recall.py` (G1 + G4), `smoke/check_determinism.py` (G3), `smoke/report_in_frame.py` (in-frame/all split).
- Modify `docs/PROGRESS.md`, `docs/tasks.json` (final task).

---

### Task 1: Determinism helper

**Files:**
- Create: `src/panopin/determinism.py`
- Test: `tests/test_determinism.py`

**Interfaces:**
- Produces: `pin(seed: int = 0) -> None` — sets torch to single-threaded + seeds numpy/torch.

- [ ] **Step 1: Write the failing test**

`tests/test_determinism.py`:
```python
def test_pin_sets_single_thread_and_seeds():
    import torch, numpy as np
    from panopin.determinism import pin
    pin(0)
    assert torch.get_num_threads() == 1
    a = np.random.rand(3)
    pin(0)
    b = np.random.rand(3)
    assert (a == b).all()   # same seed -> same numpy draw
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_determinism.py -v`
Expected: FAIL (No module named 'panopin.determinism').

- [ ] **Step 3: Write minimal implementation**

`src/panopin/determinism.py`:
```python
"""Pin run-to-run determinism for the (CPU) CPO path. Multi-threaded float
reduction order (e.g. make_pano's index_put_) is the main wobble source; a
single thread removes it. torch RNG is unused by the path but seeded for safety;
np.random matters only via read_txt_pcd's subsample permutation (sample_rate>1)."""
import numpy as np
import torch


def pin(seed=0):
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n panopin python -m pytest tests/test_determinism.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/panopin/determinism.py tests/test_determinism.py
git commit -m "feat: determinism helper (single-thread + seed)"
```

---

### Task 2: `score_room_cheap` — the cheap Tier-1 scorer

**Files:**
- Modify: `src/panopin/cpo_adapter.py`
- Test: `tests/test_cpo_adapter_cheap.py`

**Interfaces:**
- Consumes: `load_cfg` (cpo_config); CPO `data_utils.read_txt_pcd`, `dict_utils.get_init_dict_cpo`, `color_utils.color_match`, `utils.{generate_trans_points,generate_rot_points,histogram_pose_search}`, `cpo.sampling_loss.sampling_loss`.
- Produces: `score_room_cheap(cfg, pano_path, cloud_path) -> (t: np.ndarray (3,), R: np.ndarray (3,3), loss: float)` — same contract as `localize_pair`, no inlier score maps, no Adam.

- [ ] **Step 1: Add `sampling_loss` to the CPO imports in `cpo_adapter.py`**

Modify the import line so both are available:
```python
from cpo.sampling_loss import refine_pose_sampling_loss, sampling_loss
```

- [ ] **Step 2: Write the failing test**

`tests/test_cpo_adapter_cheap.py`:
```python
import numpy as np, tests.synthetic as S


def _prep(tmp_path):
    xyz, rgb = S.box_room(patch=('x1', [220, 40, 40]))
    cloud = str(tmp_path / "room.txt"); S.write_cloud_txt(cloud, xyz, rgb)
    pano = str(tmp_path / "q.png"); S.render_pano_png(pano, xyz, rgb, [2.0, 2.0, 1.5], np.eye(3))
    return pano, cloud


def test_score_room_cheap_shapes_and_finite(tmp_path):
    from panopin.cpo_config import load_cfg
    from panopin.cpo_adapter import score_room_cheap
    pano, cloud = _prep(tmp_path)
    cfg = load_cfg(sample_rate=1, num_yaw=4, num_pitch=4, num_roll=4, num_trans=10)
    t, R, loss = score_room_cheap(cfg, pano, cloud)
    assert t.shape == (3,) and R.shape == (3, 3)
    assert np.isfinite(loss)


def test_cheap_scorer_skips_inlier_detection(tmp_path, monkeypatch):
    # score_room_cheap must NOT call the expensive make_score_map_* functions.
    import utils
    from panopin.cpo_config import load_cfg
    from panopin.cpo_adapter import score_room_cheap
    def boom(*a, **k):
        raise AssertionError("inlier detection must not run in Tier-1")
    monkeypatch.setattr(utils, "make_score_map_2d", boom)
    monkeypatch.setattr(utils, "make_score_map_3d", boom)
    pano, cloud = _prep(tmp_path)
    cfg = load_cfg(sample_rate=1, num_yaw=4, num_pitch=4, num_roll=4, num_trans=10)
    score_room_cheap(cfg, pano, cloud)   # must not raise
```

- [ ] **Step 3: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_cpo_adapter_cheap.py -v`
Expected: FAIL (cannot import name 'score_room_cheap').

- [ ] **Step 4: Implement `score_room_cheap`**

Append to `src/panopin/cpo_adapter.py` (mirrors `localize_pair`'s setup, minus the inlier
score-map block and the Adam refine loop; uses `histogram_pose_search` with `img_weight=None`
then a single-forward `sampling_loss`):
```python
def score_room_cheap(cfg, pano_path, cloud_path):
    """Cheap Tier-1 room score: small-pool histogram_pose_search (no inlier score
    maps) -> single-forward sampling_loss. Returns (t, R, loss); no Adam. CPU-only."""
    device = torch.device('cpu')
    sample_rate = getattr(cfg, 'sample_rate', 1)

    xyz_np, rgb_np = data_utils.read_txt_pcd(cloud_path, sample_rate=sample_rate)
    xyz = torch.from_numpy(xyz_np).float().to(device)
    rgb = torch.from_numpy(rgb_np).float().to(device)

    orig_img = cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (2048, 1024))
    if getattr(cfg, 'match_color', False):
        mod_img = (torch.from_numpy(orig_img).float() / 255.).to(device)
        new_img = color_match(mod_img, rgb)
        orig_img = (255 * new_img.cpu().numpy()).astype(np.uint8)

    init_dict = get_init_dict_cpo(cfg)
    rot = generate_rot_points(init_dict, device=device)
    trans = generate_trans_points(xyz, init_dict, device=device)

    idh = getattr(cfg, 'init_downsample_h', 1); idw = getattr(cfg, 'init_downsample_w', 1)
    img_search = cv2.resize(orig_img, (orig_img.shape[1] // idw, orig_img.shape[0] // idh))
    img_search = (torch.from_numpy(img_search).float() / 255.).to(device)
    input_trans, input_rot = histogram_pose_search(
        img_search, xyz, rgb, trans, rot, 1,
        init_dict['num_split_h'], init_dict['num_split_w'], None, init_dict['sin_hist'])

    mdh = getattr(cfg, 'main_downsample_h', 1); mdw = getattr(cfg, 'main_downsample_w', 1)
    img_score = cv2.resize(orig_img, (orig_img.shape[1] // mdw, orig_img.shape[0] // mdh))
    img_score = (torch.from_numpy(img_score).float() / 255.).to(device)
    t_c, R_c, loss_c = sampling_loss(img_score, xyz, rgb, input_trans, input_rot, 0, cfg,
                                     return_list=True)
    t = t_c.detach().numpy().reshape(3)
    R = R_c.detach().numpy().reshape(3, 3)
    return t, R, float(loss_c.detach())
```
Note: `get_init_dict_cpo(cfg)` copies `num_yaw/num_pitch/num_roll/num_trans/num_split_h/num_split_w/trans_init_mode/sin_hist` from cfg; the cheap cfg sets the rotation/translation counts small. If any CPO call errors on a signature mismatch, align to `third_party/cpo` (reference: `cpo_adapter.py::localize_pair`, `utils.py::histogram_pose_search`, `cpo/sampling_loss.py::sampling_loss`) — do not invent behavior.

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n panopin python -m pytest tests/test_cpo_adapter_cheap.py -v`
Expected: PASS (shapes/finite + no-inlier-call). If `histogram_pose_search` rejects `img_weight=None`, pass a uniform map `torch.ones(init_dict['num_split_h'], init_dict['num_split_w'], device=device)` instead (recon: sin-weight fallback makes None valid; uniform is the equivalent).

- [ ] **Step 6: Commit**

```bash
git add src/panopin/cpo_adapter.py tests/test_cpo_adapter_cheap.py
git commit -m "feat(cpo): score_room_cheap — Tier-1 scorer without inlier detection or Adam"
```

---

### Task 3: Real-data validation gates (G1 recall, G4 cost, G3 determinism)

**Files:**
- Create: `smoke/tier1_recall.py` (G1 + G4), `smoke/check_determinism.py` (G3)

**Interfaces:**
- Consumes: `load_cfg`, `score_room_cheap` (Task 2), `determinism.pin` (Task 1); `eval/s3dis_gt.py` (GT — dev-only).

- [ ] **Step 1: Write `smoke/tier1_recall.py` (G1 recall + G4 cost)**

`smoke/tier1_recall.py`:
```python
"""G1/G4: on several in-frame Area_3 panos, does cheap Tier-1 keep the correct room
in its top-k, and how fast is it per room? GT is dev-only (never in src/panopin)."""
import argparse, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import score_room_cheap
import s3dis_gt


def resolve_pano(uuid, rgb_dir):
    for p in glob.glob(os.path.join(rgb_dir, "*.png")):
        parts = os.path.basename(p).split("_")
        if len(parts) >= 2 and parts[1] == uuid:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="office_3,office_5,office_7,conferenceRoom_1,WC_1")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--sample-rate", type=int, default=30)
    args = ap.parse_args()
    pin(0)
    cfg = load_cfg(sample_rate=args.sample_rate, num_yaw=4, num_pitch=4, num_roll=4, num_trans=10)
    g = s3dis_gt.load_config(None); a = g["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", g)
    all_rooms = s3dis_gt.candidate_rooms("Area_3", g)
    clouds = {r: os.path.join(a["rooms_dir"], r, r + ".txt") for r in all_rooms}
    hits = 0; total = 0; per_room_sec = []
    for room in [r.strip() for r in args.rooms.split(",")]:
        uuid = next((u for u, v in gt.items() if v["room"] == room), None)
        pano = resolve_pano(uuid, a["pano_rgb_dir"]) if uuid else None
        if not pano:
            print(f"{room}: no pano"); continue
        scored = []
        for cand, cloud in clouds.items():
            t0 = time.time(); _, _, loss = score_room_cheap(cfg, pano, cloud)
            per_room_sec.append(time.time() - t0); scored.append((loss, cand))
        scored.sort()
        topk = [c for _, c in scored[:args.top_k]]
        ok = room in topk; hits += ok; total += 1
        rank = [c for _, c in scored].index(room) + 1
        print(f"{room:18s} rank={rank:2d} top{args.top_k}={'HIT' if ok else 'MISS'}  best={scored[0][1]}")
    print(f"\nTier-1 recall@{args.top_k}: {hits}/{total}")
    print(f"mean sec/room: {np.mean(per_room_sec):.2f}  (G4: want < ~2 s)")
    return 0 if hits == total else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run G1/G4**

Run: `conda run -n panopin python smoke/tier1_recall.py`
Expected: recall@5 = 5/5 (or close), mean sec/room < ~2 s. **If recall is poor, widen the pool** (`--` bump `num_yaw/pitch/roll` to 6 or `num_trans` to 30) and re-run; record the smallest pool that recalls all. This is approach-B's key risk gate — do not proceed to Task 6 until Tier-1 recall is acceptable.

- [ ] **Step 3: Write `smoke/check_determinism.py` (G3)**

`smoke/check_determinism.py`:
```python
"""G3: is one real Tier-1 (and Tier-2) score identical across two runs after pin()?"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import score_room_cheap
import s3dis_gt


def main():
    g = s3dis_gt.load_config(None); a = g["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", g)
    uuid = next(u for u, v in gt.items() if v["room"] == "office_3")
    pano = next(p for p in glob.glob(os.path.join(a["pano_rgb_dir"], "*.png"))
                if os.path.basename(p).split("_")[1] == uuid)
    cloud = os.path.join(a["rooms_dir"], "office_3", "office_3.txt")
    cfg = load_cfg(sample_rate=30, num_yaw=4, num_pitch=4, num_roll=4, num_trans=10)
    losses = []
    for _ in range(2):
        pin(0)
        _, _, loss = score_room_cheap(cfg, pano, cloud)
        losses.append(loss)
    print(f"losses: {losses}  delta={abs(losses[0]-losses[1]):.6f}")
    ok = abs(losses[0] - losses[1]) < 1e-6
    print(f"G3 determinism: {'PASS (identical)' if ok else 'RESIDUAL — record and rely on Tier-2 margin'}")
    return 0 if ok else 0   # informational: never hard-fail, but print the verdict


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run G3**

Run: `conda run -n panopin python smoke/check_determinism.py`
Expected: prints identical losses (delta 0) → determinism pinned; or a small residual to record. Note the verdict in the commit message.

- [ ] **Step 5: Commit**

```bash
git add smoke/tier1_recall.py smoke/check_determinism.py
git commit -m "test(smoke): Tier-1 recall/cost (G1/G4) + determinism check (G3)"
```

---

### Task 4: Rewire `select_room` Tier-1 → `score_room_cheap`

**Files:**
- Modify: `src/panopin/select_room.py`
- Modify: `tests/test_select_room.py`

**Interfaces:**
- Consumes: `score_room_cheap` (Task 2, Tier-1), `localize_pair` (existing, Tier-2), `load_cfg`.
- Produces: `select_room(pano_path, candidate_rooms, top_k=5, sample_rate=10) -> RoomResult` (unchanged contract).

- [ ] **Step 1: Rework the mock test to mock both tier functions**

Replace `tests/test_select_room.py`'s `_fake_localizer` mechanism with separate Tier-1/Tier-2 mocks:
```python
import numpy as np
import panopin.select_room as SR


def _install(monkeypatch, tier1_loss, calls):
    def cheap(cfg, pano, cloud):            # Tier-1
        calls.append(("t1", cloud))
        return np.zeros(3), np.eye(3), float(tier1_loss[cloud])
    def full(cfg, pano, cloud):             # Tier-2
        calls.append(("t2", cloud))
        return np.zeros(3), np.eye(3), float(tier1_loss[cloud] - 0.05)
    monkeypatch.setattr(SR, "score_room_cheap", cheap)
    monkeypatch.setattr(SR, "localize_pair", full)


def test_funnel_picks_min_loss_room(monkeypatch):
    calls = []; _install(monkeypatch, {"a.txt": 0.5, "b.txt": 0.1, "c.txt": 0.9, "d.txt": 0.2}, calls)
    res = SR.select_room("q.png", {"A": "a.txt", "B": "b.txt", "C": "c.txt", "D": "d.txt"},
                         top_k=2, sample_rate=1)
    assert res.room == "B" and res.t.shape == (3,) and np.isfinite(res.loss)


def test_funnel_tiers_all_then_refines_top_k(monkeypatch):
    calls = []; _install(monkeypatch, {"a.txt": 0.5, "b.txt": 0.1, "c.txt": 0.9, "d.txt": 0.2}, calls)
    SR.select_room("q.png", {"A": "a.txt", "B": "b.txt", "C": "c.txt", "D": "d.txt"}, top_k=2, sample_rate=1)
    assert len([c for c in calls if c[0] == "t1"]) == 4                       # all rooms scored cheap
    assert sorted(c[1] for c in calls if c[0] == "t2") == ["b.txt", "d.txt"]  # only top-2 refined


def test_top_k_larger_than_candidates_is_clamped(monkeypatch):
    calls = []; _install(monkeypatch, {"a.txt": 0.3, "b.txt": 0.1}, calls)
    res = SR.select_room("q.png", {"A": "a.txt", "B": "b.txt"}, top_k=9, sample_rate=1)
    assert res.room == "B" and len([c for c in calls if c[0] == "t2"]) == 2


def test_empty_candidates_returns_none(monkeypatch):
    _install(monkeypatch, {}, [])
    assert SR.select_room("q.png", {}, top_k=5) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_select_room.py -v`
Expected: FAIL (`select_room` still imports/uses `localize_pair` for Tier-1; the `score_room_cheap` attr isn't patched into the Tier-1 path yet).

- [ ] **Step 3: Rewire `select_room.py`**

In `src/panopin/select_room.py`: import `score_room_cheap` and use it for Tier-1; keep `localize_pair` for Tier-2. Replace the two-tier body:
```python
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import score_room_cheap, localize_pair

# Tier-1 pool (small, cheap) and Tier-2 refine settings.
TIER1_POOL = {"num_yaw": 4, "num_pitch": 4, "num_roll": 4, "num_trans": 10}
TIER2_REFINE = {"num_iter": 100}
```
and in `select_room`:
```python
    cfg1 = load_cfg(sample_rate=sample_rate, **TIER1_POOL)
    cfg2 = load_cfg(sample_rate=sample_rate, **TIER2_REFINE)

    tier1 = []
    for room, cloud in candidate_rooms.items():
        _, _, loss = score_room_cheap(cfg1, pano_path, cloud)
        tier1.append((loss, room, cloud))
    tier1.sort(key=lambda x: x[0])
    survivors = tier1[:max(1, min(top_k, len(tier1)))]

    best = None
    for _, room, cloud in survivors:
        t, R, loss = localize_pair(cfg2, pano_path, cloud)
        if best is None or loss < best.loss:
            best = RoomResult(room, t, R, loss)
    return best
```
(Delete the now-unused `TIER1`/`TIER2` imports from `cpo_config` if present; keep `RoomResult`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n panopin python -m pytest tests/test_select_room.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/panopin/select_room.py tests/test_select_room.py
git commit -m "feat: select_room Tier-1 uses score_room_cheap (skip inlier detection)"
```

---

### Task 5: `solve.py` — manifest → predictions CLI

**Files:**
- Create: `src/panopin/solve.py`
- Test: `tests/test_solve.py`

**Interfaces:**
- Consumes: `select_room` (Task 4), `determinism.pin` (Task 1).
- Produces: CLI `python -m panopin.solve --manifest PATH --out PATH [--top-k 5] [--sample-rate 20] [--seed 0]`; writes predictions JSON per `eval/PREDICTIONS_SCHEMA.md`. Reads ONLY the manifest.

- [ ] **Step 1: Write the failing test**

`tests/test_solve.py`:
```python
import json, numpy as np, tests.synthetic as S


def test_solve_writes_valid_predictions(tmp_path, monkeypatch):
    import panopin.solve as SV
    # mock select_room so the test is fast and deterministic (funnel is tested elsewhere)
    from collections import namedtuple
    RR = namedtuple("RoomResult", "room t R loss")
    monkeypatch.setattr(SV, "select_room",
                        lambda img, rooms, top_k, sample_rate: RR("A", np.array([1., 2., 3.]), np.eye(3), 0.1))
    manifest = str(tmp_path / "manifest.json")
    json.dump({"area": "Synth", "panos": {"u1": {"image": "x.png"}},
               "candidate_rooms": {"A": "a.txt", "B": "b.txt"}}, open(manifest, "w"))
    out = str(tmp_path / "pred.json")
    SV.run(manifest, out, top_k=2, sample_rate=1)
    pred = json.load(open(out))
    assert pred["predictions"]["u1"]["room"] == "A"
    assert len(pred["predictions"]["u1"]["coarse_pose"]["t"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_solve.py -v`
Expected: FAIL (No module named 'panopin.solve').

- [ ] **Step 3: Write minimal implementation**

`src/panopin/solve.py`:
```python
"""PanoPin solver: anonymized manifest -> predictions JSON. Reads ONLY the manifest (D5)."""
import argparse, json
from panopin.determinism import pin
from panopin.select_room import select_room


def run(manifest_path, out_path, top_k=5, sample_rate=20, seed=0):
    pin(seed)
    man = json.load(open(manifest_path))
    rooms = man["candidate_rooms"]
    preds = {}
    for uuid, p in man["panos"].items():
        img = p.get("image")
        if not img:
            continue
        res = select_room(img, rooms, top_k=top_k, sample_rate=sample_rate)
        if res is None:
            continue
        preds[uuid] = {"room": res.room, "coarse_pose": {"t": [float(x) for x in res.t]}}
    out = {"area": man.get("area", ""), "method": "cpo_cheap_tier1_v1", "predictions": preds}
    json.dump(out, open(out_path, "w"), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=5); ap.add_argument("--sample-rate", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); run(a.manifest, a.out, a.top_k, a.sample_rate, a.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n panopin python -m pytest tests/test_solve.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/panopin/solve.py tests/test_solve.py
git commit -m "feat: solver CLI (manifest -> predictions), manifest-only + determinism"
```

---

### Task 6: Full in-frame eval + in-frame/all split report

**Files:**
- Create: `smoke/report_in_frame.py`
- Modify: `docs/PROGRESS.md`, `docs/tasks.json`

**Interfaces:**
- Consumes: `eval/make_manifest.py`, `src/panopin/solve.py`, `eval/score.py`, `smoke/check_frame_alignment.py` logic (in-frame set).

- [ ] **Step 1: Build the anonymized manifest**

Run: `conda run -n panopin python eval/make_manifest.py --area Area_3`
Expected: `wrote runs/manifest_Area_3/manifest.json (85 panos, … linked, 23 candidate rooms)`.

- [ ] **Step 2: Sanity run on a SUBSET first**

Copy `runs/manifest_Area_3/manifest.json` to `runs/manifest_head.json` and trim `panos` to ~5 entries. Run:
`PYTHONPATH=src conda run -n panopin python -m panopin.solve --manifest runs/manifest_head.json --out runs/pred_head.json --top-k 5 --sample-rate 20`
Expected: completes; `runs/pred_head.json` has room + coarse_pose per pano. Record wall-clock/pano.

- [ ] **Step 3: Full run (background; minutes-scale with cheap Tier-1)**

Run: `PYTHONPATH=src conda run -n panopin python -m panopin.solve --manifest runs/manifest_Area_3/manifest.json --out runs/pred_cheap_v1.json --top-k 5 --sample-rate 20`
Expected: `runs/pred_cheap_v1.json` covering all linked panos.

- [ ] **Step 4: Score overall**

Run: `conda run -n panopin python eval/score.py --predictions runs/pred_cheap_v1.json --area Area_3`
Expected: room accuracy (all-85). Record it.

- [ ] **Step 5: Write `smoke/report_in_frame.py` and report the split**

`smoke/report_in_frame.py`:
```python
"""In-frame vs all room-accuracy split from a predictions file. GT is dev-only.
In-frame = camera within its room cloud bbox (<= 0.10 m), per D17."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import s3dis_gt


def cloud_bbox(path, stride=40):
    mn = np.array([np.inf] * 3); mx = np.array([-np.inf] * 3)
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i % stride: continue
            p = line.split(); v = np.array([float(p[0]), float(p[1]), float(p[2])])
            mn = np.minimum(mn, v); mx = np.maximum(mx, v)
    return mn, mx


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--predictions", required=True)
    ap.add_argument("--area", default="Area_3"); args = ap.parse_args()
    g = s3dis_gt.load_config(None); a = g["s3dis"][args.area]
    gt = s3dis_gt.load_gt(args.area, g)
    pred = json.load(open(args.predictions))["predictions"]
    bbox = {}
    def inframe(u):
        room = gt[u]["room"]; c = os.path.join(a["rooms_dir"], room, room + ".txt")
        if room not in bbox: bbox[room] = cloud_bbox(c)
        mn, mx = bbox[room]; loc = np.asarray(gt[u]["location"], float)
        return float(np.maximum(np.maximum(mn - loc, loc - mx), 0.).max()) <= 0.10
    all_c = all_t = if_c = if_t = 0
    for u, g0 in gt.items():
        if u not in pred: continue
        ok = pred[u]["room"] == g0["room"]
        all_c += ok; all_t += 1
        if inframe(u):
            if_c += ok; if_t += 1
    print(f"all-panos   accuracy: {all_c}/{all_t} = {all_c/max(1,all_t):.3f}")
    print(f"in-frame    accuracy: {if_c}/{if_t} = {if_c/max(1,if_t):.3f}")
    print(f"(out-of-frame excluded from in-frame: {all_t - if_t} panos, D17)")


if __name__ == "__main__":
    main()
```
Run: `conda run -n panopin python smoke/report_in_frame.py --predictions runs/pred_cheap_v1.json`
Expected: prints in-frame and all accuracy. **In-frame accuracy is the headline number.**

- [ ] **Step 6: Record results + flip tasks.json**

Update `docs/PROGRESS.md` (new entry: in-frame + all accuracy, per-room notes on thin-margin/rotation
failures, wall-clock, chosen pool/`sample_rate`/`top_k`, G1–G4 outcomes). In `docs/tasks.json` set T3
`status:"done", passes:true, notes:"in-frame acc=<N> via runs/pred_cheap_v1.json"` **only if** in-frame
accuracy clears the 5.9 % baseline.

- [ ] **Step 7: Commit**

```bash
git add smoke/report_in_frame.py docs/PROGRESS.md docs/tasks.json
git commit -m "feat: T3 — cheap-Tier-1 in-frame eval on Area_3 (in-frame acc=<N>, all=<M>)"
```

---

## Self-Review

**1. Spec coverage:** score_room_cheap skipping inlier maps (Task 2) ✓; select_room rewire (Task 4) ✓;
determinism pin + check (Tasks 1,3) ✓; solve.py + manifest-only fairness (Task 5) ✓; in-frame/all split
reporting D17 (Task 6) ✓; validation gates G1 recall+G4 cost (Task 3 step 2), G3 determinism (Task 3
step 4), G2 agreement (covered by G1 ranks + existing office_3/office_5 probes) ✓; YAGNI scope (no
out-of-frame recovery, no GPU, no FGPL hand-off) ✓.

**2. Placeholder scan:** no TBD/TODO; each code step carries complete code. The `<N>`/`<M>` in Task 6
commit are results to fill at run time, not code placeholders. "Adapt if signature differs" notes point at
exact reference files (grounding, per D9), not vague instructions.

**3. Type consistency:** `pin(seed)->None`; `score_room_cheap(cfg,pano,cloud)->(t(3,),R(3,3),loss)` matches
`localize_pair`'s contract so `select_room` can swap Tier-1 without interface change; `select_room(...)->
RoomResult(room,t,R,loss)`; `run(manifest,out,top_k,sample_rate,seed)`. Names consistent across Tasks 1→6.
