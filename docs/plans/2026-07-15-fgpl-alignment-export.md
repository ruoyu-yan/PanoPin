# PanoPin → FGPL alignment export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/panopin/fgpl_export.py` — turn PanoPin's per-pano low-percentile color scores + coarse poses into the schema-exact `demo6_alignment.json` FGPL consumes, gating & omitting weak-lock panos with a room-anchored coverage backstop and a raw→aligned frame conversion.

**Architecture:** A pure (no-GPU, no-CPO) assembly+serialization module consuming the outputs of the shipped GPU stage `seed.localize_and_score` (`score_matrix`, `poses`). Four functions: frame inversion (with a fail-loud round-trip guard), match building (per-pano gate + coverage backstop), JSON serialization, and a convenience wrapper that reads `metadata.json`. Verified by two GPU-free test suites (synthetic unit tests + a cached-D33-data integration smoke through FGPL's own loader).

**Tech Stack:** Python 3.8 (`panopin` conda env), numpy, stdlib `json`. pytest. No new dependencies.

## Global Constraints

- **Fairness (sacred):** the module reads only `score_matrix` + `poses` + `metadata.json` — never GT, never a pano filename's room, never `camera_to_room.json` (D5). Do not import or touch `eval/` GT.
- **Env:** run everything with `conda run -n panopin python ...` (py3.8 CPU). No GPU in this plan.
- **No CPO / no torch import in `src/panopin/fgpl_export.py`** — it is pure assembly. Only tests may (guardedly) import FGPL.
- **FGPL consumer contract (verified 2026-07-15):** seeds are per-pano and positional; only `pano_name` + `camera_position` are read. Each `pano_name` may appear **at most once** in `matches` (FGPL's `positions[name]` is last-wins).
- **Frame math:** FGPL does `raw_3d = R.T @ [ax, ay, 0]` with `R = metadata['rotation_matrix']`; inverse is `camera_position = (R @ t_raw)[:2]`, exact for a yaw `R` (assert it).
- **Score:** low-percentile, `robust_score.low_percentile_scores(grids, q=robust_score.DEPLOY_Q)`, `DEPLOY_Q=20`; lower = better; genuine locks ~0.06–0.08, weak-lock ~0.12+; gate default `tau=0.10`.
- **Poses format:** `poses[pano][room] == (t, R)` tuple (the live `seed.localize_and_score` form). `t` is a 3-vector (list or np array).
- **Reuse (DRY):** use `coverage.pano_confidence` for the per-pano winner+score and `coverage.room_anchored_seeds` is the conceptual basis of the backstop (but the backstop restricts to free panos — see Task 2).
- **Commit** after each task on branch `feat/deploy-regime`.

---

### Task 1: `raw_t_to_camera_position` — frame inversion + round-trip guard

**Files:**
- Create: `src/panopin/fgpl_export.py`
- Test: `tests/test_fgpl_export.py`

**Interfaces:**
- Consumes: nothing (leaf).
- Produces: `raw_t_to_camera_position(t_raw, R_meta, tol=1e-6) -> [float, float]` — the aligned `camera_position`. Raises `ValueError` if the round-trip through FGPL's inverse (`R.T @ [ax,ay,0]`) does not recover `t_raw[:2]` within `tol` (non-yaw / malformed metadata).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fgpl_export.py
import math
import numpy as np
import pytest
from panopin.fgpl_export import raw_t_to_camera_position


def _yaw(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def test_camera_position_roundtrips_for_yaw():
    R = _yaw(37.0)
    t = [2.5, -1.3, 1.4]
    cam = raw_t_to_camera_position(t, R)
    back = (np.array(R).T @ np.array([cam[0], cam[1], 0.0]))[:2]  # FGPL's inverse
    assert np.allclose(back, [2.5, -1.3], atol=1e-6)


def test_identity_metadata_is_passthrough():
    cam = raw_t_to_camera_position([1.0, 2.0, 3.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(cam, [1.0, 2.0])


def test_non_yaw_rotation_fails_loud():
    R = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # 90 deg about x: mixes z into y
    with pytest.raises(ValueError):
        raw_t_to_camera_position([1.0, 2.0, 3.0], R)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_fgpl_export.py -v`
Expected: FAIL with `ImportError`/`cannot import name 'raw_t_to_camera_position'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/panopin/fgpl_export.py
"""PanoPin -> FGPL alignment export (2026-07-15): turn per-pano low-percentile color
scores + coarse poses into the demo6_alignment.json FGPL consumes.

Pure assembly + serialization (NO GPU, NO CPO import). Consumes the outputs of the shipped
GPU stage `seed.localize_and_score` (score_matrix {pano:{room: low-pct score}}, poses
{pano:{room:(t,R)}}). FGPL reads seeds PER PANO and positional-only (only pano_name +
camera_position); each pano may appear at most once in matches. Gate & omit weak-lock panos
with a room-anchored coverage backstop; convert PanoPin's raw-frame t to FGPL's aligned-frame
camera_position. Fair: reads only scores/poses/metadata, never GT (D5).

Deployment contract: the caller MUST set FGPL cfg["pano_names"] = the returned admitted list
(an unseeded name in pano_names -> KeyError in FGPL's loader)."""
import json
import numpy as np

from panopin import coverage


def raw_t_to_camera_position(t_raw, R_meta, tol=1e-6):
    """Raw-frame CPO translation -> FGPL aligned-frame camera_position [ax, ay].

    FGPL converts back via raw_3d = R.T @ [ax, ay, 0] (aligned_meters_to_raw_3d), so the
    inverse is camera_position = (R @ t_raw)[:2]. Exact for a yaw R (Manhattan alignment);
    a non-yaw / malformed R fails the round-trip guard and raises rather than emit a silently
    wrong seed."""
    R = np.asarray(R_meta, dtype=float)
    t = np.asarray(t_raw, dtype=float)
    cam = (R @ t)[:2]
    back = (R.T @ np.array([cam[0], cam[1], 0.0]))[:2]
    if not np.allclose(back, t[:2], atol=tol):
        raise ValueError(
            f"frame round-trip failed (metadata rotation not yaw-like?): {back} vs {t[:2]}")
    return [float(cam[0]), float(cam[1])]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n panopin python -m pytest tests/test_fgpl_export.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/panopin/fgpl_export.py tests/test_fgpl_export.py
git commit -m "feat(fgpl_export): raw->aligned camera_position with round-trip guard"
```

---

### Task 2: `build_matches` — per-pano gate + coverage backstop

**Files:**
- Modify: `src/panopin/fgpl_export.py`
- Test: `tests/test_fgpl_export.py`

**Interfaces:**
- Consumes: `raw_t_to_camera_position` (Task 1); `coverage.pano_confidence(score_matrix) -> {pano:(room, -score)}`; `coverage.room_anchored_seeds` (conceptual — backstop restricts to free panos, implemented directly here).
- Produces: `build_matches(score_matrix, poses, room_order, R_meta, tau=0.10, guarantee_coverage=True) -> (matches: list[dict], admitted_pano_names: list[str])`. Each match dict has keys `pano_name, room_idx, room_label, score, rotation_deg, camera_position`. Each pano appears at most once.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_fgpl_export.py
from panopin.fgpl_export import build_matches

_ID = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _poses(pairs):
    # pairs: {(pano, room): [x,y,z]} -> {pano:{room:(t, R)}}
    out = {}
    for (p, r), t in pairs.items():
        out.setdefault(p, {})[r] = (t, _ID)
    return out


def test_gate_admits_confident_omits_weak():
    scores = {
        "g1": {"A": 0.06, "B": 0.30},   # confident A
        "g2": {"A": 0.30, "B": 0.07},   # confident B
        "w":  {"A": 0.15, "B": 0.18},   # weak everywhere -> omitted
    }
    poses = _poses({("g1", "A"): [1, 0, 0], ("g2", "B"): [2, 0, 0], ("w", "A"): [3, 0, 0]})
    matches, admitted = build_matches(scores, poses, ["A", "B"], _ID, tau=0.10)
    assert sorted(admitted) == ["g1", "g2"]
    assert all(m["pano_name"] != "w" for m in matches)
    assert {m["room_label"] for m in matches} == {"A", "B"}


def test_backstop_covers_room_with_no_confident_pano():
    scores = {
        "g1": {"A": 0.06, "B": 0.30},   # confident A
        "w":  {"A": 0.40, "B": 0.15},   # B is uncovered; w is B's best (weak)
    }
    poses = _poses({("g1", "A"): [1, 0, 0], ("w", "B"): [5, 0, 0]})
    m_on, adm_on = build_matches(scores, poses, ["A", "B"], _ID, guarantee_coverage=True)
    assert {m["room_label"] for m in m_on} == {"A", "B"}      # backstop added B via w
    assert ("w" in adm_on)
    m_off, _ = build_matches(scores, poses, ["A", "B"], _ID, guarantee_coverage=False)
    assert {m["room_label"] for m in m_off} == {"A"}          # B dropped


def test_pano_appears_at_most_once():
    # g1 is confident in A AND is the global argmin for uncovered B; must NOT be double-emitted.
    scores = {
        "g1": {"A": 0.05, "B": 0.08},   # winner A (0.05); also lowest at B (0.08)
        "w":  {"A": 0.40, "B": 0.15},   # free pano; B's best FREE option
    }
    poses = _poses({("g1", "A"): [1, 0, 0], ("g1", "B"): [1, 1, 0],
                    ("w", "A"): [9, 0, 0], ("w", "B"): [5, 0, 0]})
    matches, admitted = build_matches(scores, poses, ["A", "B"], _ID, guarantee_coverage=True)
    assert admitted.count("g1") == 1
    assert len(admitted) == len(set(admitted))
    g1_rooms = [m["room_label"] for m in matches if m["pano_name"] == "g1"]
    assert g1_rooms == ["A"]                                   # g1 stays in its winner room
    assert {m["room_label"] for m in matches} == {"A", "B"}    # B covered by free pano w


def test_match_schema_and_room_idx():
    scores = {"g1": {"A": 0.06, "B": 0.30}, "g2": {"A": 0.30, "B": 0.07}}
    poses = _poses({("g1", "A"): [1, 2, 0], ("g2", "B"): [3, 4, 0]})
    matches, _ = build_matches(scores, poses, ["A", "B"], _ID, tau=0.10)
    for m in matches:
        assert set(m) == {"pano_name", "room_idx", "room_label", "score",
                          "rotation_deg", "camera_position"}
        assert m["room_idx"] == ["A", "B"].index(m["room_label"])
        assert m["rotation_deg"] == 0.0
        assert len(m["camera_position"]) == 2
        assert isinstance(m["score"], float)


def test_empty_score_matrix():
    assert build_matches({}, {}, ["A"], _ID) == ([], [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_fgpl_export.py -k build_matches -v`
Expected: FAIL with `cannot import name 'build_matches'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/panopin/fgpl_export.py
def build_matches(score_matrix, poses, room_order, R_meta, tau=0.10, guarantee_coverage=True):
    """Per-pano gate + coverage backstop -> (matches, admitted_pano_names).

    Gate: admit each pano at its winner (argmin) room iff winner_score <= tau; weak-lock panos
    are omitted. Backstop (guarantee_coverage): any room in room_order with no admitted pano is
    seeded by its best UNASSIGNED pano (keeps per-pano uniqueness; equals the D32 room-anchored
    pick in the common all-covered case). Each pano appears at most once."""
    if not score_matrix:
        return [], []
    assigned = {}   # pano -> room, at most one room per pano
    for pano, (room, neg_score) in coverage.pano_confidence(score_matrix).items():
        if -neg_score <= tau:
            assigned[pano] = room
    covered = set(assigned.values())
    if guarantee_coverage:
        for room in room_order:
            if room in covered:
                continue
            free = [p for p in score_matrix if p not in assigned]
            if not free:
                continue  # cannot cover without a duplicate emission; leave uncovered
            best = min(free, key=lambda p: score_matrix[p][room])
            assigned[best] = room
            covered.add(room)
    matches = []
    for pano, room in assigned.items():
        t, _R = poses[pano][room]
        cam = raw_t_to_camera_position(t, R_meta)
        matches.append({
            "pano_name": pano,
            "room_idx": room_order.index(room),
            "room_label": room,
            "score": float(score_matrix[pano][room]),
            "rotation_deg": 0.0,
            "camera_position": cam,
        })
    admitted_pano_names = [m["pano_name"] for m in matches]
    return matches, admitted_pano_names
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n panopin python -m pytest tests/test_fgpl_export.py -v`
Expected: all passed (Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/panopin/fgpl_export.py tests/test_fgpl_export.py
git commit -m "feat(fgpl_export): build_matches — per-pano gate + coverage backstop"
```

---

### Task 3: `write_alignment_json` — serialize the exact schema

**Files:**
- Modify: `src/panopin/fgpl_export.py`
- Test: `tests/test_fgpl_export.py`

**Interfaces:**
- Consumes: `matches` + `admitted_pano_names` from `build_matches` (Task 2).
- Produces: `write_alignment_json(matches, admitted_pano_names, out_path, extra_meta=None) -> None`. Writes `{"metadata": {...}, "matches": [...]}` where `metadata.pano_names == list(admitted_pano_names)`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_fgpl_export.py
import json
from panopin.fgpl_export import write_alignment_json


def test_write_alignment_json_shape(tmp_path):
    matches = [{"pano_name": "g1", "room_idx": 0, "room_label": "A", "score": 0.06,
                "rotation_deg": 0.0, "camera_position": [1.0, 2.0]}]
    out = tmp_path / "demo6_alignment.json"
    write_alignment_json(matches, ["g1"], out, extra_meta={"tau": 0.10})
    d = json.loads(out.read_text())
    assert set(d) == {"metadata", "matches"}
    assert d["metadata"]["pano_names"] == ["g1"]
    assert d["metadata"]["tau"] == 0.10
    assert d["matches"] == matches
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_fgpl_export.py -k write_alignment -v`
Expected: FAIL with `cannot import name 'write_alignment_json'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/panopin/fgpl_export.py
def write_alignment_json(matches, admitted_pano_names, out_path, extra_meta=None):
    """Serialize the exact demo6_alignment.json schema FGPL reads (align_polygons_demo6.py)."""
    meta = {"pipeline": "PanoPin color seed (gated per-pano)",
            "pano_names": list(admitted_pano_names),
            "source": "fgpl_export"}
    if extra_meta:
        meta.update(extra_meta)
    with open(out_path, "w") as f:
        json.dump({"metadata": meta, "matches": matches}, f, indent=4)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n panopin python -m pytest tests/test_fgpl_export.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/panopin/fgpl_export.py tests/test_fgpl_export.py
git commit -m "feat(fgpl_export): write_alignment_json serializer"
```

---

### Task 4: `export_alignment` — convenience wrapper (reads metadata.json)

**Files:**
- Modify: `src/panopin/fgpl_export.py`
- Test: `tests/test_fgpl_export.py`

**Interfaces:**
- Consumes: `build_matches` (Task 2), `write_alignment_json` (Task 3).
- Produces: `export_alignment(score_matrix, poses, room_order, metadata_path, out_path, tau=0.10, guarantee_coverage=True) -> admitted_pano_names: list[str]`. Reads `metadata.json['rotation_matrix']`, builds+writes the file, returns the admitted list (for FGPL `cfg["pano_names"]`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_fgpl_export.py
from panopin.fgpl_export import export_alignment


def test_export_alignment_end_to_end(tmp_path):
    scores = {"g1": {"A": 0.06, "B": 0.30}, "g2": {"A": 0.30, "B": 0.07}}
    poses = _poses({("g1", "A"): [1, 2, 0], ("g2", "B"): [3, 4, 0]})
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps({"rotation_matrix": _ID}))
    out = tmp_path / "demo6_alignment.json"
    admitted = export_alignment(scores, poses, ["A", "B"], meta_path, out, tau=0.10)
    assert sorted(admitted) == ["g1", "g2"]
    d = json.loads(out.read_text())
    assert d["metadata"]["pano_names"] == admitted
    assert d["metadata"]["tau"] == 0.10
    assert len(d["matches"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n panopin python -m pytest tests/test_fgpl_export.py -k export_alignment -v`
Expected: FAIL with `cannot import name 'export_alignment'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/panopin/fgpl_export.py
def export_alignment(score_matrix, poses, room_order, metadata_path, out_path,
                     tau=0.10, guarantee_coverage=True):
    """Read metadata.json rotation, build + write demo6_alignment.json, return admitted panos.
    The caller MUST set FGPL cfg["pano_names"] to the returned list."""
    with open(metadata_path) as f:
        R_meta = json.load(f)["rotation_matrix"]
    matches, admitted = build_matches(score_matrix, poses, room_order, R_meta,
                                      tau=tau, guarantee_coverage=guarantee_coverage)
    write_alignment_json(matches, admitted, out_path, extra_meta={"tau": tau})
    return admitted
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n panopin python -m pytest tests/test_fgpl_export.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/panopin/fgpl_export.py tests/test_fgpl_export.py
git commit -m "feat(fgpl_export): export_alignment convenience wrapper"
```

---

### Task 5: Integration smoke — cached D33 data through FGPL's own loader

**Files:**
- Create: `tests/test_fgpl_export_smoke.py`

**Interfaces:**
- Consumes: `robust_score.low_percentile_scores`, `fgpl_export.build_matches` / `write_alignment_json`; cached files `experiments/fgpl_seed/work/seeds/largeval_residuals.json` (grids) + `largeval_cache.json` (`cache[pano]["poses"][room] == {"t","R"}` dicts). Optionally imports FGPL's `load_panorama_positions`.
- Produces: no src; a test asserting positions recover through both an inline transform (always on) and FGPL's real loader (guarded, skipped if unimportable).

- [ ] **Step 1: Write the test**

```python
# tests/test_fgpl_export_smoke.py
"""GPU-free integration smoke: build a real demo6_alignment.json from the cached D33 grids +
poses, then verify FGPL recovers each admitted pano's raw position. Always checks the inline
FGPL transform; additionally checks FGPL's real load_panorama_positions when importable."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from panopin import robust_score
from panopin.fgpl_export import build_matches, write_alignment_json

_REPO = Path(__file__).resolve().parent.parent
_CACHE = _REPO / "experiments" / "fgpl_seed" / "work" / "seeds"
_FGPL = Path("/home/ruoyu/scan2measure-webframework")
_ID = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

pytestmark = pytest.mark.skipif(
    not (_CACHE / "largeval_cache.json").exists()
    or not (_CACHE / "largeval_residuals.json").exists(),
    reason="D33 largeval cache absent")


def _load():
    cache = json.loads((_CACHE / "largeval_cache.json").read_text())
    grids = json.loads((_CACHE / "largeval_residuals.json").read_text())
    scores = robust_score.low_percentile_scores(grids, q=robust_score.DEPLOY_Q)
    poses = {p: {r: (v["t"], v["R"]) for r, v in cache[p]["poses"].items()} for p in cache}
    room_order = sorted({r for p in scores for r in scores[p]})
    return cache, scores, poses, room_order


def test_smoke_positions_recover_inline(tmp_path):
    cache, scores, poses, room_order = _load()
    matches, admitted = build_matches(scores, poses, room_order, _ID, tau=0.10)
    write_alignment_json(matches, admitted, tmp_path / "demo6_alignment.json")
    # identity metadata: raw == aligned; FGPL inverse is raw = R.T @ [ax,ay,0] = [ax,ay]
    for m in matches:
        exp = np.array(cache[m["pano_name"]]["poses"][m["room_label"]]["t"])[:2]
        assert np.allclose(m["camera_position"], exp, atol=1e-6)
    # every candidate room covered by the backstop
    assert {m["room_label"] for m in matches} == set(room_order)
    # per-pano uniqueness
    assert len(admitted) == len(set(admitted))


def test_smoke_through_real_fgpl_loader(tmp_path):
    cache, scores, poses, room_order = _load()
    matches, admitted = build_matches(scores, poses, room_order, _ID, tau=0.10)
    out = tmp_path / "demo6_alignment.json"
    write_alignment_json(matches, admitted, out)
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps({"rotation_matrix": _ID}))
    sys.path.insert(0, str(_FGPL / "src" / "pose_estimation"))
    try:
        from multiroom_pose_estimation import load_panorama_positions
    except Exception as e:                       # heavy deps / repo layout
        pytest.skip(f"FGPL loader not importable here: {e}")
    positions = load_panorama_positions(str(out), str(meta), admitted)
    for m in matches:
        exp = np.array(cache[m["pano_name"]]["poses"][m["room_label"]]["t"])[:2]
        assert np.allclose(positions[m["pano_name"]], exp, atol=1e-6)
```

- [ ] **Step 2: Run the smoke test**

Run: `conda run -n panopin python -m pytest tests/test_fgpl_export_smoke.py -v`
Expected: `test_smoke_positions_recover_inline` PASSED; `test_smoke_through_real_fgpl_loader` PASSED **or** SKIPPED (FGPL import). If the cache is absent, both SKIP — regenerate via `experiments/fgpl_seed/largeval_localize.py` + `largeval_capture.py` (GPU) or note the skip.

- [ ] **Step 3: Run the full suite**

Run: `conda run -n panopin python -m pytest tests/test_fgpl_export.py tests/test_fgpl_export_smoke.py -v`
Expected: all Task 1–4 unit tests pass; smoke inline passes; real-loader passes or skips.

- [ ] **Step 4: Commit**

```bash
git add tests/test_fgpl_export_smoke.py
git commit -m "test(fgpl_export): integration smoke over cached D33 data through FGPL loader"
```

---

### Task 6: Documentation — DECISIONS, PROGRESS, tasks.json

**Files:**
- Modify: `docs/DECISIONS.md`, `docs/PROGRESS.md`, `docs/tasks.json` (if the task exists there)

**Interfaces:**
- Consumes: the shipped module + green tests.
- Produces: durable cross-session record per the project protocol (document-as-you-go).

- [ ] **Step 1: Append a DECISIONS entry**

Append to `docs/DECISIONS.md` a decision recording: (a) the FGPL consumer trace finding — seeds are **per-pano**, positional-only (`camera_position` the only load-bearing field), so the D33 hand-off is re-scoped per-room → per-pano; (b) **gate & omit** weak-lock panos at `tau=0.10` (winner low-pct score) + room-anchored **coverage backstop** restricted to free panos (per-pano uniqueness, FGPL last-wins); (c) frame conversion `camera_position = (R @ t_raw)[:2]` with a fail-loud round-trip guard; (d) the live FGPL GPU round-trip is a documented fast-follow. Reference spec `docs/specs/2026-07-15-fgpl-alignment-export-design.md`.

- [ ] **Step 2: Update PROGRESS.md**

Append a session entry: what shipped (`src/panopin/fgpl_export.py` + `tests/test_fgpl_export.py` + `tests/test_fgpl_export_smoke.py`), test evidence (paste the `pytest` summary line), where it stopped, and the NEXT step (§8 fast-follow: live FGPL GPU round-trip on a couple Area_3 rooms → then Electron/pipeline wiring). Note nothing is pushed (LOCAL on `feat/deploy-regime`).

- [ ] **Step 3: Update tasks.json if applicable**

If a matching task exists in `docs/tasks.json`, flip its `"passes"` to `true` only after pasting the green `verify` command output; otherwise add a completed task entry for the export module referencing the two test files.

- [ ] **Step 4: Commit**

```bash
git add docs/DECISIONS.md docs/PROGRESS.md docs/tasks.json
git commit -m "docs(fgpl_export): record per-pano re-scope, gate+backstop, frame guard"
```

---

## Self-Review

**Spec coverage:**
- §1 gate & omit → Task 2. §1 frame-convert → Task 1. §2 consumer contract (per-pano, camera_position only, cfg pano_names) → Tasks 2/4 + Global Constraints + smoke Task 5. §3 frame math + round-trip guard → Task 1. §4 four functions → Tasks 1–4 exactly. §5 gate + backstop + per-pano uniqueness → Task 2 (all three test cases). §6 output contract (keys, rotation_deg=0, room_idx, metadata.pano_names, return admitted) → Tasks 2/3/4. §7 test suites → Tasks 1–5. §8 fast-follow → recorded in Task 6, out of scope by design. §9 success criterion → covered by Task 5 assertions. No gaps.
- **Coverage backstop = free-pano** (spec §5 correction) is implemented and tested (Task 2 `test_pano_appears_at_most_once`).

**Placeholder scan:** No TBD/TODO; every code + test step shows complete content; no "similar to Task N".

**Type consistency:** `raw_t_to_camera_position(t_raw, R_meta, tol)` used consistently in Tasks 1/2. `build_matches(score_matrix, poses, room_order, R_meta, tau, guarantee_coverage)` signature identical across Tasks 2/4 and tests. `poses[pano][room] == (t, R)` tuple everywhere (unit tests build it via `_poses`; smoke adapts the cache `{"t","R"}` dicts to tuples). `write_alignment_json(matches, admitted_pano_names, out_path, extra_meta)` consistent Tasks 3/4. `low_percentile_scores(grids, q=robust_score.DEPLOY_Q)` matches the shipped signature.
