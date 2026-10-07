import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return p


def test_seed_from_scores_writes_alignment(tmp_path):
    """cli.seed_from_cached builds an alignment json from a score matrix + poses."""
    from panopin import cli

    scores = {"panoA": {"roomX": 0.05, "roomY": 0.40},
              "panoB": {"roomX": 0.42, "roomY": 0.06}}
    poses = {"panoA": {"roomX": ([1.0, 2.0, 0.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                       "roomY": ([9.0, 9.0, 0.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])},
             "panoB": {"roomX": ([8.0, 8.0, 0.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                       "roomY": ([3.0, 4.0, 0.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])}}
    meta = _write(tmp_path / "metadata.json",
                  {"rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    out = tmp_path / "demo6_alignment.json"

    admitted = cli.seed_from_cached(scores, poses, ["roomX", "roomY"], meta, out, tau=0.10)

    assert sorted(admitted) == ["panoA", "panoB"], admitted
    data = json.loads(out.read_text())
    assert set(data) == {"metadata", "matches"}, sorted(data)
    assert data["metadata"]["pano_names"] == admitted
    by_name = {m["pano_name"]: m for m in data["matches"]}
    assert by_name["panoA"]["room_label"] == "roomX", by_name["panoA"]
    assert by_name["panoA"]["camera_position"] == [1.0, 2.0], by_name["panoA"]


def test_cli_module_is_runnable():
    """python -m panopin.cli --help exits 0 (the entry point Stage 0 depends on)."""
    r = subprocess.run([sys.executable, "-m", "panopin.cli", "--help"],
                       cwd=str(REPO / "src"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "seed" in r.stdout, r.stdout


def test_main_routes_args_and_prints_admitted_names_one_per_line(tmp_path, monkeypatch, capsys):
    """main()'s argparse wiring + stdout contract: downstream parses stdout as the admitted
    panorama names, one per line, and nothing else. --panos and --clouds must reach
    seed_from_clouds unswapped, --tau must arrive as a float, and the exit code must be 0.

    seed_from_clouds itself runs GPU localization, so it is stubbed here; only main()'s wiring
    is under test.
    """
    from panopin import cli

    # Distinguishable contents so a --panos/--clouds swap is detectable: disjoint key sets and
    # disjoint value shapes (str paths vs a dict), not just different values at the same keys.
    panos_obj = {"panoA": "images/panoA.jpg", "panoB": "images/panoB.jpg"}
    clouds_obj = {"roomX": "clouds/roomX.ply", "roomY": "clouds/roomY.ply", "roomZ": "clouds/roomZ.ply"}
    panos_path = _write(tmp_path / "panos.json", panos_obj)
    clouds_path = _write(tmp_path / "clouds.json", clouds_obj)
    metadata_path = tmp_path / "metadata.json"  # never opened by the stub; existence not required
    out_path = tmp_path / "demo6_alignment.json"

    calls = []

    def fake_seed_from_clouds(panos, candidate_clouds, metadata, out, tau=0.10):
        calls.append(dict(panos=panos, candidate_clouds=candidate_clouds,
                           metadata=metadata, out=out, tau=tau))
        return ["panoB", "panoA"]

    monkeypatch.setattr(cli, "seed_from_clouds", fake_seed_from_clouds)

    rc = cli.main(["seed",
                   "--panos", str(panos_path),
                   "--clouds", str(clouds_path),
                   "--metadata", str(metadata_path),
                   "--out", str(out_path),
                   "--tau", "0.25"])

    assert rc == 0, rc

    captured = capsys.readouterr()
    assert captured.out == "panoB\npanoA\n", repr(captured.out)
    assert captured.err == "", repr(captured.err)

    assert len(calls) == 1, calls
    call = calls[0]
    assert call["panos"] == panos_obj, call["panos"]
    assert call["candidate_clouds"] == clouds_obj, call["candidate_clouds"]
    assert call["out"] == out_path, call["out"]
    assert call["tau"] == 0.25, call["tau"]
    assert isinstance(call["tau"], float), type(call["tau"])


def _cloud_txt(path, w, d, cx, cy, n=3000):
    import numpy as np
    rng = np.random.default_rng(0)
    xy = rng.uniform([-w / 2, -d / 2], [w / 2, d / 2], size=(n, 2)) + [cx, cy]
    rows = np.column_stack([xy, rng.uniform(0, 2.8, n), rng.integers(0, 255, (n, 3))])
    np.savetxt(path, rows, fmt="%.3f %.3f %.3f %d %d %d")
    return path


def test_segment_shapes_reads_every_candidate_cloud(tmp_path):
    from panopin import cli
    clouds = {"corr": str(_cloud_txt(tmp_path / "corr.txt", 10.0, 2.0, 5.0, 20.0)),
              "off": str(_cloud_txt(tmp_path / "off.txt", 4.0, 4.0, 1.0, 2.0))}
    shapes = cli.segment_shapes(clouds, stride=1)
    assert set(shapes) == {"corr", "off"}
    assert shapes["corr"].extent_ratio > 4 and shapes["off"].extent_ratio < 1.3
    assert abs(shapes["corr"].centroid_xy[0] - 5.0) < 0.2


def test_seed_from_cached_passes_shapes_through(tmp_path):
    from panopin import cli
    from panopin.roomseg.shape import SegmentShape
    scores = {"h": {"corr": 0.08, "off": 0.30}, "o": {"corr": 0.30, "off": 0.06}}
    poses = {"h": {"corr": ([9.0, 20.0, 1.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                   "off": ([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])},
             "o": {"corr": ([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                   "off": ([1.0, 2.0, 1.4], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])}}
    meta = _write(tmp_path / "metadata.json", {"rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    out = tmp_path / "demo6_alignment.json"
    shapes = {"corr": SegmentShape((5.0, 20.5), 5.1, 10), "off": SegmentShape((1.3, 2.2), 1.2, 10)}
    cli.seed_from_cached(scores, poses, ["corr", "off"], meta, out, tau=0.10, shapes=shapes)
    by = {m["pano_name"]: m for m in json.loads(out.read_text())["matches"]}
    assert by["h"]["camera_position"] == [5.0, 20.5] and by["h"]["seed_basis"] == "centroid"


def test_segment_shapes_skips_a_segment_too_small_to_measure(tmp_path, capsys):
    from panopin import cli
    clouds = {"tiny": str(_cloud_txt(tmp_path / "tiny.txt", 1.0, 1.0, 0.0, 0.0, n=2)),
              "off": str(_cloud_txt(tmp_path / "off.txt", 4.0, 4.0, 1.0, 2.0))}
    shapes = cli.segment_shapes(clouds)
    assert set(shapes) == {"off"}
    assert "tiny" in capsys.readouterr().err


def test_arbitrate_subcommand_writes_the_winners(tmp_path, monkeypatch):
    from panopin import cli, arbitrate as arb
    _ID = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    panos = _write(tmp_path / "panos.json", {"pA": str(tmp_path / "pA.png")})
    clouds = _write(tmp_path / "clouds.json", {"r1": str(tmp_path / "r1.txt")})
    alignment = _write(tmp_path / "demo6_alignment.json",
                       {"metadata": {}, "matches": [{"pano_name": "pA", "room_label": "r1"}]})
    cdir = tmp_path / "poses"; (cdir / "pA").mkdir(parents=True)
    doc = {"pano": "pA", "fgpl_choice": 0, "candidates": [
        {"index": 0, "origin": "xdf", "rot_idx": 0, "R": _ID, "t": [0, 0, 1.5], "cost": -1.0, "n_matched": 1, "n_tight": 1, "avg_dist": 0.1},
        {"index": 1, "origin": "seed", "rot_idx": 1, "R": _ID, "t": [5, 0, 1.5], "cost": None, "n_matched": 1, "n_tight": 1, "avg_dist": 0.1}]}
    (cdir / "pA" / "candidates.json").write_text(json.dumps(doc))
    monkeypatch.setattr(arb, "gpu_scorer", lambda cfg: (lambda pano, cloud, poses: [0.3, 0.1]))
    out = tmp_path / "arbitration.json"
    rc = cli.main(["arbitrate", "--panos", str(panos), "--clouds", str(clouds), "--alignment", str(alignment),
                   "--candidates-dir", str(cdir), "--out", str(out)])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["pA"]["index"] == 1 and res["pA"]["origin"] == "seed" and res["pA"]["t"] == [5, 0, 1.5]
    assert res["pA"]["fgpl_choice"] == {"index": 0, "score": 0.3}
