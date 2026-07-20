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
