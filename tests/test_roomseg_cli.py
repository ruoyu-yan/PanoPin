import json

import numpy as np
import open3d as o3d

from data_utils import read_txt_pcd   # vendored CPO reader (tests/conftest.py puts it on sys.path)
from panopin import cli
from tests.roomseg_synth import H, colour, two_rooms


def _ply(path, xyz, with_colour=True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if with_colour:
        pcd.colors = o3d.utility.Vector3dVector(colour(xyz) / 255.0)
    o3d.io.write_point_cloud(str(path), pcd)
    return path


def test_segment_writes_every_output_and_cpo_can_read_it(tmp_path):
    xyz = two_rooms()
    out = tmp_path / "seg"
    rc = cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", xyz)), "--out-dir", str(out),
                   "--n-panos", "2"])
    assert rc == 0
    clouds = json.loads((out / "clouds.json").read_text())
    assert sorted(clouds) == ["seg_00", "seg_01"]
    labels = np.load(out / "labels.npy")
    assert labels.shape == (len(xyz),)
    total = 0
    for name, path in clouds.items():
        pts, rgb = read_txt_pcd(path)
        assert pts.shape[1] == 3 and rgb.shape[1] == 3 and 0.0 <= rgb.min() and rgb.max() <= 1.0
        k = int(name.split("_")[1])
        assert len(pts) == int((labels == k).sum())
        total += len(pts)
    assert total == int((labels >= 0).sum())
    rep = json.loads((out / "segmentation.json").read_text())
    assert rep["K"] == 2 and rep["N"] == 2 and rep["warnings"] == []
    assert (out / "rooms.png").exists()


def test_output_points_keep_input_coordinates(tmp_path):
    xyz = two_rooms() + np.array([100.0, -50.0, 3.0])
    out = tmp_path / "seg"
    assert cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", xyz)), "--out-dir", str(out)]) == 0
    pts, _ = read_txt_pcd(str(out / "seg_00.txt"))
    assert pts[:, 0].min() > 99.0 and pts[:, 1].max() < -45.0 and pts[:, 2].max() > H + 2.9


def test_k_greater_than_n_warns_on_stderr(tmp_path, capsys):
    out = tmp_path / "seg"
    cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", two_rooms())), "--out-dir", str(out),
              "--n-panos", "1"])
    assert "K=2 > N=1" in capsys.readouterr().err


def test_stale_segment_files_are_removed(tmp_path):
    out = tmp_path / "seg"; out.mkdir()
    (out / "seg_09.txt").write_text("0 0 0 0 0 0\n")
    cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", two_rooms())), "--out-dir", str(out)])
    assert sorted(p.name for p in out.glob("seg_*.txt")) == ["seg_00.txt", "seg_01.txt"]


def test_cloud_without_colours_is_refused_and_nothing_written(tmp_path, capsys):
    out = tmp_path / "seg"
    rc = cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", two_rooms(), with_colour=False)),
                   "--out-dir", str(out)])
    assert rc == 2 and "colour" in capsys.readouterr().err
    assert not out.exists()


def test_missing_cloud_is_reported_as_not_found_and_nothing_written(tmp_path, capsys):
    out = tmp_path / "seg"
    rc = cli.main(["segment", "--cloud", str(tmp_path / "nope.ply"), "--out-dir", str(out)])
    assert rc == 2 and "not found" in capsys.readouterr().err
    assert not out.exists()


def test_refused_geometry_writes_nothing(tmp_path, capsys):
    xyz = two_rooms()
    two_storey = np.concatenate([xyz, xyz + np.array([0.0, 0.0, H + 0.2])])
    out = tmp_path / "seg"
    rc = cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", two_storey)), "--out-dir", str(out)])
    assert rc == 2 and "storey" in capsys.readouterr().err
    assert not out.exists()


def test_txt_input_and_byte_identical_reruns(tmp_path):
    xyz = two_rooms()
    txt = tmp_path / "m.txt"
    np.savetxt(txt, np.column_stack([xyz, colour(xyz)]), fmt="%.6f %.6f %.6f %d %d %d")
    a, b = tmp_path / "a", tmp_path / "b"
    assert cli.main(["segment", "--cloud", str(txt), "--out-dir", str(a)]) == 0
    assert cli.main(["segment", "--cloud", str(txt), "--out-dir", str(b)]) == 0
    for name in ["seg_00.txt", "seg_01.txt", "labels.npy", "segmentation.json", "rooms.png"]:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_baseline_flag_uses_the_hovsg_band(tmp_path):
    out = tmp_path / "seg"
    cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", two_rooms())), "--out-dir", str(out),
              "--baseline-hovsg"])
    rep = json.loads((out / "segmentation.json").read_text())
    assert rep["params"]["band_from_floor"] == 1.5
