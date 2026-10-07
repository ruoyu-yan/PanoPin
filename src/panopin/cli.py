"""Command-line entry point for PanoPin's deployable hand-off.

Stage 0 of the Scan2BIM pipeline calls this. It deliberately wraps only the
shipped, tested API (seed.seed_rooms / fgpl_export.export_alignment) and never
experiments/fgpl_seed/*, which carries hardcoded paths and cached-grid
assumptions.
"""
import argparse
import json
import sys
from pathlib import Path

from panopin import fgpl_export, seed


def segment_shapes(candidate_clouds, stride=10):
    """{room: cloud_path} -> {room: SegmentShape}, from every stride-th point of each cloud.
    Corridor segments get their FGPL seed at this centroid (fgpl_export.build_matches)."""
    from panopin.roomseg.files import read_cloud
    from panopin.roomseg.shape import segment_shape
    out = {}
    for room, path in candidate_clouds.items():
        xyz, _rgb = read_cloud(path)
        out[room] = segment_shape(xyz[::stride])
    return out


def seed_from_cached(score_matrix, poses, room_order, metadata_path, out_path, tau=0.10,
                     shapes=None):
    """Build a demo6_alignment.json from an existing score matrix + poses.

    poses maps pano -> room -> (t, R). Returns the admitted pano names, which
    the caller MUST use as FGPL's cfg["pano_names"].
    """
    return fgpl_export.export_alignment(
        score_matrix, poses, list(room_order), str(metadata_path), str(out_path), tau=tau,
        shapes=shapes)


def seed_from_clouds(panos, candidate_clouds, metadata_path, out_path, tau=0.10):
    """GPU path: localize every pano against every candidate room, then export.

    panos maps pano_id -> pano image path; candidate_clouds maps room -> cloud path.
    Room order is taken from candidate_clouds' insertion order.

    Uses localize_and_score rather than seed_rooms: the FGPL hand-off is PER-PANO
    (D34 -- FGPL localizes each pano from its own seed), while seed_rooms returns
    one seed per ROOM. The full per-(pano, room) score matrix is what build_matches
    needs.
    """
    shapes = segment_shapes(candidate_clouds)
    score_matrix, poses = seed.localize_and_score(
        panos, candidate_clouds, seed.load_cfg(sample_rate=30))
    return seed_from_cached(score_matrix, poses, list(candidate_clouds),
                            metadata_path, out_path, tau=tau, shapes=shapes)


def _segment(args):
    """`segment` subcommand: exit 0 on success, 2 when the cloud is refused (nothing written)."""
    from panopin.roomseg import DEFAULTS, HOVSG, SegmentationError
    from panopin.roomseg.files import segment_cloud

    try:
        report = segment_cloud(args.cloud, args.out_dir, args.n_panos,
                               HOVSG if args.baseline_hovsg else DEFAULTS)
    except SegmentationError as e:
        print(f"segment: {e}", file=sys.stderr)
        return 2
    for w in report["warnings"]:
        print(f"segment: warning: {w}", file=sys.stderr)
    print(f"{report['K']} segments -> {args.out_dir / 'clouds.json'}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="panopin.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("seed", help="build a demo6_alignment.json for FGPL")
    s.add_argument("--panos", required=True, type=Path,
                   help="JSON {pano_id: pano_image_path}")
    s.add_argument("--clouds", required=True, type=Path,
                   help="JSON {room: cloud_path}; key order sets room_idx")
    s.add_argument("--metadata", required=True, type=Path,
                   help="metadata.json holding rotation_matrix")
    s.add_argument("--out", required=True, type=Path,
                   help="output demo6_alignment.json path")
    s.add_argument("--tau", type=float, default=0.10,
                   help="per-pano admission threshold on the low-pct winner score")

    g = sub.add_parser("segment", help="split one merged cloud into per-room candidate clouds")
    g.add_argument("--cloud", required=True, type=Path, help="merged cloud, .ply (with colours) or .txt")
    g.add_argument("--out-dir", required=True, type=Path,
                   help="writes seg_XX.txt, clouds.json (for `seed --clouds`), labels.npy, ...")
    g.add_argument("--n-panos", type=int, default=None,
                   help="number of panoramas; warns when segments outnumber them")
    g.add_argument("--baseline-hovsg", action="store_true",
                   help="baseline B1: band [floor + 1.5, ceiling - 0.3] instead of the ceiling band")

    args = parser.parse_args(argv)
    if args.command == "segment":
        return _segment(args)

    panos = json.loads(args.panos.read_text())
    clouds = json.loads(args.clouds.read_text())
    admitted = seed_from_clouds(panos, clouds, args.metadata, args.out, tau=args.tau)
    for name in admitted:
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
