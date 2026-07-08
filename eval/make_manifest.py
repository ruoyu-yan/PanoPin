"""Emit an ANONYMIZED solver-input manifest for one area.

The manifest is the ONLY thing a PanoPin solver should read. It exposes panos by
UUID (symlinked to <uuid>.png so the room token in the original filename is hidden)
plus the candidate room point-cloud paths. Room/pose GT is withheld — score the
solver's output afterwards with eval/score.py.

Usage:
  python eval/make_manifest.py [--area Area_3] [--out-dir runs/manifest_Area_3]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3dis_gt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="Area_3")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = s3dis_gt.load_config(args.config)
    a = cfg["s3dis"][args.area]
    gt = s3dis_gt.load_gt(args.area, cfg)
    rooms = s3dis_gt.candidate_rooms(args.area, cfg)

    out_dir = Path(args.out_dir or (s3dis_gt.repo_root() / "runs" / f"manifest_{args.area}"))
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # map uuid -> original rgb path by matching the uuid token in the filename
    rgb_by_uuid = {}
    for p in Path(a["pano_rgb_dir"]).glob("*.png"):
        parts = p.name.split("_")
        if len(parts) >= 2:
            rgb_by_uuid[parts[1]] = p

    panos = {}
    n_linked = 0
    for u in gt:
        src = rgb_by_uuid.get(u)
        if src:
            link = img_dir / f"{u}.png"
            if not link.exists():
                try:
                    link.symlink_to(src)
                except FileExistsError:
                    pass
            panos[u] = {"image": str(link)}
            n_linked += 1
        else:
            panos[u] = {"image": None}

    rooms_dir = Path(a["rooms_dir"])
    candidate = {r: str(rooms_dir / r / f"{r}.txt") for r in rooms}

    manifest = {
        "area": args.area,
        "note": "Anonymized solver input. Do NOT try to recover room/pose from these paths.",
        "panos": panos,
        "candidate_rooms": candidate,
    }
    mpath = out_dir / "manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {mpath}  ({len(panos)} panos, {n_linked} images linked, "
          f"{len(candidate)} candidate rooms)")


if __name__ == "__main__":
    main()
