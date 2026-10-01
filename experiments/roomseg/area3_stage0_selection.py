"""Write Point_360/data/full_runs/<scene>/{pano_selection.json, panos/} for an Area_3 dev scene from
its scene.json manifest (s3dis/choose_stations.py: one station per room), so Point_360's
data/full_runs/_scripts/loc_inputs.py can build its Stage-0 inputs exactly as for Area_2.

Run: python3 experiments/roomseg/area3_stage0_selection.py Area_3_manhattan4
"""
import json
import shutil
import sys
from pathlib import Path

P360 = Path("/home/ruoyu/Point_360")
MANIFEST = {"Area_3_manhattan4": "_multiroom4", "Area_3_manhattan6": "_multiroom6"}
PANO_RGB = P360 / "data/s3dis/area_3_no_xyz/area_3/pano/rgb"

scene = sys.argv[1]
stations = json.loads((P360 / "data/s3dis" / MANIFEST[scene] / "scene.json").read_text())["stations"]
run = P360 / "data/full_runs" / scene
(run / "panos").mkdir(parents=True, exist_ok=True)
rooms = {}
for st in stations:
    png = st["pano_key"] + "_frame_equirectangular_domain_rgb.png"
    shutil.copy(PANO_RGB / png, run / "panos" / png)
    rooms[st["room"]] = {"pano": png, "t": st["t"], "clearance_m": st["clearance_m"],
                         "to_centroid_m": st["to_centroid_m"], "basis": st["basis"]}
(run / "pano_selection.json").write_text(json.dumps(
    {"scene": scene, "rule": "the scene.json station (Point_360 s3dis/choose_stations.py)",
     "rooms": rooms}, indent=1))
print(scene, "->", {r: v["pano"][:16] for r, v in rooms.items()})
