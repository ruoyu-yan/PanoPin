# PanoPin predictions schema

A solver reads the anonymized manifest (`eval/make_manifest.py` output) and writes a
single JSON file in this shape. Score it with `python eval/score.py --predictions <file>`.

```jsonc
{
  "area": "Area_3",
  "method": "my_method_name",          // free text, shown in the report
  "predictions": {
    "<pano_uuid>": {
      "room": "office_3",              // REQUIRED. Canonical Stanford room name (no trailing _<area>).
                                       //   Must be one of manifest.candidate_rooms.
      "coarse_pose": {                 // OPTIONAL. The seed handed to FGPL.
        "t": [x, y, z],                //   camera location in the S3DIS world frame (metres)
        "R": [[...],[...],[...]]       //   OPTIONAL 3x3 camera->world rotation (pipeline Stage-2 convention)
      }
    }
    // ... one entry per pano uuid in the manifest
  }
}
```

Notes
- **Only `room` is required.** `coarse_pose` is scored only when present (translation always;
  rotation only if `R` is given).
- Room name must be **canonical** (`office_3`, not `office_3_3`) and a member of the candidate set.
- Panos you leave out are counted as `n_missing` (they lower coverage, not accuracy directly).
- **Never** populate these fields by parsing room/pose from filenames or GT files — that is cheating
  (see the fairness rule in `CLAUDE.md`). Work only from the anonymized manifest.
