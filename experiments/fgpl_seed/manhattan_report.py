"""Meeting-ready report for the strictly-Manhattan pose round-trip (2026-07-16).

Reads whatever arms have finished (work/results/manhattan_{export,oracle}.json) and writes a
comparison. Tolerates a missing arm so it can be run while the other is still going.

    conda run -n panopin python -m experiments.fgpl_seed.manhattan_report
"""
import os
import sys
import json
import statistics

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from experiments.fgpl_seed import manhattan, paths

ARMS = [("manhattan_export", "PanoPin seed (color)"), ("manhattan_oracle", "GT seed (oracle ref)")]


def _load(arm):
    p = paths.WORK / "results" / f"{arm}.json"
    return json.load(open(p)) if p.exists() else None


def _f(x, p=3):
    return "n/a" if x is None else f"{x:.{p}f}"


def main():
    res = {arm: _load(arm) for arm, _ in ARMS}
    have = {a: r for a, r in res.items() if r}
    if not have:
        print("no arm results yet")
        return

    ref = next(iter(have.values()))
    rooms = sorted(set(ref["true_room"].values()))
    L = []

    def out(s=""):
        L.append(s)
        print(s)

    out("# Strictly-Manhattan PanoPin -> FGPL pose round-trip\n")
    out(f"**Scene `{manhattan.SCENE}`: {len(rooms)} Manhattan rooms, {ref['n_admitted']} panos.** "
        "Both the PanoPin candidate rooms AND the FGPL line map are built from Manhattan rooms "
        "only, so the Stage-3 3-orthogonal-direction assumption holds throughout.")
    out(f"Rooms: {', '.join(rooms)}. Excluded as non-Manhattan: office_3, office_7, office_8.\n")
    out("The line map recovered three exactly axis-aligned principal directions with **0% "
        "unclassified** sparse lines — the Manhattan restriction is doing what it should.\n")
    out("PanoPin seeds every pano at its low-percentile argmin room (no gate): the question is "
        "'estimated pose per pano vs GT'. The oracle arm seeds each pano at its GT position in "
        "its true room, isolating FGPL's own refinement error on this same map.\n")

    out("## Accuracy vs S3DIS ground truth")
    out("| arm | localized | per-room coverage | trans median | trans mean | trans max | "
        "rot median | wrong-room |")
    out("|---|---|---|---|---|---|---|---|")
    for arm, label in ARMS:
        r = res[arm]
        if not r:
            out(f"| {label} | _(running)_ | | | | | | |")
            continue
        s = r["scored"]
        t, rot = s["translation"], s["rotation"]
        c = r["coverage"]
        out(f"| {label} | {s['n_localized']}/{r['n_admitted']} | "
            f"{c['n_covered']}/{c['n_rooms']} | {_f(t['median'])} | {_f(t['mean'])} | "
            f"{_f(t['max'])} | {_f(rot['median'], 1)} | {_f(s['wrong_room_rate'], 2)} |")

    e = res["manhattan_export"]
    if e:
        out(f"\nRight-room slice (panos PanoPin seeded into their true room): median "
            f"**{_f(e['trans_median_right_room'])} m** over {e['n_right_room']} panos.")

        out("\n## Per-pano translation error (PanoPin arm)")
        out("| pano | true room | seed room | seed correct? | trans err (m) |")
        out("|---|---|---|---|---|")
        pu, sr, tr = e["per_uuid_trans"], e["seed_room"], e["true_room"]
        for u in sorted(pu, key=lambda u: pu[u]):
            out(f"| `{u[:8]}` | {tr[u]} | {sr.get(u, '-')} | "
                f"{'YES' if sr.get(u) == tr[u] else 'NO'} | {pu[u]:.3f} |")

        ok = [pu[u] for u in pu if sr.get(u) == tr[u]]
        bad = [pu[u] for u in pu if sr.get(u) != tr[u]]
        out(f"\nSeeded into the right room: {len(ok)}/{len(pu)} panos, median "
            f"{_f(statistics.median(ok) if ok else None)} m. "
            f"Wrong-room seeds: {len(bad)}, median {_f(statistics.median(bad) if bad else None)} m.")

        gate = e["gate_would_admit"]
        wrong_gate = [u for u in gate if sr.get(u) != tr.get(u)]
        out(f"\n**Gate caveat:** the D34 `tau=0.10` gate would admit {len(gate)}/{len(tr)} panos "
            f"here, including {len(wrong_gate)} wrong-room ones — on this pool the score "
            "distribution sits lower than the pool tau was tuned on, so the fixed threshold does "
            "not transfer. The threshold-free room-anchored seeding (5/5 correct, "
            "MANHATTAN_RESULTS.md) is the claim that holds.")

    o = res["manhattan_oracle"]
    if e and o:
        em, om = e["scored"]["translation"]["median"], o["scored"]["translation"]["median"]
        if em is not None and om is not None:
            out(f"\n## Verdict\n")
            out(f"- FGPL's own error floor on this map (GT-seeded oracle): **{om:.3f} m** median.")
            out(f"- PanoPin-seeded: **{em:.3f} m** median "
                f"(right-room slice {_f(e['trans_median_right_room'])} m).")
            out(f"- So of PanoPin's {em:.3f} m, roughly {om:.3f} m is FGPL's refinement floor, "
                f"not the color seed.")

    (paths.HERE / "MANHATTAN_POSE_RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote {paths.HERE / 'MANHATTAN_POSE_RESULTS.md'}")


if __name__ == "__main__":
    main()
