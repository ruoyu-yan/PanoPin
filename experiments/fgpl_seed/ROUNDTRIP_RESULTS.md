# PanoPin<->FGPL validation round-trip results

Scene: area3_seed_ablation (6-room subset, 12 in-frame panos). fgpl_export arm gated at tau=0.10; FGPL estimator run on the 10 admitted panos. Fair: seed from cached color scores only (no GT).

**Per-room coverage: 6/6** (a room counts iff >=1 admitted pano of that TRUE room refines into it). Per room: hallway_3=OK, office_1=OK, office_4=OK, office_5=OK, office_6=OK, office_7=OK.

| arm | n_localized | trans median | trans median (right-room) | trans mean | trans max | rot median | wrong-room |
|-----|-------------|--------------|---------------------------|------------|-----------|-----------|-----------|
| fgpl_export (gated) | 10/10 | 1.082 | 1.048 (9) | 3.044 | 21.838 | 105.2 | 0.20 |
| oracle (cached, ref) | 10/10 | 0.696 | n/a | 0.647 | 1.579 | 89.9 | 0.00 |
| p1 (cached, ref) | 10/10 | 0.998 | n/a | 5.205 | 24.426 | 105.0 | 0.20 |

Caveats: oracle/p1 cached poses ran with a 12-pano Voronoi vs this arm's 10-pano Voronoi (reference context, not a controlled ablation); Area_3 subset only; tau=0.10 is the D34 Area_3-tuned value (this run also serves as its through-FGPL precision check).
