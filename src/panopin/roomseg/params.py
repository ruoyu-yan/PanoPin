"""Every constant of the enclosure room segmenter, in metres (spec 2026-10-01 §4).

Values change ONLY on the dev scenes, and every change is logged in docs/roomseg-results.md.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Params:
    # storey heights (levels.py)
    z_bin: float = 0.02               # z-histogram bin
    peak_rel: float = 0.30            # a local z maximum is "strong" at >= this x the tallest bin
    peak_cluster_gap: float = 0.30    # strong peaks this close belong to one slab
    min_storey: float = 2.0
    max_storey: float = 6.0
    mid_slab_above_floor: float = 1.8     # a strong peak in (floor + 1.8, ceiling - 0.5)
    mid_slab_below_ceiling: float = 0.5   #   means a second storey -> refuse
    # grid and band (raster.py)
    cell: float = 0.05
    band_bottom: float = 0.60         # band = [ceiling - band_bottom, ceiling - band_top]
    band_top: float = 0.10
    band_from_floor: Optional[float] = None   # if set, band lower edge = floor + this (B1)
    max_grid_cells: int = 40_000_000  # ~316 m square at 5 cm; beyond this the extent is outliers
    footprint_close: float = 0.30
    # rooms (partition.py)
    wall_min_pts: int = 3
    gap_close_r: float = 0.15         # seals scan gaps up to ~0.3 m; must stay < half a door width
    min_room_area: float = 1.0
    min_room_halfwidth: float = 0.30
    max_fill_dist: float = 1.0
    # report warnings (core.py)
    warn_unassigned_frac: float = 0.02
    warn_small_room_area: float = 2.0


DEFAULTS = Params()
# Baseline B1 (spec §6.3 as amended in Task 7): the HOV-SG band [floor + 1.5, ceiling - 0.3].
HOVSG = Params(band_from_floor=1.5, band_top=0.30)
