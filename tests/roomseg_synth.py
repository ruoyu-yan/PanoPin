"""Synthetic indoor clouds for the room-segmentation tests: planes sampled on a regular 3 cm grid."""
import numpy as np

SP = 0.03   # sample spacing (m)
H = 2.8     # storey height (m)


def _grid(a0, a1, b0, b1):
    a, b = np.meshgrid(np.arange(a0, a1, SP), np.arange(b0, b1, SP))
    return a.ravel(), b.ravel()


def plane_z(z, x0, x1, y0, y1):
    x, y = _grid(x0, x1, y0, y1)
    return np.column_stack([x, y, np.full(x.size, z)])


def wall(axis, at, a0, a1, h=H, openings=()):
    """Vertical wall on x=at (axis='x') or y=at (axis='y'), spanning a0..a1 along the other axis.

    openings: (b0, b1, top) gaps along the wall. top=None is open floor to ceiling; otherwise it is a
    door whose lintel fills top..h.
    """
    cuts = sorted(openings)
    edges = [a0] + [v for b0, b1, _ in cuts for v in (b0, b1)] + [a1]
    pieces = [_grid(s0, s1, 0.0, h) for s0, s1 in zip(edges[0::2], edges[1::2]) if s1 > s0]
    pieces += [_grid(b0, b1, top, h) for b0, b1, top in cuts if top is not None]
    if not pieces:                       # the opening spans the whole wall
        return np.zeros((0, 3))
    s = np.concatenate([p[0] for p in pieces])
    z = np.concatenate([p[1] for p in pieces])
    if axis == "x":
        return np.column_stack([np.full(s.size, at), s, z])
    return np.column_stack([s, np.full(s.size, at), z])


def box(x0, x1, y0, y1, top):
    """Furniture: top face and four sides of a box standing on the floor."""
    return np.concatenate([plane_z(top, x0, x1, y0, y1),
                           wall("x", x0, y0, y1, h=top), wall("x", x1, y0, y1, h=top),
                           wall("y", y0, x0, x1, h=top), wall("y", y1, x0, x1, h=top)])


def shell(x1, y1, h=H, ceiling=True):
    """Floor, (ceiling,) and four outer walls of the rectangle [0, x1] x [0, y1]."""
    parts = [plane_z(0.0, 0, x1, 0, y1),
             wall("x", 0.0, 0, y1, h), wall("x", x1, 0, y1, h),
             wall("y", 0.0, 0, x1, h), wall("y", y1, 0, x1, h)]
    if ceiling:
        parts.append(plane_z(h, 0, x1, 0, y1))
    return np.concatenate(parts)


def thick_partition_x(at, y1, thick=0.2, openings=(), h=H):
    """The two faces of an interior wall centred on x=at, sharing the same openings."""
    return np.concatenate([wall("x", at - thick / 2, 0, y1, h, openings),
                           wall("x", at + thick / 2, 0, y1, h, openings)])


def two_rooms(door_top=2.1, gap=(1.5, 2.5), split=4.0, ceiling=True):
    """8 x 4 m split at x=split by a 0.2 m wall with one opening (door with lintel, or open)."""
    return np.concatenate([shell(8.0, 4.0, ceiling=ceiling),
                           thick_partition_x(split, 4.0, openings=[(gap[0], gap[1], door_top)])])


def colour(xyz):
    """Deterministic RGB (uint8) for a synthetic cloud."""
    return (np.abs(np.sin(np.asarray(xyz) * 7.0)) * 255).astype(np.uint8)
