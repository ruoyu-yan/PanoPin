import numpy as np

def box_room(w=4.0, d=4.0, h=2.8, n=40000, walls=None, seed=0, noise=45.0):
    """Colored point cloud of a box room. Walls beige by default; `walls` is an
    optional dict mapping wall-name -> rgb (e.g. {'x1': [220,40,40], 'y0': [40,40,220]})
    painting each listed wall in {'x0','x1','y0','y1','z0','z1'} that solid color.
    Any wall not named in `walls` stays beige.

    `noise`: per-point Gaussian RGB jitter (std, 0-255 scale). CPO's color_match
    (third_party/cpo/color_utils.py:_match_cumulative_cdf) indexes its per-channel
    CDF interpolation by unique-value RANK, which only coincides with the true
    integer bin index when the channel's histogram has ~no empty bins between 0
    and its max used value. Perfectly flat per-wall colors (noise=0) populate only
    2 bins out of 256, so that indexing silently aligns to the wrong bin and
    color_match returns a badly wrong color -- a real S3DIS scan is never flat
    like that, so `noise` keeps this fixture in the regime CPO expects.
    """
    rng = np.random.RandomState(seed)
    pts, cols = [], []
    beige = np.array([200, 190, 170], float)
    def face(fixed_axis, fixed_val, a_rng, b_rng, axes):
        a = rng.uniform(*a_rng, n // 6); b = rng.uniform(*b_rng, n // 6)
        p = np.zeros((n // 6, 3)); p[:, axes[0]] = a; p[:, axes[1]] = b; p[:, fixed_axis] = fixed_val
        return p
    faces = {
        'x0': face(0, 0, (0, d), (0, h), (1, 2)), 'x1': face(0, w, (0, d), (0, h), (1, 2)),
        'y0': face(1, 0, (0, w), (0, h), (0, 2)), 'y1': face(1, d, (0, w), (0, h), (0, 2)),
        'z0': face(2, 0, (0, w), (0, d), (0, 1)), 'z1': face(2, h, (0, w), (0, d), (0, 1)),
    }
    for name, p in faces.items():
        c = np.tile(beige, (len(p), 1))
        if walls is not None and name in walls:
            c[:] = np.array(walls[name], float)
        c = np.clip(c + rng.normal(0, noise, c.shape), 0, 255)
        pts.append(p); cols.append(c)
    return np.concatenate(pts), np.concatenate(cols)

def write_cloud_txt(path, xyz, rgb):
    np.savetxt(path, np.hstack([xyz, rgb]), fmt="%.4f")  # X Y Z R G B, RGB 0-255

def render_pano_png(path, xyz, rgb, trans, R):
    import cv2, _cpo_path  # noqa: F401
    from utils import make_pano
    import torch
    xt = torch.from_numpy(xyz).float(); rt = torch.from_numpy(rgb / 255.).float()
    centered = (xt - torch.tensor(trans).float()) @ torch.tensor(R).float().T
    pano = make_pano(centered, rt, resolution=(1024, 2048))   # uint8 HxWx3
    cv2.imwrite(path, cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
