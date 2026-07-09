"""Pin run-to-run determinism for the (CPU) CPO path. Multi-threaded float
reduction order (e.g. make_pano's index_put_) is the main wobble source; a
single thread removes it. torch RNG is unused by the path but seeded for safety;
np.random matters only via read_txt_pcd's subsample permutation (sample_rate>1)."""
import numpy as np
import torch


def pin(seed=0):
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)
