import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "panopin"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "cpo"))

# Vendored CPO scatters work across torch's intra-op thread pool; on the tiny
# synthetic clouds used in tests this introduces run-to-run nondeterminism in
# the loss (see task-3 brief). Tests use tiny clouds, so single-thread is fine
# and makes CPO deterministic here. Do NOT pin threads in production code
# (src/panopin/cpo_adapter.py) -- real S3DIS-scale runs need multi-thread speed.
import torch
torch.set_num_threads(1)
