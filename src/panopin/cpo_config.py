"""Build a CPO cfg namedtuple from stanford_cpo.ini, with immutable overrides."""
import os

from panopin import _cpo_path  # noqa: F401  (sys.path side effect)
from parse_utils import parse_ini

_INI = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "third_party", "cpo", "config", "stanford_cpo.ini"))

TIER1 = {"num_iter": 1, "top_k_candidate": 1}   # cheap cost at the coarse pose (all rooms)
TIER2 = {"num_iter": 100}                        # full Adam refine (top-k rooms)


def load_cfg(**overrides):
    """Parse stanford_cpo.ini into a fresh cfg namedtuple, applying overrides immutably.

    Each call re-parses the ini, so overrides never leak between calls.
    """
    cfg = parse_ini(_INI)  # namedtuple, freshly built every call
    if overrides:
        cfg = cfg._replace(**overrides)
    return cfg
