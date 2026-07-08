"""Import for side effect: puts vendored CPO on sys.path so its bare imports resolve."""
import os, sys
_CPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "cpo"))
if _CPO not in sys.path:
    sys.path.insert(0, _CPO)
