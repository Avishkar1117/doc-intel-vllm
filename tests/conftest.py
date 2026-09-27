"""Pytest config: benchmarks/ has no __init__.py (it's scripts, not a package), so its
modules aren't importable by name without this - added once here instead of a sys.path
hack repeated in every test file that needs it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
