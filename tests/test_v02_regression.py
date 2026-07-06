"""v0.2 arc regression — the five re-proof numbers must hold (seal, 2026-07-06).

Promotes the M5 integration-demo asserts (c1 ρ / c2 ρ / B1 off-diag min /
M3-2 fidelity mean / windowed Silverman p) to a pytest guard, so a v0.3 change
that breaks the v0.2 arc is caught immediately.  Figures are skipped
(``compute_checks(figures=False)``) — same numbers, no plotting/control sweep.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMOS = os.path.join(_ROOT, "demos")
if _DEMOS not in sys.path:
    sys.path.insert(0, _DEMOS)

import regenerate_m5_materials as m5  # noqa: E402


def test_v02_arc_reproof() -> None:
    checks = m5.compute_checks(figures=False)
    failures = [(name, val) for name, ok, val in m5.evaluate_asserts(checks) if not ok]
    assert not failures, f"v0.2 arc regression failed: {failures}"
