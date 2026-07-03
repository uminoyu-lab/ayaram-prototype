"""M2 unit tests for ayaram/cargo.py (blob map, genealogy, Hopfield recall).

Three required guards (指示書 共通 DoD): blob-map determinism, genealogy
consistency, clean recall.
"""

from __future__ import annotations

import numpy as np

from ayaram.cargo import (
    build_blob_map,
    build_genealogy,
    check_genealogy,
    hopfield_recall,
    nearest_pattern,
)
from ayaram.scalespace import Trajectory, sigma_grid


def _traj(id, pol, pts, birth, death, terminal, merged_into=None, sigma_merge=None):
    t = Trajectory(id=id, polarity=pol, points=pts, sigma_birth=birth,
                   sigma_death=death, terminal=terminal)
    t.merged_into = merged_into
    t.sigma_merge = sigma_merge
    return t


# --------------------------------------------------------------------------- #
# blob map: deterministic, L2-normalised, respects the alive window
# --------------------------------------------------------------------------- #
def test_blob_map_deterministic_and_normalized() -> None:
    sig = sigma_grid()
    # ink blob alive across σ_s=2 (k=16), centred at (64,64) on the 128 grid
    t = _traj(0, "ink", [(16, 64, 64, -1.0), (20, 64, 64, -0.8)], 1.5, 8.0, "vanish")
    m1 = build_blob_map([t], sig, 2.0)
    m2 = build_blob_map([t], sig, 2.0)
    assert np.array_equal(m1, m2)                      # bit-exact determinism
    assert abs(np.linalg.norm(m1) - 1.0) < 1e-5        # L2-normalised
    peak = np.unravel_index(m1.reshape(32, 32).argmax(), (32, 32))
    assert abs(peak[0] - 16) <= 1 and abs(peak[1] - 16) <= 1  # (64/4, 64/4)


def test_blob_map_respects_alive_window() -> None:
    sig = sigma_grid()
    # born at σ=4, so NOT alive at σ_s=2 -> empty map
    dead = _traj(0, "ink", [(24, 64, 64, -1.0), (28, 64, 64, -0.9)], 4.0, 16.0, "vanish")
    assert not np.any(build_blob_map([dead], sig, 2.0))
    # ground polarity is ignored (ink-only map)
    gnd = _traj(1, "ground", [(16, 30, 30, 1.0), (20, 30, 30, 0.9)], 1.0, 8.0, "vanish")
    assert not np.any(build_blob_map([gnd], sig, 2.0))


# --------------------------------------------------------------------------- #
# genealogy: parent=survivor, child=absorbed; edges==merges; roots==non-merge
# --------------------------------------------------------------------------- #
def test_genealogy_consistency() -> None:
    # t1 and t2 merge into t0 (survivor); t3 vanishes independently
    t0 = _traj(0, "ink", [(0, 10, 10, -1.0), (5, 10, 10, -1.0)], 0.5, 64.0, "survivor")
    t1 = _traj(1, "ink", [(0, 12, 12, -0.5), (2, 11, 11, -0.5)], 0.5, 1.0, "merge",
               merged_into=0, sigma_merge=1.0)
    t2 = _traj(2, "ink", [(0, 8, 8, -0.4)], 0.5, 1.4, "merge",  # ephemeral (1 point)
               merged_into=0, sigma_merge=1.4)
    t3 = _traj(3, "ink", [(0, 30, 30, -0.3), (3, 30, 30, -0.3)], 0.5, 0.9, "vanish")
    trajs = [t0, t1, t2, t3]
    tree = build_genealogy(trajs)
    chk = check_genealogy(tree, trajs)
    assert chk["ok"]
    assert chk["counts"] == {"nodes": 4, "merge": 2, "vanish": 1, "survivor": 1,
                             "edges": 2, "roots": 2}
    # ephemeral tag preserved for the 1-point trajectory
    assert next(n for n in tree["nodes"] if n["id"] == 2)["ephemeral"] is True
    assert tree["roots"] == [0, 3]  # survivor + vanished orphan


# --------------------------------------------------------------------------- #
# clean recall: stored pattern -> itself under the Modern Hopfield update
# --------------------------------------------------------------------------- #
def test_clean_recall_roundtrip() -> None:
    rng = np.random.default_rng(0)
    P = rng.standard_normal((6, 32))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    for i in range(6):
        rec = hopfield_recall(P, P[i], beta=16.0)
        assert nearest_pattern(P, rec) == i
