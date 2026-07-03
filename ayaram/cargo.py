"""v0.2 M2 — 踊り場スナップショット → Hopfield 積み荷 (cargo).

Turns the scale-space blob trajectories of a glyph into a fixed-size "blob map"
that the v0.1 Modern Hopfield (`ayaram.memory`) can store as a pattern, plus the
merge genealogy tree that M3 (downward generation) will replay.

確定事項 (G1 approved, 2026-07-03):
  * official linking = churn-reduced gap=1 (M1b), lifetime<2 dropped from the
    blob map (noise) but KEPT in the genealogy (tagged ``ephemeral``)
  * snapshots σ_s ∈ {2, 2.83, 4} (log2σ = 1, 1.5, 2); a blob is alive at σ_s iff
    σ_birth ≤ σ_s < σ_death (survivors never die)
  * blob map = 32×32 float32; each alive ink blob is an isotropic Gaussian bump
    at (x/4, y/4), width σ_s/4, amplitude |R| at σ_s (per-glyph max-normalised);
    the whole map is L2-normalised.  Pattern = flatten -> 1024 dim (= layer-0).
  * genealogy: parent = merge survivor, child = absorbed side, root = survivor.

Only *adds* capability; v0.1/v0.1.5 code and data untouched.
"""

from __future__ import annotations

import numpy as np
import torch

from .learning import modern_hopfield_update

__all__ = [
    "SNAP_LOG2",
    "snapshot_sigmas",
    "snapshot_index",
    "blob_at_slice",
    "alive_ink_blobs",
    "build_blob_map",
    "build_blob_maps",
    "build_genealogy",
    "check_genealogy",
    "hopfield_recall",
    "nearest_pattern",
]

SNAP_LOG2: tuple[float, ...] = (1.0, 1.5, 2.0)   # log2 σ_s -> σ_s = 2, 2.83, 4
GRID: int = 32
SRC: int = 128


def snapshot_sigmas(sigmas) -> list[float]:
    """The grid σ values closest to σ_s = 2^{SNAP_LOG2}."""
    sig = np.asarray(sigmas, dtype=float)
    return [float(sig[int(np.argmin(np.abs(sig - 2.0 ** l)))]) for l in SNAP_LOG2]


def snapshot_index(sigmas, sigma_s) -> int:
    return int(np.argmin(np.abs(np.asarray(sigmas, dtype=float) - float(sigma_s))))


def blob_at_slice(traj, k_s):
    """(y, x, R) of a trajectory at slice ``k_s``.

    Uses the point at k_s if present; otherwise (a gap slice) the most recent
    point with k <= k_s; otherwise the earliest point.
    """
    pts = traj.points
    exact = [p for p in pts if p[0] == k_s]
    if exact:
        _, y, x, R = exact[0]
        return y, x, R
    before = [p for p in pts if p[0] <= k_s]
    _, y, x, R = (before[-1] if before else pts[0])
    return y, x, R


def alive_ink_blobs(trajs, sigmas, sigma_s, min_lifetime=2):
    """List of (y, x, R) for ink trajectories alive at σ_s (lifetime-filtered)."""
    k_s = snapshot_index(sigmas, sigma_s)
    out = []
    for t in trajs:
        if t.polarity != "ink" or len(t.points) < min_lifetime:
            continue
        if t.sigma_birth <= sigma_s < t.sigma_death:
            out.append(blob_at_slice(t, k_s))
    return out


def build_blob_map(trajs, sigmas, sigma_s, grid=GRID, src=SRC,
                   min_lifetime=2, width=None):
    """32×32 L2-normalised ink-blob map at snapshot σ_s.

    ``width`` (blob σ in grid px) defaults to σ_s/scale (the spec's σ_s/4 on
    the 128->32 downscale); pass a value to compare the feature-scale variant.
    Returns a (grid*grid,) float32 vector.
    """
    scale = src / grid
    if width is None:
        width = float(sigma_s) / scale
    blobs = alive_ink_blobs(trajs, sigmas, sigma_s, min_lifetime)
    m = np.zeros((grid, grid), dtype=np.float64)
    if blobs:
        maxR = max(abs(R) for _, _, R in blobs) or 1.0
        gy, gx = np.mgrid[0:grid, 0:grid]
        two_w2 = 2.0 * width * width
        for (y, x, R) in blobs:
            cy, cx = y / scale, x / scale
            m += (abs(R) / maxR) * np.exp(-((gy - cy) ** 2 + (gx - cx) ** 2) / two_w2)
    n = np.linalg.norm(m)
    if n > 0:
        m /= n
    return m.reshape(-1).astype(np.float32)


def build_blob_maps(per_char_trajs, sigmas, chars, sigma_s, **kw):
    """(N, grid*grid) stack of blob maps for the given σ_s."""
    return np.stack([build_blob_map(per_char_trajs[c], sigmas, sigma_s, **kw)
                     for c in chars], axis=0)


# --------------------------------------------------------------------------- #
# merge genealogy
# --------------------------------------------------------------------------- #
def build_genealogy(trajs):
    """Genealogy forest: nodes + merge edges (parent=survivor, child=absorbed).

    Schema (指示書 確定事項):
      nodes[{id, polarity, sigma_birth, sigma_death, end_type, pos_at_death,
             ephemeral}]
      edges[{parent, child, sigma_merge}]
    ``end_type`` in {survivor, vanish, merge}; root = survivor (no parent).
    Lifetime<2 trajectories are KEPT here, tagged ``ephemeral``.
    """
    nodes, edges = [], []
    for t in trajs:
        _, y, x, _ = t.points[-1]
        nodes.append({
            "id": int(t.id), "polarity": t.polarity,
            "sigma_birth": round(float(t.sigma_birth), 4),
            "sigma_death": round(float(t.sigma_death), 4),
            "end_type": t.terminal, "pos_at_death": [int(y), int(x)],
            "ephemeral": bool(len(t.points) < 2),
        })
        if t.terminal == "merge" and t.merged_into is not None:
            edges.append({"parent": int(t.merged_into), "child": int(t.id),
                          "sigma_merge": round(float(t.sigma_merge), 4)})
    roots = sorted(n["id"] for n in nodes if n["end_type"] != "merge")
    return {"nodes": nodes, "edges": edges, "roots": roots}


def check_genealogy(tree, trajs):
    """Structural consistency checks; returns a dict of booleans + counts."""
    nodes = tree["nodes"]; edges = tree["edges"]
    ids = {n["id"] for n in nodes}
    child_ids = [e["child"] for e in edges]
    parents = {e["child"]: e["parent"] for e in edges}
    n_merge = sum(1 for n in nodes if n["end_type"] == "merge")
    n_vanish = sum(1 for n in nodes if n["end_type"] == "vanish")
    n_surv = sum(1 for n in nodes if n["end_type"] == "survivor")
    survivors = {n["id"] for n in nodes if n["end_type"] == "survivor"}
    roots = {n["id"] for n in nodes if n["id"] not in parents}

    # no cycles: walk parent links to a root without revisiting
    acyclic = True
    for start in ids:
        seen, cur = set(), start
        while cur in parents:
            if cur in seen:
                acyclic = False; break
            seen.add(cur); cur = parents[cur]
        if not acyclic:
            break

    checks = {
        "node_count_matches": len(nodes) == len(trajs),
        "edge_count_eq_merges": len(edges) == n_merge,
        "each_child_once": len(child_ids) == len(set(child_ids)),
        "all_endpoints_valid": all(e["parent"] in ids and e["child"] in ids for e in edges),
        "acyclic": acyclic,
        "survivors_are_roots": survivors <= roots,
        "counts": {"nodes": len(nodes), "merge": n_merge, "vanish": n_vanish,
                   "survivor": n_surv, "edges": len(edges), "roots": len(roots)},
    }
    checks["ok"] = all(v for k, v in checks.items()
                       if k != "counts" and isinstance(v, bool))
    return checks


# --------------------------------------------------------------------------- #
# Modern Hopfield recall (v0.1 ayaram.learning)
# --------------------------------------------------------------------------- #
def hopfield_recall(patterns, query, beta, steps=3):
    """Iterate the Modern Hopfield update; return the converged state.

    ``patterns`` (N, d) row-stacked stored patterns; ``query`` (d,).
    """
    X = torch.as_tensor(np.asarray(patterns), dtype=torch.float32).T.contiguous()
    xi = torch.as_tensor(np.asarray(query), dtype=torch.float32).clone()
    for _ in range(steps):
        xi = modern_hopfield_update(xi, X, beta=beta)
    return xi.numpy()


def nearest_pattern(patterns, vec):
    """Index of the stored pattern with max cosine to ``vec``."""
    P = np.asarray(patterns, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    vv = v / (np.linalg.norm(v) + 1e-12)
    return int(np.argmax(pn @ vv))
