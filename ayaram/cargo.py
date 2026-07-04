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
    "draw_blob_map",
    "build_blob_map",
    "build_blob_maps",
    "extract_records",
    "replay_map",
    "build_genealogy",
    "check_genealogy",
    "hopfield_recall",
    "nearest_pattern",
    "confusion_matrix",
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


def draw_blob_map(blobs_yxw, sigma, grid=GRID, src=SRC, width=None):
    """Render (y, x, weight) blobs (128-grid coords, weight>=0) into a 32×32
    L2-normalised map. weight is max-normalised, width defaults to σ/scale."""
    scale = src / grid
    if width is None:
        width = float(sigma) / scale
    m = np.zeros((grid, grid), dtype=np.float64)
    if blobs_yxw:
        maxw = max(abs(w) for _, _, w in blobs_yxw) or 1.0
        gy, gx = np.mgrid[0:grid, 0:grid]
        two_w2 = 2.0 * width * width
        for (y, x, w) in blobs_yxw:
            cy, cx = y / scale, x / scale
            m += (abs(w) / maxw) * np.exp(-((gy - cy) ** 2 + (gx - cx) ** 2) / two_w2)
    n = np.linalg.norm(m)
    if n > 0:
        m /= n
    return m.reshape(-1).astype(np.float32)


def build_blob_map(trajs, sigmas, sigma_s, grid=GRID, src=SRC,
                   min_lifetime=2, width=None):
    """32×32 L2-normalised ink-blob map at snapshot σ_s (direct-slice)."""
    blobs = alive_ink_blobs(trajs, sigmas, sigma_s, min_lifetime)
    return draw_blob_map([(y, x, abs(R)) for (y, x, R) in blobs], sigma_s,
                         grid, src, width)


def build_blob_maps(per_char_trajs, sigmas, chars, sigma_s, **kw):
    """(N, grid*grid) stack of blob maps for the given σ_s."""
    return np.stack([build_blob_map(per_char_trajs[c], sigmas, sigma_s, **kw)
                     for c in chars], axis=0)


# --------------------------------------------------------------------------- #
# M3 reverse replay: full (position+intensity series) vs tree-only (fixed pos)
# --------------------------------------------------------------------------- #
def extract_records(trajs, sigmas, min_lifetime=2, survivor_k=24):
    """Serialisable per-ink-blob records for reverse replay.

    Each record: {sigma_birth, sigma_death, series[(k,y,x,R)], pos_fixed(y,x),
    amp_fixed, ephemeral}.  ``pos_fixed``/``amp_fixed`` = position/|R| at death
    (the M3-2 "tree-only" init), except survivors, which use the σ=survivor_k
    slice (pos_at_death would sit at σ_max, outside the replay range).
    """
    recs = []
    for t in trajs:
        if t.polarity != "ink" or len(t.points) < min_lifetime:
            continue
        series = [(int(k), int(y), int(x), float(R)) for (k, y, x, R) in t.points]
        if t.terminal == "survivor":
            y, x, R = blob_at_slice(t, survivor_k)
            pos, amp = (int(y), int(x)), abs(float(R))
        else:
            _, y, x, R = t.points[-1]
            pos, amp = (int(y), int(x)), abs(float(R))
        recs.append({"sigma_birth": float(t.sigma_birth), "sigma_death": float(t.sigma_death),
                     "series": series, "pos_fixed": pos, "amp_fixed": amp,
                     "ephemeral": bool(len(t.points) < 2)})
    return recs


def _series_point(series, k_s):
    exact = [p for p in series if p[0] == k_s]
    if exact:
        return exact[0]
    before = [p for p in series if p[0] <= k_s]
    return before[-1] if before else series[0]


def replay_map(records, sigmas, sigma, mode="full", grid=GRID, src=SRC, width=None):
    """Reconstruct the 32×32 map at ``sigma`` from records (σ降順 replay).

    mode='full' : position & |R| from the stored series at σ (== direct slice;
                  the M3-1 gate).
    mode='tree' : fixed pos_fixed & amp_fixed for every alive blob — the
                  drift-discarding "tree-only" reconstruction (M3-2).
    A blob is alive iff σ_birth ≤ σ < σ_death.
    """
    k_s = snapshot_index(sigmas, sigma)
    blobs = []
    for r in records:
        if not (r["sigma_birth"] <= sigma < r["sigma_death"]):
            continue
        if mode == "full":
            _, y, x, R = _series_point(r["series"], k_s)
            blobs.append((y, x, abs(R)))
        else:  # tree
            y, x = r["pos_fixed"]
            blobs.append((y, x, r["amp_fixed"]))
    return draw_blob_map(blobs, sigma, grid, src, width)


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


def confusion_matrix(Q, P, beta):
    """W[i,j] = softmax_j(beta * <Q_i, P_j>) (M4 recall-confusion structure).

    ``Q``/``P`` are row-stacked; rows are L2-normalised internally so the logits
    are beta*cosine.  Returns an (nq, np) row-stochastic matrix.
    """
    Q = np.asarray(Q, float); P = np.asarray(P, float)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    logits = beta * (Qn @ Pn.T)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)
