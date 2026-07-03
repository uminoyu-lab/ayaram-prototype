"""Linear Gaussian scale-space for the Ayaram v0.2 "尺跨ぎ" (scale-space) organ.

Convention (Lindeberg 1994 / 1998)
----------------------------------
The diffusion equation is written in the heat-flow normalisation

    ∂_t L = ½ ∇² L ,        t = σ²

so the scale parameter ``t`` is the *variance* of the smoothing Gaussian and
``σ = sqrt(t)`` is its standard deviation.  A slice ``L(·; σ)`` is the base
image convolved with an isotropic Gaussian of standard deviation ``σ``.  Slices
are computed **independently from the base image** (never cascaded), with a
constant-0 boundary (zero padding), per the v0.2 依頼書 §1-1 / 確定事項.

Feature response (依頼書 §1, γ-normalised Laplacian, γ = 1)
-----------------------------------------------------------
    R(x; σ) = t^γ · (∇² L)(x; σ) ,   γ = 1   =>   R = σ² · ∇²L

computed as a *direct* Laplacian-of-Gaussian (LoG) — the discrete Laplacian of
the Gaussian-smoothed slice — not a difference-of-Gaussians (DoG) approximation.

Polarity convention.  Foreground ink is 1.0 on a 0.0 ground (依頼書 確定事項).
A bright ink blob is a local peak, whose Laplacian is negative, so an **ink
blob** appears as ``R < 0`` (a scale-space local minimum) and a **ground /
counter-form blob** as ``R > 0`` (a local maximum).

Public API
----------
``build_scale_space``      base image  -> (K, H, W) stack of smoothed slices
``normalized_log_response``  slices     -> (K, H, W) normalised LoG response R
``detect_extrema``           R          -> per-slice bipolar 2D extrema
``link_trajectories``        extrema    -> σ-linked blob trajectories

This module only *adds* capability; v0.1 / v0.1.5 code and data are untouched
(依頼書 §6-1 互換規律).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "sigma_grid",
    "gaussian_kernel_1d",
    "gaussian_blur",
    "build_scale_space",
    "normalized_log_response",
    "Extremum",
    "detect_extrema",
    "Trajectory",
    "link_trajectories",
]


# --------------------------------------------------------------------------- #
# scale grid
# --------------------------------------------------------------------------- #
def sigma_grid(
    sigma0: float = 0.5,
    per_octave: int = 8,
    n_slices: int = 57,
) -> np.ndarray:
    """σ_k = sigma0 · 2^(k / per_octave), k = 0 .. n_slices-1.

    Default = 依頼書 確定事項: σ0 = 0.5, 8 slices/octave, 57 slices
    -> σ ∈ [0.5, 64].
    """
    k = np.arange(n_slices, dtype=np.float64)
    return sigma0 * np.power(2.0, k / per_octave)


# --------------------------------------------------------------------------- #
# gaussian blur (independent, zero-padded, GPU-capable)
# --------------------------------------------------------------------------- #
def gaussian_kernel_1d(sigma: float, truncate: float = 4.0) -> torch.Tensor:
    """Normalised 1-D sampled Gaussian, radius = ceil(truncate·σ) (>= 1)."""
    radius = max(1, int(np.ceil(truncate * float(sigma))))
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    k = torch.exp(-(x * x) / (2.0 * float(sigma) * float(sigma)))
    k /= k.sum()
    return k


def _as_tensor(image, device=None, dtype=torch.float32) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        t = image.to(dtype)
    else:
        t = torch.as_tensor(np.asarray(image), dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def gaussian_blur(image, sigma: float, truncate: float = 4.0) -> torch.Tensor:
    """Isotropic Gaussian blur with **constant-0 padding** (separable).

    ``image`` is a 2-D (H, W) tensor/array; returns a (H, W) tensor on the same
    device.  Separable: two 1-D convolutions.  Zero padding of ``radius`` on
    each side realises the constant-0 boundary demanded by the 依頼書.
    """
    t = image if isinstance(image, torch.Tensor) else _as_tensor(image)
    if t.dim() != 2:
        raise ValueError(f"expected 2-D image, got shape {tuple(t.shape)}")
    k = gaussian_kernel_1d(sigma, truncate).to(t.device, t.dtype)
    radius = (k.numel() - 1) // 2
    x = t[None, None]  # (1,1,H,W)
    # rows (last dim): pad width, kernel (1,1,1,L)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0)), k.view(1, 1, 1, -1))
    # cols (first spatial dim): pad height, kernel (1,1,L,1)
    x = F.conv2d(F.pad(x, (0, 0, radius, radius)), k.view(1, 1, -1, 1))
    return x[0, 0]


def build_scale_space(
    image,
    sigmas,
    truncate: float = 4.0,
    device=None,
) -> torch.Tensor:
    """Stack of independently-smoothed slices L(·; σ_k), shape (K, H, W).

    Each slice is computed **from the base image** (never cascaded), so the
    semigroup structure is carried by the Gaussian family, not by chaining.
    """
    base = _as_tensor(image, device=device)
    if base.dim() != 2:
        raise ValueError(f"expected 2-D image, got shape {tuple(base.shape)}")
    slices = [gaussian_blur(base, float(s), truncate) for s in np.asarray(sigmas)]
    return torch.stack(slices, dim=0)


# --------------------------------------------------------------------------- #
# normalised Laplacian-of-Gaussian response
# --------------------------------------------------------------------------- #
_LAPLACIAN_5PT = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
)


def normalized_log_response(
    scale_space: torch.Tensor,
    sigmas,
    gamma: float = 1.0,
) -> torch.Tensor:
    """R(x; σ) = t^γ · ∇²L, with t = σ² and γ = 1 by default.

    Direct LoG: the discrete 5-point Laplacian of each already-smoothed slice
    (zero padding), scaled by t^γ.  Returns (K, H, W) with the same device/dtype
    as ``scale_space``.
    """
    ss = scale_space
    if ss.dim() != 3:
        raise ValueError(f"expected (K,H,W) scale space, got {tuple(ss.shape)}")
    lap_k = _LAPLACIAN_5PT.to(ss.device, ss.dtype).view(1, 1, 3, 3)
    x = ss[:, None]  # (K,1,H,W)
    lap = F.conv2d(F.pad(x, (1, 1, 1, 1)), lap_k)[:, 0]  # (K,H,W), zero pad
    t = torch.as_tensor(np.asarray(sigmas), dtype=ss.dtype, device=ss.device) ** 2
    scale = (t ** gamma).view(-1, 1, 1)
    return scale * lap


# --------------------------------------------------------------------------- #
# bipolar 2D extrema per slice
# --------------------------------------------------------------------------- #
@dataclass
class Extremum:
    k: int          # slice index
    sigma: float    # σ_k
    y: int
    x: int
    R: float        # response value (signed)
    polarity: str   # 'ink' (R<0) or 'ground' (R>0)


def _strict_local_min(a: np.ndarray) -> np.ndarray:
    """Boolean mask of strict 2D 8-neighbour local minima (interior only)."""
    m = np.zeros(a.shape, dtype=bool)
    c = a[1:-1, 1:-1]
    lt = np.ones_like(c, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            lt &= c < a[1 + dy : a.shape[0] - 1 + dy, 1 + dx : a.shape[1] - 1 + dx]
    m[1:-1, 1:-1] = lt
    return m


def detect_extrema(
    response: torch.Tensor,
    sigmas,
    rel_threshold: float = 0.05,
    border: int = 3,
) -> list[Extremum]:
    """Per-slice bipolar strict 8-neighbour 2D extrema of R.

    ink blob   := strict local **minimum** with R < 0
    ground blob := strict local **maximum** with R > 0

    Kept only if ``|R| >= rel_threshold · max|R|`` over the whole cube of this
    glyph (依頼書 確定事項; default 0.05, CC discretion in [0.01, 0.1]).

    Border handling (CC 工夫).  The constant-0 blur boundary makes the discrete
    Laplacian inject a huge artificial curvature at the image *frame*, and t=σ²
    amplifies it at large σ (e.g. |R|~56 at a corner vs ~0.5 for a real blob).
    Left unchecked it both hijacks the max|R| normalisation and spawns phantom
    frame blobs.  We therefore ignore a ``border``-px frame for **both** the
    normalisation and the detection — standard scale-space border discarding.
    """
    R = response.detach().cpu().numpy().astype(np.float64)
    K, H, W = R.shape
    sig = np.asarray(sigmas, dtype=np.float64)
    b = max(1, int(border))
    interior = R[:, b:H - b, b:W - b]
    thr = rel_threshold * float(np.abs(interior).max()) if interior.size else 0.0
    out: list[Extremum] = []
    for k in range(K):
        s = R[k]
        mins = _strict_local_min(s)
        maxs = _strict_local_min(-s)  # strict maxima of s
        frame = np.zeros((H, W), dtype=bool)
        frame[b:H - b, b:W - b] = True
        for (yy, xx) in zip(*np.where(frame & mins & (s < 0) & (np.abs(s) >= thr))):
            out.append(Extremum(k, float(sig[k]), int(yy), int(xx), float(s[yy, xx]), "ink"))
        for (yy, xx) in zip(*np.where(frame & maxs & (s > 0) & (np.abs(s) >= thr))):
            out.append(Extremum(k, float(sig[k]), int(yy), int(xx), float(s[yy, xx]), "ground"))
    return out


# --------------------------------------------------------------------------- #
# σ-axis linking -> trajectories
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    id: int
    polarity: str
    points: list[tuple[int, int, int, float]] = field(default_factory=list)  # (k,y,x,R)
    sigma_birth: float = 0.0
    sigma_death: float = 0.0
    terminal: str = "survivor"        # 'vanish' | 'merge' | 'survivor'
    birth_event: bool = False          # predecessor-less birth at k>0 (generation)
    sigma_merge: float | None = None
    merged_into: int | None = None

    @property
    def k_birth(self) -> int:
        return self.points[0][0]

    @property
    def k_death(self) -> int:
        return self.points[-1][0]


def link_trajectories(
    extrema: list[Extremum],
    sigmas,
    min_radius: float = 2.0,
    radius_frac: float = 0.5,
    max_gap: int = 0,
    gap_relax: bool = True,
) -> list[Trajectory]:
    """Link per-slice extrema across σ slices into trajectories.

    Rule (依頼書 確定事項):
      * nearest-neighbour, **same polarity only**
      * gate distance ``d_max = max(min_radius, radius_frac · σ_k)`` px
      * no successor -> death (σ_death = last observed σ), terminal 'vanish'
      * two -> one -> merge: absorbed side = **smaller |R|**, records σ_merge and
        is treated as a death (terminal 'merge', ``merged_into`` set)
      * no predecessor at k>0 -> birth event (``birth_event=True``, generation)
      * alive at σ_max -> right-censored survivor (terminal 'survivor')

    Matching is traj->nearest-candidate; candidates won by >1 traj trigger a
    merge (surviving = larger |R|).

    ``max_gap`` (v0.2 M1b, 採用 G1).  With ``max_gap=0`` (default) a trajectory
    unmatched at the next slice dies immediately — **identical to M1**.  With
    ``max_gap=1`` a trajectory broken at slice k may reconnect at k+2: a still-
    dormant trajectory (last seen ``lk``) stays a candidate while ``k-lk <=
    max_gap+1``, cutting the strict-extrema flicker churn.  ``gap_relax`` scales
    the gate by the slice-gap (CC 裁量): a blob skipping g slices is matched
    within ``g · d_max`` (g=1 => unchanged from M1).  A dormant trajectory that
    reaches σ_max without a last-slice observation is scored 'vanish', not
    'survivor'.
    """
    sig = np.asarray(sigmas, dtype=np.float64)
    K = len(sig)
    by_slice: dict[int, list[Extremum]] = {k: [] for k in range(K)}
    for e in extrema:
        by_slice[e.k].append(e)

    trajs: list[Trajectory] = []
    next_id = 0
    # active: id -> (traj, last_y, last_x, last_R, last_k)
    active: dict[int, tuple[Trajectory, float, float, float, int]] = {}

    for k in range(K):
        cur = by_slice[k]
        d_max = max(min_radius, radius_frac * float(sig[k]))
        # --- propose traj -> nearest same-polarity candidate within (gap-scaled) gate ---
        proposals: dict[int, list[tuple[int, float, float]]] = {}
        matched_traj: set[int] = set()
        for tid, (traj, ly, lx, lR, lk) in active.items():
            gap = k - lk  # >= 1
            gate = d_max * (gap if gap_relax else 1)
            best_j, best_d = -1, None
            for j, e in enumerate(cur):
                if e.polarity != traj.polarity:
                    continue
                d = float(np.hypot(e.y - ly, e.x - lx))
                if d <= gate and (best_d is None or d < best_d):
                    best_d, best_j = d, j
            if best_j >= 0:
                proposals.setdefault(best_j, []).append((tid, best_d, abs(lR)))
                matched_traj.add(tid)

        assigned_cand: set[int] = set()
        new_active: dict[int, tuple[Trajectory, float, float, float, int]] = {}
        for j, claimants in proposals.items():
            e = cur[j]
            # surviving traj = largest |R| (absorbing side); others absorbed
            claimants.sort(key=lambda c: c[1])           # nearest first (tiebreak)
            claimants.sort(key=lambda c: c[2], reverse=True)  # largest |R| wins
            winner_id = claimants[0][0]
            wtraj = active[winner_id][0]
            wtraj.points.append((k, e.y, e.x, e.R))
            new_active[winner_id] = (wtraj, e.y, e.x, e.R, k)
            assigned_cand.add(j)
            for loser_id, _, _ in claimants[1:]:
                ltraj = active[loser_id][0]
                ltraj.terminal = "merge"
                ltraj.sigma_merge = float(sig[k])
                ltraj.sigma_death = float(sig[k])
                ltraj.merged_into = winner_id

        # --- unmatched active trajs: carry dormant within gap grace, else vanish ---
        for tid, (traj, ly, lx, lR, lk) in active.items():
            if tid in matched_traj:
                continue
            if k - lk >= max_gap + 1:
                traj.terminal = "vanish"
                traj.sigma_death = float(sig[traj.k_death])
            else:
                new_active[tid] = (traj, ly, lx, lR, lk)  # dormant, last_k unchanged

        # --- unassigned candidates -> births ---
        for j, e in enumerate(cur):
            if j in assigned_cand:
                continue
            tr = Trajectory(
                id=next_id,
                polarity=e.polarity,
                points=[(k, e.y, e.x, e.R)],
                sigma_birth=float(sig[k]),
                birth_event=(k > 0),
            )
            trajs.append(tr)
            new_active[next_id] = (tr, e.y, e.x, e.R, k)
            next_id += 1

        active = new_active

    # --- end: last-slice observation -> survivor; stale dormant -> vanish ---
    for tid, (traj, ly, lx, lR, lk) in active.items():
        if lk == K - 1:
            traj.terminal = "survivor"
            traj.sigma_death = float(sig[-1])
        else:
            traj.terminal = "vanish"
            traj.sigma_death = float(sig[traj.k_death])

    trajs.sort(key=lambda t: t.id)
    return trajs
