"""M0 unit tests for the v0.2 Gaussian scale-space organ (ayaram/scalespace.py).

Two required tests (依頼書 M0 DoD-3, 採用):
  * test_semigroup_blur      — blur(t1) ∘ blur(t2) ≈ blur(t1+t2)
  * test_blob_feature_scale  — synthetic Gaussian blob (scale s) -> σ* ≈ s

Plus two CC guard tests protecting the M1 pipeline (polarity sign, linking).
"""

from __future__ import annotations

import numpy as np
import pytest

from ayaram.scalespace import (
    build_scale_space,
    detect_extrema,
    gaussian_blur,
    link_trajectories,
    normalized_log_response,
    sigma_grid,
)


def _centered_blob(h: int, w: int, s: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * s * s)).astype(np.float32)


# --------------------------------------------------------------------------- #
# (a) semigroup:  blur(σ1) ∘ blur(σ2) ≈ blur(sqrt(σ1²+σ2²))   (t = σ²)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s1,s2", [(3.0, 4.0), (2.0, 2.0), (5.0, 1.5)])
def test_semigroup_blur(s1: float, s2: float) -> None:
    blob = _centered_blob(64, 64, 6.0)
    s_total = float(np.hypot(s1, s2))  # sqrt(s1^2 + s2^2), i.e. t1+t2
    cascaded = gaussian_blur(gaussian_blur(blob, s1), s2)
    direct = gaussian_blur(blob, s_total)
    # interior crop keeps the constant-0 boundary from dominating the check
    c = slice(12, 52)
    max_diff = (cascaded - direct)[c, c].abs().max().item()
    assert max_diff < 1e-3, f"semigroup violated: max interior diff {max_diff:g}"


# --------------------------------------------------------------------------- #
# (b) feature scale:  Gaussian blob of std s -> normalised-LoG peak at σ* ≈ s
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s0", [4.0, 6.0, 10.0])
def test_blob_feature_scale(s0: float) -> None:
    sig = sigma_grid()  # σ ∈ [0.5, 64], 8/octave
    blob = _centered_blob(64, 64, s0)
    ss = build_scale_space(blob, sig)
    R = normalized_log_response(ss, sig)
    cy = cx = 32
    center = R[:, cy, cx].cpu().numpy()  # bright ink blob -> R < 0
    sigma_star = float(sig[int(np.argmin(center))])
    # within one eighth-octave grid step (2^(1/8) ≈ 1.09) of the true scale
    assert abs(np.log2(sigma_star / s0)) < 1.0 / 8.0 + 1e-6, (
        f"feature scale off: σ*={sigma_star:.3f} vs s={s0}"
    )


# --------------------------------------------------------------------------- #
# CC guard 1: polarity sign — ink blob is an R<0 extremum, ground is R>0
# --------------------------------------------------------------------------- #
def test_polarity_sign() -> None:
    sig = sigma_grid()
    blob = _centered_blob(64, 64, 6.0)  # bright (ink=1) blob on 0 ground
    ss = build_scale_space(blob, sig)
    R = normalized_log_response(ss, sig)
    ex = detect_extrema(R, sig, rel_threshold=0.05)
    assert ex, "expected at least one extremum for a clear blob"
    inks = [e for e in ex if e.polarity == "ink"]
    assert inks, "bright ink blob should yield an ink (R<0) extremum"
    assert all(e.R < 0 for e in inks)
    assert all(e.R > 0 for e in ex if e.polarity == "ground")
    # the strongest ink extremum sits at the blob center
    top = min(inks, key=lambda e: e.R)
    assert abs(top.y - 32) <= 1 and abs(top.x - 32) <= 1


# --------------------------------------------------------------------------- #
# CC guard 2: linking — a single blob yields one dominant ink trajectory that
# is born early and survives/vanishes coherently (no fragmentation explosion)
# --------------------------------------------------------------------------- #
def test_link_single_blob_trajectory() -> None:
    sig = sigma_grid()
    blob = _centered_blob(64, 64, 6.0)
    ss = build_scale_space(blob, sig)
    R = normalized_log_response(ss, sig)
    ex = detect_extrema(R, sig, rel_threshold=0.05)
    trajs = link_trajectories(ex, sig)
    ink = [t for t in trajs if t.polarity == "ink"]
    assert ink, "expected an ink trajectory"
    longest = max(ink, key=lambda t: len(t.points))
    # the dominant ink blob persists across many slices and stays centered
    assert len(longest.points) >= 10
    assert all(abs(y - 32) <= 2 and abs(x - 32) <= 2 for _, y, x, _ in longest.points)
    assert longest.terminal in {"vanish", "merge", "survivor"}
