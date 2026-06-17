"""Tests for the M5 ``HopfieldNetwork.learn(center_inter_inputs=...)`` option.

Locks in:

* ``center_inter_inputs=False`` (default) preserves the M3 / M4 behavior;
  ``True`` actually changes ``W_inter`` (the M4 demo helper has effectively
  moved into ``learn`` itself).

* ``center_inter_inputs=True`` is mathematically the
  zero-mean-along-batch outer product:
  ``W_inter[l] = (1/N) * (p_l - mean) @ p_{l+1}``. Verified by direct
  formula equality.

* ``HopfieldLayer.store`` is *not* affected by the centering flag --
  intra-layer recall still sees the raw patterns.

* Caveat: Modern + centering improves hierarchical layer-2 recall on the
  M4 12-kanji set; Hebb + centering degrades it. Pinned with miniature
  end-to-end recall checks so future refactors do not silently regress.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

from ayaram import core, encoding
from ayaram.memory import HopfieldNetwork


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import kanji_hierarchy_v15 as KH_V15  # noqa: E402


def _patterns(N: int, sizes: tuple[int, ...], seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N, s, generator=g) for s in sizes]


# ----- centering toggle -----------------------------------------------------


def test_center_inter_inputs_false_matches_m3_path():
    sizes = (32, 16, 8)
    patterns = _patterns(N=4, sizes=sizes, seed=0)
    a = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    b = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    a.learn(patterns, normalize_inter="spectral", center_inter_inputs=False)
    b.learn(patterns, normalize_inter="spectral")  # default
    for Wa, Wb in zip(a.W_inter, b.W_inter):
        assert torch.allclose(Wa, Wb, atol=1e-7)


def test_center_inter_inputs_true_changes_W_inter():
    sizes = (32, 16, 8)
    patterns = _patterns(N=4, sizes=sizes, seed=0)
    a = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    b = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    a.learn(patterns, center_inter_inputs=False)
    b.learn(patterns, center_inter_inputs=True)
    for Wa, Wb in zip(a.W_inter, b.W_inter):
        assert not torch.allclose(Wa, Wb)


def test_center_inter_inputs_matches_direct_formula():
    """Check the centered outer product equals the analytical
    ``(p_l - mean)^T @ p_{l+1} / N`` modulo spectral normalization."""
    sizes = (32, 16, 8)
    patterns = _patterns(N=4, sizes=sizes, seed=0)
    net = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    net.learn(patterns, normalize_inter="none", center_inter_inputs=True)
    for l in range(len(sizes) - 1):
        pl_c = patterns[l] - patterns[l].mean(dim=0, keepdim=True)
        expected = pl_c.T @ patterns[l + 1] / patterns[l].shape[0]
        assert torch.allclose(net.W_inter[l], expected, atol=1e-6)


def test_center_inter_inputs_does_not_affect_intra_layer_W():
    """``HopfieldLayer.W`` should be the same whether centering is on or
    off -- the flag is exclusively a property of the inter-layer Hebb."""
    sizes = (32, 16, 8)
    patterns = _patterns(N=4, sizes=sizes, seed=0)
    a = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    b = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    a.learn(patterns, center_inter_inputs=False)
    b.learn(patterns, center_inter_inputs=True)
    for la, lb in zip(a.layers, b.layers):
        assert torch.allclose(la.W, lb.W, atol=1e-7)


# ----- hierarchical recall mini end-to-end ---------------------------------


@pytest.fixture(scope="module")
def m4_12_setup():
    """Pre-load the M4 v15 12-kanji bitmaps + encoded patterns once."""
    bitmaps = np.load(os.path.join(_ROOT, "data", "kanji_12_32x32_v15.npy"))
    p0 = torch.from_numpy(bitmaps.reshape(bitmaps.shape[0], -1)).to(torch.float32)
    p1 = encoding.encode_batch_radical_count_v15(
        KH_V15.KANJI,
        kanji_radicals=KH_V15.KANJI_RADICALS,
        radicals=KH_V15.RADICALS,
        max_count=KH_V15.MAX_COUNT,
    )
    p2 = encoding.encode_batch_origin_v15(
        KH_V15.KANJI,
        kanji_origin=KH_V15.KANJI_ORIGIN,
        origins=KH_V15.ORIGINS,
    )
    return bitmaps, p0, p1, p2


def _run_hierarchical(mode: str, p0, p1, p2, center, bitmaps, n_steps=300):
    """Mini forward cycle, returns origin one-hot match count over 12 kanji."""
    dev = torch.device("cpu")
    net = HopfieldNetwork(mode=mode, seed=0).to(dev)
    net.learn(
        [p0.to(dev), p1.to(dev), p2.to(dev)],
        normalize_inter="spectral",
        center_inter_inputs=center,
    )
    n = bitmaps.shape[0]
    occ = bitmaps.copy()
    occ[:, :, 16:] = -1.0
    occ_flat = torch.from_numpy(occ.reshape(n, -1)).to(torch.float32).to(dev)
    sizes = net.layer_sizes
    state = core.CycleState(
        xi=[torch.zeros(n, s, device=dev) for s in sizes]
    )
    core.phase1_terrain(net, state, occ_flat, layer_idx=0)
    cfg = core.CycleConfig(
        beta=5.0,
        sigma_global=0.1,
        phase2_steps=n_steps,
        phase3_steps=50,
        inter_layer_scale=0.1,
    )
    g = torch.Generator(device=dev).manual_seed(1)
    core.phase2_fluctuation(net, state, cfg, generator=g)
    core.phase3_fixation(net, state, cfg)

    hits = 0
    for i, k in enumerate(KH_V15.KANJI):
        if encoding.origin_one_hot_match_v15(
            state.xi[2][i].detach().cpu(),
            k,
            kanji_origin=KH_V15.KANJI_ORIGIN,
            origins=KH_V15.ORIGINS,
        ):
            hits += 1
    return hits


def test_modern_with_centering_recovers_M4_layer2(m4_12_setup):
    """The headline M4 result: Modern + center_inter_inputs=True puts the
    layer-2 origin recall well above the 0.7 Part C target on the
    12-kanji set. We use reduced step counts so the test stays fast; the
    threshold below is intentionally generous (>= 7/12 ~= 0.58) to
    tolerate the reduced steps while still failing on regressions to
    M3-level (4/12) behavior."""
    bitmaps, p0, p1, p2 = m4_12_setup
    hits = _run_hierarchical("modern", p0, p1, p2, center=True, bitmaps=bitmaps)
    assert hits >= 7, f"Modern+centering only matched {hits}/12 origins"


def test_modern_without_centering_stays_at_M3_level(m4_12_setup):
    """Sanity check: turn centering off and Modern reverts to the M3-level
    layer-2 origin (around 4/12 -- only the four 天体 kanji)."""
    bitmaps, p0, p1, p2 = m4_12_setup
    hits = _run_hierarchical("modern", p0, p1, p2, center=False, bitmaps=bitmaps)
    assert hits <= 6, (
        f"Modern without centering matched {hits}/12 origins; the M3 "
        f"baseline is 4/12 and we expect centering to be the lever that "
        f"unlocks the rest"
    )


def test_hebb_with_centering_is_documented_caveat(m4_12_setup):
    """Hebb + centering degrades; this test pins the documented caveat so
    a future refactor noticing the degradation cannot remove centering
    without revisiting README v0.2 homework #9."""
    bitmaps, p0, p1, p2 = m4_12_setup
    hits = _run_hierarchical("hebb", p0, p1, p2, center=True, bitmaps=bitmaps)
    # Either similar to or below the no-centering case; we just assert
    # this is not the Modern-with-centering regime.
    assert hits < 10, (
        f"Hebb+centering matched {hits}/12 -- unexpectedly high; please "
        f"revisit README v0.2 homework #9 if this becomes Modern-like."
    )
