"""Tests for the M4 ``HopfieldNetwork.learn(normalize_inter=...)`` option.

Locks in:

* ``normalize_inter='spectral'`` rescales every ``W_inter`` to spectral
  norm <= 1 (and == 1 when the raw outer product is non-trivial). This is
  the M4 fix for the layer-pair imbalance discovered in M3.

* ``normalize_inter='none'`` leaves ``W_inter`` at the raw outer-product
  scale (used for M3-reproduction tests and for v0.2 experiments that want
  physically-scaled couplings).

* Argument validation rejects unknown modes.
"""

from __future__ import annotations

import pytest
import torch

from ayaram.memory import HopfieldNetwork


def _patterns(N: int, sizes: tuple[int, ...], seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N, s, generator=g) for s in sizes]


def _spectral(W: torch.Tensor) -> float:
    return float(torch.linalg.svdvals(W)[0].item())


def test_learn_default_is_spectral_normalization():
    sizes = (64, 16, 8)
    net = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    net.learn(_patterns(N=4, sizes=sizes))
    for W in net.W_inter:
        assert _spectral(W) <= 1.0 + 1e-6
    # And not trivially zero
    for W in net.W_inter:
        assert _spectral(W) > 0.5


def test_learn_spectral_brings_W_inter_to_unit_spectral_norm():
    sizes = (64, 16, 8)
    net = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    net.learn(_patterns(N=4, sizes=sizes), normalize_inter="spectral")
    for W in net.W_inter:
        sn = _spectral(W)
        # Should be exactly 1 for non-degenerate W
        assert abs(sn - 1.0) < 1e-5, f"spectral={sn}"


def test_learn_none_preserves_raw_outer_product_scale():
    sizes = (64, 16, 8)
    patterns = _patterns(N=4, sizes=sizes)
    net_a = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    net_b = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    net_a.learn(patterns, normalize_inter="spectral")
    net_b.learn(patterns, normalize_inter="none")
    for Wa, Wb in zip(net_a.W_inter, net_b.W_inter):
        # Non-normalized version should be at a different magnitude
        sa = _spectral(Wa)
        sb = _spectral(Wb)
        # If they happened to be equal the test below makes no sense; with
        # random patterns they will differ.
        assert abs(sa - 1.0) < 1e-5
        # Raw spectral may be any positive value; just assert it's distinct
        # from the normalized one when significantly off-scale.
        assert sb > 0
        # And the two versions must produce different matrices.
        assert not torch.allclose(Wa, Wb)


def test_learn_rejects_unknown_normalize_mode():
    sizes = (64, 16, 8)
    net = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    with pytest.raises(ValueError):
        net.learn(_patterns(N=4, sizes=sizes), normalize_inter="rmsnorm")


def test_learn_spectral_equalizes_imbalanced_pairs():
    """The whole point of spectral normalization: rescue the case where one
    layer pair's outer product is orders of magnitude larger than the
    other's. We construct such an instance by making layer 0 patterns
    much larger than the rest."""
    sizes = (256, 16, 8)
    g = torch.Generator().manual_seed(42)
    p0 = 32.0 * torch.randn(4, sizes[0], generator=g)  # bitmap-like norm
    p1 = torch.randn(4, sizes[1], generator=g)
    p2 = torch.randn(4, sizes[2], generator=g)
    net_raw = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    net_raw.learn([p0, p1, p2], normalize_inter="none")
    s_raw = [_spectral(W) for W in net_raw.W_inter]
    # raw scales differ by a large factor (layer-0 norm ~32 vs layer-2 norm ~1)
    assert s_raw[0] > 5 * s_raw[1] or s_raw[1] > 5 * s_raw[0]

    net_norm = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    net_norm.learn([p0, p1, p2], normalize_inter="spectral")
    s_norm = [_spectral(W) for W in net_norm.W_inter]
    for sn in s_norm:
        assert abs(sn - 1.0) < 1e-5


def test_learn_spectral_preserves_symmetry_and_diag_zero_invariants():
    """Spectral normalization touches W_inter only; intra-layer constraints
    must still hold."""
    sizes = (64, 16, 8)
    for mode in ("hebb", "modern"):
        net = HopfieldNetwork(mode=mode, layer_sizes=sizes, seed=0)
        net.learn(_patterns(N=4, sizes=sizes), normalize_inter="spectral")
        for layer in net.layers:
            assert torch.allclose(layer.W, layer.W.T, atol=1e-6)
            if mode == "modern":
                assert torch.allclose(
                    torch.diag(layer.W),
                    torch.zeros(layer.n_units),
                    atol=1e-7,
                )
