"""Tests for ayaram.memory: 3-layer Hopfield network, W=W^T, Modern diag=0."""

from __future__ import annotations

import pytest
import torch

from ayaram.memory import LAYER_SIZES, HopfieldLayer, HopfieldNetwork


def _rand_patterns(N: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, d, generator=g)


def test_layer_W_shape():
    layer = HopfieldLayer(64, mode="modern")
    assert layer.W.shape == (64, 64)


def test_modern_layer_W_symmetric_after_store():
    layer = HopfieldLayer(64, mode="modern")
    layer.store(_rand_patterns(8, 64))
    assert torch.allclose(layer.W, layer.W.T, atol=1e-6)


def test_modern_layer_W_zero_diagonal_after_store():
    layer = HopfieldLayer(64, mode="modern")
    layer.store(_rand_patterns(8, 64))
    assert torch.allclose(torch.diag(layer.W), torch.zeros(64), atol=1e-7)


def test_hebb_layer_W_symmetric_after_store():
    layer = HopfieldLayer(64, mode="hebb")
    layer.store(_rand_patterns(8, 64))
    assert torch.allclose(layer.W, layer.W.T, atol=1e-6)


def test_hebb_layer_diagonal_free():
    layer = HopfieldLayer(64, mode="hebb")
    layer.store(_rand_patterns(8, 64))
    # decision #6 sub-decision: Hebb keeps diagonal free
    assert torch.diag(layer.W).abs().sum() > 0


def test_modern_layer_step_shape():
    layer = HopfieldLayer(64, mode="modern")
    layer.store(_rand_patterns(4, 64))
    xi = torch.randn(64)
    out = layer.step(xi, beta=1.0)
    assert out.shape == (64,)


def test_modern_layer_step_without_store_raises():
    layer = HopfieldLayer(64, mode="modern")
    with pytest.raises(RuntimeError):
        layer.step(torch.zeros(64), beta=1.0)


def test_hebb_layer_step_shape():
    layer = HopfieldLayer(64, mode="hebb")
    layer.store(_rand_patterns(4, 64))
    out = layer.step(torch.randn(64), beta=1.0)
    assert out.shape == (64,)


def test_layer_enforce_constraints_resymmetrizes():
    layer = HopfieldLayer(32, mode="modern")
    layer.store(_rand_patterns(4, 32))
    # break the symmetry deliberately
    layer.W = layer.W + 0.5 * torch.randn(32, 32)
    layer.enforce_constraints()
    assert torch.allclose(layer.W, layer.W.T, atol=1e-6)
    assert torch.allclose(torch.diag(layer.W), torch.zeros(32), atol=1e-7)


def test_network_layer_sizes_default():
    net = HopfieldNetwork(mode="modern", seed=0)
    assert net.layer_sizes == LAYER_SIZES
    assert tuple(layer.n_units for layer in net.layers) == LAYER_SIZES


def test_network_inter_layer_shapes():
    net = HopfieldNetwork(mode="modern", seed=0)
    w0, w1 = net.W_inter
    assert w0.shape == (LAYER_SIZES[0], LAYER_SIZES[1])
    assert w1.shape == (LAYER_SIZES[1], LAYER_SIZES[2])


def test_network_initial_state_shapes():
    net = HopfieldNetwork(mode="hebb", seed=0)
    xi = net.initial_state()
    assert [t.shape[0] for t in xi] == list(LAYER_SIZES)


def test_network_store_layer0_then_step():
    net = HopfieldNetwork(mode="modern", seed=0)
    patterns = _rand_patterns(4, LAYER_SIZES[0])
    net.store_layer0(patterns)
    assert net.layers[0].has_patterns()
    out = net.layers[0].step(torch.randn(LAYER_SIZES[0]), beta=1.0)
    assert out.shape == (LAYER_SIZES[0],)


def test_network_enforce_constraints_modern_diag_zero():
    net = HopfieldNetwork(mode="modern", seed=0)
    net.store_layer0(_rand_patterns(4, LAYER_SIZES[0]))
    # break symmetry on layer 0
    net.layers[0].W = net.layers[0].W + 0.5 * torch.randn(LAYER_SIZES[0], LAYER_SIZES[0])
    net.enforce_constraints()
    for layer in net.layers:
        assert torch.allclose(layer.W, layer.W.T, atol=1e-6)
        if layer.mode == "modern":
            assert torch.allclose(
                torch.diag(layer.W), torch.zeros(layer.n_units), atol=1e-7
            )
