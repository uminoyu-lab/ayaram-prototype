"""Tests for ayaram.learning."""

from __future__ import annotations

import pytest
import torch

from ayaram import learning


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def test_modern_update_shape():
    d, N = 32, 4
    X = _rand(d, N)
    xi = _rand(d)
    out = learning.modern_hopfield_update(xi, X, beta=1.0)
    assert out.shape == (d,)


def test_modern_update_recovers_stored_pattern_at_high_beta():
    d, N = 32, 4
    g = torch.Generator().manual_seed(7)
    X = torch.randn(d, N, generator=g)
    # Use one stored pattern itself as the query: the softmax becomes peaked
    # on that pattern at high beta, so the update returns ~that same pattern.
    pat = X[:, 1]
    out = learning.modern_hopfield_update(pat, X, beta=50.0)
    diff = (out - pat).abs().max().item()
    assert diff < 1e-2, f"max diff {diff}"


def test_modern_update_rejects_wrong_shape():
    with pytest.raises(ValueError):
        learning.modern_hopfield_update(torch.zeros(8), torch.zeros(8), beta=1.0)


def test_hebb_weights_symmetric():
    patterns = _rand(16, 32)
    W = learning.hebb_weights(patterns)
    assert torch.allclose(W, W.T, atol=1e-6)


def test_hebb_weights_diagonal_free():
    patterns = _rand(16, 32)
    W = learning.hebb_weights(patterns, zero_diagonal=False)
    # at least one diagonal element should be non-zero for random patterns
    assert torch.diag(W).abs().sum() > 0


def test_hebb_weights_zero_diagonal_option():
    patterns = _rand(16, 32)
    W = learning.hebb_weights(patterns, zero_diagonal=True)
    assert torch.allclose(torch.diag(W), torch.zeros(32), atol=1e-7)


def test_hebb_update_shape():
    W = _rand(16, 16)
    W = learning.symmetrize(W)
    xi = _rand(16)
    out = learning.hebb_update(xi, W, beta=1.0)
    assert out.shape == (16,)
    # tanh range
    assert out.abs().max().item() <= 1.0 + 1e-6


def test_symmetrize_idempotent():
    W = _rand(8, 8)
    Ws = learning.symmetrize(W)
    assert torch.allclose(Ws, Ws.T, atol=1e-7)
    assert torch.allclose(learning.symmetrize(Ws), Ws, atol=1e-7)


def test_zero_diag_zeros_only_the_diagonal():
    W = _rand(8, 8)
    Z = learning.zero_diag(W)
    assert torch.allclose(torch.diag(Z), torch.zeros(8), atol=1e-7)
    # off-diagonal preserved
    mask = ~torch.eye(8, dtype=torch.bool)
    assert torch.allclose(Z[mask], W[mask], atol=1e-7)
