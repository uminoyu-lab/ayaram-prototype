"""Learning rules: Modern Hopfield continuous update + classical Hebb rule.

Design decision #2 (Aya + Yu, 2026-06-17):
  - Primary (B): Modern Hopfield continuous update (Ramsauer 2020 Theorem 1).
  - Secondary (C): classical Hebb rule, kept in parallel for comparison.

Both rules expose the same shape ``(d,) -> (d,)`` update signature so the
demos can swap them. ``ayaram.memory.HopfieldLayer`` wraps each rule with
shared bookkeeping (pattern storage, weight matrix, symmetry / diagonal
constraints from decision #6).

Conventions used throughout this module:
    d       -- per-cell dimensionality of the layer (e.g. 1024 for layer 0)
    N       -- number of stored patterns
    X       -- pattern matrix of shape (d, N); each column is one pattern
    patterns -- the same data packed as (N, d); each row is one pattern
    xi      -- state vector of shape (d,) or batched (..., d)
"""

from __future__ import annotations

import torch
from torch import Tensor


def modern_hopfield_update(xi: Tensor, X: Tensor, beta: float = 1.0) -> Tensor:
    """One step of the Modern Hopfield continuous update.

    Implements Ramsauer 2020 Theorem 1:

        xi_new = X @ softmax(beta * X.T @ xi, dim=-1)

    where ``X`` is the pattern matrix (columns = patterns) and ``beta`` is the
    inverse temperature.

    Args:
        xi:   state, shape ``(..., d)``.
        X:    pattern matrix, shape ``(d, N)``.
        beta: inverse temperature (softmax sharpness).

    Returns:
        Updated state, same shape as ``xi``.
    """
    if X.dim() != 2:
        raise ValueError(f"X must be 2-D (d, N); got shape {tuple(X.shape)}")
    z = beta * (xi @ X)
    s = torch.softmax(z, dim=-1)
    return s @ X.T


def hebb_weights(patterns: Tensor, zero_diagonal: bool = False) -> Tensor:
    """Classical Hebb weight matrix.

        W = (1/N) * sum_p p (x) p

    Args:
        patterns:      ``(N, d)`` row-stacked patterns.
        zero_diagonal: by sub-decision under decision #6, the classical Hebb
                       side keeps the diagonal free (default ``False``). The
                       Modern Hopfield side imposes ``W_ii = 0``; that is done
                       through ``HopfieldLayer``, not here.

    Returns:
        Symmetric ``(d, d)`` weight matrix.
    """
    if patterns.dim() != 2:
        raise ValueError(
            f"patterns must be 2-D (N, d); got shape {tuple(patterns.shape)}"
        )
    N = patterns.shape[0]
    if N == 0:
        raise ValueError("Cannot build Hebb weights from zero patterns")
    W = patterns.T @ patterns / N
    W = 0.5 * (W + W.T)
    if zero_diagonal:
        W = W - torch.diag(torch.diag(W))
    return W


def hebb_update(xi: Tensor, W: Tensor, beta: float = 1.0) -> Tensor:
    """One step of the classical Hopfield-Hebb update.

    Uses ``tanh(beta * W @ xi)`` -- the continuous-valued generalization of the
    bipolar sign-rule recall that reduces to ``sign(W @ xi)`` as ``beta -> inf``.
    For the kanji bitmaps (values in ``[-1, +1]``) this gives a graceful
    saturating dynamics rather than abrupt clipping.

    Args:
        xi:   state, shape ``(..., d)``.
        W:    weight matrix, shape ``(d, d)``.
        beta: gain on the pre-activation; analogous to inverse temperature.

    Returns:
        Updated state, same shape as ``xi``.
    """
    if W.dim() != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square; got shape {tuple(W.shape)}")
    return torch.tanh(beta * (xi @ W))


def symmetrize(W: Tensor) -> Tensor:
    """Project ``W`` onto the symmetric subspace: ``W <- (W + W.T) / 2``.

    Decision #6 requires this at every step.
    """
    return 0.5 * (W + W.T)


def zero_diag(W: Tensor) -> Tensor:
    """Strip the diagonal: ``W_ii <- 0``. Applied to the Modern Hopfield side
    on every step (decision #6 sub-decision)."""
    return W - torch.diag(torch.diag(W))
