"""Tests for ayaram.ising."""

from __future__ import annotations

import math

import pytest
import torch

from ayaram.ising import (
    IsingProblem,
    MaxCutProblem,
    random_erdos_renyi,
)


def _triangle_plus_one_adj() -> torch.Tensor:
    """4 nodes: 0-1-2 form a triangle, 3 hangs off node 2.

    Edges: (0,1), (0,2), (1,2), (2,3). |E| = 4.
    """
    A = torch.zeros(4, 4)
    edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A


def test_from_graph_symmetry_and_zero_diagonal():
    A = _triangle_plus_one_adj()
    p = MaxCutProblem.from_graph(A)
    assert torch.allclose(p.J, p.J.T, atol=1e-7)
    assert torch.allclose(torch.diag(p.J), torch.zeros(4), atol=1e-7)
    assert torch.allclose(p.adj, p.adj.T, atol=1e-7)
    assert torch.allclose(torch.diag(p.adj), torch.zeros(4), atol=1e-7)
    # MAX-CUT convention: J = -A
    assert torch.allclose(p.J, -p.adj, atol=1e-7)


def test_n_edges_triangle_plus_one():
    p = MaxCutProblem.from_graph(_triangle_plus_one_adj())
    assert p.n_edges == 4


def test_cut_value_known_split():
    """Hand-checked: split S = {0, 3}, T = {1, 2} on the triangle+one graph.

    Edges (0,1), (0,2), (2,3) cross the split; (1,2) does not.
    So cut = 3.
    """
    p = MaxCutProblem.from_graph(_triangle_plus_one_adj())
    s = torch.tensor([+1.0, -1.0, -1.0, +1.0])
    assert math.isclose(float(p.cut_value(s).item()), 3.0, abs_tol=1e-7)


def test_cut_value_all_same_gives_zero():
    p = MaxCutProblem.from_graph(_triangle_plus_one_adj())
    s = torch.tensor([+1.0, +1.0, +1.0, +1.0])
    assert math.isclose(float(p.cut_value(s).item()), 0.0, abs_tol=1e-7)


def test_hamiltonian_lower_for_larger_cut():
    """H decreases monotonically as cut grows -- the whole point of the
    MAX-CUT <-> Ising correspondence."""
    p = MaxCutProblem.from_graph(_triangle_plus_one_adj())
    # Edges: (0,1), (0,2), (1,2), (2,3).
    a = torch.tensor([+1.0, +1.0, +1.0, +1.0])  # cut = 0  (no edge crosses)
    b = torch.tensor([+1.0, +1.0, -1.0, -1.0])  # cut = 2  ((0,2) & (1,2))
    c = torch.tensor([+1.0, -1.0, -1.0, +1.0])  # cut = 3  ((0,1), (0,2), (2,3))
    Ha, Hb, Hc = (float(p.hamiltonian(x).item()) for x in (a, b, c))
    cuta, cutb, cutc = (float(p.cut_value(x).item()) for x in (a, b, c))
    assert cuta < cutb < cutc
    assert Ha > Hb > Hc


def test_brute_force_small():
    """N = 4 triangle+one: best cut by inspection is 3 (split nodes 0,3 vs 1,2,
    or any equivalent: edges (0,1), (0,2), (2,3) crossed, (1,2) inside)."""
    p = MaxCutProblem.from_graph(_triangle_plus_one_adj())
    best_cut, best_s = p.optimal_brute_force()
    assert math.isclose(best_cut, 3.0, abs_tol=1e-7)
    # the returned s must actually achieve that cut
    assert math.isclose(float(p.cut_value(best_s).item()), best_cut, abs_tol=1e-7)


def test_brute_force_refuses_large_N():
    g = random_erdos_renyi(21, 0.5, seed=0)
    p = MaxCutProblem.from_graph(g)
    with pytest.raises(NotImplementedError):
        p.optimal_brute_force(max_n=20)


def test_brute_force_batched_cut_value():
    """``cut_value`` must accept a batch dimension (used by the brute-force
    enumeration)."""
    p = MaxCutProblem.from_graph(_triangle_plus_one_adj())
    batch = torch.stack(
        [
            torch.tensor([+1.0, +1.0, +1.0, +1.0]),
            torch.tensor([+1.0, -1.0, -1.0, +1.0]),
        ]
    )
    cuts = p.cut_value(batch)
    assert cuts.shape == (2,)
    assert math.isclose(float(cuts[0].item()), 0.0, abs_tol=1e-7)
    assert math.isclose(float(cuts[1].item()), 3.0, abs_tol=1e-7)


def test_random_erdos_renyi_shape_and_symmetry():
    A = random_erdos_renyi(12, 0.5, seed=42)
    assert A.shape == (12, 12)
    assert torch.allclose(A, A.T)
    assert torch.allclose(torch.diag(A), torch.zeros(12))
    # entries are 0 or 1
    assert set(torch.unique(A).tolist()) <= {0.0, 1.0}


def test_random_erdos_renyi_seed_reproducible():
    A1 = random_erdos_renyi(8, 0.5, seed=7)
    A2 = random_erdos_renyi(8, 0.5, seed=7)
    A3 = random_erdos_renyi(8, 0.5, seed=8)
    assert torch.equal(A1, A2)
    assert not torch.equal(A1, A3)


def test_to_hopfield_weights_equals_J():
    p = MaxCutProblem.from_graph(_triangle_plus_one_adj())
    W = p.to_hopfield_weights()
    assert torch.equal(W, p.J)


def test_general_ising_from_couplings_symmetrizes():
    raw = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [-1.0, 3.0, 4.0],
            [0.0, 5.0, 0.0],
        ]
    )
    p = IsingProblem.from_couplings(raw)
    assert torch.allclose(p.J, p.J.T, atol=1e-7)
    assert torch.allclose(torch.diag(p.J), torch.zeros(3), atol=1e-7)
