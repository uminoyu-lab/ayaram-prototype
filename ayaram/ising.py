"""Ising / MAX-CUT problem objects for Hopfield-net annealing.

Decision (Aru M2, 2026-06-17): only MAX-CUT in v0.1, and only via the classical
Hebb side (decision #2 secondary). Modern Hopfield's pattern-attractor
machinery is not the natural language for combinatorial optimization; broader
Lucas-2014 mappings are v0.2.

Conventions
-----------
A weighted Ising Hamiltonian is

    H(s) = - (1/2) s^T J s                          (s in {-1, +1}^N)
         = - sum_{i<j} J_ij s_i s_j

with J symmetric and ``J_ii = 0``. The associated Hopfield energy
``E_H(s) = - (1/2) s^T W s`` matches ``H`` when ``W = J``, so the Hopfield
dynamics (which decrease ``E_H``) minimize ``H``.

For MAX-CUT on an undirected graph with adjacency matrix ``A_ij in {0, 1}``,

    cut(s) = sum_{(i,j) in E} (1 - s_i s_j) / 2
           = |E|/2 - (1/2) sum_{i<j} A_ij s_i s_j

so maximizing ``cut(s)`` is equivalent to minimizing
``sum_{i<j} A_ij s_i s_j``. Matching the Ising form gives ``J = -A``
(antiferromagnetic couplings on edges).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _symmetrize_zero_diag(M: Tensor) -> Tensor:
    M = 0.5 * (M + M.T)
    M = M - torch.diag(torch.diag(M))
    return M


@dataclass
class IsingProblem:
    """General Ising problem with symmetric, diagonal-free coupling ``J``."""

    J: Tensor

    @property
    def N(self) -> int:
        return int(self.J.shape[0])

    @classmethod
    def from_couplings(cls, J: Tensor) -> "IsingProblem":
        J = _symmetrize_zero_diag(J.float())
        return cls(J=J)

    def hamiltonian(self, s: Tensor) -> Tensor:
        """``H(s) = -(1/2) s^T J s``. ``s`` may be batched ``(..., N)``."""
        s = s.float()
        return -0.5 * torch.einsum("...i,ij,...j->...", s, self.J, s)

    def to_hopfield_weights(self) -> Tensor:
        """Return ``W = J`` to drop into a Hebb-mode ``HopfieldLayer``."""
        return self.J


@dataclass
class MaxCutProblem(IsingProblem):
    """MAX-CUT instance built from an undirected, unweighted adjacency matrix.

    The internal ``J`` is the antiferromagnetic coupling ``-A`` (symmetric,
    zero-diagonal). ``adj`` is also retained because ``cut_value`` and the
    edge count are most naturally read off the adjacency.
    """

    adj: Tensor

    @classmethod
    def from_graph(cls, adj: Tensor) -> "MaxCutProblem":
        adj = _symmetrize_zero_diag(adj.float())
        J = -adj
        return cls(J=J, adj=adj)

    @property
    def n_edges(self) -> int:
        return int(self.adj.triu(diagonal=1).sum().item())

    def cut_value(self, s: Tensor) -> Tensor:
        """Number of edges cut by the split ``s in {-1, +1}^N``.

        Batched: ``s`` may have shape ``(..., N)``; the return matches the
        leading dims.
        """
        s = s.float()
        # sum_{i<j} A_ij (1 - s_i s_j) / 2
        ss = torch.einsum("...i,...j->...ij", s, s)  # (..., N, N)
        upper = self.adj.triu(diagonal=1)
        return (upper * (1.0 - ss)).sum(dim=(-1, -2)) / 2.0

    def optimal_brute_force(self, max_n: int = 20) -> tuple[float, Tensor]:
        """Brute-force optimum by enumerating all ``2^N`` assignments.

        Refuses to run for ``N > max_n`` (sub-decision: M2 brute force only up
        to ``N = 20``). Use a sampling / annealing heuristic for larger
        problems; that is v0.2.
        """
        if self.N > max_n:
            raise NotImplementedError(
                f"brute force disabled for N={self.N} > max_n={max_n}; "
                f"v0.2 will provide a sampling baseline"
            )
        device = self.adj.device
        N = self.N
        # Build all 2^N sign vectors as (2^N, N).
        idx = torch.arange(1 << N, device=device)
        bits = ((idx.unsqueeze(1) >> torch.arange(N, device=device)) & 1).float()
        s_all = 2.0 * bits - 1.0  # (2^N, N), values in {-1, +1}
        cuts = self.cut_value(s_all)
        best_idx = int(cuts.argmax().item())
        return float(cuts[best_idx].item()), s_all[best_idx]


def random_erdos_renyi(n: int, p: float, seed: int) -> Tensor:
    """Symmetric 0/1 adjacency for an Erdos-Renyi graph G(n, p)."""
    g = torch.Generator().manual_seed(seed)
    upper = (torch.rand(n, n, generator=g) < p).float().triu(diagonal=1)
    return upper + upper.T
