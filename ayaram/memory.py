"""3-layer Hopfield network.

Design decision #2 (Aya + Yu, 2026-06-17): Modern Hopfield as the primary
learning rule, classical Hebb as a parallel comparison. Two parallel
``HopfieldNetwork`` instances (one per ``mode``) are constructed by the demos.

Design decision #3 (Aya + Yu, 2026-06-17): cell shape 32x32, layer widths
1024 / 256 / 64. Eight kanji are the starting test set.

Design decision #6 (Aya + Yu, 2026-06-17): v0.1 enforces only the symmetry
constraint W = W.T at every step. Modern Hopfield additionally has zero
diagonal (W_ii = 0); classical Hebb keeps the diagonal free.

Sub-decision (Aru, 2026-06-17): each layer is its own Hopfield net with its
own intra-layer weights, and adjacent layers are connected by a forward /
backward weight pair (i.e. the "A + intra-layer = C" arrangement).
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from . import learning


LAYER_SIZES: tuple[int, int, int] = (1024, 256, 64)


class HopfieldLayer(nn.Module):
    """A single Hopfield layer.

    Holds an intra-layer weight matrix ``W`` (always, so tests can inspect
    symmetry and diagonal regardless of mode) plus, for ``mode='modern'``, the
    stored pattern matrix ``X``.
    """

    def __init__(self, n_units: int, mode: str = "modern") -> None:
        super().__init__()
        if mode not in ("modern", "hebb"):
            raise ValueError("mode must be 'modern' or 'hebb'")
        self.mode = mode
        self.n_units = n_units
        self.register_buffer("W", torch.zeros(n_units, n_units))
        # X is set by store() for mode='modern'; kept as None until then.
        # Stored as a buffer (rather than nn.Parameter) since we never gradient-
        # train it here -- learning is one-shot via store().
        self.register_buffer("X", torch.zeros(n_units, 0))
        self._has_patterns = False

    def store(self, patterns: Tensor) -> None:
        """Install stored patterns.

        Args:
            patterns: shape ``(N, n_units)`` -- rows are patterns.
        """
        if patterns.dim() != 2 or patterns.shape[1] != self.n_units:
            raise ValueError(
                f"patterns must be (N, {self.n_units}); got {tuple(patterns.shape)}"
            )
        patterns = patterns.to(self.W.dtype).to(self.W.device)
        N = patterns.shape[0]
        if self.mode == "modern":
            # Conceptual W for the Modern side: (1/N) X X^T, symmetric, zero
            # diagonal. The actual update uses X directly, but we keep W around
            # for inspection / decision-#6 tests.
            W = patterns.T @ patterns / N
            W = learning.symmetrize(W)
            W = learning.zero_diag(W)
            self.W = W
            self.X = patterns.T.contiguous()  # (d, N)
        else:  # hebb
            W = learning.hebb_weights(patterns, zero_diagonal=False)
            self.W = W
        self._has_patterns = True

    def has_patterns(self) -> bool:
        return self._has_patterns

    def enforce_constraints(self) -> None:
        """Apply decision #6 every step: ``W = (W + W^T) / 2`` for both modes,
        plus ``W_ii = 0`` for the Modern side."""
        self.W = learning.symmetrize(self.W)
        if self.mode == "modern":
            self.W = learning.zero_diag(self.W)

    def step(self, xi: Tensor, beta: float = 1.0) -> Tensor:
        """Single intra-layer update of the state ``xi``."""
        if self.mode == "modern":
            if not self._has_patterns:
                raise RuntimeError(
                    "modern HopfieldLayer requires store() before step()"
                )
            return learning.modern_hopfield_update(xi, self.X, beta=beta)
        return learning.hebb_update(xi, self.W, beta=beta)


class HopfieldNetwork(nn.Module):
    """3-layer Hopfield network.

    Each layer holds its own intra-layer weights (decision #6 + sub-decision).
    Adjacent layers are coupled by rectangular weight matrices ``W_inter[l]``
    of shape ``(layer_sizes[l], layer_sizes[l+1])``. Forward propagation uses
    ``W_inter[l]``; backward propagation uses its transpose. The inter-layer
    matrices have no symmetry constraint of their own (they are rectangular),
    but ``ayaram.core`` keeps them stable -- they are random-initialized at
    construction and not learned in M1.
    """

    def __init__(
        self,
        mode: str = "modern",
        layer_sizes: Iterable[int] = LAYER_SIZES,
        inter_scale: float = 0.01,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.layer_sizes = tuple(layer_sizes)
        if len(self.layer_sizes) < 2:
            raise ValueError("need at least 2 layers")
        self.layers = nn.ModuleList(
            [HopfieldLayer(s, mode=mode) for s in self.layer_sizes]
        )
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        inter: list[Tensor] = []
        for a, b in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            inter.append(torch.randn(a, b, generator=gen) * inter_scale)
        # Stored as buffers so .to(device) propagates correctly.
        for i, w in enumerate(inter):
            self.register_buffer(f"W_inter_{i}", w)

    @property
    def W_inter(self) -> list[Tensor]:
        return [getattr(self, f"W_inter_{i}") for i in range(len(self.layer_sizes) - 1)]

    def store_layer0(self, patterns: Tensor) -> None:
        """Store patterns into layer 0 only (the input layer that the kanji
        demo uses for recall)."""
        self.layers[0].store(patterns)

    def enforce_constraints(self) -> None:
        """Apply decision #6 to every intra-layer weight matrix."""
        for layer in self.layers:
            layer.enforce_constraints()

    def initial_state(self, device: torch.device | None = None) -> list[Tensor]:
        """Zero-initialized state for every layer."""
        device = device or self.layers[0].W.device
        return [torch.zeros(s, device=device) for s in self.layer_sizes]
