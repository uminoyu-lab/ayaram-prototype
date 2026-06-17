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
        if len(self.layer_sizes) < 1:
            raise ValueError("need at least 1 layer")
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

    def learn(
        self,
        layer_patterns: list[Tensor],
        *,
        normalize_inter: str = "spectral",
        center_inter_inputs: bool = False,
    ) -> None:
        """Hierarchical Hebb learning over all layers (Aru M3, 2026-06-17;
        ``center_inter_inputs`` added Aru M5, 2026-06-17).

        Layer-internal weights are installed via the existing
        ``HopfieldLayer.store`` for each layer (which already encodes
        decision #6 -- symmetry plus Modern-side ``diag = 0``).

        Inter-layer weights are installed by the Hebb outer-product rule:

            W_inter[l] = (1/N) sum_p p_l ⊗ p_{l+1}

        with ``p_l`` the l-th layer slice of the p-th joint pattern. The
        backward path uses ``W_inter[l].T`` (consistent with the existing
        ``_inter_layer_signal`` in ``core``).

        Args:
            layer_patterns: list of length ``len(layer_sizes)``; each entry is
                            ``(N, layer_size[l])``. Same ``N`` across layers.
            normalize_inter:
                ``'spectral'`` (default, Aru M4 sub-decision):
                    rescale each ``W_inter[l]`` to spectral norm 1 after the
                    outer-product accumulation. This was a CC demo-side
                    workaround in M3 because the raw rule leaves the layer-0
                    side dominating by ~30x; M4 formalizes it inside
                    ``learn`` so all clients pick it up consistently.
                ``'none'``:
                    leave ``W_inter[l]`` at the raw scale set by the Hebb
                    outer product. Used for reproducing the pre-fix M3
                    behavior in tests and for v0.2 experiments that want
                    physically-scaled couplings.
            center_inter_inputs:
                ``False`` (default, M3 / M4 backward-compatible):
                    use the raw ``layer_patterns`` for the outer products.
                ``True`` (M5 promotion of the M4 demo helper):
                    subtract the per-dim mean across patterns before each
                    outer product:

                        W_inter[l] = (1/N) sum_p (p_l - mean_p p_l) ⊗ p_{l+1}

                    M4 showed this rescues layer-2 origin recall from
                    4/8 to 10/12 in the 12-kanji hierarchical demo by
                    cancelling the bipolar bitmap background that otherwise
                    biases every layer-0 query toward the same layer-1
                    direction. Use this for Modern-mode hierarchical demos.

                    NOTE (Hebb caveat): combining ``center_inter_inputs=True``
                    with ``mode='hebb'`` makes the ``tanh(beta * W xi)``
                    fixed point sign-flip on some queries (M4 observed
                    layer-1 cos go negative). The option is provided
                    uniformly because Modern is the headline mode, but Hebb
                    callers should set this False until v0.2 resolves the
                    interaction (README v0.2 homework #9).

                    Intra-layer ``HopfieldLayer.store`` is *not* affected --
                    layer-0 self-recall sees the raw patterns regardless of
                    this flag.
        """
        if normalize_inter not in ("spectral", "none"):
            raise ValueError(
                f"normalize_inter must be 'spectral' or 'none'; "
                f"got {normalize_inter!r}"
            )
        if len(layer_patterns) != len(self.layer_sizes):
            raise ValueError(
                f"expected {len(self.layer_sizes)} layer-pattern tensors; "
                f"got {len(layer_patterns)}"
            )
        N = layer_patterns[0].shape[0]
        for l, ps in enumerate(layer_patterns):
            if ps.dim() != 2:
                raise ValueError(f"layer {l}: patterns must be 2-D")
            if ps.shape[0] != N:
                raise ValueError(
                    f"layer {l}: got N={ps.shape[0]}, expected N={N}"
                )
            if ps.shape[1] != self.layer_sizes[l]:
                raise ValueError(
                    f"layer {l}: width {ps.shape[1]} != layer_size "
                    f"{self.layer_sizes[l]}"
                )
            self.layers[l].store(ps)

        if center_inter_inputs:
            inter_inputs = [p - p.mean(dim=0, keepdim=True) for p in layer_patterns]
        else:
            inter_inputs = list(layer_patterns)

        for l in range(len(self.layer_sizes) - 1):
            buf_name = f"W_inter_{l}"
            ref = getattr(self, buf_name)
            p_l = inter_inputs[l].to(ref.dtype).to(ref.device)
            p_lp1 = layer_patterns[l + 1].to(ref.dtype).to(ref.device)
            # (d_l, d_{l+1})
            W_new = (p_l.T @ p_lp1) / N
            if normalize_inter == "spectral":
                s = torch.linalg.svdvals(W_new)
                sn = float(s[0].item()) if s.numel() > 0 else 0.0
                if sn > 1e-9:
                    W_new = W_new * (1.0 / sn)
            setattr(self, buf_name, W_new)
        self.enforce_constraints()

    def recall(
        self,
        query: Tensor,
        *,
        config=None,
        return_all_layers: bool = True,
        generator: torch.Generator | None = None,
    ):
        """Run a single 4-phase cycle with ``query`` injected at layer 0.

        Returns a dict ``{layer_idx: state}`` if ``return_all_layers``,
        otherwise the layer-0 readout.
        """
        return self.recall_from_layer(
            query,
            layer_idx=0,
            config=config,
            return_all_layers=return_all_layers,
            generator=generator,
        )

    def recall_from_layer(
        self,
        query: Tensor,
        layer_idx: int,
        *,
        config=None,
        return_all_layers: bool = True,
        generator: torch.Generator | None = None,
    ):
        """Run a single 4-phase cycle with ``query`` injected at ``layer_idx``.

        Used by the M3 reverse demo (radical -> kanji): set ``layer_idx=1``
        and pass a radical pattern.
        """
        # Deferred import to avoid the memory <-> core cycle.
        from . import core as _core

        cfg = config if config is not None else _core.CycleConfig()
        state = _core.CycleState.from_network(self, device=query.device)
        _core.phase1_terrain(self, state, query, layer_idx=layer_idx)
        _core.phase2_fluctuation(self, state, cfg, generator=generator)
        _core.phase3_fixation(self, state, cfg)
        if return_all_layers:
            return {i: state.xi[i].clone() for i in range(len(self.layer_sizes))}
        return state.xi[0].clone()

    def enforce_constraints(self) -> None:
        """Apply decision #6 to every intra-layer weight matrix."""
        for layer in self.layers:
            layer.enforce_constraints()

    def initial_state(self, device: torch.device | None = None) -> list[Tensor]:
        """Zero-initialized state for every layer."""
        device = device or self.layers[0].W.device
        return [torch.zeros(s, device=device) for s in self.layer_sizes]
