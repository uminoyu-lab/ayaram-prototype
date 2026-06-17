"""Core: 4-phase cycle on the 3-layer MTJ array.

Design decision #1 (Aya + Yu, 2026-06-17): option C, whole-cell synchronous
switching. Per-layer barrier values come from ``ayaram.modes``.

The cycle is:
    Phase 1 (awake, terrain setup) -- map the problem onto layer-0 bias.
    Phase 2 (sleep, fluctuation)   -- lower barrier, apply Langevin / Gaussian
                                      noise so magnetization wanders the
                                      terrain stochastically.
    Phase 3 (re-awake, fixation)   -- raise barrier, freeze the state.
    Phase 4 (readout)              -- read the output (layer 0) state.

Sub-decision (Aru, 2026-06-17): default Phase 2 step count is 1000.

All updates are whole-cell synchronous (every cell, every layer, every step).
This is what makes the cycle a *cycle* rather than a layered feed-forward
pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import Tensor

from . import modes
from .memory import HopfieldNetwork


@dataclass
class CycleConfig:
    """Knobs for the 4-phase cycle."""

    beta: float = 1.0
    """Inverse temperature used by the intra-layer update (Modern softmax or
    Hebb tanh gain)."""

    sigma_global: float = 1.0
    """Scale applied on top of ``modes.sigma_local`` during Phase 2. Together
    with ``beta`` this is the knob the attention-test demo sweeps."""

    inter_layer_scale: float = 0.1
    """How much inter-layer signals contribute relative to the intra-layer
    update. Kept small so layer-0 recall is not derailed by the still-random
    inter-layer weights."""

    phase1_steps: int = 0
    """Phase 1 holds layer-0 state at the input; no updates needed."""

    phase2_steps: int = 1000
    """Sub-decision: default 1000 Langevin steps in Phase 2."""

    phase3_steps: int = 100
    """Deterministic fixation steps in Phase 3."""

    phase3_beta_boost: float = 4.0
    """Phase 3 uses ``beta * phase3_beta_boost`` to sharpen the recall."""

    dt: float = 0.1
    """Langevin step size: xi <- (1-dt) xi + dt step(xi) + sigma sqrt(2 dt) eta."""


@dataclass
class CycleState:
    """Mutable per-layer state vector list."""

    xi: list[Tensor] = field(default_factory=list)

    @classmethod
    def from_network(cls, net: HopfieldNetwork, device: torch.device | None = None) -> "CycleState":
        return cls(xi=net.initial_state(device=device))


def _inter_layer_signal(
    net: HopfieldNetwork, xi: list[Tensor], layer_idx: int
) -> Tensor:
    """Total inter-layer input arriving at ``layer_idx``.

    Forward from layer ``l-1`` uses ``W_inter[l-1]``;
    backward from layer ``l+1`` uses ``W_inter[l].T``.
    """
    parts: list[Tensor] = []
    if layer_idx > 0:
        parts.append(xi[layer_idx - 1] @ net.W_inter[layer_idx - 1])
    if layer_idx + 1 < len(net.layer_sizes):
        parts.append(xi[layer_idx + 1] @ net.W_inter[layer_idx].T)
    if not parts:
        return torch.zeros_like(xi[layer_idx])
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


def phase1_terrain(
    net: HopfieldNetwork,
    state: CycleState,
    input_bias: Tensor,
    layer_idx: int = 0,
) -> None:
    """Phase 1 (awake, terrain setup): inject ``input_bias`` at ``layer_idx``.

    The state vector at ``layer_idx`` is *set* (not updated) so subsequent
    phases see the problem terrain. Other layers are left at whatever they
    came in as.

    ``layer_idx`` defaults to 0 -- the input layer for the kanji and MAX-CUT
    demos. The M3 reverse-recall demo (radical -> kanji) uses ``layer_idx=1``
    to seed at the radical layer instead.
    """
    if layer_idx < 0 or layer_idx >= len(state.xi):
        raise IndexError(
            f"layer_idx {layer_idx} out of range for {len(state.xi)} layers"
        )
    if input_bias.shape != state.xi[layer_idx].shape:
        raise ValueError(
            f"input_bias shape {tuple(input_bias.shape)} != layer-{layer_idx} "
            f"shape {tuple(state.xi[layer_idx].shape)}"
        )
    state.xi[layer_idx] = input_bias.clone()


def phase1_learn(
    net: HopfieldNetwork,
    layer_patterns: list[Tensor],
) -> None:
    """Phase 1 (awake, learning variant): install hierarchical Hebb weights.

    Decision (Aru M3, 2026-06-17): learning lives in Phase 1 so the 4-phase
    structure is preserved (decision #1). This thin wrapper delegates to
    ``HopfieldNetwork.learn`` and exists for narrative symmetry with
    ``phase1_terrain``.

    Args:
        net:             the network whose weights will be (re)installed.
        layer_patterns:  one tensor per layer, each shape ``(N, layer_size)``.
    """
    net.learn(layer_patterns)


def phase2_fluctuation(
    net: HopfieldNetwork,
    state: CycleState,
    config: CycleConfig,
    mode: modes.Mode = modes.SLEEP,
    generator: torch.Generator | None = None,
) -> None:
    """Phase 2 (sleep): Langevin-style update with whole-cell synchronous noise.

    ``mode`` is currently a label kept for record-keeping; the v0.1 noise scale
    comes from ``config.sigma_global`` (layer-0 std) modulated by the layer
    barrier ratio via ``modes.layer_noise_ratio(l)``. See ``modes.sigma_local``
    docstring for the rationale.
    """
    net.enforce_constraints()
    dt = config.dt
    keep = 1.0 - dt
    sqrt_2dt = (2.0 * dt) ** 0.5
    for _ in range(config.phase2_steps):
        for l, layer in enumerate(net.layers):
            if not layer.has_patterns() and layer.mode == "modern":
                # Modern layer without patterns cannot do an intra-layer step;
                # treat its intra-layer drift as zero.
                drift = torch.zeros_like(state.xi[l])
            else:
                drift = layer.step(state.xi[l], beta=config.beta)
            inter = config.inter_layer_scale * _inter_layer_signal(net, state.xi, l)
            sigma_l = config.sigma_global * modes.layer_noise_ratio(l)
            eta = torch.randn(
                state.xi[l].shape,
                device=state.xi[l].device,
                dtype=state.xi[l].dtype,
                generator=generator,
            )
            state.xi[l] = keep * state.xi[l] + dt * (drift + inter) + sigma_l * sqrt_2dt * eta


def phase3_fixation(
    net: HopfieldNetwork,
    state: CycleState,
    config: CycleConfig,
) -> None:
    """Phase 3 (re-awake, fixation): deterministic high-beta sharpening."""
    net.enforce_constraints()
    beta_hi = config.beta * config.phase3_beta_boost
    dt = config.dt
    keep = 1.0 - dt
    for _ in range(config.phase3_steps):
        for l, layer in enumerate(net.layers):
            if not layer.has_patterns() and layer.mode == "modern":
                drift = torch.zeros_like(state.xi[l])
            else:
                drift = layer.step(state.xi[l], beta=beta_hi)
            inter = config.inter_layer_scale * _inter_layer_signal(net, state.xi, l)
            state.xi[l] = keep * state.xi[l] + dt * (drift + inter)


def phase4_readout(state: CycleState) -> Tensor:
    """Phase 4: read layer-0 state as the output."""
    return state.xi[0].clone()


def run_cycle(
    net: HopfieldNetwork,
    input_bias: Tensor,
    config: CycleConfig | None = None,
    initial: CycleState | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, CycleState]:
    """Run all four phases and return ``(layer-0 readout, final state)``."""
    config = config or CycleConfig()
    state = initial or CycleState.from_network(net, device=input_bias.device)
    phase1_terrain(net, state, input_bias)
    phase2_fluctuation(net, state, config, generator=generator)
    phase3_fixation(net, state, config)
    return phase4_readout(state), state
