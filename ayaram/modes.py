"""Modes: aya-awake / aya-sleep switching.

Design decision #1 (Aya + Yu, 2026-06-17): whole-cell synchronous switching.
Every cell in every layer flips between awake (high barrier, stable) and sleep
(low barrier, fluctuating) at the same time. Per-layer barrier differences are
fixed parameters that shape representation; they are not the mechanism by
which the cycle progresses.

Design decision #4 (Aya + Yu, 2026-06-17): the sleep-mode noise strength sigma
is meant to correspond to the inverse-temperature beta of softmax attention
under Ramsauer 2020 Theorem 3.

Sub-decision (Aru, 2026-06-17): per-layer barrier values
    K_u_0 = 2.0e4,  K_u_1 = 1.5e5,  K_u_2 = 1.2e6
chosen as logarithmic midpoints between the aya-sleep / aya-awake values in
the minimum-design v0.1 document.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

K_U_LAYERS: tuple[float, float, float] = (2.0e4, 1.5e5, 1.2e6)

T_AWAKE: float = 0.01
T_SLEEP: float = 1.0


def sigma_local(layer_idx: int, T_global: float) -> float:
    """Per-layer *physical* effective noise strength.

    sigma_local(l, T) = sqrt(T / K_u_layer[l]) -- the thermal fluctuation of a
    magnetic moment with anisotropy energy K_u under temperature T, up to a
    proportionality constant. This is the formula required by the sub-decision
    and is what ``ayaram.core`` reports for inspection / book-keeping.

    Note: in the v0.1 ``CycleConfig``, the user-facing noise knob
    ``sigma_global`` is interpreted as the *direct* per-step noise std at
    layer 0, with other layers scaled by ``layer_noise_ratio(l)``. This
    decouples the sweep range ``[0.01, 10]`` from the absolute size of K_u
    (which lives on a very different numerical scale).
    """
    if layer_idx < 0 or layer_idx >= len(K_U_LAYERS):
        raise IndexError(f"layer_idx must be in [0, {len(K_U_LAYERS)})")
    if T_global < 0:
        raise ValueError("T_global must be non-negative")
    return math.sqrt(T_global / K_U_LAYERS[layer_idx])


def layer_noise_ratio(layer_idx: int) -> float:
    """Per-layer noise scaling relative to layer 0.

    Equal to ``sqrt(K_U_LAYERS[0] / K_U_LAYERS[layer_idx])`` -- the ratio of
    ``sigma_local`` between layers at a common temperature. By construction
    ``layer_noise_ratio(0) = 1.0``. The remaining layers are quieter (higher
    barrier -> smaller thermal fluctuation):

        layer 1: ~0.365
        layer 2: ~0.129

    ``ayaram.core.phase2_fluctuation`` multiplies ``config.sigma_global`` by
    this ratio so that ``sigma_global`` is the literal layer-0 noise std.
    """
    if layer_idx < 0 or layer_idx >= len(K_U_LAYERS):
        raise IndexError(f"layer_idx must be in [0, {len(K_U_LAYERS)})")
    return math.sqrt(K_U_LAYERS[0] / K_U_LAYERS[layer_idx])


@dataclass(frozen=True)
class Mode:
    name: str
    T_global: float

    def sigma(self, layer_idx: int) -> float:
        return sigma_local(layer_idx, self.T_global)


AWAKE = Mode(name="aya-awake", T_global=T_AWAKE)
SLEEP = Mode(name="aya-sleep", T_global=T_SLEEP)
