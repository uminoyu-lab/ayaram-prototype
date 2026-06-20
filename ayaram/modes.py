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

K_U_REF: float = K_U_LAYERS[0]
"""v0.1.5 M1 reference anisotropy K_u_ref (J / m^3).

Equal to ``K_U_LAYERS[0]`` — the layer-0 value that v0.1's
``layer_noise_ratio(l) = sqrt(K_U_LAYERS[0] / K_U_LAYERS[l])`` is already
normalized against. Exposed as a module-level constant so the temperature-
dependent noise model in ``compute_thermal_noise_amplitude`` can name a
single source of truth.

This is CC 解釈 (M1) — the M1 brief offered three defensible paths for the
``K_u_ref`` source (direct config reference / argument / module-level
constant); a module-level constant tied to the existing K_U_LAYERS tuple
keeps one truth in one file.
"""

T_REF_KELVIN: float = 300.0
"""Reference temperature (Kelvin) used by ``compute_thermal_noise_amplitude``.
Room-temperature anchor — the story is that at T = T_REF_KELVIN the
additive thermal noise has the same dimensionless scale as the v0.1
layer-0 noise."""

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


def compute_thermal_noise_amplitude(K_u: float, T: float) -> float:
    """Dimensionless additive-thermal-noise amplitude at temperature T (K).

    v0.1.5 M1 phenomenological model:

        sigma_thermal_dimless(K_u, T) = sqrt(K_U_REF / K_u) * sqrt(T / T_REF_KELVIN)

    The K_u dependence ``sqrt(K_U_REF / K_u)`` inherits v0.1's
    ``layer_noise_ratio(l) = sqrt(K_U_LAYERS[0] / K_U_LAYERS[l])`` scaling so
    that the new term sits naturally on the same per-layer barrier hierarchy.
    The temperature dependence ``sqrt(T / T_REF_KELVIN)`` is a current-layer
    phenomenological form; physical rigor (Brown 1963 sLLG) is delegated to
    mumax3 (the aya-sleep side). M1 does NOT include the
    ``K_u_eff(T) = K_u (1 - alpha (T - T_ref))`` Sato et al. 2014 correction
    — that is reserved for v0.2 when the CoFeB temperature coefficient
    ``alpha`` is wired in.

    Sources of constants:
        K_U_REF        = K_U_LAYERS[0] = 2.0e4 (J / m^3), module-level
                         constant tied to the v0.1 layer-0 anisotropy and
                         used by ``layer_noise_ratio``. Single source of
                         truth — see the ``K_U_REF`` docstring above.
        T_REF_KELVIN   = 300.0 K, module-level constant — room-temperature
                         anchor.

    Boundary behavior (M0 bit-exact contract):
        T == 0.0  → returns ``0.0`` exactly. ``ayaram.core.phase2_fluctuation``
                    uses this to guard the v0.1 path: when ``temperature_K``
                    is exactly 0.0, no extra arithmetic, no extra RNG draws.
        T <  0.0  → ``ValueError`` (T is absolute, must be non-negative).
        T >  0.0  → positive float, multiply by ``config.sigma_global`` and
                    the per-step ``sqrt(2 dt)`` factor inside phase2 to get
                    the per-element noise std for the independent Gaussian
                    draw.

    Args:
        K_u: per-cell perpendicular-anisotropy energy density (J / m^3) at
             the reference temperature; for the 3-layer network the call
             sites use ``modes.K_U_LAYERS[layer_idx]``.
        T:   absolute temperature in Kelvin.

    Returns:
        Dimensionless amplitude — multiply by ``config.sigma_global`` for
        the per-step thermal-noise std deviation.

    References:
        Brown, W. F. (1963). "Thermal fluctuations of a single-domain
        particle." Physical Review, 130(5), 1677–1686. (First-principles
        sLLG noise term — mumax3 uses this directly; the PyTorch side here
        is the additive-Gaussian surrogate, see design_decisions.md v0.1.5.)

        Sato, H., et al. (2014). "Properties of magnetic tunnel junctions
        with a MgO/CoFeB/Ta/CoFeB/MgO recording structure down to junction
        diameter of 11 nm." Applied Physics Letters, 105(6), 062403.
        (CoFeB ``alpha = dK_u/dT`` reserved for v0.2 K_u_eff(T) correction;
        NOT used by this M1 implementation.)
    """
    if T == 0.0:
        return 0.0
    if T < 0.0:
        raise ValueError(f"Temperature must be non-negative, got T={T}")
    return (K_U_REF / K_u) ** 0.5 * (T / T_REF_KELVIN) ** 0.5
