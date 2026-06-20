"""v0.1.5 M1 — temperature_K > 0 behavior tests.

Two layers:
    A) Unit tests for ``compute_thermal_noise_amplitude`` — exact analytical
       checks at the physical anchor points (T=0 → 0, T=T_REF / K_u=K_U_REF
       → 1, etc).
    B) Integration tests confirming the additive thermal injection actually
       moves the state — T=300 K phase2_fluctuation must NOT equal T=0 K,
       and a small Modern recall must have cosine sim < 1 between T=300 and
       T=0 outputs.

The boundary contract (T=0 bit-exact, T<0 ValueError) is locked in by
``tests/test_v015_compat.py``; this file focuses on the positive-T branch.
"""

from __future__ import annotations

import math

import pytest
import torch

from ayaram import core, modes
from ayaram.memory import LAYER_SIZES, HopfieldNetwork


# ---------- A) compute_thermal_noise_amplitude unit checks --------------


def test_amplitude_T_zero_returns_exact_zero():
    """T = 0.0 returns exactly 0.0 (M0 bit-exact contract)."""
    assert modes.compute_thermal_noise_amplitude(K_u=1.0e5, T=0.0) == 0.0
    # Even for K_u == K_U_REF, T=0 must collapse to 0.0 exactly.
    assert modes.compute_thermal_noise_amplitude(K_u=modes.K_U_REF, T=0.0) == 0.0


def test_amplitude_negative_T_raises_value_error():
    with pytest.raises(ValueError):
        modes.compute_thermal_noise_amplitude(K_u=modes.K_U_REF, T=-1.0)
    with pytest.raises(ValueError):
        modes.compute_thermal_noise_amplitude(K_u=modes.K_U_REF, T=-1e-9)


def test_amplitude_at_room_temperature_and_K_u_ref_is_one():
    """T = T_REF_KELVIN, K_u = K_U_REF → amplitude = 1.0 exactly
    (room-temperature anchor of the phenomenological model)."""
    a = modes.compute_thermal_noise_amplitude(
        K_u=modes.K_U_REF, T=modes.T_REF_KELVIN
    )
    assert math.isclose(a, 1.0, rel_tol=0.0, abs_tol=0.0)


def test_amplitude_K_u_one_quarter_doubles_at_room_temperature():
    """K_u = K_U_REF / 4 at T = T_REF → sqrt(4) = 2.0 (K_u-dependence check)."""
    a = modes.compute_thermal_noise_amplitude(
        K_u=modes.K_U_REF / 4.0, T=modes.T_REF_KELVIN
    )
    assert math.isclose(a, 2.0, rel_tol=1e-12)


def test_amplitude_T_four_times_room_doubles_at_K_u_ref():
    """T = 1200 K (= 4 * T_REF), K_u = K_U_REF → sqrt(4) = 2.0 (T-dependence check)."""
    a = modes.compute_thermal_noise_amplitude(
        K_u=modes.K_U_REF, T=4.0 * modes.T_REF_KELVIN
    )
    assert math.isclose(a, 2.0, rel_tol=1e-12)


def test_amplitude_positive_for_positive_T():
    for T in (1.0, 50.0, 300.0, 500.0):
        a = modes.compute_thermal_noise_amplitude(K_u=modes.K_U_REF, T=T)
        assert a > 0.0


def test_amplitude_decreases_with_K_u_at_fixed_T():
    T = 300.0
    a0 = modes.compute_thermal_noise_amplitude(modes.K_U_LAYERS[0], T)
    a1 = modes.compute_thermal_noise_amplitude(modes.K_U_LAYERS[1], T)
    a2 = modes.compute_thermal_noise_amplitude(modes.K_U_LAYERS[2], T)
    assert a0 > a1 > a2 > 0.0


# ---------- B) phase2_fluctuation / recall integration ------------------


def _small_modern_net(seed: int = 0) -> HopfieldNetwork:
    net = HopfieldNetwork(mode="modern", seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    patterns = torch.randn(4, LAYER_SIZES[0], generator=g)
    net.store_layer0(patterns)
    return net


def test_phase2_T300_differs_from_T0():
    """T=300 K phase2 trajectory must NOT equal the T=0 trajectory."""
    net = _small_modern_net(seed=0)
    bias = torch.randn(LAYER_SIZES[0], generator=torch.Generator().manual_seed(13))
    cfg = core.CycleConfig(phase2_steps=10, sigma_global=0.1)

    st_T0 = core.CycleState.from_network(net)
    st_T300 = core.CycleState.from_network(net)
    core.phase1_terrain(net, st_T0, bias)
    core.phase1_terrain(net, st_T300, bias)

    g0 = torch.Generator().manual_seed(11)
    core.phase2_fluctuation(net, st_T0, cfg, generator=g0, temperature_K=0.0)

    g3 = torch.Generator().manual_seed(11)
    core.phase2_fluctuation(net, st_T300, cfg, generator=g3, temperature_K=300.0)

    # Layer 0 should drift visibly: not equal under torch.equal AND visibly
    # different in L1 norm.
    assert not torch.equal(st_T0.xi[0], st_T300.xi[0]), (
        "T=300 K and T=0 K phase2 produced identical layer-0 output — "
        "thermal injection is not active"
    )
    drift = (st_T0.xi[0] - st_T300.xi[0]).abs().mean().item()
    assert drift > 0.0


def test_recall_T300_cosine_recorded_small_modern():
    """Records T=300 vs T=0 cosine sim on a small Modern recall.

    Per the M1 brief cosine-sim interpretation table:
        ≥ 0.99   → 揺らぎが効いていない可能性、CC は報告書で明示
        0.7-0.99 → 数値のみ記録、M2 詳細スイープへ
        ≤ 0.7    → 揺らぎが強すぎる可能性、CC は報告書で明示

    This test asserts only the FP sanity (``cos <= 1.0 + eps``) and the M1
    invariant that the thermal injection is wired (state vector is finite).
    The numeric verdict goes into _tmp-v0.1.5-m1-report.md.
    """
    net = _small_modern_net(seed=0)
    query = net.layers[0].X[:, 0].clone()  # first stored pattern (column 0)

    g0 = torch.Generator().manual_seed(42)
    out0 = net.recall(
        query, generator=g0, return_all_layers=False, temperature_K=0.0
    )
    g3 = torch.Generator().manual_seed(42)
    out3 = net.recall(
        query, generator=g3, return_all_layers=False, temperature_K=300.0
    )

    assert torch.isfinite(out3).all(), "T=300 recall produced non-finite values"
    cos = torch.nn.functional.cosine_similarity(
        out0.flatten(), out3.flatten(), dim=0
    ).item()
    assert cos <= 1.0 + 1e-6, f"cosine sim {cos} exceeds FP sanity bound"
    print(f"\n[T=300 vs T=0 cosine sim, small Modern recall, seed=42] {cos:.9f}")


def test_recall_T300_cosine_recorded_hierarchical():
    """Same record on the 3-layer hierarchical setup (Modern, M5 defaults)."""
    seed = 1
    net = HopfieldNetwork(mode="modern", seed=seed)
    g_pat = torch.Generator().manual_seed(seed + 200)
    p0 = torch.randn(3, LAYER_SIZES[0], generator=g_pat)
    p1 = torch.randn(3, LAYER_SIZES[1], generator=g_pat)
    p2 = torch.randn(3, LAYER_SIZES[2], generator=g_pat)
    net.learn([p0, p1, p2], normalize_inter="spectral", center_inter_inputs=True)
    query = p0[0].clone()

    g0 = torch.Generator().manual_seed(seed + 1)
    out0 = net.recall(
        query, generator=g0, return_all_layers=False, temperature_K=0.0
    )
    g3 = torch.Generator().manual_seed(seed + 1)
    out3 = net.recall(
        query, generator=g3, return_all_layers=False, temperature_K=300.0
    )

    assert torch.isfinite(out3).all()
    cos = torch.nn.functional.cosine_similarity(
        out0.flatten(), out3.flatten(), dim=0
    ).item()
    assert cos <= 1.0 + 1e-6
    print(
        f"\n[T=300 vs T=0 cosine sim, hierarchical Modern, seed={seed}] "
        f"{cos:.9f}"
    )


def test_run_cycle_T300_accepts_temperature_K():
    """``core.run_cycle(..., temperature_K=300.0)`` no longer raises
    NotImplementedError (M1 lifted the guard)."""
    net = _small_modern_net(seed=0)
    bias = torch.randn(LAYER_SIZES[0])
    cfg = core.CycleConfig(phase2_steps=3, phase3_steps=3, sigma_global=0.1)
    out, _ = core.run_cycle(net, bias, config=cfg, temperature_K=300.0)
    assert out.shape == (LAYER_SIZES[0],)


def test_phase2_T0_default_signature_still_bit_exact_after_M1():
    """Sanity guard for M1 regressions: default call (no temperature_K) and
    explicit ``temperature_K=0.0`` still produce ``torch.equal`` outputs.
    Duplicates the test in test_v015_compat.py but lives here as a
    direct guard against accidental ordering changes from the M1 thermal
    branch landing in phase2_fluctuation."""
    net = _small_modern_net(seed=0)
    bias = torch.randn(LAYER_SIZES[0], generator=torch.Generator().manual_seed(99))
    cfg = core.CycleConfig(phase2_steps=15, sigma_global=0.3)

    st_a = core.CycleState.from_network(net)
    st_b = core.CycleState.from_network(net)
    core.phase1_terrain(net, st_a, bias)
    core.phase1_terrain(net, st_b, bias)

    g_a = torch.Generator().manual_seed(11)
    core.phase2_fluctuation(net, st_a, cfg, generator=g_a)

    g_b = torch.Generator().manual_seed(11)
    core.phase2_fluctuation(net, st_b, cfg, generator=g_b, temperature_K=0.0)

    for l, (a, b) in enumerate(zip(st_a.xi, st_b.xi)):
        assert torch.equal(a, b), (
            f"M1 regression: layer {l} max abs diff = "
            f"{(a - b).abs().max().item()}"
        )
