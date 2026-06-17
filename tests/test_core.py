"""Tests for ayaram.core: 4-phase cycle."""

from __future__ import annotations

import torch

from ayaram import core, modes
from ayaram.memory import LAYER_SIZES, HopfieldNetwork


def _small_net(mode: str = "modern", seed: int = 0) -> HopfieldNetwork:
    net = HopfieldNetwork(mode=mode, seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    patterns = torch.randn(4, LAYER_SIZES[0], generator=g)
    net.store_layer0(patterns)
    return net


def test_initial_state_shapes_match_network():
    net = _small_net()
    st = core.CycleState.from_network(net)
    assert [t.shape[0] for t in st.xi] == list(LAYER_SIZES)


def test_phase1_sets_layer0_bias_exactly():
    net = _small_net()
    st = core.CycleState.from_network(net)
    bias = torch.randn(LAYER_SIZES[0])
    core.phase1_terrain(net, st, bias)
    assert torch.allclose(st.xi[0], bias, atol=0)


def test_phase1_does_not_touch_other_layers():
    net = _small_net()
    st = core.CycleState.from_network(net)
    before = [t.clone() for t in st.xi[1:]]
    core.phase1_terrain(net, st, torch.randn(LAYER_SIZES[0]))
    for a, b in zip(st.xi[1:], before):
        assert torch.equal(a, b)


def test_phase2_produces_state_change():
    """Phase 2 must visibly perturb the state -- it is the fluctuation phase."""
    net = _small_net()
    st = core.CycleState.from_network(net)
    bias = torch.randn(LAYER_SIZES[0])
    core.phase1_terrain(net, st, bias)
    before0 = st.xi[0].clone()
    cfg = core.CycleConfig(phase2_steps=20)
    g = torch.Generator().manual_seed(2)
    core.phase2_fluctuation(net, st, cfg, mode=modes.SLEEP, generator=g)
    # state must move
    assert not torch.allclose(st.xi[0], before0, atol=1e-6)


def test_phase2_zero_temperature_is_deterministic():
    """sigma_global=0 turns Phase 2 into deterministic gradient flow."""
    net = _small_net()
    st_a = core.CycleState.from_network(net)
    st_b = core.CycleState.from_network(net)
    bias = torch.randn(LAYER_SIZES[0])
    core.phase1_terrain(net, st_a, bias)
    core.phase1_terrain(net, st_b, bias)
    cfg = core.CycleConfig(phase2_steps=10, sigma_global=0.0)
    core.phase2_fluctuation(net, st_a, cfg, mode=modes.SLEEP)
    core.phase2_fluctuation(net, st_b, cfg, mode=modes.SLEEP)
    for a, b in zip(st_a.xi, st_b.xi):
        assert torch.allclose(a, b, atol=1e-6)


def test_phase3_is_deterministic_given_same_state():
    net = _small_net()
    st_a = core.CycleState.from_network(net)
    st_b = core.CycleState.from_network(net)
    bias = torch.randn(LAYER_SIZES[0])
    core.phase1_terrain(net, st_a, bias)
    core.phase1_terrain(net, st_b, bias)
    cfg = core.CycleConfig(phase3_steps=10)
    core.phase3_fixation(net, st_a, cfg)
    core.phase3_fixation(net, st_b, cfg)
    for a, b in zip(st_a.xi, st_b.xi):
        assert torch.allclose(a, b, atol=1e-6)


def test_run_cycle_returns_layer0_readout_shape():
    net = _small_net()
    bias = torch.randn(LAYER_SIZES[0])
    cfg = core.CycleConfig(phase2_steps=10, phase3_steps=5)
    out, st = core.run_cycle(net, bias, config=cfg)
    assert out.shape == (LAYER_SIZES[0],)
    assert torch.equal(out, st.xi[0])


def test_run_cycle_enforces_W_symmetry_modern_diag_zero():
    """After a full cycle the M0 + M1 invariants must still hold."""
    net = _small_net(mode="modern")
    bias = torch.randn(LAYER_SIZES[0])
    cfg = core.CycleConfig(phase2_steps=5, phase3_steps=5)
    core.run_cycle(net, bias, config=cfg)
    for layer in net.layers:
        assert torch.allclose(layer.W, layer.W.T, atol=1e-6)
        if layer.mode == "modern":
            assert torch.allclose(
                torch.diag(layer.W), torch.zeros(layer.n_units), atol=1e-7
            )
