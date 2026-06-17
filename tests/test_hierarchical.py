"""Tests for the M3 hierarchical recall pieces.

Covers data/kanji_hierarchy.py (encoding dictionaries), ayaram/encoding.py
(pattern encoders), the new HopfieldNetwork.learn() / recall() /
recall_from_layer() methods, and the Phase 1 ``learn`` variant.
"""

from __future__ import annotations

import pytest
import torch

from ayaram import core, encoding
from ayaram.memory import HopfieldLayer, HopfieldNetwork

from data.kanji_hierarchy import (
    KANJI,
    KANJI_ORIGIN,
    KANJI_RADICALS,
    ORIGINS,
    RADICALS,
    validate,
)


# ----- dictionaries --------------------------------------------------------


def test_hierarchy_validation_passes():
    validate()


def test_kanji_count_and_radical_origin_consistency():
    assert len(KANJI) == 8
    assert len(RADICALS) == 4
    assert len(ORIGINS) == 3
    # Each kanji has at least one radical
    for k in KANJI:
        assert len(KANJI_RADICALS[k]) >= 1
    # Each kanji's origin is one of the three categories
    for k in KANJI:
        assert KANJI_ORIGIN[k] in ORIGINS


# ----- encoders ------------------------------------------------------------


def test_encode_radical_木_simple_onehot():
    p = encoding.encode_radical("木")
    assert p.shape == (encoding.LAYER1_DIM,)
    assert float(p[0].item()) == 1.0
    assert float(p[1:].abs().sum().item()) == 0.0


def test_encode_radical_林_double_activation():
    p = encoding.encode_radical("林")
    # 林 -> {木: 2} -> first radical-dim = 2, rest 0
    assert float(p[0].item()) == 2.0
    assert float(p[1:].abs().sum().item()) == 0.0


def test_encode_radical_明_multi_hot():
    p = encoding.encode_radical("明")
    # 明 -> {日: 1, 月: 1} -> dims 1 and 2 each = 1
    assert float(p[0].item()) == 0.0
    assert float(p[1].item()) == 1.0
    assert float(p[2].item()) == 1.0
    assert float(p[3].item()) == 0.0


def test_encode_radical_晶_triple_activation():
    p = encoding.encode_radical("晶")
    assert float(p[1].item()) == 3.0
    p[1] = 0
    assert float(p.abs().sum().item()) == 0.0


def test_encode_origin_one_hot_植物():
    p = encoding.encode_origin("木")
    assert p.shape == (encoding.LAYER2_DIM,)
    assert float(p[0].item()) == 1.0
    assert float(p[1:].abs().sum().item()) == 0.0


def test_encode_origin_one_hot_天体():
    p = encoding.encode_origin("月")
    # 月 -> 天体 (index 1)
    assert float(p[1].item()) == 1.0
    p[1] = 0
    assert float(p.abs().sum().item()) == 0.0


def test_encode_batch_shapes():
    p1 = encoding.encode_batch_radical(KANJI)
    p2 = encoding.encode_batch_origin(KANJI)
    assert p1.shape == (8, encoding.LAYER1_DIM)
    assert p2.shape == (8, encoding.LAYER2_DIM)


def test_decode_radical_round_trip():
    p = encoding.encode_radical("明")
    d = encoding.decode_radical(p)
    assert d["木"] == 0.0
    assert d["日"] == 1.0
    assert d["月"] == 1.0
    assert d["火"] == 0.0


def test_origin_one_hot_match_true_and_false():
    p_true = encoding.encode_origin("月")
    assert encoding.origin_one_hot_match(p_true, "月") is True
    # corrupted: change argmax
    p_wrong = encoding.encode_origin("木")
    assert encoding.origin_one_hot_match(p_wrong, "月") is False


def test_radical_set_match_full_and_partial():
    # 明 has two radicals (日, 月); a pattern with both is a full match
    full = encoding.encode_radical("明")
    assert encoding.radical_set_match(full, "明") == 1.0
    # zero out one radical -> half match
    half = full.clone()
    half[1] = 0.0  # erase 日
    assert encoding.radical_set_match(half, "明") == 0.5


# ----- learning + recall ---------------------------------------------------


def _build_small_patterns(N: int = 4, sizes: tuple[int, ...] = (16, 8, 4)):
    """Random non-trivial multi-layer patterns for HopfieldNetwork.learn."""
    g = torch.Generator().manual_seed(0)
    return [torch.randn(N, s, generator=g) for s in sizes]


def test_network_learn_installs_intra_layer_weights():
    sizes = (16, 8, 4)
    net = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    patterns = _build_small_patterns(N=4, sizes=sizes)
    # before: all zeros
    for layer in net.layers:
        assert torch.equal(layer.W, torch.zeros(layer.n_units, layer.n_units))
    net.learn(patterns)
    # after: each layer has non-trivial W
    for layer in net.layers:
        assert layer.W.abs().sum().item() > 0


def test_network_learn_installs_inter_layer_weights():
    sizes = (16, 8, 4)
    net = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    patterns = _build_small_patterns(N=4, sizes=sizes)
    before = [w.clone() for w in net.W_inter]
    net.learn(patterns)
    after = net.W_inter
    for b, a in zip(before, after):
        assert not torch.allclose(b, a)


def test_network_learn_preserves_symmetry_and_modern_diag_zero():
    sizes = (16, 8, 4)
    for mode in ("hebb", "modern"):
        net = HopfieldNetwork(mode=mode, layer_sizes=sizes, seed=0)
        patterns = _build_small_patterns(N=4, sizes=sizes)
        net.learn(patterns)
        for layer in net.layers:
            assert torch.allclose(layer.W, layer.W.T, atol=1e-6)
            if mode == "modern":
                assert torch.allclose(
                    torch.diag(layer.W),
                    torch.zeros(layer.n_units),
                    atol=1e-7,
                )


def test_network_recall_returns_all_layers_when_requested():
    sizes = (16, 8, 4)
    net = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    patterns = _build_small_patterns(N=4, sizes=sizes)
    net.learn(patterns)
    q = torch.randn(sizes[0])
    out = net.recall(
        q,
        config=core.CycleConfig(phase2_steps=5, phase3_steps=5, sigma_global=0.0),
        return_all_layers=True,
    )
    assert isinstance(out, dict)
    assert sorted(out.keys()) == [0, 1, 2]
    for l, x in out.items():
        assert x.shape == (sizes[l],)


def test_network_recall_returns_layer0_only():
    sizes = (16, 8, 4)
    net = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    net.learn(_build_small_patterns(N=4, sizes=sizes))
    q = torch.randn(sizes[0])
    out = net.recall(
        q,
        config=core.CycleConfig(phase2_steps=5, phase3_steps=5, sigma_global=0.0),
        return_all_layers=False,
    )
    assert torch.is_tensor(out)
    assert out.shape == (sizes[0],)


def test_network_recall_from_layer_seeds_at_specified_layer():
    sizes = (16, 8, 4)
    net = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    patterns = _build_small_patterns(N=4, sizes=sizes)
    net.learn(patterns)
    q1 = torch.randn(sizes[1])  # seed at layer 1
    out = net.recall_from_layer(
        q1,
        layer_idx=1,
        config=core.CycleConfig(phase2_steps=5, phase3_steps=5, sigma_global=0.0),
    )
    # All three layers should be present
    assert sorted(out.keys()) == [0, 1, 2]
    # And the layer-1 state shouldn't be identically zero (we seeded it)
    assert out[1].abs().sum().item() > 0


# ----- phase1_learn convenience wrapper -----------------------------------


def test_phase1_learn_delegates_to_network_learn():
    sizes = (16, 8, 4)
    net_a = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    net_b = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    patterns = _build_small_patterns(N=4, sizes=sizes)
    net_a.learn(patterns)
    core.phase1_learn(net_b, patterns)
    for la, lb in zip(net_a.layers, net_b.layers):
        assert torch.allclose(la.W, lb.W, atol=1e-7)
    for wa, wb in zip(net_a.W_inter, net_b.W_inter):
        assert torch.allclose(wa, wb, atol=1e-7)


def test_phase1_terrain_layer_idx_seeds_correct_layer():
    sizes = (16, 8, 4)
    net = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    state = core.CycleState.from_network(net)
    seed = torch.randn(sizes[1])
    core.phase1_terrain(net, state, seed, layer_idx=1)
    assert torch.allclose(state.xi[1], seed, atol=0)
    assert state.xi[0].abs().sum().item() == 0
    assert state.xi[2].abs().sum().item() == 0


def test_phase1_terrain_layer_idx_out_of_range_raises():
    sizes = (16, 8, 4)
    net = HopfieldNetwork(mode="hebb", layer_sizes=sizes, seed=0)
    state = core.CycleState.from_network(net)
    with pytest.raises(IndexError):
        core.phase1_terrain(net, state, torch.zeros(4), layer_idx=3)
