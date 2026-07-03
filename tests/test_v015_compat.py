"""v0.1.5 M0 / M1 bit-exact compatibility tests.

DoD: ``temperature_K=0.0`` is a strict no-op — every recall / cycle output
along that path must be ``torch.equal`` to the v0.1 path (max abs diff = 0.0,
no tolerance).

We anchor the bit-exact guarantee by computing the SAME trajectory two ways:
    A) without ``temperature_K`` (old v0.1 signature)
    B) with explicit ``temperature_K=0.0`` (new v0.1.5 signature)
``torch.equal`` requires identical dtype, shape, AND every element bit-for-bit.

By construction ``ayaram.core.phase2_fluctuation`` performs zero extra
arithmetic and draws zero extra RNG samples when ``temperature_K == 0.0``,
so this anchors the v0.1 commit ``2d0932b`` reference behavior.

M1 adds the ``v15_modern_seed0_cosine.pt`` snapshot test below — the fixture
was generated from the v0.1 worktree at commit 2d0932b by
``scripts/generate_v15_cosine_fixture.py`` (CPU-locked) and re-running the
same demo path on the current branch with ``temperature_K=0.0`` (the default)
must reproduce it bit-exactly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from ayaram import core, modes
from ayaram.memory import LAYER_SIZES, HopfieldNetwork


def _small_modern_net(seed: int = 0) -> HopfieldNetwork:
    net = HopfieldNetwork(mode="modern", seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    patterns = torch.randn(4, LAYER_SIZES[0], generator=g)
    net.store_layer0(patterns)
    return net


def _hierarchical_net(seed: int = 0, mode: str = "modern") -> HopfieldNetwork:
    net = HopfieldNetwork(mode=mode, seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    p0 = torch.randn(3, LAYER_SIZES[0], generator=g)
    p1 = torch.randn(3, LAYER_SIZES[1], generator=g)
    p2 = torch.randn(3, LAYER_SIZES[2], generator=g)
    net.learn([p0, p1, p2], normalize_inter="spectral", center_inter_inputs=True)
    return net, p0, p1, p2


# ---------- recall() bit-exact ------------------------------------------


def test_recall_T0_bit_exact_with_v01_signature():
    """recall(query) == recall(query, temperature_K=0.0), max abs diff = 0.0."""
    net = _small_modern_net(seed=0)
    query = net.layers[0].X[:, 0].clone()  # first stored pattern as column

    g_v01 = torch.Generator().manual_seed(42)
    out_v01 = net.recall(query, generator=g_v01, return_all_layers=True)

    g_v15 = torch.Generator().manual_seed(42)
    out_v15 = net.recall(
        query, generator=g_v15, return_all_layers=True, temperature_K=0.0
    )

    for layer_idx in out_v01:
        a = out_v01[layer_idx]
        b = out_v15[layer_idx]
        assert torch.equal(a, b), (
            f"layer {layer_idx}: max abs diff = "
            f"{(a - b).abs().max().item()} (expected 0.0)"
        )


def test_recall_from_layer_T0_bit_exact_with_v01_signature():
    """recall_from_layer(layer_idx=1) bit-exact same with explicit T=0."""
    net, _, p1, _ = _hierarchical_net(seed=1, mode="modern")
    query = p1[0].clone()

    g_v01 = torch.Generator().manual_seed(7)
    out_v01 = net.recall_from_layer(query, layer_idx=1, generator=g_v01)

    g_v15 = torch.Generator().manual_seed(7)
    out_v15 = net.recall_from_layer(
        query, layer_idx=1, generator=g_v15, temperature_K=0.0
    )

    for layer_idx in out_v01:
        a = out_v01[layer_idx]
        b = out_v15[layer_idx]
        assert torch.equal(a, b), (
            f"layer {layer_idx}: max abs diff = "
            f"{(a - b).abs().max().item()} (expected 0.0)"
        )


# ---------- phase2_fluctuation() bit-exact -------------------------------


def test_phase2_fluctuation_T0_bit_exact_with_v01_signature():
    """phase2_fluctuation default (no temperature_K) and explicit T=0 produce
    identical post-Phase-2 state — strict torch.equal, max abs diff = 0.0."""
    net = _small_modern_net(seed=0)
    bias = torch.randn(LAYER_SIZES[0], generator=torch.Generator().manual_seed(99))
    cfg = core.CycleConfig(phase2_steps=15, sigma_global=0.3)

    st_v01 = core.CycleState.from_network(net)
    st_v15 = core.CycleState.from_network(net)
    core.phase1_terrain(net, st_v01, bias)
    core.phase1_terrain(net, st_v15, bias)

    g_v01 = torch.Generator().manual_seed(11)
    core.phase2_fluctuation(net, st_v01, cfg, generator=g_v01)

    g_v15 = torch.Generator().manual_seed(11)
    core.phase2_fluctuation(net, st_v15, cfg, generator=g_v15, temperature_K=0.0)

    for l, (a, b) in enumerate(zip(st_v01.xi, st_v15.xi)):
        assert torch.equal(a, b), (
            f"layer {l}: max abs diff = "
            f"{(a - b).abs().max().item()} (expected 0.0)"
        )


# ---------- v15 demo path (Modern, seed=0) bit-exact --------------------


def test_v15_modern_demo_path_T0_bit_exact():
    """Replicates demos/hierarchical_kanji_v15.py _run_forward for Modern /
    seed=0 on a small batch — the demo's explicit phase1+phase2+phase3 path
    must give max abs diff = 0.0 between v0.1 signature and T=0 v0.1.5
    signature. Anchors "hierarchical_kanji_v15.py Modern seed=0 bit-exact"
    DoD without needing to load the full 12-kanji dataset."""
    n = 3
    seed = 0
    g_patterns = torch.Generator().manual_seed(seed + 200)
    p0 = torch.randn(n, LAYER_SIZES[0], generator=g_patterns)
    p1 = torch.randn(n, LAYER_SIZES[1], generator=g_patterns)
    p2 = torch.randn(n, LAYER_SIZES[2], generator=g_patterns)
    query = torch.randn(n, LAYER_SIZES[0], generator=g_patterns)

    def _run(temperature_K: float | None) -> list[torch.Tensor]:
        net = HopfieldNetwork(mode="modern", seed=seed)
        net.learn([p0, p1, p2], normalize_inter="spectral", center_inter_inputs=True)
        state = core.CycleState(
            xi=[torch.zeros(n, s) for s in net.layer_sizes]
        )
        core.phase1_terrain(net, state, query, layer_idx=0)
        cfg = core.CycleConfig(
            beta=5.0,
            sigma_global=0.1,
            phase2_steps=20,
            phase3_steps=10,
            inter_layer_scale=0.1,
        )
        g = torch.Generator().manual_seed(seed + 1)
        if temperature_K is None:
            core.phase2_fluctuation(net, state, cfg, generator=g)
        else:
            core.phase2_fluctuation(
                net, state, cfg, generator=g, temperature_K=temperature_K
            )
        core.phase3_fixation(net, state, cfg)
        return [t.clone() for t in state.xi]

    xi_v01 = _run(temperature_K=None)
    xi_v15 = _run(temperature_K=0.0)

    for l, (a, b) in enumerate(zip(xi_v01, xi_v15)):
        assert torch.equal(a, b), (
            f"layer {l}: max abs diff = "
            f"{(a - b).abs().max().item()} (expected 0.0)"
        )


# ---------- guard: negative T rejected at both layers -------------------


def test_negative_temperature_rejected():
    net = _small_modern_net(seed=0)
    bias = torch.randn(LAYER_SIZES[0])
    cfg = core.CycleConfig(phase2_steps=1)
    state = core.CycleState.from_network(net)
    core.phase1_terrain(net, state, bias)
    with pytest.raises(ValueError):
        core.phase2_fluctuation(net, state, cfg, temperature_K=-1.0)


# ---------- modes.compute_thermal_noise_amplitude signature -------------


def test_compute_thermal_noise_amplitude_docstring_cites_references():
    """M0 / M1 contract: callable exists, T=0 returns 0.0 exactly, and the
    docstring still cites the physical sources (Brown 1963 and Sato 2014,
    even though the Sato alpha correction is deferred to v0.2)."""
    assert callable(modes.compute_thermal_noise_amplitude)
    assert modes.compute_thermal_noise_amplitude(K_u=1.0e5, T=0.0) == 0.0
    doc = modes.compute_thermal_noise_amplitude.__doc__ or ""
    assert "Sato" in doc and "2014" in doc, "docstring must cite Sato et al. 2014"
    assert "Brown" in doc and "1963" in doc, "docstring must cite Brown 1963"


def test_thermal_sweep_demo_importable():
    """demos/thermal_sweep.py must import cleanly (skeleton-only is fine)."""
    import importlib

    mod = importlib.import_module("demos.thermal_sweep")
    assert hasattr(mod, "T_LIST_DEFAULT")
    assert mod.T_LIST_DEFAULT[0] == 0.0
    for name in ("run_kanji_sweep", "run_attention_sweep", "plot_results", "main"):
        assert callable(getattr(mod, name))


# ---------- v15 demo Modern seed=0 cosine fixture (two-stage anchor) ----


def test_v15_modern_seed0_matches_v01_fixture():
    """Re-runs the M4-12 Modern seed=0 demo path with temperature_K=0.0 and
    asserts ``torch.equal`` against the fixture ``v15_modern_seed0_cosine.pt``,
    which was produced by running the same script
    (``scripts/generate_v15_cosine_fixture.py``) from a git worktree at
    v0.1 commit 2d0932b. Cross-check (CC 解釈, M1): the v0.1-worktree fixture
    and a re-run on this branch were proven ``torch.equal`` at fixture
    creation time; this test perpetuates that guarantee."""
    import numpy as np  # noqa: F401 (used inside script generate())
    from scripts.generate_v15_cosine_fixture import FIXTURE_PATH, generate

    assert os.path.exists(FIXTURE_PATH), (
        f"fixture missing: {FIXTURE_PATH}. Regenerate via "
        "`uv run python scripts/generate_v15_cosine_fixture.py`."
    )
    expected = torch.load(FIXTURE_PATH, weights_only=True)
    actual = generate()

    assert actual["kanji"] == expected["kanji"], "kanji label list drifted"
    for layer in (0, 1, 2):
        key = f"layer{layer}_cos"
        a, b = actual[key], expected[key]
        assert a.dtype == b.dtype == torch.float64
        assert torch.equal(a, b), (
            f"layer {layer}: max abs diff = "
            f"{(a - b).abs().max().item()} (expected 0.0)"
        )
