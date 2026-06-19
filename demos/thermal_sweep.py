"""v0.1.5 M2 demo: temperature sweep over the kanji + attention experiments.

Scaffold landed in v0.1.5 M0; bodies will be implemented in M2.
The sweep axis is temperature T (Kelvin) — shared with the mumax3 aya-sleep
side, see _to-cc-v0.1.5-m0.md "Designed判断 / 中心パラメータ".

Three questions this demo will answer (M2 / M3 / M4):
    (a) which of the 12 kanji shift hit/miss as T varies (v0.1: 川 / 山 missed)
    (b) how Theorem 3's bit-exact equivalence degrades with T (cos sim, KL)
    (c) does the PyTorch curve "resonate" with the mumax3 sLLG curve at the
        same T

T_LIST_DEFAULT is the canonical sweep grid used by all three runs.
"""

from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ayaram import core, modes  # noqa: E402
from ayaram.memory import HopfieldNetwork  # noqa: E402


T_LIST_DEFAULT: list[float] = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
"""Default temperature sweep grid in Kelvin. T=0.0 is the v0.1 bit-exact
compatibility anchor; the rest cover sub-room, room, and elevated regimes
relevant to CoFeB MTJ operating temperatures (Sato et al. 2014)."""

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def run_kanji_sweep(
    T_list: list[float] | None = None,
    seed: int = 0,
) -> dict:
    """Sweep the 12-kanji hierarchical recall over temperature T (Kelvin).

    For each T in ``T_list``, runs the v15 Modern + Hebb hierarchical recall
    and records per-kanji layer-0/1/2 cosine and origin hit-rate. The T=0
    point must reproduce v0.1's 10/12 origin hit rate exactly (bit-exact
    anchor, verified by tests/test_v015_compat.py).

    Args:
        T_list: list of temperatures (K) to sweep. ``None`` -> T_LIST_DEFAULT.
        seed:   passed to ``HopfieldNetwork(seed=...)`` and the Phase 2
                Generator (seed + 1, matching demos/hierarchical_kanji_v15.py).

    Returns:
        dict with keys ``"T"``, ``"modern"``, ``"hebb"`` — each carrying
        per-T per-kanji metrics suitable for ``plot_results``.

    M0 status: skeleton only. M2 will fill the body.
    """
    raise NotImplementedError("M2 で実装")


def run_attention_sweep(
    T_list: list[float] | None = None,
) -> dict:
    """Sweep the Theorem 3 attention-equivalence check over temperature T.

    The v0.1 check (M1 attention_test.py) reports max abs diff = 0.0 between
    Modern Hopfield's softmax-equivalent attention and a reference softmax
    layer. v0.1.5 measures how that bit-exact equivalence degrades as the
    thermal fluctuation amplitude grows with T (question (b)).

    Args:
        T_list: list of temperatures (K). ``None`` -> T_LIST_DEFAULT.

    Returns:
        dict with keys ``"T"``, ``"max_abs_diff"``, ``"cos_sim"``,
        ``"kl_divergence"`` — each a list aligned with ``T_list``.

    M0 status: skeleton only. M3 will fill the body.
    """
    raise NotImplementedError("M2 で実装")  # body deferred to M3


def plot_results(results: dict) -> None:
    """Render the kanji-sweep and attention-sweep results as PNG.

    Will save (M2 / M3):
        thermal_sweep_kanji.png       — per-kanji origin hit-rate vs T
        thermal_sweep_attention.png   — max abs diff / cos sim / KL vs T

    Args:
        results: dict matching ``run_kanji_sweep`` or
                 ``run_attention_sweep`` schema.

    M0 status: skeleton only. M2 will fill the body.
    """
    raise NotImplementedError("M2 で実装")


def main() -> None:
    """End-to-end demo entry point.

    Plan (M2 / M3 / M4):
        1. run_kanji_sweep(T_LIST_DEFAULT) -> save figure
        2. run_attention_sweep(T_LIST_DEFAULT) -> save figure
        3. M4: overlay mumax3 aya-sleep CSV on a common T axis.

    M0 status: skeleton only.
    """
    raise NotImplementedError("M2 で実装")


if __name__ == "__main__":
    main()
