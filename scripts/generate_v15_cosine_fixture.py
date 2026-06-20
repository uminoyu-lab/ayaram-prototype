"""v0.1.5 M1 fixture-generation: per-kanji cosine snapshot of the
``hierarchical_kanji_v15.py`` Modern-mode seed=0 path (M4-12).

Two-stage defensibility (per _to-cc-v0.1.5-m1.md):
    (a) Run on the v0.1 worktree at commit 2d0932b → produce fixture.
    (b) Run on the current branch (v0.1.5-thermal-fluctuation) with the
        default temperature_K=0.0 path → reproduce identical fixture.
    (c) tests/test_v015_compat.py asserts torch.equal between the
        committed (a) fixture and a fresh in-test (b) run, perpetually.

CC 解釈 (M1):
    - Device locked to CPU regardless of CUDA availability — torch.randn
      on CUDA can differ across driver / device versions, while CPU
      randn is deterministic across machines (within the same PyTorch
      version). The bit-exact contract we care about is "T=0 path
      identical to v0.1"; locking to CPU is the cross-machine anchor.
    - Fixture format: torch.save of a dict containing the 12 kanji
      labels (list of str) and three float64 tensors layer0_cos /
      layer1_cos / layer2_cos. float64 preserves the Python-float
      values that the demo's _cosine helper returns.
    - File path: tests/fixtures/v15_modern_seed0_cosine.pt
    - Demo params copied verbatim from demos/hierarchical_kanji_v15.py
      main() defaults: beta=5.0, sigma_global=0.1, phase2_steps=500,
      phase3_steps=100, inter_layer_scale=0.1, seed=0.

Run:
    uv run python scripts/generate_v15_cosine_fixture.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ayaram import encoding  # noqa: E402
from demos.hierarchical_kanji_v15 import (  # noqa: E402
    _run_forward,
    KH_V15,
    KANJI_PATH_V15,
)


FIXTURE_PATH = os.path.join(
    _ROOT, "tests", "fixtures", "v15_modern_seed0_cosine.pt"
)


def generate() -> dict:
    """Run the M4-12 Modern seed=0 path on CPU and return the cosine dict."""
    device = torch.device("cpu")
    bitmaps = np.load(KANJI_PATH_V15)
    p1 = encoding.encode_batch_radical_count_v15(
        KH_V15.KANJI,
        kanji_radicals=KH_V15.KANJI_RADICALS,
        radicals=KH_V15.RADICALS,
        max_count=KH_V15.MAX_COUNT,
    ).to(device)
    p2 = encoding.encode_batch_origin_v15(
        KH_V15.KANJI,
        kanji_origin=KH_V15.KANJI_ORIGIN,
        origins=KH_V15.ORIGINS,
    ).to(device)
    r = _run_forward(
        "M4-12",
        "modern",
        KH_V15.KANJI,
        bitmaps,
        p1,
        p2,
        kanji_radicals=KH_V15.KANJI_RADICALS,
        kanji_origin=KH_V15.KANJI_ORIGIN,
        radicals=KH_V15.RADICALS,
        origins=KH_V15.ORIGINS,
        beta=5.0,
        sigma_global=0.1,
        phase2_steps=500,
        phase3_steps=100,
        inter_layer_scale=0.1,
        device=device,
        seed=0,
    )
    return {
        "kanji": list(r.kanji),
        "layer0_cos": torch.tensor(r.layer0_cos, dtype=torch.float64),
        "layer1_cos": torch.tensor(r.layer1_cos, dtype=torch.float64),
        "layer2_cos": torch.tensor(r.layer2_cos, dtype=torch.float64),
    }


def main() -> None:
    fixture = generate()
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    torch.save(fixture, FIXTURE_PATH)
    print(f"wrote {FIXTURE_PATH}")
    for i, k in enumerate(fixture["kanji"]):
        l0 = float(fixture["layer0_cos"][i])
        l1 = float(fixture["layer1_cos"][i])
        l2 = float(fixture["layer2_cos"][i])
        print(f"  {k}: l0={l0:.6f} l1={l1:.6f} l2={l2:.6f}")


if __name__ == "__main__":
    main()
