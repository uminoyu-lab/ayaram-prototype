"""Generate the M4 12-kanji 32x32 bitmap dataset.

Extends the M3 selection (木日月火林明炎晶) with four new kanji
森水川山 from `data/kanji_hierarchy_v15`.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from generate_kanji import render_kanji  # noqa: E402

# Single source of truth: pull the 12-kanji list from the v15 hierarchy.
from kanji_hierarchy_v15 import KANJI as KANJI_V15  # noqa: E402

SIZE: int = 32
DEFAULT_OUT = os.path.join(_HERE, "kanji_12_32x32_v15.npy")


def build_dataset() -> np.ndarray:
    out = np.stack([render_kanji(ch, size=SIZE) for ch in KANJI_V15], axis=0)
    assert out.shape == (len(KANJI_V15), SIZE, SIZE)
    assert out.dtype == np.float32
    return out


def main(out_path: str = DEFAULT_OUT) -> None:
    data = build_dataset()
    np.save(out_path, data)
    print(f"Saved {data.shape} {data.dtype} -> {out_path}")
    print(f"kanji order: {KANJI_V15}")
    print(f"value range: [{data.min():.3f}, {data.max():.3f}]")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    main(target)
