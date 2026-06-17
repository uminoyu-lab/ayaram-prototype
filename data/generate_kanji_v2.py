"""Generate the M3 8-kanji 32x32 bitmap dataset.

Decision (Aru M3, 2026-06-17): the M1 selection (人木口川火山日月) is
superseded by 木日月火林明炎晶 so that the radical / origin hierarchy is
non-trivial (4 single-radical kanji + 4 composite kanji).

The M1 dataset is intentionally kept on disk (no edit to ``generate_kanji.py``
or ``kanji_8_32x32.npy``) so M1 demos remain reproducible.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from generate_kanji import render_kanji  # noqa: E402

# Import the canonical kanji list from the hierarchy module (single source of
# truth).
from kanji_hierarchy import KANJI as KANJI_V2  # noqa: E402

SIZE: int = 32
DEFAULT_OUT = os.path.join(_HERE, "kanji_8_32x32_v2.npy")


def build_dataset() -> np.ndarray:
    out = np.stack([render_kanji(ch, size=SIZE) for ch in KANJI_V2], axis=0)
    assert out.shape == (len(KANJI_V2), SIZE, SIZE)
    assert out.dtype == np.float32
    return out


def main(out_path: str = DEFAULT_OUT) -> None:
    data = build_dataset()
    np.save(out_path, data)
    print(f"Saved {data.shape} {data.dtype} -> {out_path}")
    print(f"kanji order: {KANJI_V2}")
    print(f"value range: [{data.min():.3f}, {data.max():.3f}]")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    main(target)
