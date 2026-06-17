"""Hand-built radical / origin hierarchy for the M3 demo.

Source of truth for the M3 hierarchical recall experiment. CHISE-based
automatic decomposition is v0.2 (see README "v0.2 への宿題").

Layer correspondences (Aru M3 decisions, 2026-06-17):
    Layer 0 (1024 dim) -- 32x32 bitmap of one of 8 kanji
    Layer 1 (256 dim)  -- multi-hot integer activation over 4 radicals;
                          the first 4 dimensions are the radicals in the
                          order [木, 日, 月, 火].
    Layer 2 (64 dim)   -- one-hot over 3 origin categories;
                          the first 3 dimensions are [植物, 天体, 元素].
"""

from __future__ import annotations

KANJI: tuple[str, ...] = ("木", "日", "月", "火", "林", "明", "炎", "晶")

RADICALS: tuple[str, ...] = ("木", "日", "月", "火")
ORIGINS: tuple[str, ...] = ("植物", "天体", "元素")

# Aru M3 spec: integer multi-hot, count = how many copies of the radical the
# kanji contains. 木 and 林 then share the 木-axis direction (林 having
# magnitude 2 along it); that colinearity is intentional -- it is the
# "hierarchical abstraction" that Layer 1 is supposed to capture.
KANJI_RADICALS: dict[str, dict[str, int]] = {
    "木": {"木": 1},
    "日": {"日": 1},
    "月": {"月": 1},
    "火": {"火": 1},
    "林": {"木": 2},
    "明": {"日": 1, "月": 1},
    "炎": {"火": 2},
    "晶": {"日": 3},
}

KANJI_ORIGIN: dict[str, str] = {
    "木": "植物",
    "林": "植物",
    "日": "天体",
    "月": "天体",
    "明": "天体",
    "晶": "天体",
    "火": "元素",
    "炎": "元素",
}

RADICAL_ORIGIN: dict[str, str] = {
    "木": "植物",
    "日": "天体",
    "月": "天体",
    "火": "元素",
}


def validate() -> None:
    """Sanity check that the dictionaries are internally consistent."""
    assert set(KANJI_RADICALS.keys()) == set(KANJI), "KANJI_RADICALS keys mismatch"
    assert set(KANJI_ORIGIN.keys()) == set(KANJI), "KANJI_ORIGIN keys mismatch"
    assert set(RADICAL_ORIGIN.keys()) == set(RADICALS), "RADICAL_ORIGIN keys mismatch"
    for k, radicals in KANJI_RADICALS.items():
        for r in radicals:
            assert r in RADICALS, f"unknown radical {r} in {k}"
    for k, origin in KANJI_ORIGIN.items():
        assert origin in ORIGINS, f"unknown origin {origin} for {k}"


if __name__ == "__main__":
    validate()
    print("kanji:", KANJI)
    print("radicals:", RADICALS)
    print("origins:", ORIGINS)
    print("validation passed")
