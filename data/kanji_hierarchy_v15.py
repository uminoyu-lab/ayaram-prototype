"""M4 expanded hierarchy: 12 kanji, 7 radicals, 4 origins.

Decision (Aru M4, 2026-06-17): keep the M3 set and add four more to test
(a) the new orthogonal (radical, count) encoding under count = 3 (森 = 3×木
adds count-3 outside the 晶 = 3×日 case from M3) and (b) Hopfield capacity
when the radical alphabet and origin categories grow.

Selection rationale (CC):

  * Carry the M3 8 (木 日 月 火 林 明 炎 晶) unchanged so the M3-vs-M4
    head-to-head at the same kanji set is clean.

  * 森 = 3 × 木 (植物). Mirrors 晶 = 3 × 日 (M3 already has this on the
    日 axis); having a count-3 case on both 木 and 日 axes lets us check
    the Option B unary-count encoding under two different radicals.

  * 水 (元素). Single-radical, natural Five-Phases element. Adds a new
    radical without bringing in a new origin category.

  * 川 (地形) and 山 (地形). Single-radical kanji that naturally form a
    fourth origin category 地形 (terrain). Lets M4 demonstrate that the
    origin layer can host more than three categories without losing recall
    accuracy.

Result: 12 kanji, 7 radicals (木 日 月 火 水 川 山), 4 origins (植物 / 天体
/ 元素 / 地形). Origin distribution: 3 / 4 / 3 / 2.
"""

from __future__ import annotations

KANJI: tuple[str, ...] = (
    # M3 carry-over (8)
    "木", "日", "月", "火",
    "林", "明", "炎", "晶",
    # M4 new (4)
    "森", "水", "川", "山",
)

RADICALS: tuple[str, ...] = ("木", "日", "月", "火", "水", "川", "山")

ORIGINS: tuple[str, ...] = ("植物", "天体", "元素", "地形")

# Multi-hot integer counts -- same semantics as M3 (data/kanji_hierarchy.py)
# but extended. The Option B encoder in ayaram.encoding turns these into
# orthogonal (radical multi-hot, count_unary) vectors.
KANJI_RADICALS: dict[str, dict[str, int]] = {
    # M3 8
    "木": {"木": 1},
    "日": {"日": 1},
    "月": {"月": 1},
    "火": {"火": 1},
    "林": {"木": 2},
    "明": {"日": 1, "月": 1},
    "炎": {"火": 2},
    "晶": {"日": 3},
    # M4 new
    "森": {"木": 3},
    "水": {"水": 1},
    "川": {"川": 1},
    "山": {"山": 1},
}

KANJI_ORIGIN: dict[str, str] = {
    "木": "植物", "林": "植物", "森": "植物",
    "日": "天体", "月": "天体", "明": "天体", "晶": "天体",
    "火": "元素", "炎": "元素", "水": "元素",
    "川": "地形", "山": "地形",
}

RADICAL_ORIGIN: dict[str, str] = {
    "木": "植物",
    "日": "天体",
    "月": "天体",
    "火": "元素",
    "水": "元素",
    "川": "地形",
    "山": "地形",
}

# Max count across all kanji -- used to size the count_unary block of the
# Option B encoder. 森 and 晶 are the count-3 cases.
MAX_COUNT: int = 3


def validate() -> None:
    assert set(KANJI_RADICALS.keys()) == set(KANJI), "KANJI_RADICALS keys mismatch"
    assert set(KANJI_ORIGIN.keys()) == set(KANJI), "KANJI_ORIGIN keys mismatch"
    assert set(RADICAL_ORIGIN.keys()) == set(RADICALS), "RADICAL_ORIGIN keys mismatch"
    for k, radicals in KANJI_RADICALS.items():
        for r, c in radicals.items():
            assert r in RADICALS, f"unknown radical {r} in {k}"
            assert 1 <= c <= MAX_COUNT, f"count {c} for {k}:{r} out of range"
    for k, origin in KANJI_ORIGIN.items():
        assert origin in ORIGINS, f"unknown origin {origin} for {k}"


if __name__ == "__main__":
    validate()
    print(f"kanji ({len(KANJI)}): {KANJI}")
    print(f"radicals ({len(RADICALS)}): {RADICALS}")
    print(f"origins ({len(ORIGINS)}): {ORIGINS}")
    print(f"max count: {MAX_COUNT}")
    print("validation passed")
