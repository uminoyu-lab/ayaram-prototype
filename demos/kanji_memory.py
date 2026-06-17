"""Demo: kanji associative recall.

Stores 8 simple kanji (decision #3, e.g. 一 二 三 川 水 火 山 日) as bitmap
patterns and recalls the full kanji from a partially occluded input.

Design decision #5 (Aya + Yu, 2026-06-17): inputs are grayscale bitmaps
(16 x 16 or 32 x 32), option A. SVG / CHISE-based inputs are a v0.2 follow-up,
not v0.1 scope.

M1: implement the storage / partial-recall / visualization loop. The
grayscale bitmaps will likely be rendered with cairosvg from glyph outlines
or, in the simplest case, hand-authored arrays.
"""


def main() -> None:
    """Entry point. M1 will populate this."""
    raise NotImplementedError("Kanji memory demo is planned for M1.")


if __name__ == "__main__":
    main()
