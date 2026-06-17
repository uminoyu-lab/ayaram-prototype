"""Generate the 8-kanji 32x32 bitmap dataset.

Design decision #3 + #5 + sub-decision (Aya + Yu + Aru, 2026-06-17):
    8 kanji: 人 木 口 川 火 山 日 月
    32x32 grayscale
    values normalized to [-1, +1] (background -1, foreground +1)

The script is checked in so the dataset is reproducible from source.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

KANJI: tuple[str, ...] = ("人", "木", "口", "川", "火", "山", "日", "月")
SIZE: int = 32
FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\NotoSansJP-VF.ttf",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
)
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "kanji_8_32x32.npy")


def _pick_font(point_size: int) -> ImageFont.FreeTypeFont:
    last_err: Exception | None = None
    for path in FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            return ImageFont.truetype(path, point_size)
        except OSError as e:
            last_err = e
            continue
    raise RuntimeError(
        "No usable Japanese font found among "
        + ", ".join(FONT_CANDIDATES)
        + (f"; last error: {last_err}" if last_err else "")
    )


def render_kanji(ch: str, size: int = SIZE, point_size: int = 26) -> np.ndarray:
    """Render a single kanji into a (size, size) float32 array in [-1, +1]."""
    font = _pick_font(point_size)
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), ch, fill=255, font=font)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return arr


def build_dataset() -> np.ndarray:
    out = np.stack([render_kanji(ch) for ch in KANJI], axis=0)
    assert out.shape == (len(KANJI), SIZE, SIZE)
    assert out.dtype == np.float32
    return out


def main(out_path: str = DEFAULT_OUT) -> None:
    data = build_dataset()
    np.save(out_path, data)
    print(f"Saved {data.shape} {data.dtype} -> {out_path}")
    print(f"value range: [{data.min():.3f}, {data.max():.3f}]")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    main(target)
