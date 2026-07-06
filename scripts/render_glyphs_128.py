"""Render the v0.2 24-kanji 128x128 glyph dataset for the scale-space organ.

依頼書 確定事項 (2026-07-03):
  * 24 kanji = existing 12 (木日月火林明炎晶森水川山)
                + new 12 (一十口田回国岩品語銀樹鬱)
  * render at 512x512 with anti-aliasing, then area/box downscale to 128x128
  * ink = 1.0 (foreground), background = 0.0, float32 in [0, 1]
  * font = same family as v0.1 (NotoSansJP-VF first); recorded in metadata

Outputs (data/glyphs_128/):
  * glyphs_128.npz      — arrays{'glyphs': (24,128,128) f32, plus per-char keys}
  * metadata.json       — font, render params, char list + codepoints, env, QC
  * contact_sheet.png   — 24-glyph montage for visual QC

欠字 (font non-coverage) is detected by comparing each rendered glyph to the
font's .notdef rendering (a private-use codepoint) and to the empty image; any
match is flagged and, per the 依頼書 STOP rule, the caller must halt.

Only *adds* data; v0.1/v0.1.5 datasets are untouched (依頼書 §6-1).
"""

from __future__ import annotations

import json
import os
import platform
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_ROOT, "data", "glyphs_128")

# --- kanji set (order fixed: existing 12 then new 12) ------------------------
KANJI_EXISTING: tuple[str, ...] = tuple("木日月火林明炎晶森水川山")
KANJI_NEW: tuple[str, ...] = tuple("一十口田回国岩品語銀樹鬱")
KANJI: tuple[str, ...] = KANJI_EXISTING + KANJI_NEW

# --- render params (アル起案 §7-4) ------------------------------------------
SUPERSAMPLE: int = 512
TARGET: int = 128
POINT_SIZE: int = 384         # ~0.75 * 512, centred by tight bbox
NOTDEF_CODEPOINT: str = ""  # private-use: reliably absent -> .notdef

# same font family as v0.1 (data/generate_kanji.py), NotoSansJP-VF first
FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\NotoSansJP-VF.ttf",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
)


def pick_font(point_size: int) -> tuple[ImageFont.FreeTypeFont, str]:
    last_err: Exception | None = None
    for path in FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            return ImageFont.truetype(path, point_size), path
        except OSError as e:
            last_err = e
    raise RuntimeError(
        "No usable Japanese font among " + ", ".join(FONT_CANDIDATES)
        + (f"; last error: {last_err}" if last_err else "")
    )


def _render_supersampled(ch: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Grayscale (L) 512x512, glyph centred by tight bbox, ink=255 on 0."""
    img = Image.new("L", (SUPERSAMPLE, SUPERSAMPLE), color=0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (SUPERSAMPLE - w) // 2 - bbox[0]
    y = (SUPERSAMPLE - h) // 2 - bbox[1]
    draw.text((x, y), ch, fill=255, font=font)
    return img


def render_glyph(ch: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Return (128,128) float32 in [0,1], ink=1.0 foreground, via BOX downscale."""
    hi = _render_supersampled(ch, font)
    lo = hi.resize((TARGET, TARGET), resample=Image.BOX)  # area averaging
    return (np.asarray(lo, dtype=np.float32) / 255.0)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    font, font_path = pick_font(POINT_SIZE)

    notdef = render_glyph(NOTDEF_CODEPOINT, font)  # font's missing-glyph render
    notdef_ink = float((notdef > 0.5).mean())

    glyphs = np.stack([render_glyph(ch, font) for ch in KANJI], axis=0)
    assert glyphs.shape == (len(KANJI), TARGET, TARGET)
    assert glyphs.dtype == np.float32

    # --- QC -----------------------------------------------------------------
    qc_rows = []
    missing: list[str] = []
    for i, ch in enumerate(KANJI):
        g = glyphs[i]
        ink = float((g > 0.5).mean())          # fraction of foreground pixels
        empty = bool(g.max() < 0.5)
        # 欠字: matches the font's .notdef render (tofu) or is blank
        looks_notdef = bool(np.array_equal(g > 0.5, notdef > 0.5) and notdef_ink > 0)
        is_missing = empty or looks_notdef
        if is_missing:
            missing.append(ch)
        qc_rows.append({
            "char": ch,
            "codepoint": f"U+{ord(ch):04X}",
            "ink_rate": round(ink, 4),
            "empty": empty,
            "looks_notdef": looks_notdef,
            "ink_in_1_50pct": bool(0.01 <= ink <= 0.50),
        })

    out_of_range = [r["char"] for r in qc_rows if not r["ink_in_1_50pct"] and r["char"] not in missing]

    metadata = {
        "version": "v0.2-M0",
        "n_glyphs": len(KANJI),
        "chars": list(KANJI),
        "chars_existing": list(KANJI_EXISTING),
        "chars_new": list(KANJI_NEW),
        "codepoints": [f"U+{ord(c):04X}" for c in KANJI],
        "size": TARGET,
        "supersample": SUPERSAMPLE,
        "downscale": "PIL.Image.BOX (area averaging)",
        "point_size": POINT_SIZE,
        "ink_value": 1.0,
        "background_value": 0.0,
        "dtype": "float32",
        "value_range": [float(glyphs.min()), float(glyphs.max())],
        "font_path": font_path,
        "font_candidates": list(FONT_CANDIDATES),
        "notdef_codepoint": "U+E000",
        "notdef_ink_rate": round(notdef_ink, 4),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        "qc": {
            "missing_glyphs": missing,
            "ink_out_of_1_50pct": out_of_range,
            "rows": qc_rows,
            "pass": (len(missing) == 0),
        },
    }

    npz_path = os.path.join(OUT_DIR, "glyphs_128.npz")
    named = {"glyphs": glyphs}
    for i, ch in enumerate(KANJI):
        named[f"U{ord(ch):04X}"] = glyphs[i]
    np.savez_compressed(npz_path, **named)

    meta_path = os.path.join(OUT_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    _contact_sheet(glyphs, os.path.join(OUT_DIR, "contact_sheet.png"))

    print(f"font: {font_path}")
    print(f"saved {glyphs.shape} {glyphs.dtype} -> {npz_path}")
    print(f"value range: [{glyphs.min():.3f}, {glyphs.max():.3f}]")
    print("ink rates: " + ", ".join(f"{c}={r['ink_rate']:.3f}" for c, r in zip(KANJI, qc_rows)))
    if missing:
        print(f"!! 欠字 (STOP): {missing}")
        return 2
    if out_of_range:
        print(f"!! ink rate outside [1%,50%] (visual check): {out_of_range}")
    print("QC pass" if metadata["qc"]["pass"] else "QC FAIL")
    return 0


def _contact_sheet(glyphs: np.ndarray, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk_prop = None
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
                cjk_prop = font_manager.FontProperties(fname=fp)
            except (RuntimeError, OSError):
                cjk_prop = None
            break

    n = glyphs.shape[0]
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(glyphs[i], cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(f"{KANJI[i]} U+{ord(KANJI[i]):04X}", fontsize=7,
                         fontproperties=cjk_prop)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
