"""Layer-1 / layer-2 pattern encoders for the M3 hierarchy demo.

Aru M3 decision (2026-06-17): use the first ``len(RADICALS)`` dimensions of
layer 1 as radical activations (multi-hot, integer counts) and the first
``len(ORIGINS)`` dimensions of layer 2 as origin one-hots. Remaining
dimensions are zero. Random orthogonal embeddings are an alternative
explored in v0.2 if the direct encoding hits capacity issues.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

import torch
from torch import Tensor

# data/kanji_hierarchy.py lives at the project root; expose it as a top-level
# module by prepending the project root to sys.path. The repository is
# local-only (no install), so this is the simplest cross-module import.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.kanji_hierarchy import (  # noqa: E402
    KANJI,
    KANJI_ORIGIN,
    KANJI_RADICALS,
    ORIGINS,
    RADICALS,
)

LAYER1_DIM: int = 256
LAYER2_DIM: int = 64


def encode_radical(kanji: str, dim: int = LAYER1_DIM) -> Tensor:
    """Layer-1 multi-hot integer activation for one kanji.

    The first ``len(RADICALS)`` dims are the radicals in the canonical order;
    the remaining ``dim - len(RADICALS)`` dims are zero.
    """
    if kanji not in KANJI_RADICALS:
        raise KeyError(f"unknown kanji: {kanji}")
    p = torch.zeros(dim, dtype=torch.float32)
    for radical, count in KANJI_RADICALS[kanji].items():
        i = RADICALS.index(radical)
        p[i] = float(count)
    return p


def encode_origin(kanji: str, dim: int = LAYER2_DIM) -> Tensor:
    """Layer-2 one-hot for the origin category of one kanji."""
    if kanji not in KANJI_ORIGIN:
        raise KeyError(f"unknown kanji: {kanji}")
    p = torch.zeros(dim, dtype=torch.float32)
    i = ORIGINS.index(KANJI_ORIGIN[kanji])
    p[i] = 1.0
    return p


def encode_batch_radical(
    kanji_iter: Iterable[str] = KANJI, dim: int = LAYER1_DIM
) -> Tensor:
    return torch.stack([encode_radical(k, dim=dim) for k in kanji_iter])


def encode_batch_origin(
    kanji_iter: Iterable[str] = KANJI, dim: int = LAYER2_DIM
) -> Tensor:
    return torch.stack([encode_origin(k, dim=dim) for k in kanji_iter])


# -- decoders for evaluation / reporting -------------------------------------


def decode_radical(p: Tensor) -> dict[str, float]:
    """Return the (radical -> activation) dictionary read off the first few
    dims of a layer-1 state. Useful for inspecting recall output."""
    out: dict[str, float] = {}
    for i, r in enumerate(RADICALS):
        out[r] = float(p[i].item())
    return out


def decode_origin(p: Tensor) -> tuple[str, float]:
    """Return the (argmax origin, activation) read off the first 3 dims of a
    layer-2 state. Useful for one-hot accuracy reporting."""
    probs = p[: len(ORIGINS)]
    idx = int(probs.argmax().item())
    return ORIGINS[idx], float(probs[idx].item())


def origin_one_hot_match(p: Tensor, kanji: str) -> bool:
    """True if ``argmax(p[:3])`` equals the kanji's true origin index."""
    true_idx = ORIGINS.index(KANJI_ORIGIN[kanji])
    pred_idx = int(p[: len(ORIGINS)].argmax().item())
    return true_idx == pred_idx


def radical_set_match(p: Tensor, kanji: str, threshold: float = 0.5) -> float:
    """Fraction of the kanji's true radicals that pass ``threshold`` in ``p``.

    Returns a soft score in ``[0, 1]``: 1.0 means every radical the kanji has
    is active above threshold (regardless of magnitude). Useful for layer-1
    recall accuracy reporting.
    """
    target_radicals = KANJI_RADICALS[kanji].keys()
    if not target_radicals:
        return 1.0
    hits = 0
    for r in target_radicals:
        idx = RADICALS.index(r)
        if float(p[idx].item()) >= threshold:
            hits += 1
    return hits / len(target_radicals)


# ===========================================================================
# v15 (M4) -- orthogonal (radical multi-hot + count_unary) encoders.
# ===========================================================================
#
# Decision (Aru M4, 2026-06-17): Option B unary-count encoder. M3 used integer
# multi-hot (林 = [2, 0, 0, 0]) which made 木, 林, 森 colinear under Modern
# Hopfield's softmax: higher-magnitude patterns systematically stole the
# attention from the queried lower-magnitude one. Option B splits the encoding
# into a presence multi-hot plus a per-radical unary count block, which makes
# every kanji a distinct (no longer parallel) point in pattern space.
#
# Dim layout for ``encode_radical_count_v15``, given ``R`` radicals and
# ``max_count`` count levels:
#
#     dims 0 .. R - 1                           : radical multi-hot (presence)
#     dims R .. R + R*max_count - 1             : count_unary[r][k] = 1 iff
#                                                  count of radical r >= k+1
#
# Example with R = 7, max_count = 3:
#     木  (1 × 木)  -> radical[0]=1, count_unary[0][0]=1
#     林  (2 × 木)  -> radical[0]=1, count_unary[0][0..1]=1
#     森  (3 × 木)  -> radical[0]=1, count_unary[0][0..2]=1
#     晶  (3 × 日)  -> radical[1]=1, count_unary[1][0..2]=1
#     明  (1×日 + 1×月) -> radical[1]=1, radical[2]=1,
#                          count_unary[1][0]=1, count_unary[2][0]=1
#
# Pairwise cosines are no longer 1.0 even on count siblings:
#     cos(木, 林) = 2 / (sqrt(2) * sqrt(3)) ~= 0.816
#     cos(林, 森) = 3 / (sqrt(3) * sqrt(4)) ~= 0.866
# and Modern Hopfield's softmax can now break the tie on magnitude.


def encode_radical_count_v15(
    kanji: str,
    kanji_radicals: dict[str, dict[str, int]],
    radicals: tuple[str, ...],
    max_count: int = 3,
    dim: int = LAYER1_DIM,
) -> Tensor:
    """Option B orthogonal encoder for one kanji.

    Args:
        kanji:           the kanji character to encode.
        kanji_radicals:  dict mapping kanji -> {radical: count}.
        radicals:        tuple of radical strings (positional indexing).
        max_count:       max count level to encode unarily (3 for v0.1).
        dim:             layer-1 dim. Must be at least ``R + R*max_count``.

    Returns:
        ``(dim,)`` float32 tensor with values in ``{0, 1}``.
    """
    if kanji not in kanji_radicals:
        raise KeyError(f"unknown kanji: {kanji}")
    R = len(radicals)
    needed = R + R * max_count
    if dim < needed:
        raise ValueError(f"dim={dim} too small; need at least {needed}")
    p = torch.zeros(dim, dtype=torch.float32)
    for r_name, count in kanji_radicals[kanji].items():
        if r_name not in radicals:
            raise KeyError(f"unknown radical {r_name!r} for {kanji}")
        r_idx = radicals.index(r_name)
        p[r_idx] = 1.0
        for k in range(min(int(count), max_count)):
            p[R + r_idx * max_count + k] = 1.0
    return p


def encode_origin_v15(
    kanji: str,
    kanji_origin: dict[str, str],
    origins: tuple[str, ...],
    dim: int = LAYER2_DIM,
) -> Tensor:
    """v15 one-hot origin encoder. Same scheme as M3 but parameterized by
    the (kanji_origin, origins) dictionaries so the M4 hierarchy can supply
    them without M3 hard-coding."""
    if kanji not in kanji_origin:
        raise KeyError(f"unknown kanji: {kanji}")
    if dim < len(origins):
        raise ValueError(f"dim={dim} too small for {len(origins)} origins")
    p = torch.zeros(dim, dtype=torch.float32)
    i = origins.index(kanji_origin[kanji])
    p[i] = 1.0
    return p


def encode_batch_radical_count_v15(
    kanji_iter,
    kanji_radicals: dict[str, dict[str, int]],
    radicals: tuple[str, ...],
    max_count: int = 3,
    dim: int = LAYER1_DIM,
) -> Tensor:
    return torch.stack(
        [
            encode_radical_count_v15(
                k, kanji_radicals=kanji_radicals, radicals=radicals,
                max_count=max_count, dim=dim,
            )
            for k in kanji_iter
        ]
    )


def encode_batch_origin_v15(
    kanji_iter,
    kanji_origin: dict[str, str],
    origins: tuple[str, ...],
    dim: int = LAYER2_DIM,
) -> Tensor:
    return torch.stack(
        [
            encode_origin_v15(
                k, kanji_origin=kanji_origin, origins=origins, dim=dim,
            )
            for k in kanji_iter
        ]
    )


def origin_one_hot_match_v15(
    p: Tensor,
    kanji: str,
    kanji_origin: dict[str, str],
    origins: tuple[str, ...],
) -> bool:
    true_idx = origins.index(kanji_origin[kanji])
    pred_idx = int(p[: len(origins)].argmax().item())
    return true_idx == pred_idx


def radical_set_match_v15(
    p: Tensor,
    kanji: str,
    kanji_radicals: dict[str, dict[str, int]],
    radicals: tuple[str, ...],
    threshold: float = 0.5,
) -> float:
    target = kanji_radicals[kanji].keys()
    if not target:
        return 1.0
    hits = 0
    for r in target:
        idx = radicals.index(r)
        if float(p[idx].item()) >= threshold:
            hits += 1
    return hits / len(target)
