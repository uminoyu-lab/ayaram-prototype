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
