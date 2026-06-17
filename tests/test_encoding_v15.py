"""Tests for the M4 v15 orthogonal (radical, count) encoder.

Locks in:

* Layout: dim layout matches the docstring contract (presence first,
  count_unary second).

* Unary count semantics: ``count_unary[r][k] = 1`` iff the kanji has
  ``count(r) >= k + 1``.

* Non-colinearity of the M3 degenerate cases (木 / 林 / 森 on the 木 axis,
  日 / 晶 on the 日 axis): pairwise cosines are < 1 strict, and each
  kanji's full active-dim set is unique.

* Modern Hopfield can break the count ties: querying with the canonical
  pattern of the largest kanji on a count axis picks itself out, not its
  siblings.
"""

from __future__ import annotations

import math

import torch

from ayaram import encoding
from data.kanji_hierarchy_v15 import (
    KANJI as KANJI_V15,
    KANJI_RADICALS as KR_V15,
    KANJI_ORIGIN as KO_V15,
    MAX_COUNT,
    ORIGINS as O_V15,
    RADICALS as R_V15,
)


def _encode_one(k: str) -> torch.Tensor:
    return encoding.encode_radical_count_v15(
        k, kanji_radicals=KR_V15, radicals=R_V15, max_count=MAX_COUNT
    )


def test_v15_dim_layout_presence_then_count_unary():
    R = len(R_V15)
    # 木: count(木) = 1 -> radical[0]=1, count_unary[木][0]=1
    p_ki = _encode_one("木")
    assert float(p_ki[0].item()) == 1.0          # radical 木
    assert float(p_ki[R + 0 * MAX_COUNT + 0].item()) == 1.0  # count_木_>=1
    # All other "useful" dims should be 0.
    assert float(p_ki[R + 0 * MAX_COUNT + 1].item()) == 0.0  # count_木_>=2
    assert float(p_ki[R + 0 * MAX_COUNT + 2].item()) == 0.0  # count_木_>=3
    for r in range(1, R):
        assert float(p_ki[r].item()) == 0.0
        for k in range(MAX_COUNT):
            assert float(p_ki[R + r * MAX_COUNT + k].item()) == 0.0


def test_v15_count_unary_monotone_for_森():
    """森 has count(木) = 3 -> all three count_unary[木][k] dims active."""
    p = _encode_one("森")
    R = len(R_V15)
    assert float(p[0].item()) == 1.0
    assert float(p[R + 0 * MAX_COUNT + 0].item()) == 1.0  # >=1
    assert float(p[R + 0 * MAX_COUNT + 1].item()) == 1.0  # >=2
    assert float(p[R + 0 * MAX_COUNT + 2].item()) == 1.0  # >=3


def test_v15_multi_radical_明():
    """明 = 1 × 日 + 1 × 月."""
    p = _encode_one("明")
    R = len(R_V15)
    assert float(p[R_V15.index("日")].item()) == 1.0
    assert float(p[R_V15.index("月")].item()) == 1.0
    assert float(p[R_V15.index("木")].item()) == 0.0
    # count >=1 for both 日 and 月
    assert float(p[R + R_V15.index("日") * MAX_COUNT + 0].item()) == 1.0
    assert float(p[R + R_V15.index("月") * MAX_COUNT + 0].item()) == 1.0


def test_v15_resolves_the_M3_colinearity_for_木_林_森():
    """The whole point of Option B: 木, 林, 森 are no longer parallel."""
    pk = _encode_one("木")
    pr = _encode_one("林")
    pm = _encode_one("森")
    # all three must be distinct
    assert not torch.equal(pk, pr)
    assert not torch.equal(pr, pm)
    assert not torch.equal(pk, pm)

    def cos(a, b):
        return float((a @ b / (a.norm() * b.norm())).item())

    # cos < 1 strict in all pairs
    assert cos(pk, pr) < 0.99
    assert cos(pr, pm) < 0.99
    assert cos(pk, pm) < 0.99
    # M3 would have given exact 1.0 here; M4 should give the analytic
    # cos = sqrt(2/3) for 木/林, sqrt(3/4) for 林/森, sqrt(2/4) for 木/森.
    assert math.isclose(cos(pk, pr), math.sqrt(2.0 / 3.0), rel_tol=1e-4)
    assert math.isclose(cos(pr, pm), math.sqrt(3.0 / 4.0), rel_tol=1e-4)
    assert math.isclose(cos(pk, pm), math.sqrt(2.0 / 4.0), rel_tol=1e-4)


def test_v15_modern_softmax_picks_largest_on_count_axis():
    """For the 木 axis (木 / 林 / 森), Modern's softmax with the 森 query
    must pick 森, not 林 or 木. This is what M3's parallel encoding
    explicitly could not do."""
    X = torch.stack([_encode_one(k) for k in KANJI_V15], dim=0).T  # (d, N)
    q_森 = _encode_one("森")
    scores = X.T @ q_森  # (N,)
    best = int(scores.argmax().item())
    assert KANJI_V15[best] == "森"

    q_林 = _encode_one("林")
    scores = X.T @ q_林
    best = int(scores.argmax().item())
    # 林 has score 3, 森 has score 3 (since 林 is subset of 森's dims) -- a
    # tie is OK, but neither should be beaten by an unrelated kanji.
    assert KANJI_V15[best] in {"林", "森"}


def test_v15_origin_encoder_one_hot_dim():
    p = encoding.encode_origin_v15(
        "川", kanji_origin=KO_V15, origins=O_V15,
    )
    assert p.shape == (encoding.LAYER2_DIM,)
    # 川 -> 地形 (last new category at index 3)
    assert float(p[3].item()) == 1.0
    # All other origin dims zero
    for j in range(len(O_V15)):
        if j != 3:
            assert float(p[j].item()) == 0.0


def test_v15_batch_encoders_match_per_item():
    p1 = encoding.encode_batch_radical_count_v15(
        KANJI_V15, kanji_radicals=KR_V15, radicals=R_V15, max_count=MAX_COUNT
    )
    p2 = encoding.encode_batch_origin_v15(
        KANJI_V15, kanji_origin=KO_V15, origins=O_V15,
    )
    assert p1.shape == (len(KANJI_V15), encoding.LAYER1_DIM)
    assert p2.shape == (len(KANJI_V15), encoding.LAYER2_DIM)
    for i, k in enumerate(KANJI_V15):
        assert torch.equal(
            p1[i],
            encoding.encode_radical_count_v15(
                k, kanji_radicals=KR_V15, radicals=R_V15, max_count=MAX_COUNT
            ),
        )
        assert torch.equal(
            p2[i],
            encoding.encode_origin_v15(
                k, kanji_origin=KO_V15, origins=O_V15
            ),
        )
