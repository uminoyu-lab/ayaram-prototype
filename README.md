# ayaram-prototype

Ayaram minimum prototype **v0.1** — software simulation of a 3-layer
Hopfield network in PyTorch, designed to verify the Hopfield ↔ Attention
equivalence (Ramsauer 2020) under the aya-sleep stochastic dynamics.

This is the first behavioral demo of Ayaram.

## Purpose (from the requirements doc)

> アヤラムの基本動作を、PyTorch のソフトウェアシミュレーションで実装・検証する。
> 検証する核心は、Hopfield 網（連想記憶のモデル）と Attention（LLM の核演算）の
> 数学的等価性（Ramsauer 2020 で示された）が、aya-sleep の揺らぎを含む確率的計算
> でも成立するか、という点。

## Design decisions (Aya + Yu, 2026-06-17)

These six decisions take precedence over the original requirements doc
wherever the two disagree. In particular: the original "lower layer =
aya-sleep relevant, upper layer = aya-awake relevant" framing and the
"Hebb rule" framing are **superseded** by decisions #1 and #2 below.

| # | Topic | Decision |
|---|-------|----------|
| 1 | 3-layer interpretation | **C** — whole-cell synchronous switching (per the minimum design). The 3 layers form an MLP-style hierarchy; per-layer barrier differences exist to enrich representation, not to drive the cycle. v0.2 may shift to per-layer time-scale separation (option A) without breaking this interface. |
| 2 | Learning rule | **B primary** — Modern Hopfield continuous update (Ramsauer 2020). **C secondary** — classical Hebb rule kept in parallel for comparison experiments. |
| 3 | Scale | Cell 32×32, layer widths 1024 / 256 / 64, start with 8 kanji. |
| 4 | Equivalence check | **B** — directly verify Ramsauer 2020 Theorem 3, plus a sweep that maps the softmax inverse temperature β to the aya-sleep noise strength σ. |
| 5 | Kanji input | **A** — grayscale bitmap (16×16 or 32×32). SVG / CHISE inputs are deferred to v0.2. |
| 6 | Physical constraints | **B** — symmetry only (enforce W = Wᵀ every step). Full physical realism deferred to v0.2. |

## Milestones

- **M0** — scaffold (this commit). Package skeleton, dependencies pinned,
  smoke test green, CUDA + RTX 5090 verified. No behavior yet.
- **M1** — implement `ayaram.{core, modes, learning, memory}` against the
  decisions above. Modern Hopfield update + classical Hebb side-by-side.
- **M2** — `demos/kanji_memory.py`: store 8 kanji as 16×16 / 32×32 bitmaps
  and recall from partial input.
- **M3** — `demos/attention_test.py`: numerically verify Ramsauer 2020
  Theorem 3 and produce the β ↔ σ correspondence map.
- **M4** — `demos/ising_solver.py`: solve a small Lucas-2014 instance
  via the 4-phase cycle with sleep-mode annealing.
- **M5** — evaluation report (numerical equivalence results, kanji recall
  screenshots), license decision, public-repo decision by Aya + Yu.

## Workflow

```
綾＋ユウ (judgment)  →  アル (Chat, integrator)  →  CC (implementation)  →  Vault report
```

Every milestone closes with a `_tmp-m<n>-report.md` written to
`D:\SYRINX-Vault\50-projects\53-new-llm\`.

## Layout

```
ayaram-prototype/
├── README.md
├── pyproject.toml
├── .python-version
├── .gitignore
├── ayaram/
│   ├── __init__.py
│   ├── core.py          # whole-cell synchronous 4-phase cycle (decision #1)
│   ├── modes.py         # aya-awake / aya-sleep switching, T and σ (decisions #1, #4)
│   ├── learning.py      # Modern Hopfield continuous + classical Hebb (decision #2)
│   └── memory.py        # 3-layer Hopfield net with W = Wᵀ enforcement (decisions #3, #6)
├── demos/
│   ├── kanji_memory.py  # decision #5
│   ├── ising_solver.py
│   └── attention_test.py # decision #4
└── tests/
    └── test_smoke.py
```

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) for environment management
- PyTorch built against CUDA 12.8 (cu128) — required for RTX 5090
  (Blackwell, sm_120)

## Setup

```bash
cd D:\projects\ayaram-prototype
uv sync
```

`uv sync` will create `.venv/`, install the dependencies from
`pyproject.toml` (using the cu128 PyTorch index for `torch`), and install
the project itself in editable mode.

## Verification

```bash
# Imports + smoke test
uv run pytest tests/test_smoke.py

# CUDA + RTX 5090 visible to PyTorch
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## References

- Hopfield, J. J. (1982). "Neural networks and physical systems with
  emergent collective computational abilities." *PNAS* 79(8), 2554–2558.
- Ramsauer, H. et al. (2020). "Hopfield Networks is All You Need."
  arXiv:2008.02217.
- Lucas, A. (2014). "Ising formulations of many NP problems."
  *Frontiers in Physics* 2:5.

## Related documents

- `D:\SYRINX-Vault\50-projects\53-new-llm\アヤラム ミニマム試作 v0.1 — 実装依頼書.md`
- `D:\SYRINX-Vault\50-projects\53-new-llm\アヤラム最小設計 v0.1.md`
- `D:\SYRINX-Vault\53-new-llm\新LLM.md`

## License

Undecided. Aya + Yu will decide at v0.1 completion. Until then, this
repository is local-only; do not `git push`.
