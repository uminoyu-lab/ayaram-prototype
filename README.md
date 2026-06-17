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
- **M1 — DONE** — implement `ayaram.{core, modes, learning, memory}`,
  Modern Hopfield + classical Hebb side by side, kanji associative recall,
  Ramsauer 2020 Theorem 3 verification, β ↔ σ correspondence map.
- **M2 — DONE** — `ayaram.ising` (Ising / MAX-CUT problem objects) and
  `demos/ising_solver.py` solving Lucas-2014 MAX-CUT via the 4-phase cycle
  on the classical Hebb side.
- **M3 — DONE** — `ayaram/encoding.py`, hierarchical Hebb learning
  (`HopfieldNetwork.learn`), Phase 1 `learn=True` variant, and
  `demos/hierarchical_kanji.py` for the forward (kanji → radical → origin)
  and reverse (radical → kanji) demos plus per-layer K_u dynamics.
- **M5 — planned** — evaluation report (numerical equivalence results,
  kanji recall screenshots, MAX-CUT statistics, M3 hierarchical limits),
  license decision, public-repo decision by Aya + Yu.

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
│   ├── memory.py        # 3-layer Hopfield net with W = Wᵀ enforcement (decisions #3, #6)
│   ├── ising.py         # MAX-CUT / Ising problem objects for the 4-phase cycle
│   └── encoding.py      # layer-1 radical + layer-2 origin encoders (M3)
├── data/
│   ├── __init__.py
│   ├── generate_kanji.py        # reproducible build of the M1 kanji bitmap dataset
│   ├── kanji_8_32x32.npy        # M1: 人木口川火山日月  (8, 32, 32) float32 [-1, +1]
│   ├── kanji_hierarchy.py       # M3 radical / origin dictionaries
│   ├── generate_kanji_v2.py     # M3 bitmap generator (re-uses generate_kanji.render_kanji)
│   └── kanji_8_32x32_v2.npy     # M3: 木日月火林明炎晶  (8, 32, 32) float32 [-1, +1]
├── demos/
│   ├── kanji_memory.py        # decision #5
│   ├── attention_test.py      # decision #4
│   ├── ising_solver.py        # Lucas-2014 MAX-CUT via Hebb-mode 4-phase cycle
│   ├── hierarchical_kanji.py  # M3 forward / reverse / dynamics demo
│   └── output/                # generated artifacts (gitignored)
└── tests/
    ├── test_smoke.py
    ├── test_modes.py
    ├── test_learning.py
    ├── test_memory.py
    ├── test_core.py
    ├── test_ising.py
    └── test_hierarchical.py
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

## v0.2 への宿題（M2 時点で確定）

1. **物理層対応の再設計**（M1 疑問 #1、M2 で A 案を追認）
   - v0.1 では `CycleConfig.sigma_global` を「layer-0 での実効ノイズ std」と
     工学的に解釈し、他層は `modes.layer_noise_ratio(l) = sqrt(K_u_0 / K_u_l)`
     でスケールダウン。物理 `sigma_local(l, T) = sqrt(T/K_u_l)` は記録用に
     保持するのみで、`run_cycle` の数値には乗っていない。
   - v0.2 では `sigma_global * sigma_local(l, T_global)` を物理通りに掛け、
     温度 `T_global` 自体を sweep 軸として扱う。`AWAKE` / `SLEEP` の T 値
     から固定の二点で動かしているのも、連続スイープに切り替える。
   - これは mumax3 連携の前提条件。
2. **β 依存性の容量極限実験**（M1 疑問 #2）
   - 類似字形（一・二・三、口・日・目 等）を含む漢字セットで Modern
     Hopfield の β 依存性を観察。
   - 漢字数を増やして指数容量（Ramsauer 2020 Theorem 2 系）の限界点を
     探索し、N が増えるにつれ β-σ 相転移しきいがどう動くかを記録。
3. **複数 seed 化による robustness 主張**（M1 疑問 #3）
   - M5 評価レポート時、`demos/kanji_memory.py` を複数 seed（≥ 10）で
     再測定。
   - v0.1 で見えた「Modern cos 1.000、Hebb cos 0.903」を統計量として
     報告できる形にする。
4. **MAX-CUT 以外の Lucas-2014 マッピング**（M2 で意図的に v0.1 から除外）
   - TSP、3-SAT、グラフ彩色などを `ayaram.ising` の同じ枠組みで実装し、
     4-phase cycle の汎用性を示す。
   - 必要なら ``IsingProblem`` の subclass を増やすだけで足りる API に
     なっているかを v0.2 着手時に再確認。
5. **MAX-CUT 焼き鈍しスケジュール**（M2 疑問 #3）
   - Phase 2 内で σ を時間で減衰させる焼き鈍し（temperature schedule）を
     組み込み、N=32 以上の MAX-CUT で M2 が頭打ちした「lift over random
     1.25×」を改善する。
   - 現状の 4-phase cycle に新たな「sweep」サブフェーズを足すか、Phase 2
     の σ パラメータを step 関数化するかは v0.2 着手時の設計判断。
6. **CHISE 自動部首分解 + 階層表現の見直し**（M3 で手作り辞書 + 整数
   multi-hot を採用した結果、複数の限界が見えた）
   - M3 では `data/kanji_hierarchy.py` の手作り辞書を使ったが、v0.2 では
     CHISE（漢字構造データベース）経由の自動分解に置き換える。意符＋音符
     の二軸を扱う「漢字最小元素方針」に乗せる。
   - 同時に、整数 multi-hot 表現（林=[2,0,0,0]、晶=[0,3,0,0] 等）が
     Modern Hopfield の softmax で **高 magnitude colinear pattern が
     attention を奪い、低 magnitude pattern が想起されない** という基本
     的な縮退を引き起こすことを M3 で実証（`_tmp-m3-report.md` 参照）。
     v0.2 では (a) 直交埋め込み（(radical, count) 毎に独立 dim）、
     (b) 学習時の pattern norm 正規化、いずれかへの移行が必要。
7. **層 0 の bipolar bitmap が層間 Hebb 学習を背景ノイズで汚染する問題**
   （M3 で実証、`_tmp-m3-report.md` 参照）
   - `{-1, +1}` 表現では背景 (-1) 同士の内積が信号同士の内積を凌駕し、
     層 1 への inter-layer 信号が「全パターン平均（M3 では天体に重い）」
     に流される。M3 デモでは inter weights を spectral norm 1 に再正規化
     する CC 工夫で部分緩和したが、本質解決には至っていない。
   - v0.2 候補：
     (a) 学習時のみ `{0, 1}` 表現で W_inter を計算し、状態は `{-1, +1}`
         のまま運用するハイブリッド。
     (b) 学習時に各パターンを zero-center してから outer product を取る。
     (c) 物理層対応 (v0.2 宿題 #1) と合わせて、MTJ の double-well を
         素直に反映した連続値表現に移行する。

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
