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
- **M4 — DONE** — Option B orthogonal `(radical, count)` encoding
  (`encode_radical_count_v15`), `HopfieldNetwork.learn(normalize_inter=...)`
  formalization, 12-kanji expanded set (`data/kanji_hierarchy_v15.py`,
  added 森・水・川・山, new origin 地形), `demos/hierarchical_kanji_v15.py`
  with M3 vs M4(8) vs M4(12) bar-chart comparison. Layer-2 origin recall
  improved from M3 4/8 to M4-12 10/12 (83 %).
- **M5 — planned** — evaluation report (numerical equivalence results,
  kanji recall screenshots, MAX-CUT statistics, M3.5 hierarchical results),
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
│   ├── generate_kanji.py            # reproducible build of the M1 kanji bitmap dataset
│   ├── kanji_8_32x32.npy            # M1: 人木口川火山日月  (8, 32, 32) float32 [-1, +1]
│   ├── kanji_hierarchy.py           # M3 radical / origin dictionaries
│   ├── generate_kanji_v2.py         # M3 bitmap generator
│   ├── kanji_8_32x32_v2.npy         # M3: 木日月火林明炎晶  (8, 32, 32) float32 [-1, +1]
│   ├── kanji_hierarchy_v15.py       # M4 extended dictionaries (12 kanji, 7 radicals, 4 origins)
│   ├── generate_kanji_v15.py        # M4 bitmap generator
│   └── kanji_12_32x32_v15.npy       # M4: + 森 水 川 山 (12, 32, 32) float32 [-1, +1]
├── demos/
│   ├── kanji_memory.py             # decision #5
│   ├── attention_test.py           # decision #4
│   ├── ising_solver.py             # Lucas-2014 MAX-CUT via Hebb-mode 4-phase cycle
│   ├── hierarchical_kanji.py       # M3 forward / reverse / dynamics demo
│   ├── hierarchical_kanji_v15.py   # M4 forward + M3 vs M4 comparison
│   └── output/                     # generated artifacts (gitignored)
└── tests/
    ├── test_smoke.py
    ├── test_modes.py
    ├── test_learning.py
    ├── test_memory.py
    ├── test_core.py
    ├── test_ising.py
    ├── test_hierarchical.py
    ├── test_encoding_v15.py
    └── test_learn_spectral.py
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

## v0.2 への宿題（M4 時点で更新）

M4 で fix 扱いとなった項目：
- **Inter-layer weight の spectral 正規化**：M3 で CC 工夫として導入したものを
  M4 で `HopfieldNetwork.learn(normalize_inter='spectral')` のオプションとして
  正式化（デフォルト）。旧挙動を再現したいときは `normalize_inter='none'`。
- **整数 multi-hot 同軸縮退**：M4 で **Option B（直交 radical + 部首毎
  unary count）encoding** を導入し、Modern Hopfield の softmax が高
  magnitude colinear pattern に attention を奪われる縮退を解消（`林` と
  `木` が別ベクトル空間方向に分離、テスト `test_v15_*` で断言）。

M4 で部分着手、v0.2 で物理層と統合する項目：
- **層 0 bipolar bitmap の背景バイアス**：M4 demo（`demos/hierarchical_kanji_v15.py`）
  で **入力 zero-centering を inter-layer Hebb 学習側にだけ適用**（状態 alphabet は
  `{-1, +1}` のまま）する CC 工夫で、layer-2 字源 origin 一致を M3 の 4/8 から
  M4 12 字版で **10/12 = 83%** に改善。ただし `HopfieldNetwork.learn` 本体には
  まだ組込んでいないので、v0.2 で正式化 + 物理層対応との整合を取る。

残る v0.2 宿題：

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
6. **CHISE 自動部首分解 + 階層表現の更なる見直し**（M4 では手作り辞書
   `data/kanji_hierarchy_v15.py` で十分機能した）
   - 意符＋音符の二軸を扱う「漢字最小元素方針」を CHISE 経由で自動化、
     v0.1 の手作り 12 字を ≥ 100 字に拡張。
7. **`{0, 1}` alphabet と zero-centering の選択**（M3 で {0, 1} を試行
   して層 0 Hebb が劣化、M4 で zero-centering に変更して成功）
   - v0.2 では物理層対応と合わせて、状態 alphabet を MTJ の double-well
     を素直に反映した連続値（`tanh` の像）に統一するのが筋。
8. **逆方向想起**（M3 で実質動かず、M4 で意図的に v0.1 から除外）
   - 層 1 部首パターンを入力 → 層 0 で対応漢字が想起される、という方向を
     v0.2 で実装。M4 の Option B encoding + zero-centering の組合せが
     逆方向にも効くかの検証も兼ねる。
9. **Hebb モードの zero-centered 学習耐性**（M4 で観察、Modern は改善、
   Hebb は劣化）
   - M4 の `_learn_with_centered_inter` を Hebb モードで使うと `tanh(βWξ)`
     の saturation 点が崩れ、recall が反転する場合がある。Modern と Hebb
     で同じ centering を共用するのが本当に正しいか、別経路にすべきかを
     v0.2 で再検討。

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
