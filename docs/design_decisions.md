# Design Decisions — Ayaram Prototype v0.1

各マイルストーンで綾＋ユウが確定した判断と、CC（実装側）で取った解釈・
逸脱を 1 か所にまとめた付録。M5 評価レポートの参照元。

凡例：
- **採用** … 綾＋ユウ判断、CC は実装するのみ
- **CC 解釈** … 仕様内裁量で CC が取った設計判断（明示）
- **CC 工夫** … 仕様逸脱と CC が判定した変更（docstring + 報告書で明示）

## M0（2026-06-17、依頼書受領）— v0.1 設計判断 6 項目

| # | 論点 | 採用 |
|---|---|---|
| 1 | 3 レイヤーの解釈 | **C**：全セル同期切替（最小設計通り）。3 レイヤーは MLP 的階層化、バリア違いは表現の豊かさのための仕掛け。v0.2 で時間スケール分離（A）への移行を妨げない設計 |
| 2 | 学習則 | **B 主軸**：Modern Hopfield の continuous update（Ramsauer 2020）。**C 補助**：古典ヘブ則も比較実験用に並走 |
| 3 | スケール | セル 32×32、層 1024 / 256 / 64、漢字 8 字から開始 |
| 4 | 等価性検証 | **B**：Ramsauer 2020 Theorem 3 直接検証 + softmax 温度 β と aya-sleep ノイズ強度 σ の対応マップ |
| 5 | 漢字入力 | **A**：bitmap グレースケール（16×16 か 32×32）。v0.2 で SVG / CHISE へ |
| 6 | 物理制約 | **B**：対称性のみ（W = Wᵀ を毎ステップ強制）。v0.2 で完全制約へ |

## M1（実装）— サブ判断

### CC 解釈

- **σ_global を「layer 0 実効ノイズ std」として再定義**：物理式
  `sigma_local(l, T) = sqrt(T / K_u_l)` は記録用に保持しつつ、`run_cycle` の
  数値には `sigma_global * layer_noise_ratio(l)` を掛ける。これで Part B の
  β-σ sweep が意味のあるレンジで動く（K_u を直接掛けると σ=10 でも実効
  ノイズ ≪ 1 に潰れる）。
- **`modes.layer_noise_ratio(l) = sqrt(K_u_0 / K_u_l)`**：層 0 を 1.0 基準、
  上層は K_u 比でスケールダウン。
- **層別 batched recall**：M1 `attention_test.py` Part B は kanji を batch
  dim で並列化（30 倍高速、Theorem 3 とは無関係）。

### M1 で見えたこと

- Theorem 3 が **bit-exact** で一致（max abs diff = 0.0e+00）。
- σ ≈ 1.5 で recall 相転移、それ以下は完全再生、以上は 0.85 床。
- β 依存性なし（8 字が十分離れているため）。

## M2（MAX-CUT）— 判断

### 採用

- N ∈ {8, 16, 32}、Erdős–Rényi p = 0.5、10 試行 / N、σ = 1.0、古典ヘブ側のみ。
- 結果：N=8 平均近似比 1.000、N=16 = 0.986、N=32 lift over random 1.25×。

### CC 解釈

- **MaxCutProblem.J = −adj**（反強磁性結合）：Hopfield energy
  E = −(1/2) sᵀ W s と Ising H(s) を一致させる規約として明示。
- **HopfieldNetwork の最小レイヤー数を 2 → 1 に緩和**：MAX-CUT は単層 N 個
  ノードで使うため。kanji 3 層構造を壊さない範囲の緩和。
- **networkx は使わず matplotlib のみで graph viz**：仕様「依存パッケージを
  勝手に追加しない」を踏まえ、循環レイアウト + 辺色塗りを自作。

### M2 CC 所見（綾＋ユウから 1 ずんだ）

「Modern Hopfield のイジング最適化への自然な拡張は v0.1 では困難。Modern は
`X · softmax(β Xᵀ ξ)` という**stored pattern 集合の凸結合の最近傍を返す
operator** であり、組合せ最適化の dynamics ではない。Lucas-2014 他問題を
Modern に乗せるなら、stored pattern X として候補集合を持たせるパラダイム
転換が必要。」

→ v0.2 宿題 #4 として README に明示。

## M3（階層連想）— 判断

### 採用

- (b) 階層連想（漢字 → 部首 → 字源）を M3 主軸、(c) 時間スケール分離は v0.2 へ。
- inter-layer 重み学習は (α) ヘブ則拡張で **Phase 1 に組込**。
- 漢字 8 字：木 日 月 火 林 明 炎 晶。部首 4、字源 3。
- 整数 multi-hot：林 = `[2, 0, 0, 0]`、晶 = `[0, 3, 0, 0]` 等。

### CC 工夫（仕様逸脱）

- **W_inter spectral 正規化（demo helper `_normalize_inter_weights`）**：
  W_01 と W_12 の spectral norm が ~30 倍非対称で、同じ inter_layer_scale を
  両方に適用すると 0-1 pair が divergence、1-2 pair が silence。spectral norm
  1 に揃える正規化を demo 内で実施。**M4 で `learn(normalize_inter='spectral')`
  として本体組込済**。

### M3 で見えた問題

- Layer 2 字源 origin one-hot 一致 = **4/8 = 50 %**（DoD 0.9 未達）。
- 原因：(1) 整数 multi-hot の同軸縮退 + (2) bipolar bitmap 背景バイアスの連鎖。

### CC v0.2 候補

- v0.2 候補 #6：直交 (radical, count) 埋め込みで縮退解消
- v0.2 候補 #7-(b)：学習時 zero-centering で背景バイアス除去

これらが M4・M5 で順次実装される伏線。

## M4（直交 encoding + 字数拡張）— 判断

### 採用

- (c) **両方やる**：前半 encoding 改善、後半 12〜16 字。
- Option B unary count encoding（綾推奨、CC 推奨と一致）。
- M3 の 8 字を維持 + 森 水 川 山 → 計 12 字、新 radical 3 + 新 origin 地形。
- spectral 正規化を `learn(normalize_inter='spectral')` で正式化（M3 で CC 工夫
  だったものを本体組込、デフォルト）。

### CC 工夫（仕様逸脱、宣言）

- **`demos/hierarchical_kanji_v15.py` 内に `_learn_with_centered_inter`
  helper**：M3 v0.2 候補 #7-(b) の zero-centering を demo 側で先取り実装。
  「{0, 1} alphabet は v0.2 へ」の禁止項目とは別軸（状態 alphabet は
  `{-1, +1}` のまま、`W_inter` 計算時のみ centering）であることを明示。
  **M5 で `learn(center_inter_inputs=True)` として本体組込済**。

### M4 で見えたこと

- Layer 2 字源 origin = **10/12 = 83 %**（DoD 0.7 余裕クリア）。
- 失敗 2 字は地形（最小 class 2 字）で、majority bias が attractor pull に
  直接効くという観察。
- Hebb モードで centering は degradation（v0.2 宿題 #9）。

### M4 CC 所見（綾＋ユウから 2 ずんだ）

「M3 v0.2 候補 #7-(b) の zero-centering を M4 で demo helper として先取り
実装、仕様逸脱を誠実に明示。これが layer 2 字源 50% → 83% という新 LLM の
本質的進展に直結。」

## M5（v0.1 完成）— 判断

### 採用

- 疑問 #1：主指標を l1_cos に切替（set match は補助）。
- 疑問 #2：(α) `learn(center_inter_inputs=True)` API 追加、デフォルト False
  （後方互換）。
- 疑問 #3：(I) 逆方向想起を試す（30 分実装）。
- 疑問 #4：M5 評価レポートは CC が数値・図表、起案はアル + ユウ + 綾。

### CC 解釈

- **Modern モードでの自動有効化はしない**：明示 API の方が読みやすく、
  Hebb での degradation も docstring caveat で読み手に伝わる。

### M5 で見えたこと

- `center_inter_inputs=True` で M4-12 origin = 10/12 を本体組込で再現
  （`test_modern_with_centering_recovers_M4_layer2`）。
- 逆方向想起：**Modern top-3 radical-hit = 2/7（部分成功）**、Hebb 0/7。
  木・火 radical（複数 kanji が共有）でだけ動く構造的観察 → v0.2 宿題 #8。

## まとめ：v0.1 中に着地済の改善

| 由来 | 対策 | 着地マイルストーン |
|---|---|---|
| M1 σ-K_u スケール問題 | `layer_noise_ratio(l)` 追加、σ_global を layer-0 std として再定義 | M1 内 |
| M3 W_inter pair 非対称 | demo `_normalize_inter_weights` | M3 |
| ↑ 本体組込 | `learn(normalize_inter='spectral')` デフォルト | M4 |
| M3 整数 multi-hot 同軸縮退 | Option B `(radical, count_unary)` 直交 encoding | M4 |
| M3 bipolar 背景バイアス | demo `_learn_with_centered_inter` | M4 |
| ↑ 本体組込 | `learn(center_inter_inputs=True)` オプション | M5 |

## v0.2 への引き継ぎ

`README.md` の `## v0.2 への宿題（M5 時点で再整理）` セクションに 9 項目を
「M○○ で見えた問題 / v0.2 でどう解決見込み」形式で記載。

## v0.1.5 M0（2026-06-19、scaffold）— 判断

`_to-cc-v0.1.5-m0.md` 受領。温度 T（K）を共通軸として K_u 温度依存ノイズの
scaffold を入れる段階。実装本体は M1。M0 は signature・docstring・
雛形・bit-exact 互換テストに絞る。

### 採用（綾＋ユウ＋アル）

- 熱揺らぎ粒度 = 最小（K_u に温度依存ノイズのみ）
- 物理厳密性は mumax3 に譲り、PyTorch 側は機能層
- 中心パラメータ = 温度 T（K 単位、両系統共通軸）
- `compute_thermal_noise_amplitude(K_u, T)` を `ayaram/modes.py` に追加、
  実装は `raise NotImplementedError("M1 で実装")`、docstring に Sato et al.
  2014（CoFeB 温度係数）と Brown 1963（熱揺らぎ第一原理）を明記
- `recall(..., temperature_K=0.0)` 引数追加。デフォルト 0.0 で v0.1 と
  bit-exact 一致する経路を保証

### CC 解釈

- **`recall` の所在**：指示書は「`ayaram/core.py` の `recall(...)` 系メソッド」
  と書かれていたが、実際の `recall` / `recall_from_layer` は
  `ayaram/memory.py` の `HopfieldNetwork` クラス上に住んでいる。両メソッドに
  `temperature_K: float = 0.0` を追加し、`ayaram/core.py` 側は
  `phase2_fluctuation(..., *, temperature_K)` と `run_cycle(..., *, temperature_K)`
  に同等の引数を生やして経路を貫通させた。指示の趣旨（recall API に温度を
  乗せる）に沿った最短経路と CC 判定。
- **`temperature_K` を keyword-only にした**：`phase2_fluctuation` の既存
  positional 引数（`mode`, `generator`）の後ろに `*` 区切りで配置。
  既存テスト `test_core.py` が positional / keyword 両方で
  `phase2_fluctuation(net, st, cfg, mode=...)` を呼んでおり、
  後方互換を最大化するため。
- **`temperature_K > 0.0` で `NotImplementedError("M1 で実装")`**：v0.1.5
  M1 が実装する側の "panic" 経路。`temperature_K < 0.0` は `ValueError`
  （物理的にも未定義のため M0 で弾く）。
- **bit-exact テスト戦略**：「v0.1 commit `2d0932b` と bit-exact 一致」を
  unit test で直接突き合わせるには別 worktree が要るため、`temperature_K=0.0`
  経路で **計算もRNG drawも一切追加しない** 設計にし、その不変条件を
  「signature 違いの 2 経路が `torch.equal`（許容誤差なし）」で抑える。
  4 件の `torch.equal` テスト（`recall` / `recall_from_layer` / `phase2`
  単体 / v15 demo の縮小版）で M0 DoD「max abs diff = 0.0 厳密」を満たす。
- **`demos/thermal_sweep.py`**：`run_attention_sweep` も body は同じく
  `raise NotImplementedError("M2 で実装")` にしてある（attention sweep は
  M3 担当だが、関数 stub 自体は M0 scaffold に含まれる）。

### CC 工夫

- なし。M0 は scaffold のみで仕様逸脱なし。

### M0 で見えたこと

- 既存 97 テスト + 新規 9 テスト = 全 106 テスト pass、~3.2 s。
- `temperature_K=0.0` no-op 設計は素直に通る。M1 で K_u(T) ノイズ本体を
  入れる際、`temperature_K > 0.0` 経路の `NotImplementedError` を本実装に
  置換するだけで済む形になっている。
