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

## v0.1.5 M1（2026-06-20、K_u(T) 温度依存ノイズ実装本体）— 判断

`_to-cc-v0.1.5-m1.md` 受領。M0 scaffold の中身を埋める段階。

### 採用（綾＋ユウ＋アル）

- **加算型注入**：独立 Gaussian 熱外乱を v0.1 ノイズに加算（乗算でも混合でもない）。
  「温度ダイヤルが 1 つ生えた」物語性と T=0 bit-exact 自明性を担保。
- **戻り値次元 = dimensionless**：M0 方針継承、`sigma_global` と同じ次元で
  合成（`config.sigma_global * thermal_amp * sqrt(2 dt) * eta_thermal`）。
- **物理式 = `sqrt(K_U_REF / K_u) * sqrt(T / T_REF_KELVIN)`**：v0.1 の
  `layer_noise_ratio(l) = sqrt(K_U_LAYERS[0] / K_U_LAYERS[l])` を K_u 軸として
  継承、T 軸は現象論的な平方根スケーリング。
- **T_REF_KELVIN = 300.0 K**：室温基準で物語的に "v0.1 と同程度の揺らぎ" が
  T=300 K で再現される設計。
- **K_u_eff(T) 補正なし**：M1 では K_u_l 固定、Sato et al. 2014 の α は
  v0.2 で導入予定（docstring で明示）。
- **不変条件規律**：`temperature_K == 0.0` で追加 RNG draw ゼロ・追加演算
  ゼロを厳格維持。`if temperature_K > 0.0:` の guard で T=0 経路を保護。

### CC 解釈

- **`K_U_REF` を module-level 定数として `ayaram/modes.py` に追加**：M1 brief
  が提示した 3 つの defensible 経路（config 直接参照 / 引数追加 /
  module-level 定数）のうち、module-level 定数が「ayaram.modes に 1 か所
  だけ K_u_ref が住む」状態を作れて読みやすい。`K_U_REF = K_U_LAYERS[0]`
  にバインドして単一情報源化、`compute_thermal_noise_amplitude` docstring で
  経路明示。
- **`T_REF_KELVIN = 300.0` も同様に module-level 定数**：物理式中のマジック
  ナンバーを廃したい。`compute_thermal_noise_amplitude` から参照。
- **K_u_l の取得経路**：`phase2_fluctuation` 内で
  `modes.K_U_LAYERS[l]` を直接 index 参照。`layer_noise_ratio(l)` が
  既に同じ index で K_u_l を引いているので、コードの読み筋を保つ。
- **加算順序**：bit-exact 不変条件を最大限尊重するため、既存の v0.1 表現
  `state.xi[l] = keep * xi + dt * (drift + inter) + sigma_l * sqrt_2dt * eta`
  は **そのまま** 残し、`if temperature_K > 0.0:` の分岐で
  `state.xi[l] = state.xi[l] + (sigma_global * thermal_amp * sqrt_2dt * eta_thermal)`
  を追加加算する形にした。T=0 path は文字通り 1 行も変わらない。
- **fixture 二段階照合の細部**：
  - 取得経路：`scripts/generate_v15_cosine_fixture.py`（共通スクリプト）を
    `git worktree add -d <tmp> 2d0932b` で取った v0.1 worktree と現 branch の
    両方で実行、両 fixture を `torch.equal` で照合（→ max abs diff = 0.0、
    fixture 生成時刻の Python 3.11 / 3.12 venv 差異を超えて bit-exact 一致）。
  - device：**CPU 固定**（torch.randn は CUDA だと driver / device 版で
    変わりうる、CPU は cross-machine deterministic）。
  - format：`torch.save` の dict `{"kanji": list[str], "layer{0,1,2}_cos":
    torch.float64 tensor}`。float64 で Python float（demo の `_cosine` が返す）
    の bit-exact 値を保存。
  - 比較：`torch.equal`（精度 0、許容誤差なし）。
  - 所在：fixture = `tests/fixtures/v15_modern_seed0_cosine.pt`、生成
    script = `scripts/generate_v15_cosine_fixture.py`、test =
    `tests/test_v015_compat.py::test_v15_modern_seed0_matches_v01_fixture`。
- **`torch.load(..., weights_only=True)`**：fixture は dict + list[str] +
  float64 tensor のみで weights_only allow-list 内、安全側に倒す。
- **M0 で書いた `NotImplementedError` ガードテスト 2 件は削除**：
  `test_positive_temperature_raises_not_implemented_in_phase2` と
  `..._in_recall` は M1 で挙動が正しく実装された時点で意味を失うため、
  T=300 の動作確認テスト（`tests/test_v015_thermal.py`）に置換。
  `compute_thermal_noise_amplitude_signature_exists` も同様に「T=0→0.0、
  docstring に Brown 1963 / Sato 2014 を含む」を確認する形に改名・改訂。
- **`if T == 0.0:` の float 完全一致比較**：`0.0` は IEEE 754 で唯一の
  exact 表現なので、`T == 0.0` の比較は安全。これで M0 不変条件
  「T=0 で 0.0 厳密」を保つ。

### CC 工夫

- **なし**。M1 も仕様内の CC 解釈で収まった。

### M1 で見えたこと

- 全 117 テスト pass（v0.1 baseline 97 + M0 9 + M1 11）、~4.1 s。
- v15 hierarchical demo Modern seed=0 の per-kanji cosine（12 字 × 3 層 =
  36 値）は v0.1 (2d0932b) と現 branch の T=0 path で **全完全一致**
  （torch.equal、max abs diff = 0.0）。fixture は永続テストとして
  `tests/test_v015_compat.py::test_v15_modern_seed0_matches_v01_fixture`
  に固定。
- **T=300 K vs T=0 K の Modern recall cosine sim**:
  - small Modern recall（seed=42、4-pattern、layer 0 のみ）:**0.999999881**
  - hierarchical Modern（seed=1、3-layer、M5 default）:**1.000000000**
  - 両者とも M1 brief table の「≥ 0.99」帯。M1 報告書で「揺らぎが効いて
    いない可能性、ダイヤルの意味なし」を明示する義務がある。
- 解釈仮説（M2 詳細スイープ前の暫定）：Modern Hopfield は softmax(β X^T ξ)
  が stored pattern 集合の凸結合の最近傍を返す operator なので、phase 3
  fixation で stored pattern にスナップバックする。phase 2 中の熱外乱が
  十分大きくないと最終 cosine sim は 1.0 に張りつく。M2 で
  `T = [0, 100, 200, 300, 400, 500] K` × 12 漢字 × 複数 seed のスイープを
  かけたとき、より大きい T で「外れる」漢字が出るかが本物の問いになる。

---

# Design Decisions — v0.2 スケール空間統合（尺跨ぎ）

依頼書「アヤラム v0.2 — スケール空間統合(尺跨ぎ) 実装依頼書」(status: approved)
および指示書 `_to-cc-v0.2-m0-m1.md` に基づく。branch: `v0.2-scalespace`
（main HEAD `d404124` = v0.1.5 合流 merge commit から分岐）。

凡例は v0.1 と同一（**採用** / **CC 解釈** / **CC 工夫**）。

## v0.2 M0（2026-07-03）— scaffold + 128×128 字形

### 採用（依頼書 確定事項）

| 論点 | 内容 |
|---|---|
| 規約 | Lindeberg 流 ∂t L = ½∇²L、t = σ²。`scalespace.py` docstring に明記 |
| 漢字セット | 既存12字（木日月火林明炎晶森水川山）+ 追加12字（一十口田回国岩品語銀樹鬱）= 24字 |
| レンダリング | 512×512 AA → 128×128 area/box（PIL `Image.BOX`）縮小。ink=1.0 / 背景 0.0 / float32 [0,1] |
| σ グリッド | σ_k = 0.5·2^(k/8)、k=0..56（σ∈[0.5,64]、8点/oct、57スライス） |
| ぼかし | 各スライスをベース画像から独立算出（累積不可）。境界=定数0 padding |
| 応答 | 正規化ラプラシアン R = t^γ·∇²L、γ=1。DoG でなく LoG 直接 |
| 極値 | スライス内 2D 8近傍 strict 極値、両極性（ink R<0 / 地 R>0）、閾値 \|R\|≥0.05×max\|R\| |
| 互換規律 | 既存 117 テスト無変更。新機能は `ayaram/scalespace.py` + scripts + data 増築のみ |

### CC 解釈（仕様内裁量）

- **font 継承**：v0.1 `data/generate_kanji.py` の候補列（NotoSansJP-VF 優先、
  YuGothM / meiryo / msgothic fallback）をそのまま踏襲。実採用は
  NotoSansJP-VF.ttf。`metadata.json` に font_path・候補列を記録。
- **Gaussian は spatial separable conv（torch, zero-pad）**：scipy 不在のため。
  依頼書要求の「定数0 padding」を `F.pad` の zero padding で厳密実現。σ=64 でも
  kernel radius=4σ を image 外までゼロ拡張して valid-conv、同サイズ出力。FFT を
  採らないのは FFT が周期境界になり定数0規約に反するため。
- **Laplacian は 5点 stencil（zero-pad）**の直接 LoG。
- **データ format = 圧縮 npz**（`glyphs_128.npz`：`glyphs` (24,128,128) f32 +
  文字別 `U{codepoint}` キー）。format は依頼書で CC 裁量。
- **リンク突合は traj→最近傍候補**方式。1候補を複数 traj が獲得したとき merge
  とし、生存側 = \|R\| 大（吸収側 = \|R\| 小、依頼書規約通り）。

### CC 工夫（仕様逸脱・明示）

- **frame（境界フレーム）極値の除外**：定数0 blur 境界により離散 Laplacian が
  画像フレームで巨大な人工曲率を生み、t=σ² が大 σ で増幅する（σ=64 で角
  \|R\|≈56 vs 実 blob ≈0.5）。放置すると (1) max\|R\| 正規化を乗っ取り、
  (2) フレーム上に phantom blob を量産する。よって `detect_extrema` は
  border（既定 3px）フレームを **正規化・検出の両方から除外**する。標準的な
  scale-space の境界破棄に相当。σ² 正規化と定数0規約は温存し、検出段のみで対処。
  → docstring 明記済み。M1 で border 感度も報告。

### M0 で見えたこと

- 全 24 字 QC pass（欠字 0、ink率 [2.1%, 13.0%] ⊂ [1%,50%]）。29 画の鬱も
  正常描画（contact_sheet.png で目視確認）。
- テスト 125 pass（v0.1/v0.1.5 の 117 を無変更で維持 + scalespace 8）。
- 単体2件（採用）：半群性 blur(t1)∘blur(t2)≈blur(t1+t2) は内部で max diff
  ≈2e-5、合成 blob 特徴スケール σ*≈s は 1/8 oct 以内で一致。CC guard 2件
  （極性符号・単一 blob リンク）を追加。

## v0.2 M1（2026-07-03）— 漢字の熱溶解

### 採用（依頼書 §3-4 + 綾修正 A/B）

- DoD-1：pooled log(σ_death) の帯クラスタ検定。
- DoD-2：P(n) 滞在時間。**(A)** ink 極性（R<0）の n のみで算出、地極性は参考併記。
  **(B)** n=1 初到達以降の σ 域を除外、n≥2 近傍で比較。
  anchor {明2 林2 炎2 晶3 森3 岩2 品3 語2 銀2 国2 回2}、曖昧字 {川 田 樹 鬱} は観測のみ。
- DoD-3：全 24 字、畳み込み決定論・seed 不在。DoD-5：実行時間報告。

### CC 解釈

- **device = CPU**：M0 probe で CPU（0.8 s/字）が CUDA（1.65 s/字）より速く、かつ
  CUDA は float 非決定性で extrema 数が run 間で微差。DoD-3「決定論・seed 不在」を
  満たすため CPU を採用。唯一の RNG は DoD-1 Silverman bootstrap（seed 記録）。
- **DoD-1 形式検定 = Silverman(1981) smoothed-bootstrap 多峰性検定**。
  H0: log(σ_death) 密度は単峰（m≤1）。h1 = 単峰化する最小 bandwidth、分散補正
  smoothed bootstrap（B=1000）で h1 における多峰率を p とする。**小さい p が H0 を棄却**
  （多峰＝帯あり）。scipy 不在のため KDE・mode 計数・h1 二分探索・bootstrap を numpy 自作。
- **DoD-2「初到達」= σ 増加方向で最初に n_ink==1 となるスライス**。その手前
  [0, k_first1) で P(n) を集計。C=2 字は実質 P(2)>P(3)、C=3 字は P(3)>P(2)∧P(3)>P(4)。
- **σ_death**：vanish/merge を death、survivor（σ_max 到達）は right-censored として
  ヒストグラム本体から分離（依頼書 打ち切り規約）。

### M1 で見えたこと（両様記録）

- **DoD-1 = 帯あり（強い）**：pooled ink σ_death（n=2087）は**二峰**。rule-of-thumb
  KDE で 2 mode、Silverman p=0.000（<0.05）で単峰を棄却。低 σ 峰（log2σ≈−0.6、σ≈0.66）は
  AA/微細テクスチャの即死、高 σ の広い帯（log2σ≈2〜2.5、σ≈4〜6）が構造 blob の溶解。
- **DoD-2 = 部分的**：anchor 11 字中 **5 字 pass**（林 炎 晶 森 岩）、6 字 fail
  （明 回 国 品 語 銀）。C 段が「好まれる滞在」になるのは一部の字のみ。fail は中間 σ で
  n が C より大きい値に長く滞在するため。曖昧字 4 字は観測のみ（同図掲載）。
- **coarse-scale bounce（要注意）**：n_ink(σ) は σ≈16 付近で最小化後、σ∈[32,64]
  （log2 5〜6）で**再上昇**する字が多い。128px canvas に対し σ≥32 は Gaussian support が
  画像を超え、有限領域境界が phantom coarse 極値を生む。実効解析域は σ∈[0.5,~16]。
  DoD-2 の窓（first n=1 で切断、σ≈16〜22）はこの領域をほぼ除外している。
- **linking churn（要注意）**：death/birth 散布図は全 σ で生成イベント（birth k>0）が多数。
  スライス内 strict 極値が slice 間で明滅し、trajectory が短く分断（死→再生）される。
  集約 σ_death 帯は堅牢だが個々の軌跡連続性は断片的。1-slice gap 許容や sub-pixel 極値で
  churn 低減余地（M2 検討）。

### 閾値感度（0.03 / 0.05 / 0.08）

metadata.json `threshold_sensitivity` に記録。閾値を上げると総 trajectory 数は減るが、
pooled σ_death の中央値・IQR（log2）は大きくは動かず、二峰構造は閾値に頑健。

## v0.2 M1b（2026-07-03）— window確定・churn低減・綾機構仮説

base `119e162`(M1)の上に積む。追加データ取得なし、glyphs_128.npz からの再計算のみ。

### 採用（G1 approved）

- **解析窓 σ∈[1,16]**（log2σ∈[0,4]）。σ<1=テクスチャ域（分離記録・非破棄）、σ>16=survivor 吸収。
- **適用の分離**：(1)DoD-1 主判定=窓のみ（churn 不適用）。(3)DoD-2=窓＋churn 低減。効果を混ぜない。
- **gap linking**：1-slice gap（`max_gap=1`）。(4)寿命<2 スライスは n(σ) 集計から除外。
- **穴/開放タグ**：ベース白領域を連結成分ラベリング、画像縁連結=開放・非連結=穴。地 blob は
  σ_birth 中心座標のラベルでタグ、**合算 n(σ) は穴由来のみ**算入・開放由来は参考。
- **合算判定**：主=「段の出現」（ink 単独に無い n≥2 滞在極大が合算で立つか）、副=n と構造整合。

### CC 解釈

- **gap ゲート = gap 数でスケール**：g スライス跨ぐ再連結は `g·d_max` 以内（g=1 は M1 と同一）。
- **link_trajectories(max_gap=0) は M1 と挙動完全一致**を再現性ゲートで実証（traj=6794 厳密一致）。
  dormant traj が σ_max まで残り最終スライス未観測なら survivor でなく vanish 採点。
- **穴検出**＝ベース bg（glyph<0.5）を 4 連結ラベリング（ink 8 連結の補）、縁連結=開放。
- **囲み溶解点**＝各 σ スライスの L を 0.5 二値化、穴中心が縁連結になる最初の σ（should、CC 実装）。
- **DoD-2「初到達」＝窓内で最初に n_ink==1 のスライス**。窓内・初到達手前で P(n) 集計。
- **段の出現**＝P_sum の局所極大整数のうち P_ink に無いもの（`_local_max_ns` 差分）。

### M1b で見えたこと（両様記録）

- **再現性ゲート PASS**：gap0/窓なし/寿命なし/thr0.05/border3 で traj=6794・中央値 log2σ=1.75・
  二峰・Silverman reject を厳密再現。以降の差分は改訂由来と保証。
- **(1) 窓 DoD-1＝単峰**：σ∈[1,16] に絞ると pooled ink log(σ_death) は **Silverman p=0.187、
  reject=False**（rot mode は 2 だが広い単一山の微小凹凸）。churn 低減参考版も p=0.531・単峰。
  → **M1 の二峰は「テクスチャ(σ<1) vs 構造」であって「画 vs 部品」ではない**。構造帯内部に
  画帯・部品帯の σ 軸分離は無い（texture 死 603 個を窓で除外）。
- **(3) DoD-2 改訂＝3/11 pass、fail は全て type-(b)**：pass=**林 炎 森**（反復ソリッド部首 木/火）。
  fail_b（P(C)=0）=明 晶 岩 品 語 銀 国 回。M1 で pass だった **晶・岩が窓＋churn で fail に反転**
  （M1 の pass はテクスチャ/明滅由来を含んでいた）。type-(a) はゼロ＝「C 段が均される」のでなく
  「C 段が存在しない」。**ソリッド部首字は段を持ち、リング部首字は段を持たない**の対比が明瞭。
- **(4) 綾機構仮説の判定**：
  - **前半（LoG blob は塊検出器、リング部品は ink の塊として立たない）＝支持**。全リング字が
    ink 単独で C 段なし（最小証言台の **口 も P(2)=0**）。ソリッド反復の 森(3)/林炎(2) だけ段を持つ。
    品(口×3) fail が仮説の想定通り、対する 森(木×3) pass が対照。
  - **後半（地＝穴の側に立つ→合算で段が立つ）＝不支持**。合算しても低 n（2,3）の滞在は 0 のまま
    （口＋穴でも P_sum(2)=0）。理由＝**穴は σ≈2 で溶解**（囲み溶解点、log2σ≈1.0〜1.13）し、地 blob は
    細スケールにしか居ないため、構造帯(σ≈4〜6)に部品数段を作らない。
  - **晶（対照）＝仮説の 準塊 予測を裏切る**：日＝棒入りリング＝準塊なら pass 寄りのはずが fail_b。
    128px では日の内部白も穴として振る舞い、塊化しない。
  - 総括＝**単体リング字（口 日 月 田）・複合字（回 国 品 語 銀 明）とも同じ挙動**：ink で段立たず、
    合算でも段立たず。仮説の「検出器 vs 部品」骨子は支持、「穴で救済」は解像度・溶解 σ の壁で不成立。
- **(5) border 感度**：border=2/3/5 で 窓中央値 log2σ=**2.125**・二峰性・DoD-2 pass=3 が**完全一致**。
  border=3 採用は主要指標に無影響、妥当。

## v0.2 M2（2026-07-03）— 踊り場スナップショット → Hopfield 積み荷

base `d4936ef`(M1b)の上。再計算のみ（正式 linking=gap=1）。scripts/run_m2_cargo.py。

### 採用（G1 approved）

- 正式 linking=churn 低減 gap=1。寿命<2 は blob map から除外・系譜木では ephemeral タグで保持。
- スナップショット σ_s∈{2, 2.83, 4}（log2σ=1,1.5,2）。生存＝σ_birth≤σ_s<σ_death。
- blob map=32×32、生存 ink blob を中心(x/4,y/4)・幅 σ_s/4 の等方ガウス、振幅=|R|（字内 max 正規化）、
  全体 L2 正規化、flatten=1024 次元=layer0。
- Hopfield=v0.1 `ayaram/memory.py` Modern（`learning.modern_hopfield_update`）。
- f_c=(穴数,has_hole,囲み溶解点 log2σ 中央値)。24 字 z-score→単位ノルム→w 倍連結。既定 w=0.3、感度{0.1,0.3,1.0}。
- 系譜木 JSON：nodes[id,polarity,σ_birth,σ_death,end_type,pos_at_death,ephemeral]、
  edges[parent=生存側,child=吸収側,σ_merge]、root=survivor。

### CC 解釈・CC 工夫

- **β=16**（v0.1 既定 1.0 から引き上げ、CC 裁量＋記録）。理由＝blob map は全正・単位ノルムで相互
  cosine が高く（σ_s=4 で max 0.917）、24 パターンの自己想起 gate に高 β が要る。σ_s=2/2.83 は β=8 でも
  24/24 だが σ_s=4 が β=16 を要するため全 σ_s 統一で 16。
- **σ_s* 選定**＝ペアワイズ mean off-cosine（低いほど良）を主、ノイズ付き想起（η∈{1,2,3,5}、seed 記録）を
  従。両者とも σ_s=2（log2=1）が最良。
- **ノイズ規約**＝各パターンに N(0,(η/√d)²) 加算後正規化、5 反復、seed=20260703。
- **B2 Mann-Whitney U**＝tie 補正正規近似で自作（scipy 不在）。片側 H1: 共有>非共有。
- **系譜木 root**＝親を持たない node（survivor＋孤立 vanish）。survivor⊆roots を整合テストで確認。

### M2 で見えたこと（両様記録）

- **A2 自己想起 gate＝PASS**：全 σ_s で 24/24。ベクトル化健全。
- **A3 σ_s*=2.0（log2=1）**：mean off-cosine 0.153<0.235<0.376、ノイズ想起も η=5 で 0.975 vs σ_s=4 の
  0.183。σ_s=2 が最も識別的（細かい blob が多く字形が richに残る）。
- **A4 系譜木＝全 24 字 consistency OK**（edges==merge、roots==survivor+vanish、閉路なし）。M3 が根→葉で
  下れる形で `merge_trees.json` 固定化。
- **B1 尺跨ぎ想起＝成功**：3×3 行列の非対角が **0.92〜1.00**（chance 4.2% の 20 倍超）。細クエリで粗記憶／
  粗クエリで細記憶を、正解字にほぼ確実に落とせる。四演算子「尺跨ぎ」の最初の動作テストは**明確に肯定**。
- **B2 部品干渉＝「形を見ている」確定**：強共有 11 ペアの cosine 平均 0.131 は非共有 0.154 を**下回り**
  （MW p=0.807、H1 共有>非共有 不支持）、形状類似対照 3 ペア（日-月/田-回/田-国）は 0.349 と**大幅に上回る**。
  綾の前置き通り「**この位置固定表現は部品共有でなく形（位置の重なり）を見ている**」——対照>共有の
  確定条件を満たす。部品情報の有無ではなく、この表現では部品共有が近接に変換されないと限定判定。
- **B3 リング字併用＝改善なし**：ring margin は blob 単独 0.606、w=0.3 で 0.607（±0）、w=1.0 で 0.396
  （悪化）。f_c（穴特徴）はリング字識別を改善しない。M1b の「穴は細スケール特徴で部品構造を担わない」と整合。

### M3 への含意

- 系譜木は root(survivor)→葉(細 blob) で下る「粗が細に割れる経路図」。M3 下り生成は survivor の粗 blob
  map を起点に、σ_merge を逆順に辿って子 blob を展開する入口になる。ephemeral node は M3 側で filter 可能。
- blob map が「形（位置）」表現である以上、M3 の下り生成も位置ベース。部品意味は別チャネルが要る（B2/B3 帰結）。

## v0.2 M3（2026-07-03）— 系譜木の逆再生（溶けた字の帰り道）

base `3aaabac`(M2)の上。再計算のみ。scripts/run_m3_replay.py。射程＝**形の尺跨ぎ限定**、
**「記録の再生」であって「生成」ではない**（確率生成は M3b 以降）。

### 採用（G1 approved）

- 再生 σ 格子＝M1 と同一、σ∈[0.5,4]（k=0..24）を σ 降順。再生の状態規約＝M2 A1 の全 σ 拡張。
- M3-1 完全再生 gate＝位置・強度系列フル使用で cosine≥0.999 全字・全 σ（fail=STOP）。
- M3-2 木のみ再生＝root 初期状態＋σ_merge＋pos_at_death＋吸収時|R|、分離後は位置・強度固定。
- M3-3＝ephemeral 込み/抜き二重再生。識別想起は記憶=M2 正式積み荷（σ_s=2 抜き）固定、
  クエリ σ∈{2,1,0.707}×込み/抜き=6 セル。σ=0.707 は壁越え（テクスチャ帯）として区別記載。
- f_c 検証器＝リング字で B̂ 二値化＋穴ラベリング→穴出現 σ を M1b 溶解点と比較。積み荷 f_c は w=0 運用。

### CC 解釈・CC 工夫

- **β=16 標準**（M2 踏襲）。M3-0 で σ_s 別必要 β 下限＝σ_s=2/2.83→8、σ_s=4→16 を実測。
- **二値化＝0.5×max(map)**（L2 正規化 map に絶対閾値不適のため max 比、綾修正 B）。記録済。
  一致許容 |log2σ 差|≤0.25。
- **survivor 初期値**＝σ=4 スライス（k=24）の位置・|R|（pos_at_death は σ_max にあり範囲外のため）。
- replay_map(mode='full') は build_blob_map（直接スライス）とビット一致（M3-1 gate の土台、単体テスト化）。
- ephemeral=len(points)<2（単スライス明滅）。込み=min_lifetime1、抜き=min_lifetime2。

### M3 で見えたこと（両様記録）

- **M3-0 β sweep＝「β の必要量は尺の関数」を支持**：B1 平均は β≤4 で chance(0.042)、β=8 で 0.72、
  β=16 で 0.965。自己想起の必要 β 下限は σ_s=2/2.83 で 8、**σ_s=4（粗）で 16**。粗い尺ほどパターンが
  接近し高 β を要する＝逆温度 β の出番が尺で決まる。**「温度=尺」M4 仮説の傍証の芽**（M4 で本検証）。
- **M3-1 完全再生 gate＝PASS**（min cosine=1.000000、全字・全 σ）。データ完全性＋再生器の正しさを担保。
- **M3-2 木のみ再生＝「道は覚え、歩幅を忘れる」**：忠実度は σ=4 で 0.69、σ=1 で 0.50、σ=0.5 で 0.23 と
  σ 降下で単調低下（mean 全域 0.526）。ドリフト（各 σ の実位置）を捨てた分だけ細スケールでズレる。
  だが識別想起は σ=2 抜きで **0.792**（Hopfield が最近傍へ引くため、忠実度 0.53 でも字は当たる）。
- **M3-3 込み/抜き＝差ゼロ**：6 セルとも incl==excl（σ=2/1→0.792、σ=0.707→0.708、全て chance 4.2% 超）。
  理由＝ephemeral は再生域 σ∈[0.5,4] の外に住む（下記）。壁越えセル(σ=0.707)が高いのは**木のみ再生が
  位置固定**のため σ=0.707 でも blob 中心が σ=2 と同じで σ_s=2 記憶に乗る——真のテクスチャ帯構造の
  壁越えを測っているのではない点を明記（限定）。
- **ephemeral σ_death 分布＝「テクスチャ帯住人」前提は不成立**：n=470、log2σ 中央値 **4.12（σ≈17）**、
  IQR[0.53, 5.75]、texture 帯（log2σ<0）は **24%** のみ。ephemeral は単スライス明滅で、大半は**粗スケール**の
  ちらつき。**M3b で軸を切り直す判断材料**（綾修正 C の事後検証＝重なり薄い）。
- **f_c 検証器＝0/10（表現由来の負結果）**：木のみ再生の二値化 map にリング字の穴は**出現しない**。
  追加検証＝直接スライス map でも、閾値 0.15〜0.5 でも、1px クロージングでも穴は形成されず。
  ＝**ink blob map は疎な点表現で、閉曲線（囲み）を再構成しない**。穴は M1b で地(ground)極性の特徴、
  ink 積み荷には元から無い（M2 B2 の「形＝位置を見る」と一貫）。帰り道の穴検証には ground チャネル or
  塗り stroke 表現が要る（v0.3）。

### M3b / v0.3 への含意

- **M3b（確率生成）**：木の統計から細部をサンプルする。ephemeral は粗スケール住人と判明したので、
  M3b の「細部」軸は ephemeral でなく別（σ<1 の真テクスチャ trajectory）に取り直す。
- **v0.3（位置正規化 / 意味）**：ink 積み荷は位置・形の表現で穴も部品意味も持たない（M2 B2＋M3 f_c）。
  部品意味・囲み検証には ground 極性チャネルか位置正規化表現が要る。M3 は「形の帰り道」までを確定。

## v0.2 M4（2026-07-04）— 温度と尺（天秤の測定）＝ v0.2 最終マイルストーン

base `76cc151`(M3)の上。再計算のみ。scripts/run_m4_temperature.py。

### Positioning（三温度、綾裁定・report 必須転記）

温度は三つ：**σ(入力空間の拡散熱＝尺)、β⁻¹(想起演算の温度)、K_u(T)(状態空間の Langevin＝物理の熱)**。
M4 主検証＝β↔σ、対照＝ノイズ温度。**β↔σ の橋が架かっても「温度＝尺」の完全証明ではない。物理熱
K_u(T) との本橋は v0.3 以降**。

### 採用（G1 approved）

- β グリッド 2^(m/2) m=0..10（11 点）、σ_q k={16..32}（σ=2..8、9 点）。W[i,j]=softmax_j(β⟨q_i,p_j⟩)。
- (c1) 構造対応 σ*(β)=argmin_σ D(W_β,W_σ)、**D は非対角のみ・行 JS 平均**、argmin は近傍 3 点二次補間で連続化。
- (c2) 等識別 β_c(σ_q)=識別率 0.9 達成の最小 β（線形補間）。単調性＝Spearman（|ρ|≥0.8, p<0.05）。
- 対照＝K_u(T) 反復想起（q_{t+1}=Hopfield 出力＋N(0,T/√d)、10 step、T 8 点×seed 5）。判定対象外の参考。

### CC 解釈・CC 工夫

- **D=行ごと非対角分布の JS 平均**（推奨通り）。offdiag 抽出＝各行から対角除去して再正規化。
- **連続化＝log2σ 上で近傍 3 点の放物線頂点**（A>0 のときのみ、範囲クリップ）。
- **識別率＝反復 Modern Hopfield 想起（3 step）の argmax 正解率**（一段 softmax の argmax は β 非依存で不適、
  反復想起は β 依存で σ_q ごとに 0.9 交差が定義できる）。基準 0.9（裁量内）。
- Spearman は tie 平均順位＋t 近似 p を numpy 自作（scipy 不在）。
- **confusion_matrix を ayaram/cargo.py に昇格**（単体テスト化：決定論・行確率・β 鋭化）。

### M4 で見えたこと（両様記録）

- **再現性ゲート PASS**：β=16 自己想起 24/24、B1 非対角 min 0.917（M2 一致）。
- **M4-0 真の壁越え**：直接スライス B(σ_q) クエリ→σ_s=2 記憶。σ=0.707（テクスチャ帯）=**0.875**、
  σ=1.0=**1.00**（寿命フィルタ有無で同値）。chance 4.2% を大きく超え踊り場(1.00)に迫る。**M3-3 の 0.708 は
  位置固定の擬似値**だった——真値はより高く、テクスチャ帯クエリでも σ_s=2 記憶を引ける。
- **(a) 想起を熱する**：clean 想起は β≤2.83 で chance、β=4 で 0.12、β≥5.66 で 1.0。**β≈4-6 の覚醒閾**（M3-0 追認）。
- **(b) 入力を熱する**：B(σ_q) 識別（β=16）は σ=2 で 1.0 → σ=4 で 0.92 → σ=8 で 0.17。入力の溶けで単調低下。
- **(c1) 構造対応 σ*(β)＝右下がり（HIT）**：ρ=**−0.863**、p<0.0001。低 β（熱い想起）は高 σ（溶けた入力）の
  混同構造に、高 β（冷たい想起）は σ=2（crisp）に対応。**pre-register σ↑⇔β*↓ 的中**。＝「似姿の軸」
  （溶けを真似る温度）。両極限（β→0, σ→∞）が一様分布へ収束するアル論証を支持。
- **(c2) 等識別 β_c(σ_q)＝右上がり（HIT）**：ρ=**1.000**、p<0.0001（有効 5/9 点、σ_q>4.75 は β=32 でも
  0.9 未達で除外）。β_c=log2 2.44→3.8 と単調増。溶けた入力を 0.9 で見分けるには冷たい想起が要る。
  **pre-register σ↑⇔β_c↑ 的中（綾の退路なし主張）**。＝「代償の軸」（溶けに耐える温度）。
- **両皿確定**：c1（似姿・右下がり）と c2（代償・右上がり）が**逆向きで両方単調**＝「天秤」が二本の腕として
  定量化。支点は交点付近（log2β≈2.5、log2σ≈1）。理論注記通り「溶け具合のマッチング」と「溶けに耐える
  冷たさ」は別物で向きが逆。
- **対照 K_u(T)＝溶けない（NULL、参考）**：T∈[0,0.5] で反復想起の正解率は**全域 1.0**（森の cosine 軌跡も
  0.995 で平坦）。softmax が毎 step 記憶へスナップバックし、Langevin ノイズは想起を溶かさない
  （v0.1.5 M1 の追認）。**注意**：正解率 1.0 ゆえ W_T は恒等（非対角情報ゼロ）で、(c1) 距離空間への配置は
  offdiag 一様化の副作用で β=1（最熱端）に落ちる——これは人工物で「K_u≈熱い想起」を意味しない。物理熱の
  混同構造は本レンジで存在せず、**K_u↔β/σ 本橋は未測定（v0.3+）**。

### v0.2 総括への含意（M5 評価レポート向け）

- v0.2 は**「溶ける（M1）→帯（M1b）→積む（M2）→帰る（M3）→天秤（M4）」**の弧を閉じた。中核発見＝
  (1) 構造帯は単帯で部品は blob 数に出ない（M1b）、(2) blob map は形＝位置を見て部品を見ない（M2 B2）、
  (3) 尺跨ぎ想起は成立（M2 B1）、(4) 木のみ再生は道を覚え歩幅を忘れる（M3-2）、(5) **β↔σ 天秤が二本腕で
  定量化（M4 c1/c2 両 HIT）**。
- **β=尺の関数が本検証で確定**。逆温度 β と入力尺 σ は混同構造上で対応写像を持つ（似姿）かつ等識別の
  代償で逆向きに連動。M4 は「温度＝尺」を β↔σ の水準で確立し、物理熱 K_u との本橋を v0.3 の宿題として残す。

## v0.2 README（2026-07-06）— 対外の顔＋B2 補遺の出典確認

main `88d8638`（封印済）上の文書作業。README に v0.2 セクション追加（v0.1 記述は履歴保存）＋
B2 補遺 3 値を `results/v0.2_m2/blob_maps.npz`（σ_s=2）から実測照合。

### 採用

- README v0.2 セクション（四行結論＋六動詞、天秤図、再現コマンド 2 種、tag v0.2.0＋Arc 8 commit、
  §9-3 限定転記、書誌 4 点）。図参照は `results/m5_materials/m5_s1_correspondence.png` を直接（git 追跡・
  origin/main 到達確認済、追加 commit 不要）。

### CC 解釈

- 図は results 直参照（docs/ へのコピーはせず）。README から相対パスで GitHub がレンダリング。remote 到達を
  `git cat-file -e origin/main:...` で確認。

### B2 補遺の出典確認（M5 §6-2 / §6-4）— **blob_maps.npz(σ_s=2) 実測、authoritative**

強共有 11 ペア cosine（σ_s=2、L2 正規化 map の内積）：
木-林=0.1343 / 木-森=0.1432 / 林-森=0.1537 / 火-炎=0.3497 / 日-明=0.0287 / 日-晶=0.0473 /
明-晶=0.2107 / 月-明=0.0450 / 口-品=0.0525 / 口-回=0.0904 / 山-岩=0.1855。
- **強共有平均=0.131**（M5 一致）、**非共有平均=0.1536**（M5 0.154 一致、非共有=276−11=265 ペア・対照 3 含む）、
  **単体字-複合字ペア数=9**（M5 一致。複合-複合は林-森・明-晶 の 2 件）。
- **不一致 2 件（M5 §6-4 の裁定往復由来 個別値、repo 未照合値）**：
  林-森 M5=0.209 vs 実測 **0.1537**、明-晶 M5=0.116 vs 実測 **0.2107**。
  → aggregates と結論（対照>強共有、位置の目）は無傷。footnote 2 値のみ M5 側で要訂正（Vault で アル→綾、
  CC は M5 不触）。repo 側の出典は本記録の実測値で閉じる。
