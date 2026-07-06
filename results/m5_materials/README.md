# v0.2 M5 素材 — 弧の再演 (regenerated figures)

`python demos/regenerate_m5_materials.py` の 1 コマンドで全 13 図を再生成する。
**過去の results/* からのコピーではなく、M1–M4 パイプラインの再計算**。実行そのものが
「溶ける → 帯 → 積む → 帰る → 天秤」の統合デモであり、5 点の再演 assert で決定論を再実証する。

> **注記**：再演は正式 linking (gap=1)。M1 原 report の数値 (gap=0 時代) との微差は規約差であり既知。

## 図 ↔ 骨格 § 対応表

| ファイル | 内容 | 骨格 § |
|---|---|---|
| m5_s1_correspondence.png | 天秤図 (c1 似姿の軸 σ*(β) の D 谷 ＋ c2 代償の軸 β_c(σ) 右上がり) | §1・§8 |
| m5_s4_cargo_contactsheet.png | 積み荷 blob map (24 字 × σ_s=2/2.83/4) | §4-3 |
| m5_s5_meltdown_utsu.png | 鬱 熱溶解フィルムストリップ (1 oct 毎) | §5 |
| m5_s5_meltdown_mori.png | 森 熱溶解フィルムストリップ | §5 |
| m5_s5_death_hist.png | σ_death ヒスト (窓前=二峰 / 窓後 [1,16]=単峰) | §5 |
| m5_s5_hole_dissolution.png | リング字 10 の囲み溶解点 (国 = σ≈1.0 例外) | §5-3 |
| m5_s6_b1_matrix.png | B1 尺跨ぎ想起 3×3 (非対角 0.92–1.0) | §6 |
| m5_s6_b2_groups.png | B2 三群 (対照>強共有 = 形を見る) | §6 |
| m5_s7_replay_mori.png | 森 逆再生フィルムストリップ (σ=4→0.5) | §7 |
| m5_s7_fidelity.png | M3-2 木のみ再生 忠実度 (24 字＋mean=0.526) | §7 |
| m5_s7_unreturnable.png | 品・回 戻れない字 (原 map vs 逆再生終端) | §7 |
| m5_s7_ephemeral_dist.png | ephemeral σ_death 分布 (前提崩れ) | §7-2 |
| m5_s8_awakening.png | 覚醒閾 (β≈4-6) | §8 |
| m5_s8_control_flat.png | 対照 K_u(T) 平坦線 | §8 |

## 再演 assert (STOP on mismatch)

c1 ρ=−0.863±0.005 / c2 ρ=1.000 / B1 非対角 min=0.917±0.005 /
M3-2 忠実度 mean=0.526±0.005 / 窓内 Silverman p>0.05。結果は `regen_report.json`。
