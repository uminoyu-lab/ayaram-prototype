"""v0.2 M5 — regenerate all evaluation-report figures in one command.

This is the integration demo of the whole v0.2 arc:
    溶ける(M1) → 帯(M1b) → 積む(M2) → 帰る(M3) → 天秤(M4)
Nothing is copied from results/*; every figure is RE-COMPUTED by re-running the
existing M1–M4 pipeline functions.  The run doubles as a determinism re-proof:
five key numbers are asserted against the milestone reports (STOP on mismatch).

Official linking is gap=1 (M1b onward). Small differences from the *original*
M1 report (gap=0 era) are regime differences and are expected/known.

Usage:  python demos/regenerate_m5_materials.py   ->  results/m5_materials/
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_ROOT, "scripts")
for p in (_ROOT, _SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from ayaram.scalespace import (  # noqa: E402
    build_scale_space, detect_extrema, link_trajectories,
    normalized_log_response, sigma_grid,
)
from ayaram.cargo import (  # noqa: E402
    GRID, build_blob_map, build_blob_maps, extract_records, replay_map,
    hopfield_recall, nearest_pattern, snapshot_sigmas,
)
from ayaram.learning import modern_hopfield_update  # noqa: E402
from run_m1_heat_dissolution import silverman_multimodality, deaths, _cjk_prop, _kde  # noqa: E402
from run_m1b_analysis import (  # noqa: E402
    hole_map, dissolution_sigma, partition_deaths, MAX_GAP,
)
from run_m2_cargo import SHARED_PAIRS, CONTRAST_PAIRS  # noqa: E402
from run_m4_temperature import (  # noqa: E402
    BETA_GRID, SIGMA_KS, T_GRID, SEEDS, BETA, IDENT_RATE, CTRL_STEPS,
    recall_acc, conf_matrix, D_confusion, spearman, cont_argmin, interp_threshold,
)

OUT = os.path.join(_ROOT, "results", "m5_materials")
GLYPHS = os.path.join(_ROOT, "data", "glyphs_128", "glyphs_128.npz")
OCT_KS = list(range(0, 57, 8))          # 1 slice / octave (meltdown filmstrip)
REPLAY_KS = [24, 20, 16, 12, 8, 4, 0]   # σ=4→0.5 (reverse replay)
KRANGE = list(range(0, 25))             # σ∈[0.5,4]


def _set_cjk_font():
    """Register a CJK font as the global default so all figure text renders."""
    from matplotlib import font_manager
    for fp in (r"C:\Windows\Fonts\NotoSansJP-VF.ttf", r"C:\Windows\Fonts\YuGothM.ttc",
               r"C:\Windows\Fonts\meiryo.ttc", r"C:\Windows\Fonts\msgothic.ttc"):
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
                name = font_manager.FontProperties(fname=fp).get_name()
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
            except (RuntimeError, OSError):
                pass
            return


def main():
    os.makedirs(OUT, exist_ok=True)
    _set_cjk_font()
    prop = _cjk_prop()
    t0 = time.time()

    dat = np.load(GLYPHS, allow_pickle=False)
    glyphs = dat["glyphs"]
    chars = json.load(open(os.path.join(_ROOT, "data", "glyphs_128", "metadata.json"),
                           encoding="utf-8"))["chars"]
    sig = sigma_grid(); N = len(chars); log_sig = np.log2(sig)
    idx = {c: i for i, c in enumerate(chars)}

    # ---- shared recompute: trajectories (gap=1), scale spaces ----
    per, ss_np = {}, {}
    for i, ch in enumerate(chars):
        ss = build_scale_space(glyphs[i], sig, device="cpu")
        R = normalized_log_response(ss, sig)
        ex = detect_extrema(R, sig, 0.05, 3)
        per[ch] = link_trajectories(ex, sig, max_gap=MAX_GAP)
        ss_np[ch] = ss.cpu().numpy()
    snaps = snapshot_sigmas(sig)
    maps_by_sig = {s: build_blob_maps(per, sig, chars, s) for s in snaps}
    P2 = maps_by_sig[2.0]

    checks = {}

    # ===================== §4 cargo contact sheet (13) =====================
    _fig_cargo_sheet(chars, maps_by_sig, snaps, prop, os.path.join(OUT, "m5_s4_cargo_contactsheet.png"))

    # ===================== §5 meltdown filmstrips + death hist + holes =====
    _filmstrip_L(ss_np["鬱"], sig, OCT_KS, "鬱  熱溶解 (heat dissolution)",
                 os.path.join(OUT, "m5_s5_meltdown_utsu.png"), prop)
    _filmstrip_L(ss_np["森"], sig, OCT_KS, "森  熱溶解 (heat dissolution)",
                 os.path.join(OUT, "m5_s5_meltdown_mori.png"), prop)
    # σ_death histogram: pre-window (all ink) vs post-window [1,16]
    pre = np.concatenate([deaths(per[c], "ink") for c in chars]); pre = pre[pre > 0]
    post = np.concatenate([partition_deaths(per[c], "ink")[0] for c in chars]); post = post[post > 0]
    _fig_death_hist(np.log2(pre), np.log2(post), log_sig, os.path.join(OUT, "m5_s5_death_hist.png"))
    # windowed Silverman (assert v)
    sv = silverman_multimodality(np.log2(post))
    checks["windowed_silverman_p"] = sv["silverman_p_value"]
    # hole dissolution (ring 10)
    ring = list("口日月田回国品語銀明")
    hole_rows = {}
    for ch in ring:
        i = idx[ch]
        diss = [h["dissolution_sigma"] for h in dissolution_sigma(ss_np[ch], glyphs[i], sig)
                if h["dissolution_sigma"]]
        hole_rows[ch] = [float(np.log2(x)) for x in diss]
    _fig_hole_dissolution(hole_rows, prop, os.path.join(OUT, "m5_s5_hole_dissolution.png"))

    # ===================== §6 B1 matrix + B2 groups =====================
    B1 = np.zeros((3, 3))
    for qi, sq in enumerate(snaps):
        for mi, sm in enumerate(snaps):
            B1[qi, mi] = recall_acc(maps_by_sig[sq], maps_by_sig[sm], BETA)
    checks["b1_offdiag_min"] = float(B1[~np.eye(3, dtype=bool)].min())
    _fig_b1(B1, snaps, os.path.join(OUT, "m5_s6_b1_matrix.png"))
    Pn = P2 / (np.linalg.norm(P2, axis=1, keepdims=True) + 1e-12)
    cosf = lambda a, b: float(Pn[idx[a]] @ Pn[idx[b]])
    shared = [cosf(a, b) for a, b in SHARED_PAIRS]
    sset = {frozenset(p) for p in SHARED_PAIRS}
    allp = [(chars[i], chars[j]) for i in range(N) for j in range(i + 1, N)]
    nonshared = [cosf(a, b) for a, b in allp if frozenset((a, b)) not in sset]
    contrast = {f"{a}-{b}": cosf(a, b) for a, b in CONTRAST_PAIRS}
    _fig_b2(shared, nonshared, contrast, os.path.join(OUT, "m5_s6_b2_groups.png"))

    # ===================== §7 reverse replay + fidelity + unreturnable + ephemeral
    recs = {c: extract_records(per[c], sig, min_lifetime=2) for c in chars}
    _filmstrip_replay("森", recs["森"], sig, REPLAY_KS,
                      os.path.join(OUT, "m5_s7_replay_mori.png"), prop)
    fid_mat = np.vstack([_tree_fidelity(per[c], recs[c], sig) for c in chars])
    mean_fid = np.nanmean(fid_mat, axis=0)
    checks["m32_fidelity_mean"] = float(np.nanmean(mean_fid))
    _fig_fidelity(log_sig[KRANGE], fid_mat, mean_fid, os.path.join(OUT, "m5_s7_fidelity.png"))
    _fig_unreturnable(["品", "回"], per, recs, sig, idx, P2, prop,
                      os.path.join(OUT, "m5_s7_unreturnable.png"))
    ephem = np.array([t.sigma_death for c in chars for t in per[c]
                      if t.polarity == "ink" and len(t.points) < 2])
    _fig_ephemeral(np.log2(ephem[ephem > 0]), os.path.join(OUT, "m5_s7_ephemeral_dist.png"))

    # ===================== §8 awakening + correspondence + control =====================
    acc_b = [recall_acc(P2, P2, b) for b in BETA_GRID]
    _fig_awakening([np.log2(b) for b in BETA_GRID], acc_b, os.path.join(OUT, "m5_s8_awakening.png"))

    # c1 / c2 (the balance beam)
    Wb = {b: conf_matrix(P2, P2, b) for b in BETA_GRID}
    Bq = {k: np.stack([build_blob_map(per[c], sig, sig[k], min_lifetime=2) for c in chars])
          for k in SIGMA_KS}
    Ws = {k: conf_matrix(Bq[k], P2, BETA) for k in SIGMA_KS}
    log2s = [float(np.log2(sig[k])) for k in SIGMA_KS]
    D_grid, sstar = [], []
    for b in BETA_GRID:
        ds = [D_confusion(Wb[b], Ws[k]) for k in SIGMA_KS]
        D_grid.append(ds); sstar.append(cont_argmin(log2s, ds))
    c1_rho, _ = spearman(BETA_GRID, sstar)
    betas_l2 = [float(np.log2(b)) for b in BETA_GRID]
    beta_c = []
    for k in SIGMA_KS:
        accs = [recall_acc(Bq[k], P2, b) for b in BETA_GRID]
        beta_c.append(interp_threshold(betas_l2, accs, IDENT_RATE))
    valid = [(np.log2(sig[k]), bc) for k, bc in zip(SIGMA_KS, beta_c) if bc is not None]
    c2_rho, _ = spearman([v[0] for v in valid], [v[1] for v in valid])
    checks["c1_rho"] = float(c1_rho); checks["c2_rho"] = float(c2_rho)
    _fig_correspondence(D_grid, sstar, log2s, BETA_GRID, beta_c, c1_rho, c2_rho,
                        os.path.join(OUT, "m5_s1_correspondence.png"))

    # control K_u(T) flat line (seed 5)
    X = torch.as_tensor(P2, dtype=torch.float32).T.contiguous(); d = P2.shape[1]
    means, sds = [], []
    for ti, T in enumerate(T_GRID):
        accs = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 100 + ti); ok = 0
            for i in range(N):
                xi = torch.as_tensor(P2[i], dtype=torch.float32).clone()
                for _ in range(CTRL_STEPS):
                    xi = modern_hopfield_update(xi, X, beta=BETA)
                    xi = xi + torch.as_tensor(rng.standard_normal(d) * (T / math.sqrt(d)), dtype=torch.float32)
                ok += int(nearest_pattern(P2, xi.numpy()) == i)
            accs.append(ok / N)
        means.append(float(np.mean(accs))); sds.append(float(np.std(accs)))
    _fig_control(T_GRID, means, sds, os.path.join(OUT, "m5_s8_control_flat.png"))

    # ===================== assert re-proof (STOP on mismatch) =====================
    asserts = [
        ("c1_rho == -0.863 ± 0.005", abs(checks["c1_rho"] - (-0.863)) <= 0.005, checks["c1_rho"]),
        ("c2_rho == 1.000", abs(checks["c2_rho"] - 1.000) <= 1e-6, checks["c2_rho"]),
        ("b1_offdiag_min == 0.917 ± 0.005", abs(checks["b1_offdiag_min"] - 0.9166667) <= 0.005, checks["b1_offdiag_min"]),
        ("m32_fidelity_mean == 0.526 ± 0.005", abs(checks["m32_fidelity_mean"] - 0.526) <= 0.005, checks["m32_fidelity_mean"]),
        ("windowed_silverman_p > 0.05", checks["windowed_silverman_p"] > 0.05, checks["windowed_silverman_p"]),
    ]
    runtime = time.time() - t0
    result = {"checks": checks, "asserts": [{"name": n, "pass": bool(p), "value": float(v)} for n, p, v in asserts],
              "all_pass": all(p for _, p, _ in asserts), "runtime_sec": round(runtime, 2),
              "n_figures": 13, "linking": "official gap=1"}
    json.dump(result, open(os.path.join(OUT, "regen_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    _write_readme(OUT)

    for n, p, v in asserts:
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  (got {v:.4f})")
    if not result["all_pass"]:
        print("STOP: re-proof assert mismatch (reproducibility break)")
        return 1
    print(f"M5 materials: 13 figures regenerated, all asserts PASS, {runtime:.1f}s -> {OUT}")
    return 0


# --------------------------------------------------------------------------- helpers
def _tree_fidelity(trajs, recs, sig):
    cur = []
    for k in KRANGE:
        s = sig[k]
        direct = build_blob_map(trajs, sig, s, min_lifetime=2)
        tree = replay_map(recs, sig, s, mode="tree")
        na, nb = np.linalg.norm(direct), np.linalg.norm(tree)
        cur.append(float(direct @ tree / (na * nb + 1e-12)) if na > 1e-9 and nb > 1e-9 else np.nan)
    return np.array(cur)


def _filmstrip_L(ss, sig, ks, title, path, prop):
    fig, axes = plt.subplots(1, len(ks), figsize=(2 * len(ks), 2.4))
    for ax, k in zip(axes, ks):
        ax.imshow(ss[k], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"σ={sig[k]:.1f}", fontsize=8); ax.axis("off")
    fig.suptitle(title, fontproperties=prop, fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _filmstrip_replay(ch, recs, sig, ks, path, prop):
    fig, axes = plt.subplots(1, len(ks), figsize=(2 * len(ks), 2.4))
    for ax, k in zip(axes, ks):
        m = replay_map(recs, sig, sig[k], mode="tree").reshape(GRID, GRID)
        ax.imshow(m, cmap="magma"); ax.set_title(f"σ={sig[k]:.2f}", fontsize=8); ax.axis("off")
    fig.suptitle(f"{ch}  逆再生 (reverse replay σ:4→0.5)", fontproperties=prop, fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _fig_cargo_sheet(chars, maps_by_sig, snaps, prop, path):
    fig, axes = plt.subplots(4, 6, figsize=(15, 10))
    for ax, ch, i in zip(axes.flat, chars, range(len(chars))):
        strip = np.hstack([maps_by_sig[s][i].reshape(GRID, GRID) for s in snaps])
        ax.imshow(strip, cmap="magma"); ax.set_title(ch, fontproperties=prop, fontsize=10); ax.axis("off")
    fig.suptitle("§4 積み荷 blob map — 各字 σ_s = 2 / 2.83 / 4 (左→右)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_death_hist(pre_l2, post_l2, log_sig, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(pre_l2, bins=40, color="C3", alpha=0.8)
    ax[0].set_title("§5 σ_death 窓前 (all ink, 二峰=テクスチャ vs 構造)")
    ax[0].set_xlabel("log2 σ_death")
    ax[1].hist(post_l2, bins=30, color="C0", alpha=0.8)
    ax[1].axvspan(0, 4, color="C2", alpha=0.06)
    ax[1].set_title("§5 σ_death 窓後 [1,16] (構造帯=単峰)"); ax[1].set_xlabel("log2 σ_death")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_hole_dissolution(rows, prop, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ys = list(rows.keys())
    for yi, ch in enumerate(ys):
        for v in rows[ch]:
            ax.plot(v, yi, "o", color="C0", ms=6, alpha=0.7)
    ax.axvline(1.0, color="0.6", ls="--", label="σ≈2 (囲み溶解 中央)")
    ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys, fontproperties=prop)
    ax.set_xlabel("log2 σ (囲み溶解点)"); ax.legend(fontsize=8)
    ax.set_title("§5-3 リング字の囲み溶解点 (国は σ≈1.0 = log2 0 の例外)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_b1(B1, snaps, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(B1, cmap="viridis", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{B1[i,j]:.2f}", ha="center", va="center",
                    color="w" if B1[i, j] < 0.6 else "k")
    labs = [f"{np.log2(s):.1f}" for s in snaps]
    ax.set_xticks(range(3)); ax.set_xticklabels([f"mem {l}" for l in labs])
    ax.set_yticks(range(3)); ax.set_yticklabels([f"qry {l}" for l in labs])
    ax.set_title("§6 B1 尺跨ぎ想起 (log2 σ_s; chance=4.2%)")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_b2(shared, nonshared, contrast, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(nonshared, bins=30, density=True, alpha=0.4, color="0.6", label="非共有")
    ax.hist(shared, bins=12, density=True, alpha=0.5, color="C3", label="強共有(11)")
    for name, v in contrast.items():
        ax.axvline(v, color="C0", ls="--", lw=1)
        ax.text(v, ax.get_ylim()[1] * 0.9, name, rotation=90, fontsize=7, color="C0")
    ax.set_xlabel("blob-map cosine (σ_s=2)"); ax.set_ylabel("density")
    ax.set_title("§6 B2 三群 — 対照(形状類似)>強共有 ⇒ 形を見て部品を見ない")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_fidelity(log2, fid_mat, mean_fid, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in fid_mat:
        ax.plot(log2, row, color="0.85", lw=0.5)
    ax.plot(log2, mean_fid, "C3", lw=2.5, label=f"mean = {np.nanmean(mean_fid):.3f}")
    ax.set_xlabel("log2 σ (逆再生 4→0.5)"); ax.set_ylabel("cosine B̂_tree vs B")
    ax.set_title("§7 M3-2 木のみ再生 忠実度 (道を覚え歩幅を忘れる)")
    ax.set_ylim(0, 1.02); ax.invert_xaxis(); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_unreturnable(chs, per, recs, sig, idx, P2, prop, path):
    fig, axes = plt.subplots(len(chs), 2, figsize=(6, 3 * len(chs)))
    for r, ch in enumerate(chs):
        orig = P2[idx[ch]].reshape(GRID, GRID)
        final = replay_map(recs[ch], sig, sig[0], mode="tree").reshape(GRID, GRID)
        axes[r, 0].imshow(orig, cmap="magma"); axes[r, 0].axis("off")
        axes[r, 0].set_title(f"{ch} 原 map σ_s=2", fontproperties=prop, fontsize=9)
        axes[r, 1].imshow(final, cmap="magma"); axes[r, 1].axis("off")
        axes[r, 1].set_title(f"{ch} 逆再生終端 σ=0.5", fontproperties=prop, fontsize=9)
    fig.suptitle("§7 戻れない字 (リング字は囲みが戻らない)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_ephemeral(ed_l2, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ed_l2, bins=30, color="C4", alpha=0.8)
    ax.axvline(0, color="k", ls=":", label="log2σ=0 (σ=1, テクスチャ境界)")
    ax.axvline(float(np.median(ed_l2)), color="C3", ls="--", label=f"median={np.median(ed_l2):.2f}")
    ax.set_xlabel("log2 σ_death (ephemeral)"); ax.set_ylabel("count")
    ax.set_title("§7-2 ephemeral σ_death — 大半は粗スケール(前提崩れ)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_awakening(log2b, acc, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(log2b, acc, "C0-o")
    ax.axhline(1 / 24, color="k", ls=":", label="chance 4.2%")
    ax.set_xlabel("log2 β"); ax.set_ylabel("clean 自己想起 正解率")
    ax.set_title("§8 覚醒閾 — β≈4-6 で系が目覚める"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_correspondence(D_grid, sstar, log2s, betas, beta_c, c1_rho, c2_rho, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    im = ax[0].imshow(np.array(D_grid), aspect="auto", cmap="viridis_r",
                      extent=[log2s[0], log2s[-1], np.log2(betas[-1]), np.log2(betas[0])])
    ax[0].plot(sstar, [np.log2(b) for b in betas], "w.-", lw=1.5, label="σ*(β)")
    ax[0].set_xlabel("log2 σ"); ax[0].set_ylabel("log2 β")
    ax[0].set_title(f"§1/§8 天秤・似姿の軸 σ*(β)\nc1 ρ={c1_rho:.3f} (右下がり)")
    ax[0].legend(fontsize=8); fig.colorbar(im, ax=ax[0])
    bc = [b if b is not None else np.nan for b in beta_c]
    ax[1].plot(log2s, bc, "C2-o")
    ax[1].set_xlabel("log2 σ_q"); ax[1].set_ylabel("log2 β_c (ident≥0.9)")
    ax[1].set_title(f"§1/§8 天秤・代償の軸 β_c(σ)\nc2 ρ={c2_rho:.3f} (右上がり)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_control(T_grid, means, sds, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(T_grid, means, yerr=sds, fmt="C0-o", capsize=3)
    ax.axhline(1 / 24, color="k", ls=":", label="chance 4.2%")
    ax.set_xscale("symlog", linthresh=0.005); ax.set_ylim(0, 1.05)
    ax.set_xlabel("K_u ノイズ温度 T"); ax.set_ylabel("正解率 (±SD, 5 seed)")
    ax.set_title("§8 対照 K_u(T) — 平坦 1.0 (物理熱は溶かさない)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _write_readme(out):
    txt = """# v0.2 M5 素材 — 弧の再演 (regenerated figures)

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
"""
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    raise SystemExit(main())
