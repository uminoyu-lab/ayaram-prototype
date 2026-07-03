"""v0.2 M1 — 漢字の熱溶解実験 (kanji heat-dissolution).

Builds the 24-kanji × 57-σ Gaussian scale-space, tracks bipolar blobs across
scale, and characterises how ink structure dissolves as σ grows.

依頼書 §3-4 + 指示書 M1 DoD 1-5 (綾修正 A/B 反映 2026-07-03):
  DoD-1  band cluster of pooled log(σ_death): KDE + formal multimodality test
  DoD-2  order/dwell P(n): (A) ink polarity only, (B) exclude σ ≥ first-n=1;
         anchor set {明2 林2 炎2 晶3 森3 岩2 品3 語2 銀2 国2 回2}, ambiguous
         {川 田 樹 鬱} observation-only
  DoD-3  all 24 chars, deterministic conv (no seed), env recorded
  DoD-5  runtime reported

Outputs -> results/v0.2_m1/.  Deterministic (CPU); the only RNG is the DoD-1
Silverman bootstrap, whose seed is recorded in metadata.
"""

from __future__ import annotations

import json
import os
import platform
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from ayaram.scalespace import (  # noqa: E402
    build_scale_space,
    detect_extrema,
    link_trajectories,
    normalized_log_response,
    sigma_grid,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
GLYPHS = os.path.join(_ROOT, "data", "glyphs_128", "glyphs_128.npz")
OUT = os.path.join(_ROOT, "results", "v0.2_m1")

# expected component counts (アル起案, 指示書 DoD-2). ambiguous = observation-only
ANCHORS: dict[str, int] = {
    "明": 2, "林": 2, "炎": 2, "晶": 3, "森": 3, "岩": 2,
    "品": 3, "語": 2, "銀": 2, "国": 2, "回": 2,
}
AMBIGUOUS: tuple[str, ...] = ("川", "田", "樹", "鬱")

DEVICE = "cpu"                     # deterministic + fastest here (see M0 probe)
THRESHOLDS = (0.03, 0.05, 0.08)    # 0.05 default + 2 sensitivity points
DEFAULT_THR = 0.05
BOOT_SEED = 20260703
BOOT_N = 1000


# --------------------------------------------------------------------------- #
# per-glyph scale-space + tracking
# --------------------------------------------------------------------------- #
def analyse_glyph(img, sig, rel_threshold):
    ss = build_scale_space(img, sig, device=DEVICE)
    R = normalized_log_response(ss, sig)
    ex = detect_extrema(R, sig, rel_threshold=rel_threshold)
    trajs = link_trajectories(ex, sig)
    return ss, R, ex, trajs


def n_of_sigma(extrema, K, polarity=None):
    n = np.zeros(K, dtype=int)
    for e in extrema:
        if polarity is None or e.polarity == polarity:
            n[e.k] += 1
    return n


def deaths(trajs, polarity, include_survivor=False):
    """σ_death values (vanish+merge). survivors right-censored, excluded by default."""
    out = []
    for t in trajs:
        if t.polarity != polarity:
            continue
        if t.terminal == "survivor" and not include_survivor:
            continue
        out.append(t.sigma_death)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# DoD-1: KDE + Silverman multimodality bootstrap
# --------------------------------------------------------------------------- #
def _kde(data, grid, h):
    u = (grid[:, None] - data[None, :]) / h
    return np.exp(-0.5 * u * u).sum(1) / (len(data) * h * np.sqrt(2 * np.pi))


def _count_modes(density):
    # interior strict local maxima
    d = density
    return int(np.sum((d[1:-1] > d[:-2]) & (d[1:-1] > d[2:])))


def _critical_bandwidth(data, k, grid, lo=1e-3, hi=None, iters=60):
    """Smallest h whose Gaussian KDE has <= k modes (modes monotone in h)."""
    if hi is None:
        hi = (data.max() - data.min()) + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _count_modes(_kde(data, grid, mid)) <= k:
            hi = mid
        else:
            lo = mid
    return hi


def silverman_multimodality(data, seed=BOOT_SEED, B=BOOT_N, ngrid=512):
    """Silverman (1981) smoothed-bootstrap test.

    H0: the density of log(σ_death) is **unimodal** (m <= 1).
    H1: multimodal (m >= 2).
    Statistic: p = fraction of variance-corrected smoothed-bootstrap resamples
    (drawn with the critical bandwidth h1 = smallest bandwidth making the
    observed KDE unimodal) that are themselves multimodal at h1.
    Direction (standard Silverman): **small p rejects H0** — a small p means the
    observed multimodality survives smoothing at h1, so it is not a small-sample
    artifact.  p < 0.05 => reject unimodality (bands present).
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    grid = np.linspace(data.min() - 1.0, data.max() + 1.0, ngrid)
    h1 = _critical_bandwidth(data, 1, grid)
    m_rot = _count_modes(_kde(data, grid, 1.06 * data.std() * n ** (-1 / 5)))
    rng = np.random.default_rng(seed)
    var = data.var()
    xbar = data.mean()
    multimodal = 0
    for _ in range(B):
        idx = rng.integers(0, n, n)
        eps = rng.standard_normal(n)
        y = xbar + (data[idx] - xbar + h1 * eps) / np.sqrt(1.0 + h1 * h1 / var)
        g = np.linspace(y.min() - 1.0, y.max() + 1.0, ngrid)
        if _count_modes(_kde(y, g, h1)) > 1:
            multimodal += 1
    p = multimodal / B
    return {
        "n": n,
        "critical_bandwidth_h1": float(h1),
        "modes_at_rule_of_thumb": m_rot,
        "silverman_p_value": p,
        "reject_unimodal_5pct": bool(p < 0.05),
        "bootstrap_B": B,
        "seed": seed,
        "null_hypothesis": "log(sigma_death) density is unimodal (m<=1)",
        "interpretation": "small p rejects unimodality; p<0.05 => multimodal (bands present)",
    }


# --------------------------------------------------------------------------- #
# plotting helpers
# --------------------------------------------------------------------------- #
def _cjk_prop():
    for fp in (r"C:\Windows\Fonts\NotoSansJP-VF.ttf", r"C:\Windows\Fonts\YuGothM.ttc",
               r"C:\Windows\Fonts\meiryo.ttc", r"C:\Windows\Fonts\msgothic.ttc"):
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
                return font_manager.FontProperties(fname=fp)
            except (RuntimeError, OSError):
                return None
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "filmstrips"), exist_ok=True)
    prop = _cjk_prop()
    t_start = time.time()

    data = np.load(GLYPHS, allow_pickle=False)
    glyphs = data["glyphs"]
    meta_g = json.load(open(os.path.join(_ROOT, "data", "glyphs_128", "metadata.json"),
                             encoding="utf-8"))
    chars = meta_g["chars"]
    sig = sigma_grid()
    K = len(sig)
    log_sig = np.log2(sig)

    # ---- default-threshold full analysis, per char ----
    per_char = {}
    traj_rows = []
    for i, ch in enumerate(chars):
        ss, R, ex, trajs = analyse_glyph(glyphs[i], sig, DEFAULT_THR)
        n_ink = n_of_sigma(ex, K, "ink")
        n_gnd = n_of_sigma(ex, K, "ground")
        d_ink = deaths(trajs, "ink")
        d_gnd = deaths(trajs, "ground")
        n_surv = sum(1 for t in trajs if t.terminal == "survivor")
        n_birthev = sum(1 for t in trajs if t.birth_event)
        per_char[ch] = dict(
            i=i, n_ink=n_ink, n_gnd=n_gnd, d_ink=d_ink, d_gnd=d_gnd,
            trajs=trajs, ss=ss, n_surv=n_surv, n_birthev=n_birthev,
        )
        for t in trajs:
            traj_rows.append((ch, t.id, t.polarity, round(t.sigma_birth, 4),
                              round(t.sigma_death, 4), t.terminal,
                              int(t.birth_event), len(t.points)))

    # ---- trajectory summary table (CSV) ----
    with open(os.path.join(OUT, "trajectory_summary.csv"), "w", encoding="utf-8") as f:
        f.write("char,id,polarity,sigma_birth,sigma_death,terminal,birth_event,n_points\n")
        for r in traj_rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # ---- DoD-1: pooled ink σ_death band ----
    pooled = np.concatenate([per_char[ch]["d_ink"] for ch in chars])
    pooled_log = np.log2(pooled[pooled > 0])
    dod1 = silverman_multimodality(pooled_log)
    _plot_dod1(pooled_log, dod1, OUT)

    # ---- σ_death histogram (pooled + small multiples) ----
    _plot_death_hist(chars, per_char, log_sig, OUT, prop)

    # ---- n(σ) step grid ----
    _plot_nsigma(chars, per_char, log_sig, OUT, prop)

    # ---- death/birth scatter char × log σ ----
    _plot_scatter(chars, per_char, OUT, prop)

    # ---- filmstrips (1 slice / octave) ----
    oct_k = list(range(0, K, 8))
    for ch in chars:
        _filmstrip(ch, per_char[ch], sig, oct_k, OUT, prop)

    # ---- DoD-2: P(n) dwell, anchor local-max test ----
    dod2 = _dod2(chars, per_char, K)
    _plot_dod2(chars, per_char, dod2, K, OUT, prop)

    # ---- threshold sensitivity (0.03 / 0.05 / 0.08) ----
    sens = _threshold_sensitivity(chars, glyphs, sig)

    runtime = time.time() - t_start
    metadata = {
        "milestone": "v0.2-M1",
        "device": DEVICE,
        "n_chars": len(chars),
        "sigma_grid": {"sigma0": 0.5, "per_octave": 8, "n_slices": K,
                       "sigma_min": float(sig[0]), "sigma_max": float(sig[-1])},
        "default_threshold": DEFAULT_THR,
        "thresholds_tested": list(THRESHOLDS),
        "anchors": ANCHORS,
        "ambiguous": list(AMBIGUOUS),
        "runtime_sec": round(runtime, 2),
        "env": {"python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__},
        "dod1": dod1,
        "dod2": dod2,
        "threshold_sensitivity": sens,
        "seed_note": "convolution deterministic (no seed); only DoD-1 bootstrap uses seed",
    }
    json.dump(metadata, open(os.path.join(OUT, "metadata.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"M1 done in {runtime:.1f}s  device={DEVICE}")
    print(f"DoD-1 pooled ink deaths n={dod1['n']} modes(rot)={dod1['modes_at_rule_of_thumb']} "
          f"silverman_p={dod1['silverman_p_value']:.3f}")
    print(f"DoD-2 anchors passed {dod2['n_pass']}/{dod2['n_anchor']}")
    return 0


# --------------------------------------------------------------------------- #
def _plot_dod1(pooled_log, dod1, out):
    grid = np.linspace(pooled_log.min() - 1, pooled_log.max() + 1, 512)
    h1 = dod1["critical_bandwidth_h1"]
    h_rot = 1.06 * pooled_log.std() * len(pooled_log) ** (-1 / 5)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pooled_log, bins=40, density=True, alpha=0.3, color="0.6")
    ax.plot(grid, _kde(pooled_log, grid, h_rot), "b-", label=f"KDE rot h={h_rot:.2f}")
    ax.plot(grid, _kde(pooled_log, grid, h1), "r--", lw=1, label=f"KDE crit h1={h1:.2f}")
    ax.set_xlabel("log2 σ_death (ink, pooled)")
    ax.set_ylabel("density")
    ax.set_title(f"DoD-1 pooled ink σ_death  modes(rot)={dod1['modes_at_rule_of_thumb']}  "
                 f"Silverman p={dod1['silverman_p_value']:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "dod1_kde.png"), dpi=150); plt.close(fig)


def _plot_death_hist(chars, per_char, log_sig, out, prop):
    fig, axes = plt.subplots(4, 6, figsize=(15, 9))
    edges = np.linspace(log_sig[0], log_sig[-1], 25)
    for ax, ch in zip(axes.flat, chars):
        d = per_char[ch]["d_ink"]
        if len(d):
            ax.hist(np.log2(d[d > 0]), bins=edges, color="C3", alpha=0.8)
        ax.set_title(ch, fontproperties=prop, fontsize=10)
        ax.set_xlim(log_sig[0], log_sig[-1]); ax.tick_params(labelsize=6)
    fig.suptitle("σ_death histogram per char (ink, log2 σ; survivors excluded)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "sigma_death_hist.png"), dpi=140); plt.close(fig)


def _plot_nsigma(chars, per_char, log_sig, out, prop):
    fig, axes = plt.subplots(4, 6, figsize=(15, 9))
    for ax, ch in zip(axes.flat, chars):
        pc = per_char[ch]
        ax.step(log_sig, pc["n_ink"], where="mid", color="C3", lw=1.2, label="ink")
        ax.step(log_sig, pc["n_gnd"], where="mid", color="C0", lw=0.8, alpha=0.7, label="ground")
        ax.set_yscale("symlog")
        ax.set_title(ch, fontproperties=prop, fontsize=10)
        ax.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("n(σ) alive-blob count per char (log2 σ x-axis, symlog y)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "n_sigma_grid.png"), dpi=140); plt.close(fig)


def _plot_scatter(chars, per_char, out, prop):
    fig, ax = plt.subplots(figsize=(10, 7))
    for row, ch in enumerate(chars):
        for t in per_char[ch]["trajs"]:
            if t.polarity != "ink":
                continue
            x = np.log2(t.sigma_death)
            if t.terminal == "survivor":
                ax.plot(x, row, ">", color="0.5", ms=3)
            elif t.birth_event:
                ax.plot(x, row, "^", color="C2", ms=3, alpha=0.6)  # generation
            else:
                ax.plot(x, row, ".", color="C3", ms=3, alpha=0.5)  # death
    ax.set_yticks(range(len(chars)))
    ax.set_yticklabels(chars, fontproperties=prop)
    ax.set_xlabel("log2 σ")
    ax.set_title("ink death (red .), generation (green ^, birth k>0), survivor (grey >)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "death_scatter.png"), dpi=150); plt.close(fig)


def _filmstrip(ch, pc, sig, oct_k, out, prop):
    ss = pc["ss"].cpu().numpy()
    fig, axes = plt.subplots(1, len(oct_k), figsize=(2 * len(oct_k), 2.4))
    for ax, k in zip(axes, oct_k):
        ax.imshow(ss[k], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"σ={sig[k]:.1f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(ch, fontproperties=prop, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "filmstrips", f"U{ord(ch):04X}.png"), dpi=120)
    plt.close(fig)


def _dod2(chars, per_char, K):
    """P(n) dwell with (A) ink polarity, (B) exclude σ >= first n=1 arrival.

    'first arrival' := smallest slice k (increasing σ) with n_ink(k) == 1.
    Dwell measured over k in [0, k_first1); compared among neighbour integers.
    Anchor passes if P(C) is a strict local max over {C-1, C, C+1}.
    """
    result = {"rule": "P(n)=# slices with n_ink==n over k<k_first1; "
                      "k_first1 = first slice (increasing σ) with n_ink==1; "
                      "anchor pass = P(C) strict local max over {C-1,C,C+1}",
              "chars": {}, "n_anchor": len(ANCHORS), "n_pass": 0}
    for ch in chars:
        n_ink = per_char[ch]["n_ink"]
        ones = np.where(n_ink == 1)[0]
        k1 = int(ones[0]) if len(ones) else K
        window = n_ink[:k1]
        maxn = int(window.max()) if len(window) else 0
        P = {int(n): int(np.sum(window == n)) for n in range(2, maxn + 1)}
        entry = {"k_first1": k1, "P": P}
        if ch in ANCHORS:
            C = ANCHORS[ch]
            pc_ = P.get(C, 0)
            passed = pc_ > P.get(C - 1, 0) and pc_ > P.get(C + 1, 0) and pc_ > 0
            entry.update({"expected_C": C, "P_C": pc_,
                          "P_Cm1": P.get(C - 1, 0), "P_Cp1": P.get(C + 1, 0),
                          "pass": bool(passed)})
            result["n_pass"] += int(passed)
        elif ch in AMBIGUOUS:
            entry["role"] = "ambiguous (observation only)"
        result["chars"][ch] = entry
    return result


def _plot_dod2(chars, per_char, dod2, K, out, prop):
    anchors = [c for c in chars if c in ANCHORS]
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    for ax, ch in zip(axes.flat, anchors):
        e = dod2["chars"][ch]
        ns = sorted(e["P"])
        vals = [e["P"][n] for n in ns]
        C = e["expected_C"]
        colors = ["C1" if n == C else "0.6" for n in ns]
        ax.bar(ns, vals, color=colors)
        ax.set_title(f"{ch} C={C} {'OK' if e['pass'] else 'x'}",
                     fontproperties=prop, fontsize=10)
        ax.set_xlabel("n"); ax.tick_params(labelsize=6)
    for ax in axes.flat[len(anchors):]:
        ax.axis("off")
    fig.suptitle(f"DoD-2 P(n) dwell for anchors (orange = expected C; n=1+ excluded). "
                 f"pass {dod2['n_pass']}/{dod2['n_anchor']}")
    fig.tight_layout(); fig.savefig(os.path.join(out, "dod2_pn.png"), dpi=140); plt.close(fig)


def _threshold_sensitivity(chars, glyphs, sig):
    K = len(sig)
    table = {}
    for thr in THRESHOLDS:
        n_traj = 0
        deaths_all = []
        for i, ch in enumerate(chars):
            ss = build_scale_space(glyphs[i], sig, device=DEVICE)
            R = normalized_log_response(ss, sig)
            ex = detect_extrema(R, sig, rel_threshold=thr)
            trajs = link_trajectories(ex, sig)
            n_traj += len(trajs)
            deaths_all.extend(deaths(trajs, "ink"))
        d = np.asarray(deaths_all)
        dl = np.log2(d[d > 0]) if len(d) else np.array([0.0])
        table[str(thr)] = {"total_trajectories": n_traj,
                           "ink_deaths": int(len(d)),
                           "median_log2_sigma_death": float(np.median(dl)),
                           "iqr_log2": [float(np.percentile(dl, 25)),
                                        float(np.percentile(dl, 75))]}
    return table


if __name__ == "__main__":
    raise SystemExit(main())
