"""v0.2 M1b — 解析追補: window確定 / churn低減 / 綾機構仮説判定.

No new data: recompute from data/glyphs_128/glyphs_128.npz (指示書 M1b).
Deterministic (CPU; CUDA unused, M1 踏襲). Outputs -> results/v0.2_m1b/.

Pipeline (指示書 作業項目):
  gate  reproducibility: gap=0 / no-window / no-lifetime / thr .05 / border 3
        must reproduce M1 (traj≈6794, median log2σ=1.75, bimodal, reject) else STOP
  (1)   DoD-1 re-test on window σ>=1 (window-only main; churn-reduced reference)
  (2)   window [1,16] applied to main figures (n(σ), death hist, death scatter)
  (3)   churn reduction (gap=1 + lifetime>=2) -> DoD-2 re-judge + fail class (a)/(b)
  (4)   polarity-sum series (ink + hole-derived ground) + 綾 mechanism-hypothesis
  (5)   border sensitivity 2/3/5

確定事項 (G1 approved):
  window   σ in [1,16] (log2 in [0,4]); σ<1 texture (separate); σ>16 -> survivor
  split    DoD-1 main = window only; DoD-2 = window + churn reduction
  gap      1-slice gap linking (max_gap=1); gate scaled by gap (CC 裁量)
  lifetime traj with <2 slices dropped from n(σ) aggregation only
  hole tag base white regions labelled; border-connected = open, else = hole;
           ground blob tagged by σ_birth-centre label; only hole-derived summed
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections import deque

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ayaram.scalespace import (  # noqa: E402
    build_scale_space,
    detect_extrema,
    link_trajectories,
    normalized_log_response,
    sigma_grid,
)
from run_m1_heat_dissolution import (  # noqa: E402
    ANCHORS,
    AMBIGUOUS,
    _cjk_prop,
    _kde,
    deaths,
    silverman_multimodality,
)

GLYPHS = os.path.join(_ROOT, "data", "glyphs_128", "glyphs_128.npz")
OUT = os.path.join(_ROOT, "results", "v0.2_m1b")

DEVICE = "cpu"
DEFAULT_THR = 0.05
WIN_LO, WIN_HI = 1.0, 16.0            # structural analysis window (σ)
MAX_GAP = 1                           # churn reduction: 1-slice gap
MIN_LIFETIME = 2                      # drop <2-slice trajectories from n(σ)
BORDERS = (2, 3, 5)
BASELINE = {"total_traj": 6794, "median_log2_sigma_death": 1.75}

# ring-component chars (10) + control 晶 for the 合算 hypothesis (指示書 (4))
RING_CHARS = tuple("口日月田回国品語銀明") + ("晶",)
SINGLE_RING = tuple("口日月田")        # 単体リング字
COMPLEX_RING = tuple("回国品語銀明")   # 複合字


# --------------------------------------------------------------------------- #
# window-aware death partitioning
# --------------------------------------------------------------------------- #
def partition_deaths(trajs, polarity):
    """Return (structural[1,16], texture<1, censored>16) ink/ground σ_death."""
    struct, texture, censored = [], [], []
    for t in trajs:
        if t.polarity != polarity or t.terminal == "survivor":
            continue
        s = t.sigma_death
        if s < WIN_LO:
            texture.append(s)
        elif s > WIN_HI:
            censored.append(s)
        else:
            struct.append(s)
    return np.asarray(struct), np.asarray(texture), np.asarray(censored)


def n_from_trajs(trajs, K, polarity, min_lifetime=1, ids=None):
    """n(σ) alive-blob count from trajectory points (churn/lifetime aware)."""
    n = np.zeros(K, dtype=int)
    for t in trajs:
        if t.polarity != polarity or len(t.points) < min_lifetime:
            continue
        if ids is not None and t.id not in ids:
            continue
        for (k, _, _, _) in t.points:
            n[k] += 1
    return n


# --------------------------------------------------------------------------- #
# connected components (no scipy): background hole vs open labelling
# --------------------------------------------------------------------------- #
def _label(mask, conn=4):
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=int)
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn == 8:
        nb += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    cur = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lab[y, x] == 0:
                cur += 1
                q = deque([(y, x)]); lab[y, x] = cur
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in nb:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = cur; q.append((ny, nx))
    return lab, cur


def hole_map(glyph, thr=0.5):
    """Label background (glyph<thr); border-touching label=open, else=hole.

    Returns (label_map, hole_labels:set, open_labels:set, n_holes).
    Background labelled with 4-connectivity (complement to 8-conn ink).
    """
    bg = glyph < thr
    lab, n = _label(bg, conn=4)
    H, W = glyph.shape
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    holes = set(range(1, n + 1)) - border
    return lab, holes, border, len(holes)


def _bg_border_connected(ink_mask, probe):
    """True if the background pixel `probe` connects to the image border."""
    bg = ~ink_mask
    if not bg[probe]:
        return True  # probe became ink -> treat as dissolved/merged
    H, W = bg.shape
    seen = np.zeros_like(bg)
    q = deque()
    for x in range(W):
        if bg[0, x]: q.append((0, x)); seen[0, x] = True
        if bg[H - 1, x]: q.append((H - 1, x)); seen[H - 1, x] = True
    for y in range(H):
        if bg[y, 0]: q.append((y, 0)); seen[y, 0] = True
        if bg[y, W - 1]: q.append((y, W - 1)); seen[y, W - 1] = True
    while q:
        cy, cx = q.popleft()
        if (cy, cx) == probe:
            return True
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and bg[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True; q.append((ny, nx))
    return seen[probe]


def dissolution_sigma(ss, glyph, sig, thr=0.5):
    """For each base hole, the first σ where its centre becomes border-connected."""
    lab, holes, _, _ = hole_map(glyph, thr)
    out = []
    for hl in holes:
        ys, xs = np.where(lab == hl)
        cy, cx = int(round(ys.mean())), int(round(xs.mean()))
        # snap centre into the hole if the mean lands on ink
        if lab[cy, cx] != hl:
            cy, cx = int(ys[0]), int(xs[0])
        sd = None
        for k in range(ss.shape[0]):
            if _bg_border_connected(ss[k] >= thr, (cy, cx)):
                sd = float(sig[k]); break
        out.append({"center": [cy, cx], "dissolution_sigma": sd})
    return out


# --------------------------------------------------------------------------- #
# P(n) dwell within window [1,16], excluding σ >= first-n==1  (綾修正 A/B)
# --------------------------------------------------------------------------- #
def dwell_Pn(n_ink, sig):
    widx = [k for k in range(len(sig)) if WIN_LO <= sig[k] <= WIN_HI]
    k1 = next((k for k in widx if n_ink[k] == 1), None)
    ks = [k for k in widx if (k1 is None or k < k1)]
    if not ks:
        return {}, k1
    window = n_ink[ks]
    maxn = int(window.max())
    return {int(nn): int((window == nn).sum()) for nn in range(2, maxn + 1)}, k1


def anchor_verdict(P, C):
    pc = P.get(C, 0)
    passed = pc > 0 and pc > P.get(C - 1, 0) and pc > P.get(C + 1, 0)
    if passed:
        cls = "pass"
    elif pc > 0:
        cls = "a"   # C dwell exists but not local max
    else:
        cls = "b"   # C dwell absent
    return bool(passed), cls, pc


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    prop = _cjk_prop()
    t0 = time.time()

    d = np.load(GLYPHS, allow_pickle=False)
    glyphs = d["glyphs"]
    chars = json.load(open(os.path.join(_ROOT, "data", "glyphs_128", "metadata.json"),
                           encoding="utf-8"))["chars"]
    sig = sigma_grid(); K = len(sig); log_sig = np.log2(sig)

    # ---- gate + baseline (gap=0) and churn-reduced (gap=1) per char ----
    base, churn = {}, {}
    total_traj0 = 0
    for i, ch in enumerate(chars):
        ss = build_scale_space(glyphs[i], sig, device=DEVICE)
        R = normalized_log_response(ss, sig)
        ex = detect_extrema(R, sig, rel_threshold=DEFAULT_THR, border=3)
        t0g = link_trajectories(ex, sig, max_gap=0)
        t1g = link_trajectories(ex, sig, max_gap=MAX_GAP)
        total_traj0 += len(t0g)
        base[ch] = dict(ss=ss, trajs=t0g)
        churn[ch] = dict(trajs=t1g)

    pooled0 = np.concatenate([deaths(base[ch]["trajs"], "ink") for ch in chars])
    med0 = float(np.median(np.log2(pooled0[pooled0 > 0])))
    gate_s = silverman_multimodality(np.log2(pooled0[pooled0 > 0]))
    gate_pass = (total_traj0 == BASELINE["total_traj"]
                 and abs(med0 - BASELINE["median_log2_sigma_death"]) < 1e-9
                 and gate_s["modes_at_rule_of_thumb"] >= 2
                 and gate_s["reject_unimodal_5pct"])
    gate = {"total_traj": total_traj0, "median_log2_sigma_death": med0,
            "modes": gate_s["modes_at_rule_of_thumb"],
            "reject_unimodal": gate_s["reject_unimodal_5pct"], "pass": bool(gate_pass)}
    if not gate_pass:
        json.dump({"gate": gate, "STOP": "reproducibility gate failed"},
                  open(os.path.join(OUT, "metadata.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("STOP: reproducibility gate FAILED", gate)
        return 1
    print(f"gate PASS  traj={total_traj0} med={med0}")

    # ================= (1) DoD-1 windowed (main = window only) =================
    struct_pool, texture_pool = [], []
    for ch in chars:
        s, tx, _ = partition_deaths(base[ch]["trajs"], "ink")
        struct_pool.append(s); texture_pool.append(tx)
    struct_pool = np.concatenate(struct_pool)
    texture_pool = np.concatenate(texture_pool)
    win_log = np.log2(struct_pool)
    dod1_win = silverman_multimodality(win_log)

    # churn-reduced reference (gap=1 + lifetime>=2), same window
    struct_cr = []
    for ch in chars:
        keep = [t for t in churn[ch]["trajs"] if len(t.points) >= MIN_LIFETIME]
        s, _, _ = partition_deaths(keep, "ink")
        struct_cr.append(s)
    struct_cr = np.concatenate(struct_cr)
    dod1_cr = silverman_multimodality(np.log2(struct_cr))

    _plot_kde(win_log, dod1_win, "DoD-1 windowed σ∈[1,16] (main, gap=0)",
              os.path.join(OUT, "dod1_kde_windowed.png"), texture_pool)
    _plot_kde(np.log2(struct_cr), dod1_cr,
              "DoD-1 windowed + churn-reduced (reference)",
              os.path.join(OUT, "dod1_kde_windowed_churnreduced.png"), None)

    # ================= (2) window applied to main figures =================
    _fig_nsigma(chars, base, churn, log_sig, prop, os.path.join(OUT, "n_sigma_grid_windowed.png"))
    _fig_death_hist(chars, base, log_sig, prop, os.path.join(OUT, "sigma_death_hist_windowed.png"))
    _fig_scatter(chars, base, prop, os.path.join(OUT, "death_scatter_windowed.png"))

    # ================= (3) churn-reduced DoD-2 + fail class =================
    dod2 = {"chars": {}, "n_anchor": len(ANCHORS), "n_pass": 0, "fail_a": [], "fail_b": []}
    for ch in chars:
        keep = [t for t in churn[ch]["trajs"] if len(t.points) >= MIN_LIFETIME]
        n_ink = n_from_trajs(keep, K, "ink", MIN_LIFETIME)
        P, k1 = dwell_Pn(n_ink, sig)
        entry = {"P": P, "k_first1": k1}
        if ch in ANCHORS:
            C = ANCHORS[ch]
            passed, cls, pc = anchor_verdict(P, C)
            entry.update(expected_C=C, P_C=pc, cls=cls, passed=passed,
                         P_Cm1=P.get(C - 1, 0), P_Cp1=P.get(C + 1, 0))
            dod2["n_pass"] += int(passed)
            if cls == "a": dod2["fail_a"].append(ch)
            if cls == "b": dod2["fail_b"].append(ch)
        elif ch in AMBIGUOUS:
            entry["role"] = "ambiguous"
        dod2["chars"][ch] = entry
    _fig_dod2(chars, dod2, prop, os.path.join(OUT, "dod2_pn_revised.png"))

    # ================= (4) polarity-sum series + 綾 hypothesis =================
    sum_res = {"chars": {}}
    for ch in RING_CHARS:
        i = chars.index(ch)
        lab, holes, _, n_holes = hole_map(glyphs[i])
        keep = [t for t in churn[ch]["trajs"] if len(t.points) >= MIN_LIFETIME]
        # tag ground trajs by birth-centre label
        hole_ids, open_ids = set(), set()
        for t in keep:
            if t.polarity != "ground":
                continue
            _, y, x, _ = t.points[0]
            (hole_ids if lab[y, x] in holes else open_ids).add(t.id)
        n_ink = n_from_trajs(keep, K, "ink", MIN_LIFETIME)
        n_hole = n_from_trajs(keep, K, "ground", MIN_LIFETIME, ids=hole_ids)
        n_open = n_from_trajs(keep, K, "ground", MIN_LIFETIME, ids=open_ids)
        n_sum = n_ink + n_hole
        P_ink, _ = dwell_Pn(n_ink, sig)
        P_sum, _ = dwell_Pn(n_sum, sig)
        # "段の出現": a local-max integer in P_sum absent as local-max in P_ink
        steps_ink = _local_max_ns(P_ink)
        steps_sum = _local_max_ns(P_sum)
        new_steps = sorted(set(steps_sum) - set(steps_ink))
        diss = dissolution_sigma(base[ch]["ss"].cpu().numpy(), glyphs[i], sig)
        sum_res["chars"][ch] = {
            "n_holes": n_holes, "P_ink": P_ink, "P_sum": P_sum,
            "steps_ink": steps_ink, "steps_sum": steps_sum, "new_steps": new_steps,
            "dissolution_sigma": diss,
            "n_ink": n_ink.tolist(), "n_hole": n_hole.tolist(), "n_open": n_open.tolist(),
        }
    _fig_sum_series(RING_CHARS, sum_res, log_sig, prop, os.path.join(OUT, "polarity_sum_series.png"))

    # ================= (5) border sensitivity =================
    border_sens = {}
    for b in BORDERS:
        pool, npass = [], 0
        for i, ch in enumerate(chars):
            ss = base[ch]["ss"]
            R = normalized_log_response(ss, sig)
            ex = detect_extrema(R, sig, rel_threshold=DEFAULT_THR, border=b)
            tb = link_trajectories(ex, sig, max_gap=MAX_GAP)
            keep = [t for t in tb if len(t.points) >= MIN_LIFETIME]
            s, _, _ = partition_deaths([t for t in tb if t.terminal != "survivor"], "ink")
            pool.append(s)
            if ch in ANCHORS:
                n_ink = n_from_trajs(keep, K, "ink", MIN_LIFETIME)
                P, _ = dwell_Pn(n_ink, sig)
                npass += int(anchor_verdict(P, ANCHORS[ch])[0])
        pool = np.concatenate(pool); pl = np.log2(pool[pool > 0])
        s = silverman_multimodality(pl)
        border_sens[str(b)] = {"median_log2_sigma_death": float(np.median(pl)),
                               "modes_rot": s["modes_at_rule_of_thumb"],
                               "reject_unimodal": s["reject_unimodal_5pct"],
                               "dod2_pass": npass}

    runtime = time.time() - t0
    meta = {
        "milestone": "v0.2-M1b", "device": DEVICE, "runtime_sec": round(runtime, 2),
        "window": {"sigma_lo": WIN_LO, "sigma_hi": WIN_HI},
        "max_gap": MAX_GAP, "min_lifetime": MIN_LIFETIME,
        "reproducibility_gate": gate,
        "dod1_windowed_main": dod1_win,
        "dod1_windowed_churnreduced_ref": dod1_cr,
        "texture_deaths_excluded": int(len(texture_pool)),
        "dod2_revised": {k: dod2[k] for k in ("n_anchor", "n_pass", "fail_a", "fail_b")},
        "dod2_per_char": {ch: {kk: dod2["chars"][ch][kk]
                               for kk in ("expected_C", "P_C", "cls", "passed")
                               if kk in dod2["chars"][ch]}
                          for ch in ANCHORS},
        "polarity_sum": {ch: {"n_holes": sum_res["chars"][ch]["n_holes"],
                              "steps_ink": sum_res["chars"][ch]["steps_ink"],
                              "steps_sum": sum_res["chars"][ch]["steps_sum"],
                              "new_steps": sum_res["chars"][ch]["new_steps"],
                              "dissolution_sigma": sum_res["chars"][ch]["dissolution_sigma"]}
                         for ch in RING_CHARS},
        "border_sensitivity": border_sens,
        "env": {"python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__},
        "seed_note": "conv deterministic; only Silverman bootstrap uses seed",
    }
    json.dump(meta, open(os.path.join(OUT, "metadata.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"M1b done {runtime:.1f}s")
    print(f"(1) DoD-1 win main modes={dod1_win['modes_at_rule_of_thumb']} "
          f"p={dod1_win['silverman_p_value']:.3f} reject={dod1_win['reject_unimodal_5pct']}; "
          f"churn-ref modes={dod1_cr['modes_at_rule_of_thumb']} reject={dod1_cr['reject_unimodal_5pct']}")
    print(f"(3) DoD-2 revised pass {dod2['n_pass']}/{dod2['n_anchor']} "
          f"fail_a={dod2['fail_a']} fail_b={dod2['fail_b']}")
    for ch in RING_CHARS:
        r = sum_res["chars"][ch]
        print(f"(4) {ch} holes={r['n_holes']} ink_steps={r['steps_ink']} "
              f"sum_steps={r['steps_sum']} new={r['new_steps']}")
    print(f"(5) border {border_sens}")
    return 0


def _local_max_ns(P):
    """integers n whose dwell is a strict local max over {n-1,n,n+1}."""
    out = []
    for n in P:
        if P[n] > 0 and P[n] > P.get(n - 1, 0) and P[n] > P.get(n + 1, 0):
            out.append(int(n))
    return sorted(out)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _plot_kde(logd, res, title, path, texture):
    grid = np.linspace(logd.min() - 1, logd.max() + 1, 512)
    h1 = res["critical_bandwidth_h1"]
    h_rot = 1.06 * logd.std() * len(logd) ** (-1 / 5)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(logd, bins=35, density=True, alpha=0.3, color="0.6")
    ax.plot(grid, _kde(logd, grid, h_rot), "b-", label=f"KDE rot h={h_rot:.2f}")
    ax.plot(grid, _kde(logd, grid, h1), "r--", lw=1, label=f"crit h1={h1:.2f}")
    if texture is not None and len(texture):
        ax.axvspan(np.log2(texture).min(), 0, color="orange", alpha=0.08,
                   label=f"texture σ<1 (n={len(texture)}, excl.)")
    ax.set_xlabel("log2 σ_death (ink)"); ax.set_ylabel("density")
    ax.set_title(f"{title}\nmodes={res['modes_at_rule_of_thumb']} "
                 f"Silverman p={res['silverman_p_value']:.3f} "
                 f"reject_unimodal={res['reject_unimodal_5pct']}", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _fig_nsigma(chars, base, churn, log_sig, prop, path):
    fig, axes = plt.subplots(4, 6, figsize=(15, 9))
    for ax, ch in zip(axes.flat, chars):
        n0 = n_from_trajs(base[ch]["trajs"], len(log_sig), "ink", 1)
        keep = [t for t in churn[ch]["trajs"] if len(t.points) >= MIN_LIFETIME]
        n1 = n_from_trajs(keep, len(log_sig), "ink", MIN_LIFETIME)
        ax.step(log_sig, n0, where="mid", color="0.7", lw=0.8, label="ink gap0")
        ax.step(log_sig, n1, where="mid", color="C3", lw=1.2, label="ink churn-red")
        ax.axvspan(0, 4, color="C2", alpha=0.06)
        ax.set_yscale("symlog"); ax.set_xlim(log_sig[0], log_sig[-1])
        ax.set_title(ch, fontproperties=prop, fontsize=10); ax.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=5)
    fig.suptitle("n(σ) ink: gap0 (grey) vs churn-reduced (red); window σ∈[1,16] shaded")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_death_hist(chars, base, log_sig, prop, path):
    fig, axes = plt.subplots(4, 6, figsize=(15, 9))
    edges = np.linspace(0, 4, 21)
    for ax, ch in zip(axes.flat, chars):
        s, _, _ = partition_deaths(base[ch]["trajs"], "ink")
        if len(s):
            ax.hist(np.log2(s), bins=edges, color="C3", alpha=0.8)
        ax.set_xlim(0, 4); ax.set_title(ch, fontproperties=prop, fontsize=10)
        ax.tick_params(labelsize=6)
    fig.suptitle("σ_death histogram per char, windowed σ∈[1,16] (log2 σ), ink")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_scatter(chars, base, prop, path):
    fig, ax = plt.subplots(figsize=(10, 7))
    for row, ch in enumerate(chars):
        for t in base[ch]["trajs"]:
            if t.polarity != "ink" or t.terminal == "survivor":
                continue
            s = t.sigma_death
            if s < WIN_LO or s > WIN_HI:
                continue
            x = np.log2(s)
            c = "C2" if t.birth_event else "C3"
            ax.plot(x, row, ".", color=c, ms=3, alpha=0.5)
    ax.set_yticks(range(len(chars))); ax.set_yticklabels(chars, fontproperties=prop)
    ax.set_xlim(0, 4); ax.set_xlabel("log2 σ (window [1,16])")
    ax.set_title("ink death within window (red=continued, green=post-birth)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _fig_dod2(chars, dod2, prop, path):
    anchors = [c for c in chars if c in ANCHORS]
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    for ax, ch in zip(axes.flat, anchors):
        e = dod2["chars"][ch]; C = e["expected_C"]
        ns = sorted(e["P"]) or [2]
        ax.bar(ns, [e["P"].get(n, 0) for n in ns],
               color=["C1" if n == C else "0.6" for n in ns])
        tag = {"pass": "OK", "a": "a", "b": "b"}[e["cls"]]
        ax.set_title(f"{ch} C={C} [{tag}]", fontproperties=prop, fontsize=10)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(anchors):]:
        ax.axis("off")
    fig.suptitle(f"DoD-2 revised (churn-reduced, window). pass {dod2['n_pass']}/{dod2['n_anchor']} "
                 f"| fail_a={dod2['fail_a']} fail_b={dod2['fail_b']}")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _fig_sum_series(ring_chars, sum_res, log_sig, prop, path):
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    for ax, ch in zip(axes.flat, ring_chars):
        r = sum_res["chars"][ch]
        ni = np.array(r["n_ink"]); nh = np.array(r["n_hole"]); ns = ni + nh
        ax.step(log_sig, ni, where="mid", color="C3", lw=1.1, label="ink")
        ax.step(log_sig, nh, where="mid", color="C0", lw=0.9, label="ground(hole)")
        ax.step(log_sig, ns, where="mid", color="k", lw=1.3, label="sum")
        ax.step(log_sig, np.array(r["n_open"]), where="mid", color="0.7", lw=0.6,
                ls=":", label="ground(open) ref")
        ax.axvspan(0, 4, color="C2", alpha=0.06)
        ax.set_yscale("symlog"); ax.set_xlim(log_sig[0], log_sig[-1])
        ax.set_title(f"{ch} holes={r['n_holes']} new={r['new_steps']}",
                     fontproperties=prop, fontsize=9); ax.tick_params(labelsize=6)
    for ax in axes.flat[len(ring_chars):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=5)
    fig.suptitle("polarity-sum n(σ): ink + hole-ground = sum (open-ground ref, dotted). window shaded")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
