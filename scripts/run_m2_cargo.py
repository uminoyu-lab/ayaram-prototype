"""v0.2 M2 — 踊り場スナップショット → Hopfield 積み荷 (Phase A + Phase B).

Recompute-only from data/glyphs_128/glyphs_128.npz (official linking = gap=1,
M1b). Deterministic CPU. Outputs -> results/v0.2_m2/.

Phase A: A1 blob maps (24×3σ_s) -> A2 store + 24/24 clean-recall GATE (STOP if
fail) -> A3 discrimination -> σ_s* -> A4 merge genealogy JSON + consistency.
Phase B (auto after gate): B1 3×3 scale-crossing recall (chance 4.2% + diag
ratio) -> B2 part interference (Mann-Whitney, position-fixed caveat) -> B3 f_c
concat, w in {0.1,0.3,1.0}.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ayaram.scalespace import (  # noqa: E402
    build_scale_space, detect_extrema, link_trajectories,
    normalized_log_response, sigma_grid,
)
from ayaram.cargo import (  # noqa: E402
    GRID, SNAP_LOG2, build_blob_map, build_genealogy, check_genealogy,
    hopfield_recall, nearest_pattern, snapshot_sigmas,
)
from run_m1_heat_dissolution import _cjk_prop  # noqa: E402
from run_m1b_analysis import hole_map, dissolution_sigma, MAX_GAP, MIN_LIFETIME  # noqa: E402

GLYPHS = os.path.join(_ROOT, "data", "glyphs_128", "glyphs_128.npz")
OUT = os.path.join(_ROOT, "results", "v0.2_m2")

BETA = 16.0            # raised from v0.1 default 1.0 (see design_decisions); gate needs it
RECALL_STEPS = 3
NOISE_LEVELS = (1.0, 2.0, 3.0, 5.0)   # harsh enough to separate σ_s (β=16 recall is robust)
NOISE_SEED = 20260703
NOISE_REPEATS = 5
W_GRID = (0.1, 0.3, 1.0)

# 強共有 11 pairs + 形状類似対照 3 (指示書 付録)
SHARED_PAIRS = [("木", "林"), ("木", "森"), ("林", "森"), ("火", "炎"),
                ("日", "明"), ("日", "晶"), ("明", "晶"), ("月", "明"),
                ("口", "品"), ("口", "回"), ("山", "岩")]
CONTRAST_PAIRS = [("日", "月"), ("田", "回"), ("田", "国")]
RING_CHARS = tuple("口日月田回国品語銀明")


def _phi(z):  # standard normal CDF
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_whitney_u(a, b):
    """One-sided Mann-Whitney U for H1: a > b. Tie-corrected normal approx."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    Ra = ranks[:na].sum()
    Ua = Ra - na * (na + 1) / 2.0
    mu = na * nb / 2.0
    # tie correction
    _, tc = np.unique(allv, return_counts=True)
    tie = np.sum(tc ** 3 - tc)
    N = na + nb
    var = na * nb / 12.0 * ((N + 1) - tie / (N * (N - 1)))
    z = (Ua - mu) / math.sqrt(var) if var > 0 else 0.0
    return {"U": float(Ua), "z": float(z), "p_one_sided_a_gt_b": float(1.0 - _phi(z)),
            "median_a": float(np.median(a)), "median_b": float(np.median(b))}


def noisy_recall_accuracy(P, beta, eta, seed):
    rng = np.random.default_rng(seed)
    N, d = P.shape
    ok = 0
    for r in range(NOISE_REPEATS):
        for i in range(N):
            eps = rng.standard_normal(d) / math.sqrt(d) * eta
            q = P[i] + eps
            q = q / (np.linalg.norm(q) + 1e-12)
            if nearest_pattern(P, hopfield_recall(P, q, beta)) == i:
                ok += 1
    return ok / (N * NOISE_REPEATS)


def main():
    os.makedirs(OUT, exist_ok=True)
    prop = _cjk_prop()
    t0 = time.time()

    dat = np.load(GLYPHS, allow_pickle=False)
    glyphs = dat["glyphs"]
    chars = json.load(open(os.path.join(_ROOT, "data", "glyphs_128", "metadata.json"),
                           encoding="utf-8"))["chars"]
    sig = sigma_grid(); K = len(sig)

    # ---- recompute trajectories (official gap=1) + keep ss for f_c ----
    per, ss_np, holes_per, diss_per = {}, {}, {}, {}
    for i, ch in enumerate(chars):
        ss = build_scale_space(glyphs[i], sig, device="cpu")
        R = normalized_log_response(ss, sig)
        ex = detect_extrema(R, sig, 0.05, 3)
        per[ch] = link_trajectories(ex, sig, max_gap=MAX_GAP)
        ss_np[ch] = ss.cpu().numpy()
        _, holes, _, nh = hole_map(glyphs[i])
        holes_per[ch] = nh
        diss = [h["dissolution_sigma"] for h in dissolution_sigma(ss_np[ch], glyphs[i], sig)
                if h["dissolution_sigma"]]
        diss_per[ch] = float(np.median(np.log2(diss))) if diss else 0.0

    snaps = snapshot_sigmas(sig)  # [2.0, 2.83, 4.0]

    # ================= A1 blob maps =================
    maps = {}
    for s in snaps:
        maps[s] = np.stack([build_blob_map(per[ch], sig, s) for ch in chars], axis=0)
    npz = {f"sig_{l}": maps[s] for l, s in zip(SNAP_LOG2, snaps)}
    np.savez_compressed(os.path.join(OUT, "blob_maps.npz"), chars=np.array(chars), **npz)
    for l, s in zip(SNAP_LOG2, snaps):
        _contact(maps[s], chars, f"blob map σ_s={s:.2f} (log2={l})",
                 os.path.join(OUT, f"blob_maps_log2_{l}.png"), prop)

    # ================= A2 clean-recall gate =================
    gate = {}
    for l, s in zip(SNAP_LOG2, snaps):
        P = maps[s]
        ok = sum(nearest_pattern(P, hopfield_recall(P, P[i], BETA)) == i for i in range(len(chars)))
        gate[f"log2_{l}"] = {"clean_recall": f"{ok}/{len(chars)}", "pass": ok == len(chars)}
    gate_pass = all(v["pass"] for v in gate.values())
    if not gate_pass:
        json.dump({"A2_gate": gate, "STOP": "clean-recall gate failed"},
                  open(os.path.join(OUT, "metadata.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("STOP: A2 clean-recall gate FAILED", gate)
        return 1
    print("A2 gate PASS", {k: v["clean_recall"] for k, v in gate.items()})

    # ================= A3 discrimination + σ_s* =================
    a3 = {}
    for l, s in zip(SNAP_LOG2, snaps):
        P = maps[s]
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        C = Pn @ Pn.T
        off = C[~np.eye(len(chars), dtype=bool)]
        acc = {f"eta_{eta}": noisy_recall_accuracy(P, BETA, eta, NOISE_SEED)
               for eta in NOISE_LEVELS}
        a3[f"log2_{l}"] = {"mean_off_cosine": float(off.mean()),
                           "max_off_cosine": float(off.max()),
                           "noisy_recall": acc,
                           "noisy_recall_avg": float(np.mean(list(acc.values())))}
    # σ_s* = highest avg noisy recall; tie-break lowest mean off-cosine
    best = max(SNAP_LOG2, key=lambda l: (a3[f"log2_{l}"]["noisy_recall_avg"],
                                         -a3[f"log2_{l}"]["mean_off_cosine"]))
    sig_star = float(2.0 ** best)
    a3["sigma_star_log2"] = best
    a3["sigma_star"] = sig_star
    _plot_a3(a3, os.path.join(OUT, "a3_discrimination.png"))
    print(f"A3 σ_s* = {sig_star} (log2={best})")

    # ================= A4 genealogy =================
    trees, geno_checks = {}, {}
    for ch in chars:
        tr = build_genealogy(per[ch])
        chk = check_genealogy(tr, per[ch])
        trees[ch] = tr; geno_checks[ch] = chk
    geno_ok = all(c["ok"] for c in geno_checks.values())
    json.dump({"schema": "nodes[id,polarity,sigma_birth,sigma_death,end_type,"
                         "pos_at_death,ephemeral]; edges[parent,child,sigma_merge]; "
                         "parent=survivor,child=absorbed,root=survivor",
               "trees": trees},
              open(os.path.join(OUT, "merge_trees.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"A4 genealogy consistency all_ok={geno_ok}")

    # ================= B1 scale-crossing recall matrix =================
    B = len(snaps)
    mat = np.zeros((B, B))
    for qi, sq in enumerate(snaps):        # query scale
        for mi, sm in enumerate(snaps):    # memory scale
            Pm = maps[sm]; Pq = maps[sq]
            ok = sum(nearest_pattern(Pm, hopfield_recall(Pm, Pq[i], BETA)) == i
                     for i in range(len(chars)))
            mat[qi, mi] = ok / len(chars)
    diag = np.diag(mat)
    ratio = mat / (diag[:, None] + 1e-12)  # off/diag per row
    b1 = {"matrix_query_by_memory": mat.tolist(), "chance": 1.0 / len(chars),
          "offdiag_over_diag": ratio.tolist(),
          "snap_log2": list(SNAP_LOG2),
          "note": "rows=query σ_s', cols=memory σ_s; diag=same-scale"}
    _plot_b1(mat, os.path.join(OUT, "b1_scale_crossing.png"))
    print(f"B1 diag={diag} chance={1/len(chars):.3f}")

    # ================= B2 part interference (at σ_s*) =================
    Pstar = maps[sig_star]
    Pn = Pstar / (np.linalg.norm(Pstar, axis=1, keepdims=True) + 1e-12)
    idx = {c: i for i, c in enumerate(chars)}
    cos = lambda a, b: float(Pn[idx[a]] @ Pn[idx[b]])
    shared_cos = [cos(a, b) for a, b in SHARED_PAIRS]
    allpairs = [(chars[i], chars[j]) for i in range(len(chars)) for j in range(i + 1, len(chars))]
    sharedset = {frozenset(p) for p in SHARED_PAIRS}
    nonshared_cos = [cos(a, b) for a, b in allpairs if frozenset((a, b)) not in sharedset]
    mw = mann_whitney_u(shared_cos, nonshared_cos)
    contrast = {f"{a}-{b}": cos(a, b) for a, b in CONTRAST_PAIRS}
    contrast_mean = float(np.mean(list(contrast.values())))
    shared_mean = float(np.mean(shared_cos))
    # mixed-query convergence sharpness
    mixed = {}
    for a, b in SHARED_PAIRS[:4] + CONTRAST_PAIRS:
        q = Pstar[idx[a]] + Pstar[idx[b]]; q = q / (np.linalg.norm(q) + 1e-12)
        rec = hopfield_recall(Pstar, q, BETA)
        recn = rec / (np.linalg.norm(rec) + 1e-12)
        sims = sorted(((float(Pn[i] @ recn), chars[i]) for i in range(len(chars))), reverse=True)
        mixed[f"{a}+{b}"] = {"top": sims[0][1], "top_cos": round(sims[0][0], 3),
                             "second": sims[1][1], "second_cos": round(sims[1][0], 3)}
    b2 = {"shared_mean_cos": shared_mean, "nonshared_mean_cos": float(np.mean(nonshared_cos)),
          "contrast_pairs": contrast, "contrast_mean_cos": contrast_mean,
          "mann_whitney": mw,
          "form_not_parts_confirmed": bool(contrast_mean > shared_mean),
          "mixed_query": mixed}
    _plot_b2(shared_cos, nonshared_cos, contrast, os.path.join(OUT, "b2_interference.png"))
    print(f"B2 shared_mean={shared_mean:.3f} nonshared={np.mean(nonshared_cos):.3f} "
          f"contrast_mean={contrast_mean:.3f} MW p={mw['p_one_sided_a_gt_b']:.4f} "
          f"form_not_parts={b2['form_not_parts_confirmed']}")

    # ================= B3 f_c concat, w sweep =================
    fc_raw = np.array([[holes_per[c], 1.0 if holes_per[c] > 0 else 0.0, diss_per[c]]
                       for c in chars], dtype=float)
    mu = fc_raw.mean(0); sd = fc_raw.std(0); sd[sd == 0] = 1.0
    fc_z = (fc_raw - mu) / sd
    fc_unit = fc_z / (np.linalg.norm(fc_z, axis=1, keepdims=True) + 1e-12)
    ring_idx = [idx[c] for c in RING_CHARS]

    def ring_margin(P):
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        C = Pn @ Pn.T
        np.fill_diagonal(C, -1)
        return float(np.mean([1.0 - C[i].max() for i in ring_idx]))  # 1 - nearest-other cos

    Pstar_only = maps[sig_star]
    b3 = {"blob_only_ring_margin": ring_margin(Pstar_only), "w": {}}
    for w in W_GRID:
        Pcat = np.concatenate([Pstar_only, w * fc_unit], axis=1)
        b3["w"][str(w)] = {"ring_margin": ring_margin(Pcat),
                           "improvement": ring_margin(Pcat) - b3["blob_only_ring_margin"]}
    print("B3 ring margins:", b3["blob_only_ring_margin"],
          {w: b3["w"][str(w)]["ring_margin"] for w in W_GRID})

    runtime = time.time() - t0
    meta = {"milestone": "v0.2-M2", "device": "cpu", "runtime_sec": round(runtime, 2),
            "beta": BETA, "recall_steps": RECALL_STEPS, "snap_log2": list(SNAP_LOG2),
            "snap_sigmas": snaps, "grid": GRID,
            "A2_gate": gate, "A3": a3, "sigma_star": sig_star,
            "A4_genealogy_all_ok": geno_ok,
            "A4_counts": {c: geno_checks[c]["counts"] for c in chars},
            "B1": b1, "B2": b2, "B3": b3,
            "noise": {"levels": list(NOISE_LEVELS), "repeats": NOISE_REPEATS, "seed": NOISE_SEED},
            "env": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": np.__version__},
            "fc_note": "f_c=(holes,has_hole,dissolution_median_log2); z-scored, unit-normed, w-scaled"}
    json.dump(meta, open(os.path.join(OUT, "metadata.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"M2 done {runtime:.1f}s")
    return 0


# --------------------------------------------------------------------------- #
def _contact(P, chars, title, path, prop):
    fig, axes = plt.subplots(4, 6, figsize=(13, 9))
    for ax, ch, i in zip(axes.flat, chars, range(len(chars))):
        ax.imshow(P[i].reshape(GRID, GRID), cmap="magma")
        ax.set_title(ch, fontproperties=prop, fontsize=10); ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_a3(a3, path):
    ls = list(SNAP_LOG2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar([str(l) for l in ls], [a3[f"log2_{l}"]["mean_off_cosine"] for l in ls], color="C0")
    ax[0].set_title("mean off-diagonal cosine (lower=better)"); ax[0].set_xlabel("log2 σ_s")
    for eta_i, eta in enumerate(NOISE_LEVELS):
        ax[1].plot(ls, [a3[f"log2_{l}"]["noisy_recall"][f"eta_{eta}"] for l in ls],
                   "-o", label=f"η={eta}")
    ax[1].axvline(a3["sigma_star_log2"], color="k", ls="--", alpha=0.5, label="σ_s*")
    ax[1].set_title("noisy recall accuracy"); ax[1].set_xlabel("log2 σ_s"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _plot_b1(mat, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    color="w" if mat[i, j] < 0.6 else "k", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"mem {l}" for l in SNAP_LOG2])
    ax.set_yticks(range(3)); ax.set_yticklabels([f"qry {l}" for l in SNAP_LOG2])
    ax.set_title("B1 scale-crossing recall (query→memory)\nlog2 σ_s; chance=4.2%")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _plot_b2(shared, nonshared, contrast, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(nonshared, bins=30, density=True, alpha=0.4, color="0.6", label="non-shared")
    ax.hist(shared, bins=12, density=True, alpha=0.5, color="C3", label="strong-shared(11)")
    for name, v in contrast.items():
        ax.axvline(v, color="C0", ls="--", lw=1)
        ax.text(v, ax.get_ylim()[1] * 0.9, name, rotation=90, fontsize=7, color="C0")
    ax.set_xlabel("blob-map cosine (σ_s*)"); ax.set_ylabel("density")
    ax.set_title("B2 part interference: shared vs non-shared (contrast dashed)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
