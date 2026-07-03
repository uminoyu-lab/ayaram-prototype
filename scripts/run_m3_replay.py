"""v0.2 M3 — 系譜木の逆再生 (reverse-playback of the melted glyph's return path).

射程 = 形の尺跨ぎ限定 (G1). 「記録の再生」であって「生成」ではない (M3b 以降で確率生成).
Recompute-only from glyphs_128.npz (official linking gap=1). Deterministic CPU.
Outputs -> results/v0.2_m3/.

M3-0 β sweep {1,2,4,8,16}: σ_s別必要β下限 + B1 β依存
M3-1 gate: full replay vs direct cosine>=0.999 all chars/σ  (STOP if fail)
M3-2 tree-only replay: fidelity curve (24) + identification
M3-3 incl/excl dual: fidelity + 6-cell identification (σ∈{2,1,0.707}×incl/excl)
f_c verifier: ring hole-appearance σ vs M1b dissolution σ
+ ephemeral σ_death dist + reverse filmstrips (§10 material).
"""

from __future__ import annotations

import json
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
    GRID, build_blob_map, build_blob_maps, extract_records, hopfield_recall,
    nearest_pattern, replay_map, snapshot_sigmas, snapshot_index,
)
from run_m1_heat_dissolution import _cjk_prop  # noqa: E402
from run_m1b_analysis import hole_map, dissolution_sigma, MAX_GAP  # noqa: E402

GLYPHS = os.path.join(_ROOT, "data", "glyphs_128", "glyphs_128.npz")
OUT = os.path.join(_ROOT, "results", "v0.2_m3")

BETA = 16.0
BETAS = (1.0, 2.0, 4.0, 8.0, 16.0)
GATE_COS = 0.999
K_MAX = 24                       # σ ≤ 4 replay range (k=0..24)
RING = tuple("口日月田回国品語銀明")
BIN_RATIO = 0.5                  # binarisation = BIN_RATIO * max(map); recorded (綾修正 B)
MATCH_TOL = 0.25                 # |log2σ_appear - log2σ_dissolve| tolerance (2 grid)
FILM_CHARS = ("森", "品", "回", "鬱", "木", "語")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "filmstrips"), exist_ok=True)
    prop = _cjk_prop()
    t0 = time.time()

    dat = np.load(GLYPHS, allow_pickle=False)
    glyphs = dat["glyphs"]
    chars = json.load(open(os.path.join(_ROOT, "data", "glyphs_128", "metadata.json"),
                           encoding="utf-8"))["chars"]
    sig = sigma_grid(); K = len(sig)
    ks_range = list(range(0, K_MAX + 1))
    snaps = snapshot_sigmas(sig)           # [2.0, 2.83, 4.0]

    per, ss_np = {}, {}
    rec_excl, rec_incl, ephem_deaths = {}, {}, []
    for i, ch in enumerate(chars):
        ss = build_scale_space(glyphs[i], sig, device="cpu")
        R = normalized_log_response(ss, sig)
        ex = detect_extrema(R, sig, 0.05, 3)
        tr = link_trajectories(ex, sig, max_gap=MAX_GAP)
        per[ch] = tr; ss_np[ch] = ss.cpu().numpy()
        rec_excl[ch] = extract_records(tr, sig, min_lifetime=2)
        rec_incl[ch] = extract_records(tr, sig, min_lifetime=1)
        ephem_deaths += [t.sigma_death for t in tr if t.polarity == "ink" and len(t.points) < 2]

    # M2 official cargo memory (σ_s=2, excl)
    maps_by_sig = {s: build_blob_maps(per, sig, chars, s) for s in snaps}
    P2 = maps_by_sig[2.0]

    # ================= M3-0 β sweep =================
    sweep = {"self_recall": {}, "b1_by_beta": {}, "min_beta_for_2424": {}}
    for l, s in zip((1.0, 1.5, 2.0), snaps):
        P = maps_by_sig[s]
        rec = {}
        for b in BETAS:
            ok = sum(nearest_pattern(P, hopfield_recall(P, P[i], b)) == i for i in range(len(chars)))
            rec[str(b)] = f"{ok}/{len(chars)}"
        sweep["self_recall"][f"log2_{l}"] = rec
        mb = next((b for b in BETAS if rec[str(b)] == f"{len(chars)}/{len(chars)}"), None)
        sweep["min_beta_for_2424"][f"log2_{l}"] = mb
    for b in BETAS:
        M = np.zeros((3, 3))
        for qi, sq in enumerate(snaps):
            for mi, sm in enumerate(snaps):
                Pm, Pq = maps_by_sig[sm], maps_by_sig[sq]
                M[qi, mi] = sum(nearest_pattern(Pm, hopfield_recall(Pm, Pq[i], b)) == i
                                for i in range(len(chars))) / len(chars)
        sweep["b1_by_beta"][str(b)] = M.tolist()
    _plot_sweep(sweep, os.path.join(OUT, "m30_beta_sweep.png"))
    print("M3-0 min_beta:", sweep["min_beta_for_2424"])

    # ================= M3-1 full-replay gate =================
    gate_min = 1.0; gate_fail = []
    for ch in chars:
        for k in ks_range:
            s = sig[k]
            direct = build_blob_map(per[ch], sig, s, min_lifetime=2)
            full = replay_map(rec_excl[ch], sig, s, mode="full")
            na, nb = np.linalg.norm(direct), np.linalg.norm(full)
            if na < 1e-9 and nb < 1e-9:
                continue
            c = float(direct @ full / (na * nb + 1e-12))
            gate_min = min(gate_min, c)
            if c < GATE_COS:
                gate_fail.append((ch, round(float(s), 3), round(c, 5)))
    gate = {"min_cosine": gate_min, "threshold": GATE_COS,
            "pass": len(gate_fail) == 0, "fails": gate_fail[:10]}
    if not gate["pass"]:
        json.dump({"M3_1_gate": gate, "STOP": "full-replay gate failed"},
                  open(os.path.join(OUT, "metadata.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("STOP: M3-1 gate FAILED", gate["fails"][:3])
        return 1
    print(f"M3-1 gate PASS min_cos={gate_min:.6f}")

    # ================= M3-2 tree-only replay fidelity =================
    log2 = np.log2([sig[k] for k in ks_range])

    def fidelity_curve(trajs, recs_char, direct_ml):
        cur = []
        for k in ks_range:
            s = sig[k]
            direct = build_blob_map(trajs, sig, s, min_lifetime=direct_ml)
            tree = replay_map(recs_char, sig, s, mode="tree")
            na, nb = np.linalg.norm(direct), np.linalg.norm(tree)
            cur.append(float(direct @ tree / (na * nb + 1e-12)) if na > 1e-9 and nb > 1e-9 else np.nan)
        return np.array(cur)

    fid_excl = {ch: fidelity_curve(per[ch], rec_excl[ch], 2) for ch in chars}
    fid_mat = np.vstack([fid_excl[c] for c in chars])
    mean_fid = np.nanmean(fid_mat, axis=0)
    _plot_fidelity(log2, fid_mat, mean_fid, chars, os.path.join(OUT, "m32_fidelity.png"), prop)

    # ================= M3-3 incl/excl + 6-cell identification =================
    def fid_mean(recs_dict, ml):
        return np.nanmean(np.vstack([fidelity_curve(per[ch], recs_dict[ch], ml)
                                     for ch in chars]), axis=0)
    mean_fid_incl = fid_mean(rec_incl, 1)
    mean_fid_excl = mean_fid  # already excl

    cells = {}
    for cond, recs in (("excl", rec_excl), ("incl", rec_incl)):
        for lab, kq in (("sigma_2.0", 16), ("sigma_1.0", 8), ("sigma_0.707", 4)):
            acc = 0
            for i, ch in enumerate(chars):
                q = replay_map(recs[ch], sig, sig[kq], mode="tree")
                if np.linalg.norm(q) < 1e-9:
                    continue
                if nearest_pattern(P2, hopfield_recall(P2, q, BETA)) == i:
                    acc += 1
            cells[f"{cond}_{lab}"] = acc / len(chars)
    _plot_m33(log2, mean_fid_excl, mean_fid_incl, cells, os.path.join(OUT, "m33_incl_excl.png"))
    print("M3-3 cells:", {k: round(v, 3) for k, v in cells.items()})

    # M3-2 identification = the σ=2 excl cell
    m32_ident = cells["excl_sigma_2.0"]

    # ================= f_c verifier: hole appearance vs dissolution =================
    fc = {}
    for ch in RING:
        i = chars.index(ch)
        # M1b dissolution σ (median log2) from 128-space
        diss = [h["dissolution_sigma"] for h in dissolution_sigma(ss_np[ch], glyphs[i], sig)
                if h["dissolution_sigma"]]
        diss_l2 = float(np.median(np.log2(diss))) if diss else None
        # hole appearance in tree replay (descending σ): first k (high→low) with a hole
        appear_l2 = None
        for k in range(K_MAX, -1, -1):
            m = replay_map(rec_excl[ch], sig, sig[k], mode="tree").reshape(GRID, GRID)
            if m.max() <= 0:
                continue
            _, holes, _, nh = hole_map(m, thr=BIN_RATIO * m.max())
            if nh >= 1:
                appear_l2 = float(np.log2(sig[k])); break
        match = (diss_l2 is not None and appear_l2 is not None
                 and abs(appear_l2 - diss_l2) <= MATCH_TOL)
        fc[ch] = {"dissolution_log2": diss_l2, "appear_log2": appear_l2, "match": bool(match)}
    fc_hits = sum(1 for v in fc.values() if v["match"])
    print(f"f_c verifier: {fc_hits}/{len(RING)} within {MATCH_TOL} log2")

    # ================= ephemeral σ_death distribution =================
    ed = np.log2(np.array([d for d in ephem_deaths if d > 0]))
    ephem = {"n": int(ed.size), "log2_median": float(np.median(ed)),
             "log2_q25": float(np.percentile(ed, 25)), "log2_q75": float(np.percentile(ed, 75)),
             "frac_texture_below_log2_0": float(np.mean(ed < 0))}

    # ================= reverse filmstrips =================
    film_ks = [24, 20, 16, 12, 8, 4, 0]
    for ch in FILM_CHARS:
        _filmstrip(ch, rec_excl[ch], sig, film_ks, prop, os.path.join(OUT, "filmstrips", f"U{ord(ch):04X}.png"))

    runtime = time.time() - t0
    meta = {
        "milestone": "v0.2-M3", "device": "cpu", "runtime_sec": round(runtime, 2),
        "positioning": "形の尺跨ぎ限定; 記録の再生であって生成ではない (M3b以降で確率生成)",
        "beta": BETA, "replay_sigma_range": [float(sig[0]), float(sig[K_MAX])],
        "M3_0_beta_sweep": sweep,
        "M3_1_gate": gate,
        "M3_2": {"mean_fidelity_curve_log2sigma": log2.tolist(),
                 "mean_fidelity": mean_fid.tolist(),
                 "per_char_mean_fidelity": {c: float(np.nanmean(fid_excl[c])) for c in chars},
                 "identification_sigma2_excl": m32_ident},
        "M3_3": {"mean_fidelity_incl": mean_fid_incl.tolist(),
                 "mean_fidelity_excl": mean_fid_excl.tolist(),
                 "identification_6cell": cells,
                 "wall_crossing_cells": ["excl_sigma_0.707", "incl_sigma_0.707"]},
        "fc_verifier": {"binarisation": f"{BIN_RATIO}*max(map)", "match_tol_log2": MATCH_TOL,
                        "hits": fc_hits, "n": len(RING), "per_char": fc},
        "ephemeral_sigma_death": ephem,
        "env": {"python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__},
    }
    json.dump(meta, open(os.path.join(OUT, "metadata.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"M3 done {runtime:.1f}s | M3-2 mean fid={np.nanmean(mean_fid):.3f} "
          f"ident σ2 excl={m32_ident:.3f} | ephem log2 median={ephem['log2_median']:.2f} "
          f"texture frac={ephem['frac_texture_below_log2_0']:.2f}")
    return 0


# --------------------------------------------------------------------------- #
def _plot_sweep(sweep, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for l in (1.0, 1.5, 2.0):
        ys = [int(sweep["self_recall"][f"log2_{l}"][str(b)].split("/")[0]) for b in BETAS]
        ax[0].plot([str(b) for b in BETAS], ys, "-o", label=f"σ_s log2={l}")
    ax[0].axhline(24, color="k", ls=":", alpha=0.4); ax[0].set_ylabel("self-recall /24")
    ax[0].set_xlabel("β"); ax[0].set_title("M3-0 self-recall vs β"); ax[0].legend(fontsize=7)
    for b in BETAS:
        M = np.array(sweep["b1_by_beta"][str(b)])
        ax[1].plot([str(b)], [M[~np.eye(3, dtype=bool)].mean()], "o")
    xs = [str(b) for b in BETAS]
    ax[1].plot(xs, [np.array(sweep["b1_by_beta"][str(b)])[~np.eye(3, dtype=bool)].mean() for b in BETAS], "-o")
    ax[1].set_xlabel("β"); ax[1].set_ylabel("B1 mean off-diagonal recall")
    ax[1].set_title("M3-0 scale-crossing vs β")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _plot_fidelity(log2, fid_mat, mean_fid, chars, path, prop):
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in fid_mat:
        ax.plot(log2, row, color="0.8", lw=0.5)
    ax.plot(log2, mean_fid, "C3", lw=2.5, label="mean (24 chars)")
    ax.set_xlabel("log2 σ (descending replay 4→0.5)"); ax.set_ylabel("cosine B̂_tree vs B")
    ax.set_title("M3-2 tree-only replay fidelity (drift discarded)")
    ax.set_ylim(0, 1.02); ax.legend(); ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _plot_m33(log2, fid_excl, fid_incl, cells, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(log2, fid_excl, "C0-o", ms=3, label="excl (no ephemeral)")
    ax[0].plot(log2, fid_incl, "C1-o", ms=3, label="incl (ephemeral)")
    ax[0].set_xlabel("log2 σ"); ax[0].set_ylabel("mean fidelity cosine")
    ax[0].set_title("M3-3 fidelity: incl vs excl (same-condition pairs)")
    ax[0].invert_xaxis(); ax[0].legend(fontsize=8)
    labels = ["σ=2.0\n(踊り場)", "σ=1.0\n(壁上)", "σ=0.707\n(壁越え)"]
    x = np.arange(3)
    ax[1].bar(x - 0.2, [cells[f"excl_sigma_{v}"] for v in ("2.0", "1.0", "0.707")],
              0.4, label="excl", color="C0")
    ax[1].bar(x + 0.2, [cells[f"incl_sigma_{v}"] for v in ("2.0", "1.0", "0.707")],
              0.4, label="incl", color="C1")
    ax[1].axhline(1 / 24, color="k", ls=":", label="chance 4.2%")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("identification accuracy (mem=σ_s2 excl)")
    ax[1].set_title("M3-3 6-cell (wall-crossing = σ=0.707)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _filmstrip(ch, recs, sig, film_ks, prop, path):
    fig, axes = plt.subplots(1, len(film_ks), figsize=(2 * len(film_ks), 2.4))
    for ax, k in zip(axes, film_ks):
        m = replay_map(recs, sig, sig[k], mode="tree").reshape(GRID, GRID)
        ax.imshow(m, cmap="magma"); ax.set_title(f"σ={sig[k]:.2f}", fontsize=8); ax.axis("off")
    fig.suptitle(f"{ch}  (reverse replay σ:4→0.5)", fontproperties=prop, fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
