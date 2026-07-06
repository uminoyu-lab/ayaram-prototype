"""v0.2 M4 — 温度と尺 (the balance beam). Final v0.2 milestone.

Three temperatures (positioning, 綾裁定): σ (input diffusion heat = scale),
β⁻¹ (recall-operator temperature), K_u(T) (state-space Langevin heat = physics).
Main test = β↔σ bridge; control = K_u(T) noise sweep (M2a lineage). β↔σ bridging
is NOT full proof of "temperature=scale"; the physical bridge is v0.3+.

Recompute-only from glyphs_128.npz. Deterministic CPU (control uses recorded seeds).
Outputs -> results/v0.2_m4/.
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
    build_blob_map, build_blob_maps, confusion_matrix, hopfield_recall,
    nearest_pattern, snapshot_sigmas, snapshot_index,
)
from ayaram.learning import modern_hopfield_update  # noqa: E402
from run_m1_heat_dissolution import _cjk_prop  # noqa: E402
from run_m1b_analysis import MAX_GAP  # noqa: E402

import torch  # noqa: E402

GLYPHS = os.path.join(_ROOT, "data", "glyphs_128", "glyphs_128.npz")
OUT = os.path.join(_ROOT, "results", "v0.2_m4")

BETA = 16.0
BETA_GRID = [2.0 ** (m / 2) for m in range(11)]        # 1 .. 32, 11 pts
SIGMA_KS = [16, 18, 20, 22, 24, 26, 28, 30, 32]        # σ=2..8, 9 pts (W_σ / c2)
WALL_KS = {"sigma_0.707": 4, "sigma_1.0": 8}
IDENT_RATE = 0.9                                        # c2 baseline (CC [0.8,0.95])
T_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
SEEDS = [11, 22, 33, 44, 55]
RECALL_STEPS = 3
CTRL_STEPS = 10
CTRL_CHARS = ("森", "回", "鬱")


# --------------------------------------------------------------------------- #
def conf_matrix(Q, P, beta):
    return confusion_matrix(Q, P, beta)


def recall_acc(Q, P, beta, steps=RECALL_STEPS):
    return np.mean([nearest_pattern(P, hopfield_recall(P, Q[i], beta, steps)) == i
                    for i in range(len(Q))])


def _js(p, q):
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _offdiag_rows(W):
    """Drop the diagonal element from each row and renormalise."""
    n = W.shape[0]
    out = np.zeros((n, n - 1))
    for i in range(n):
        row = np.delete(W[i], i)
        s = row.sum()
        out[i] = row / s if s > 0 else np.ones(n - 1) / (n - 1)
    return out


def D_confusion(Wa, Wb):
    """Mean per-row JS divergence over off-diagonal recall distributions."""
    A, B = _offdiag_rows(Wa), _offdiag_rows(Wb)
    return float(np.mean([_js(A[i], B[i]) for i in range(A.shape[0])]))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v)); r[order] = np.arange(1, len(v) + 1)
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt)); np.add.at(s, inv, r)
        return (s / cnt)[inv]
    rx, ry = rank(x), rank(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    if abs(r) >= 1 or n <= 2:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return r, p


def cont_argmin(xs, ds):
    a = int(np.argmin(ds))
    if 0 < a < len(ds) - 1:
        x0, x1, x2 = xs[a - 1], xs[a], xs[a + 1]
        y0, y1, y2 = ds[a - 1], ds[a], ds[a + 1]
        den = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if den != 0:
            A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
            B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
            if A > 0:
                v = -B / (2 * A)
                return float(min(max(v, xs[0]), xs[-1]))
    return float(xs[a])


def interp_threshold(betas_log2, accs, target):
    """Smallest log2β where acc crosses `target` (linear interp); None if never."""
    for i in range(len(accs)):
        if accs[i] >= target:
            if i == 0:
                return betas_log2[0]
            a0, a1 = accs[i - 1], accs[i]
            b0, b1 = betas_log2[i - 1], betas_log2[i]
            if a1 == a0:
                return b1
            return b0 + (target - a0) * (b1 - b0) / (a1 - a0)
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    prop = _cjk_prop()
    t0 = time.time()

    dat = np.load(GLYPHS, allow_pickle=False)
    glyphs = dat["glyphs"]
    chars = json.load(open(os.path.join(_ROOT, "data", "glyphs_128", "metadata.json"),
                           encoding="utf-8"))["chars"]
    sig = sigma_grid(); N = len(chars)
    snaps = snapshot_sigmas(sig)

    per = {}
    for i, ch in enumerate(chars):
        ss = build_scale_space(glyphs[i], sig, device="cpu")
        R = normalized_log_response(ss, sig)
        ex = detect_extrema(R, sig, 0.05, 3)
        per[ch] = link_trajectories(ex, sig, max_gap=MAX_GAP)

    maps_by_sig = {s: build_blob_maps(per, sig, chars, s) for s in snaps}
    P2 = maps_by_sig[2.0]                       # memory (σ_s=2, excl)
    P2n = P2 / (np.linalg.norm(P2, axis=1, keepdims=True) + 1e-12)

    # ---- 第1手 reproducibility gate ----
    self_ok = sum(nearest_pattern(P2, hopfield_recall(P2, P2[i], BETA)) == i for i in range(N))
    B1 = np.zeros((3, 3))
    for qi, sq in enumerate(snaps):
        for mi, sm in enumerate(snaps):
            Pm, Pq = maps_by_sig[sm], maps_by_sig[sq]
            B1[qi, mi] = sum(nearest_pattern(Pm, hopfield_recall(Pm, Pq[i], BETA)) == i
                             for i in range(N)) / N
    gate = {"self_recall": f"{self_ok}/{N}", "b1_diag": np.diag(B1).tolist(),
            "b1_offdiag_min": float(B1[~np.eye(3, dtype=bool)].min()),
            "pass": self_ok == N and np.allclose(np.diag(B1), 1.0)}
    if not gate["pass"]:
        json.dump({"gate": gate, "STOP": "reproducibility gate failed"},
                  open(os.path.join(OUT, "metadata.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("STOP: gate failed", gate); return 1
    print("gate PASS", gate["self_recall"], "b1off_min", gate["b1_offdiag_min"])

    # ================= M4-0 true wall-crossing =================
    wall = {}
    for name, k in WALL_KS.items():
        for ml, tag in ((2, "filter"), (1, "nofilter")):
            Q = np.stack([build_blob_map(per[c], sig, sig[k], min_lifetime=ml) for c in chars])
            wall[f"{name}_{tag}"] = float(recall_acc(Q, P2, BETA))
    wall["chance"] = 1.0 / N
    wall["plateau_ref_B1_diag"] = 1.0
    print("M4-0 wall-crossing:", {k: round(v, 3) for k, v in wall.items()})

    # ================= (a) W_β + accuracy =================
    Wb = {f"{b:.4f}": conf_matrix(P2n, P2n, b) for b in BETA_GRID}
    acc_b = [float(recall_acc(P2, P2, b)) for b in BETA_GRID]

    # ================= (b) W_σ (β=16) + accuracy =================
    Bq = {k: np.stack([build_blob_map(per[c], sig, sig[k], min_lifetime=2) for c in chars])
          for k in SIGMA_KS}
    Bqn = {k: Bq[k] / (np.linalg.norm(Bq[k], axis=1, keepdims=True) + 1e-12) for k in SIGMA_KS}
    Ws = {k: conf_matrix(Bqn[k], P2n, BETA) for k in SIGMA_KS}
    acc_s = [float(recall_acc(Bq[k], P2, BETA)) for k in SIGMA_KS]

    np.savez_compressed(os.path.join(OUT, "W_series.npz"),
                        betas=np.array(BETA_GRID), sigma_ks=np.array(SIGMA_KS),
                        **{f"Wb_{i}": Wb[f"{b:.4f}"] for i, b in enumerate(BETA_GRID)},
                        **{f"Ws_{i}": Ws[k] for i, k in enumerate(SIGMA_KS)})

    # ================= (c1) structure map σ*(β) =================
    log2_sig = [float(np.log2(sig[k])) for k in SIGMA_KS]
    sigma_star, D_grid = [], []
    for b in BETA_GRID:
        ds = [D_confusion(Wb[f"{b:.4f}"], Ws[k]) for k in SIGMA_KS]
        D_grid.append(ds)
        sigma_star.append(cont_argmin(log2_sig, ds))
    rho_c1, p_c1 = spearman(BETA_GRID, sigma_star)   # expect NEGATIVE (σ↑⇔β*↓)
    c1 = {"log2beta": [float(np.log2(b)) for b in BETA_GRID],
          "sigma_star_log2": sigma_star, "spearman_rho": rho_c1, "spearman_p": p_c1,
          "predicted": "σ↑⇔β*↓ (decreasing; rho<0)",
          "hit": bool(rho_c1 < 0 and abs(rho_c1) >= 0.8 and p_c1 < 0.05)}

    # ================= (c2) equi-discrimination β_c(σ_q) =================
    betas_log2 = [float(np.log2(b)) for b in BETA_GRID]
    beta_c, acc_grid = [], []
    for k in SIGMA_KS:
        accs = [float(recall_acc(Bq[k], P2, b)) for b in BETA_GRID]
        acc_grid.append(accs)
        beta_c.append(interp_threshold(betas_log2, accs, IDENT_RATE))
    valid = [(np.log2(sig[k]), bc) for k, bc in zip(SIGMA_KS, beta_c) if bc is not None]
    if len(valid) >= 3:
        rho_c2, p_c2 = spearman([v[0] for v in valid], [v[1] for v in valid])
    else:
        rho_c2, p_c2 = float("nan"), float("nan")
    c2 = {"log2sigma": log2_sig, "beta_c_log2": beta_c, "ident_rate": IDENT_RATE,
          "spearman_rho": rho_c2, "spearman_p": p_c2, "n_valid": len(valid),
          "predicted": "σ↑⇔β_c↑ (increasing; rho>0)",
          "hit": bool(len(valid) >= 3 and rho_c2 > 0 and abs(rho_c2) >= 0.8 and p_c2 < 0.05)}
    _plot_c(c1, c2, D_grid, log2_sig, BETA_GRID, os.path.join(OUT, "m4_correspondence.png"))
    _plot_ab(BETA_GRID, acc_b, log2_sig, acc_s, os.path.join(OUT, "m4_accuracy.png"))
    print(f"(c1) rho={rho_c1:.3f} p={p_c1:.4f} hit={c1['hit']} | "
          f"(c2) rho={rho_c2:.3f} p={p_c2:.4f} hit={c2['hit']} nvalid={len(valid)}")

    # ================= control: K_u(T) noise sweep =================
    X = torch.as_tensor(P2, dtype=torch.float32).T.contiguous()
    d = P2.shape[1]
    ctrl = {"T_grid": T_GRID, "acc_mean": [], "acc_sd": [], "seeds": SEEDS}
    WT_by_T = {}
    for ti, T in enumerate(T_GRID):
        accs = []
        conf = np.zeros((N, N))
        for si, seed in enumerate(SEEDS):
            rng = np.random.default_rng(seed * 100 + ti)
            ok = 0
            for i in range(N):
                xi = torch.as_tensor(P2[i], dtype=torch.float32).clone()
                for _ in range(CTRL_STEPS):
                    xi = modern_hopfield_update(xi, X, beta=BETA)
                    xi = xi + torch.as_tensor(rng.standard_normal(d) * (T / math.sqrt(d)),
                                              dtype=torch.float32)
                j = nearest_pattern(P2, xi.numpy())
                conf[i, j] += 1
                ok += int(j == i)
            accs.append(ok / N)
        ctrl["acc_mean"].append(float(np.mean(accs)))
        ctrl["acc_sd"].append(float(np.std(accs)))
        WT_by_T[T] = conf / conf.sum(axis=1, keepdims=True)
    # qualitative trajectories (seed 0) for CTRL_CHARS
    traj = {}
    for ch in CTRL_CHARS:
        i = chars.index(ch)
        rng = np.random.default_rng(SEEDS[0])
        xi = torch.as_tensor(P2[i], dtype=torch.float32).clone()
        cs = []
        for _ in range(CTRL_STEPS):
            xi = modern_hopfield_update(xi, X, beta=BETA)
            xi = xi + torch.as_tensor(rng.standard_normal(d) * (0.1 / math.sqrt(d)), dtype=torch.float32)
            v = xi.numpy()
            cs.append(float(P2[i] @ v / (np.linalg.norm(P2[i]) * np.linalg.norm(v) + 1e-12)))
        traj[ch] = cs
    # place W_T in the (c1) distance space: nearest β per T (observation only)
    wt_nearest_beta = {}
    for T in T_GRID:
        ds = [D_confusion(WT_by_T[T], Wb[f"{b:.4f}"]) for b in BETA_GRID]
        wt_nearest_beta[str(T)] = float(np.log2(BETA_GRID[int(np.argmin(ds))]))
    _plot_ctrl(ctrl, traj, os.path.join(OUT, "m4_control.png"), prop)
    print("control acc@T:", {T: round(a, 2) for T, a in zip(T_GRID, ctrl["acc_mean"])})

    runtime = time.time() - t0
    meta = {"milestone": "v0.2-M4", "device": "cpu", "runtime_sec": round(runtime, 2),
            "positioning": "three temperatures: σ(input diffusion=scale), β⁻¹(recall op), "
                           "K_u(T)(Langevin=physics). β↔σ bridge != full 'temperature=scale' "
                           "proof; physical bridge is v0.3+",
            "beta": BETA, "beta_grid": BETA_GRID, "sigma_ks": SIGMA_KS,
            "reproducibility_gate": gate,
            "M4_0_wall_crossing": wall,
            "a_acc_beta": acc_b, "b_acc_sigma": acc_s,
            "c1_structure_map": c1, "c2_equi_discrimination": c2,
            "D_matrix_beta_by_sigma": D_grid,
            "control_KuT": {**ctrl, "wt_nearest_log2beta": wt_nearest_beta,
                            "trajectories_T0.1": traj},
            "env": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform()},
            "seeds_note": "control uses recorded seeds; all else CPU-deterministic"}
    json.dump(meta, open(os.path.join(OUT, "metadata.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"M4 done {runtime:.1f}s | c1 hit={c1['hit']} c2 hit={c2['hit']}")
    return 0


# --------------------------------------------------------------------------- #
def _plot_ab(betas, acc_b, log2_sig, acc_s, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot([np.log2(b) for b in betas], acc_b, "C0-o")
    ax[0].axhline(1 / 24, color="k", ls=":", label="chance"); ax[0].axhline(0.9, color="r", ls="--", alpha=0.4)
    ax[0].set_xlabel("log2 β"); ax[0].set_ylabel("recall accuracy")
    ax[0].set_title("(a) heat the recall: clean-query acc vs β"); ax[0].legend(fontsize=7)
    ax[1].plot(log2_sig, acc_s, "C3-o")
    ax[1].axhline(1 / 24, color="k", ls=":"); ax[1].axhline(0.9, color="r", ls="--", alpha=0.4)
    ax[1].set_xlabel("log2 σ_q"); ax[1].set_ylabel("recall accuracy (β=16)")
    ax[1].set_title("(b) heat the input: B(σ_q) acc vs σ")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _plot_c(c1, c2, D_grid, log2_sig, betas, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    im = ax[0].imshow(np.array(D_grid), aspect="auto", cmap="viridis_r",
                      extent=[log2_sig[0], log2_sig[-1], np.log2(betas[-1]), np.log2(betas[0])])
    ax[0].plot(c1["sigma_star_log2"], [np.log2(b) for b in betas], "w.-", lw=1.5, label="σ*(β)")
    ax[0].set_xlabel("log2 σ"); ax[0].set_ylabel("log2 β")
    ax[0].set_title(f"(c1) structure map D(W_β,W_σ)\nρ={c1['spearman_rho']:.2f} "
                    f"p={c1['spearman_p']:.3f} hit={c1['hit']}")
    ax[0].legend(fontsize=8); fig.colorbar(im, ax=ax[0])
    bc = [b if b is not None else np.nan for b in c2["beta_c_log2"]]
    ax[1].plot(log2_sig, bc, "C2-o")
    ax[1].set_xlabel("log2 σ_q"); ax[1].set_ylabel("log2 β_c (ident≥0.9)")
    ax[1].set_title(f"(c2) equi-discrimination\nρ={c2['spearman_rho']:.2f} "
                    f"p={c2['spearman_p']:.3f} hit={c2['hit']}")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _plot_ctrl(ctrl, traj, path, prop):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].errorbar(ctrl["T_grid"], ctrl["acc_mean"], yerr=ctrl["acc_sd"], fmt="C0-o", capsize=3)
    ax[0].axhline(1 / 24, color="k", ls=":", label="chance")
    ax[0].set_xlabel("K_u noise T"); ax[0].set_ylabel("recall accuracy (±SD, 5 seeds)")
    ax[0].set_xscale("symlog", linthresh=0.005)
    ax[0].set_title("control: K_u(T) noise sweep (iterated recall)"); ax[0].legend(fontsize=7)
    for ch, cs in traj.items():
        ax[1].plot(range(1, len(cs) + 1), cs, "-o", ms=3, label=ch)
    for ln in ax[1].get_legend_handles_labels()[1]:
        pass
    ax[1].legend(prop=prop, fontsize=9)
    ax[1].set_xlabel("iteration t"); ax[1].set_ylabel("cosine to correct (T=0.1)")
    ax[1].set_title("qualitative trajectories")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
