"""Demo: MAX-CUT via the 4-phase cycle on a 1-layer Hopfield-Hebb net.

Decision (Aru M2, 2026-06-17):
    - N in {8, 16, 32}, Erdos-Renyi p = 0.5, 10 independent graphs per N
    - classical Hebb side only (W = -A, antiferromagnetic) -- Modern Hopfield
      attractor structure is not the right tool for combinatorial optimization
      in v0.1
    - Phase 1 terrain = Ising coupling already installed on W
      Phase 2 = 1000 Langevin steps, sigma_global = 1.0
      Phase 3 = 100 deterministic high-beta steps for fixation
      Phase 4 = sign(layer-0 readout)
    - DoD: mean approximation ratio >= 0.9 against brute-force optimum at
      N = 8, 16 (10-trial average)

For N = 32 brute force is intractable (2^32 = 4.3e9 enumerations); report the
raw cut value, the ratio to the trivial upper bound |E|, and the improvement
factor over a random cut (|E|/2).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ayaram import core  # noqa: E402
from ayaram.ising import MaxCutProblem, random_erdos_renyi  # noqa: E402
from ayaram.memory import HopfieldNetwork  # noqa: E402


OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


@dataclass
class TrialResult:
    N: int
    seed: int
    n_edges: int
    cut: float
    optimal_cut: float | None  # None when brute force is skipped
    ratio: float | None        # cut / optimal_cut when available
    cut_over_edges: float      # cut / |E|, an absolute upper bound
    cut_over_half_edges: float # cut / (|E|/2), the "lift over random" factor


def _build_solver_net(W: torch.Tensor, device: torch.device, seed: int) -> HopfieldNetwork:
    """Build a 1-layer Hebb HopfieldNetwork and install the Ising W directly."""
    N = W.shape[0]
    net = HopfieldNetwork(mode="hebb", layer_sizes=(N,), seed=seed).to(device)
    layer = net.layers[0]
    # Bypass store() -- the Ising W is not a Hebb-of-patterns matrix.
    layer.W = W.to(device).clone()
    layer._has_patterns = True
    layer.enforce_constraints()  # symmetrize as a safety belt
    return net


def solve_maxcut(
    problem: MaxCutProblem,
    *,
    beta: float = 1.0,
    sigma_global: float = 1.0,
    phase2_steps: int = 1000,
    phase3_steps: int = 100,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[float, torch.Tensor]:
    """Run a single 4-phase cycle on the problem and return ``(cut, s)``."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = problem.to_hopfield_weights().to(dev)
    net = _build_solver_net(W, dev, seed=seed)
    cfg = core.CycleConfig(
        beta=beta,
        sigma_global=sigma_global,
        phase2_steps=phase2_steps,
        phase3_steps=phase3_steps,
    )
    g_init = torch.Generator(device=dev).manual_seed(seed)
    init_bias = 0.01 * torch.randn(problem.N, device=dev, generator=g_init)
    g_noise = torch.Generator(device=dev).manual_seed(seed + 1)
    readout, _ = core.run_cycle(net, init_bias, config=cfg, generator=g_noise)
    s = torch.sign(readout)
    s = torch.where(s == 0, torch.ones_like(s), s)
    cut = float(problem.cut_value(s).item())
    return cut, s


def benchmark(
    Ns: Iterable[int] = (8, 16, 32),
    n_trials: int = 10,
    p_edge: float = 0.5,
    *,
    beta: float = 1.0,
    sigma_global: float = 1.0,
    phase2_steps: int = 1000,
    phase3_steps: int = 100,
    brute_force_max_n: int = 20,
    device: torch.device | None = None,
) -> list[TrialResult]:
    """Run ``n_trials`` independent graphs at every ``N`` and collect results."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[TrialResult] = []
    for N in Ns:
        for trial in range(n_trials):
            seed = 1000 * N + trial
            adj = random_erdos_renyi(N, p_edge, seed=seed)
            problem = MaxCutProblem.from_graph(adj.to(dev))
            cut, _s = solve_maxcut(
                problem,
                beta=beta,
                sigma_global=sigma_global,
                phase2_steps=phase2_steps,
                phase3_steps=phase3_steps,
                device=dev,
                seed=seed,
            )
            if N <= brute_force_max_n:
                opt_cut, _ = problem.optimal_brute_force(max_n=brute_force_max_n)
                ratio = cut / opt_cut if opt_cut > 0 else 0.0
            else:
                opt_cut = None
                ratio = None
            n_edges = problem.n_edges
            results.append(
                TrialResult(
                    N=N,
                    seed=seed,
                    n_edges=n_edges,
                    cut=cut,
                    optimal_cut=opt_cut,
                    ratio=ratio,
                    cut_over_edges=cut / n_edges if n_edges > 0 else 0.0,
                    cut_over_half_edges=cut / (n_edges / 2) if n_edges > 0 else 0.0,
                )
            )
    return results


def _summarize(results: list[TrialResult]) -> dict[int, dict]:
    by_N: dict[int, list[TrialResult]] = {}
    for r in results:
        by_N.setdefault(r.N, []).append(r)
    summary: dict[int, dict] = {}
    for N, rs in by_N.items():
        ratios = [r.ratio for r in rs if r.ratio is not None]
        cuts = [r.cut for r in rs]
        edges = [r.n_edges for r in rs]
        lifts = [r.cut_over_half_edges for r in rs]
        summary[N] = {
            "n_trials": len(rs),
            "mean_ratio": float(np.mean(ratios)) if ratios else None,
            "min_ratio": float(np.min(ratios)) if ratios else None,
            "max_ratio": float(np.max(ratios)) if ratios else None,
            "mean_cut": float(np.mean(cuts)),
            "mean_edges": float(np.mean(edges)),
            "mean_lift_over_random": float(np.mean(lifts)),
        }
    return summary


# ----- visualization -------------------------------------------------------


def _circular_layout(n: int) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def _plot_graph(ax, adj: np.ndarray, s: np.ndarray, title: str) -> None:
    n = adj.shape[0]
    pos = _circular_layout(n)
    # edges
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                cut = s[i] * s[j] < 0
                color = "tab:red" if cut else "lightgray"
                lw = 1.6 if cut else 0.7
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    color=color,
                    linewidth=lw,
                    zorder=1,
                )
    # nodes
    pos_s = pos[s > 0]
    pos_t = pos[s < 0]
    ax.scatter(pos_s[:, 0], pos_s[:, 1], c="tab:blue", s=180, zorder=2, edgecolor="k")
    ax.scatter(pos_t[:, 0], pos_t[:, 1], c="tab:orange", s=180, zorder=2, edgecolor="k")
    for i, (x, y) in enumerate(pos):
        ax.text(x * 1.13, y * 1.13, str(i), ha="center", va="center", fontsize=9)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)


def make_figure(
    results: list[TrialResult],
    representative_adj: np.ndarray,
    representative_s: np.ndarray,
    representative_label: str,
    out_path: str,
) -> None:
    summary = _summarize(results)
    Ns = sorted(summary.keys())

    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, len(Ns), height_ratios=[1.4, 1.0], hspace=0.35, wspace=0.25)

    # Top: representative graph spanning the full top row.
    ax_top = fig.add_subplot(gs[0, :])
    _plot_graph(ax_top, representative_adj, representative_s, representative_label)

    # Bottom: per-N box / strip plots of either ratio (where available) or lift.
    for i, N in enumerate(Ns):
        ax = fig.add_subplot(gs[1, i])
        rs = [r for r in results if r.N == N]
        if rs[0].ratio is not None:
            data = [r.ratio for r in rs]
            label = "approx ratio (cut / opt)"
            ax.axhline(0.9, color="tab:red", linestyle="--", linewidth=0.8, label="DoD 0.9")
            ax.set_ylim(0.5, 1.05)
        else:
            data = [r.cut_over_edges for r in rs]
            label = "cut / |E|"
            ax.set_ylim(0.0, 1.05)
        ax.boxplot([data], widths=0.45, showmeans=True)
        # also scatter the individual points
        xs = np.full_like(np.asarray(data, dtype=float), 1.0) + np.random.uniform(
            -0.08, 0.08, size=len(data)
        )
        ax.scatter(xs, data, alpha=0.7, color="tab:blue", zorder=3)
        ax.set_xticks([1])
        ax.set_xticklabels([f"N={N}"])
        ax.set_ylabel(label)
        if rs[0].ratio is not None:
            ax.set_title(f"N={N}: mean={summary[N]['mean_ratio']:.3f}")
            ax.legend(loc="lower right", fontsize=8)
        else:
            ax.set_title(
                f"N={N}: cut/|E|={summary[N]['mean_cut'] / summary[N]['mean_edges']:.3f}"
            )

    fig.suptitle("MAX-CUT via Hopfield + 4-phase cycle (sigma_global=1.0)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved figure -> {out_path}")


def main() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    t0 = time.time()
    results = benchmark(
        Ns=(8, 16, 32),
        n_trials=10,
        p_edge=0.5,
        beta=1.0,
        sigma_global=1.0,
        phase2_steps=1000,
        phase3_steps=100,
        device=dev,
    )
    summary = _summarize(results)
    for N, s in summary.items():
        print(
            f"N={N}: trials={s['n_trials']}, "
            f"mean_ratio={s['mean_ratio']}, "
            f"min={s['min_ratio']}, max={s['max_ratio']}, "
            f"mean_cut={s['mean_cut']:.2f}, mean_edges={s['mean_edges']:.1f}, "
            f"lift_over_random={s['mean_lift_over_random']:.3f}"
        )

    # Pick an N=16 representative (median ratio) for the top-row visual.
    n16 = [r for r in results if r.N == 16]
    n16_sorted = sorted(n16, key=lambda r: (r.ratio or 0.0))
    rep = n16_sorted[len(n16_sorted) // 2]
    rep_adj = random_erdos_renyi(rep.N, 0.5, seed=rep.seed).numpy()
    rep_problem = MaxCutProblem.from_graph(torch.from_numpy(rep_adj).to(dev))
    _cut, rep_s = solve_maxcut(
        rep_problem,
        beta=1.0,
        sigma_global=1.0,
        phase2_steps=1000,
        phase3_steps=100,
        device=dev,
        seed=rep.seed,
    )
    rep_label = (
        f"N=16 representative (seed={rep.seed}): "
        f"cut={rep.cut:.0f}/optimal={rep.optimal_cut:.0f} "
        f"(ratio={rep.ratio:.3f})"
    )
    out_png = os.path.join(OUT_DIR, "ising_maxcut.png")
    make_figure(results, rep_adj, rep_s.detach().cpu().numpy(), rep_label, out_png)

    print(f"total elapsed: {time.time() - t0:.1f}s")
    return {"summary": summary, "out_png": out_png}


if __name__ == "__main__":
    main()
