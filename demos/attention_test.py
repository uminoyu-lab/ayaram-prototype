"""Demo: Hopfield = Attention equivalence check (decision #4, Aru 2026-06-17).

Part A -- Direct numerical verification of Ramsauer 2020 Theorem 3.
Part B -- 20x20 beta-sigma map: store the 8 kanji, recall from a half-occluded
          input under the 4-phase cycle, and record recall accuracy
          (cosine similarity) over a log-grid of (beta, sigma).
"""

from __future__ import annotations

import os
import sys
import time
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make ``import ayaram`` work when running this file as a script from the repo.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ayaram import core, learning, modes  # noqa: E402
from ayaram.memory import HopfieldNetwork  # noqa: E402


OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
KANJI_PATH = os.path.join(_ROOT, "data", "kanji_8_32x32.npy")


# ----- Part A: Theorem 3 ----------------------------------------------------


class TheoremResult(NamedTuple):
    max_abs_diff: float
    mean_abs_diff: float
    d: int
    N: int
    beta: float


def verify_theorem3(
    d: int = 64,
    N: int = 16,
    beta: float = 0.5,
    seed: int = 42,
    device: str = "cpu",
) -> TheoremResult:
    """Verify ``X @ softmax(beta * X^T xi) == softmax(beta * xi X^T) @ X^T``.

    The first form is Ramsauer 2020 Theorem 1 (Modern Hopfield update). The
    second form is standard scaled-dot-product attention with
    ``Q = xi, K = V = X^T`` and the softmax scale replaced by beta. Theorem 3
    asserts they are identical.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(d, N, generator=g, device=device)
    xi = torch.randn(d, generator=g, device=device)

    out_hopfield = learning.modern_hopfield_update(xi, X, beta=beta)

    Q = xi.unsqueeze(0)  # (1, d)
    K = X.T  # (N, d)
    V = X.T  # (N, d)
    attn = torch.softmax(beta * (Q @ K.T), dim=-1)  # (1, N)
    out_attn = (attn @ V).squeeze(0)  # (d,)

    diff = (out_hopfield - out_attn).abs()
    return TheoremResult(
        max_abs_diff=float(diff.max().item()),
        mean_abs_diff=float(diff.mean().item()),
        d=d,
        N=N,
        beta=beta,
    )


# ----- Part B: beta-sigma map ----------------------------------------------


def _half_occlude(pattern: torch.Tensor, side: str = "right") -> torch.Tensor:
    """Mask half of a (32, 32) pattern to the background value (-1).

    ``side`` selects which half to keep visible: ``'left'``, ``'right'``,
    ``'top'``, ``'bottom'``.
    """
    out = pattern.clone()
    if side == "right":
        out[:, 16:] = -1.0
    elif side == "left":
        out[:, :16] = -1.0
    elif side == "bottom":
        out[16:, :] = -1.0
    elif side == "top":
        out[:16, :] = -1.0
    else:
        raise ValueError(f"unknown occlusion side: {side}")
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a.norm()
    nb = b.norm()
    if na.item() == 0 or nb.item() == 0:
        return 0.0
    return float((a @ b / (na * nb)).item())


def _recall_accuracy(
    patterns_flat: torch.Tensor,
    beta: float,
    sigma_global: float,
    *,
    phase2_steps: int,
    phase3_steps: int,
    device: torch.device,
    seed: int,
) -> float:
    """Mean cosine similarity between recalled and target patterns.

    All 8 kanji are processed in parallel as a batched state. The intra-layer
    Modern update and the Langevin step are linear-algebra primitives over
    the layer dimension, so batching is a pure speedup; no semantics change.
    """
    net = HopfieldNetwork(mode="modern", seed=seed).to(device)
    net.store_layer0(patterns_flat)
    cfg = core.CycleConfig(
        beta=beta,
        sigma_global=sigma_global,
        phase2_steps=phase2_steps,
        phase3_steps=phase3_steps,
    )
    n_patterns, d0 = patterns_flat.shape
    d1, d2 = net.layer_sizes[1], net.layer_sizes[2]
    state = core.CycleState(
        xi=[
            torch.zeros(n_patterns, d0, device=device),
            torch.zeros(n_patterns, d1, device=device),
            torch.zeros(n_patterns, d2, device=device),
        ]
    )
    occ = patterns_flat.view(n_patterns, 32, 32).clone()
    occ[:, :, 16:] = -1.0
    occ = occ.view(n_patterns, d0)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    core.phase1_terrain(net, state, occ)
    core.phase2_fluctuation(net, state, cfg, generator=gen)
    core.phase3_fixation(net, state, cfg)
    readout = state.xi[0]  # (n_patterns, d0)
    sims = torch.nn.functional.cosine_similarity(readout, patterns_flat, dim=-1)
    return float(sims.mean().item())


def part_b_beta_sigma_map(
    patterns: np.ndarray,
    beta_range: tuple[float, float] = (0.1, 100.0),
    sigma_range: tuple[float, float] = (0.01, 10.0),
    n_beta: int = 20,
    n_sigma: int = 20,
    phase2_steps: int = 1000,
    phase3_steps: int = 100,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the (beta, sigma) sweep and return ``(betas, sigmas, accuracy)``."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    patterns_flat = (
        torch.from_numpy(patterns.reshape(patterns.shape[0], -1))
        .to(torch.float32)
        .to(dev)
    )
    betas = np.logspace(np.log10(beta_range[0]), np.log10(beta_range[1]), n_beta)
    sigmas = np.logspace(np.log10(sigma_range[0]), np.log10(sigma_range[1]), n_sigma)
    acc = np.zeros((n_sigma, n_beta), dtype=np.float32)
    print(
        f"[Part B] grid {n_sigma}x{n_beta} = {n_sigma * n_beta} points "
        f"on {dev}, phase2={phase2_steps}, phase3={phase3_steps}"
    )
    t0 = time.time()
    for i, sigma in enumerate(sigmas):
        for j, beta in enumerate(betas):
            acc[i, j] = _recall_accuracy(
                patterns_flat,
                beta=float(beta),
                sigma_global=float(sigma),
                phase2_steps=phase2_steps,
                phase3_steps=phase3_steps,
                device=dev,
                seed=seed + i * n_beta + j,
            )
        elapsed = time.time() - t0
        done = (i + 1) * n_beta
        total = n_sigma * n_beta
        print(
            f"  row {i + 1}/{n_sigma}: sigma={sigma:.3g}, "
            f"{done}/{total} points, {elapsed:.1f}s"
        )
    return betas, sigmas, acc


def plot_heatmap(
    betas: np.ndarray,
    sigmas: np.ndarray,
    acc: np.ndarray,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(
        acc,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=(
            np.log10(betas[0]),
            np.log10(betas[-1]),
            np.log10(sigmas[0]),
            np.log10(sigmas[-1]),
        ),
        vmin=-1.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("recall cosine similarity")
    ax.set_xlabel("log10(beta)  (softmax inverse temperature)")
    ax.set_ylabel("log10(sigma_global)  (aya-sleep noise scale)")
    ax.set_title(
        f"beta-sigma map: recall cos sim over {acc.shape[0]}x{acc.shape[1]} grid"
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved heatmap -> {out_path}")


def part_c_beta_finescan(
    patterns: np.ndarray,
    *,
    n_beta: int = 20,
    beta_range: tuple[float, float] = (1.0, 10.0),
    sigma_fixed: float = 1.5,
    phase2_steps: int = 1000,
    phase3_steps: int = 100,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Part C: narrow beta sweep at the noise transition (sigma ~ 1.5).

    M1 saw beta have essentially no effect across the full Part B grid because
    the 8 kanji are too well separated for the softmax temperature to matter.
    Part C zooms into the only regime where beta might do something: right at
    the noise threshold (sigma ~= 1.5), where the recall is partially
    degraded and selecting the right pattern among the noisy alternatives
    matters more.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    patterns_flat = (
        torch.from_numpy(patterns.reshape(patterns.shape[0], -1))
        .to(torch.float32)
        .to(dev)
    )
    betas = np.logspace(np.log10(beta_range[0]), np.log10(beta_range[1]), n_beta)
    acc = np.zeros(n_beta, dtype=np.float32)
    print(
        f"[Part C] beta finescan: {n_beta} points at sigma={sigma_fixed}, "
        f"on {dev}, phase2={phase2_steps}"
    )
    t0 = time.time()
    for j, beta in enumerate(betas):
        acc[j] = _recall_accuracy(
            patterns_flat,
            beta=float(beta),
            sigma_global=sigma_fixed,
            phase2_steps=phase2_steps,
            phase3_steps=phase3_steps,
            device=dev,
            seed=seed + j,
        )
    print(f"  done in {time.time() - t0:.1f}s")
    return betas, acc


def plot_finescan(
    betas: np.ndarray,
    acc: np.ndarray,
    sigma_fixed: float,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(betas, acc, marker="o", color="tab:blue")
    ax.set_xlabel("beta  (softmax inverse temperature)")
    ax.set_ylabel("recall cosine similarity")
    ax.set_title(f"Part C: beta finescan at sigma = {sigma_fixed}")
    ax.set_ylim(min(0.5, float(acc.min()) - 0.05), 1.02)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved finescan -> {out_path}")


def main(
    *,
    n_beta: int = 20,
    n_sigma: int = 20,
    phase2_steps: int = 1000,
    phase3_steps: int = 100,
    run_part_c: bool = True,
) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    # Part A
    print("[Part A] Theorem 3 verification")
    res = verify_theorem3()
    print(
        f"  d={res.d}, N={res.N}, beta={res.beta}: "
        f"max|diff|={res.max_abs_diff:.3e}, mean|diff|={res.mean_abs_diff:.3e}"
    )

    # Part B
    print("[Part B] beta-sigma sweep")
    patterns = np.load(KANJI_PATH)
    betas, sigmas, acc = part_b_beta_sigma_map(
        patterns,
        n_beta=n_beta,
        n_sigma=n_sigma,
        phase2_steps=phase2_steps,
        phase3_steps=phase3_steps,
    )
    out_png = os.path.join(OUT_DIR, "beta_sigma_map.png")
    plot_heatmap(betas, sigmas, acc, out_png)
    np.savez(
        os.path.join(OUT_DIR, "beta_sigma_map.npz"),
        betas=betas,
        sigmas=sigmas,
        accuracy=acc,
    )
    summary = {
        "theorem3_max_abs_diff": res.max_abs_diff,
        "theorem3_mean_abs_diff": res.mean_abs_diff,
        "acc_max": float(acc.max()),
        "acc_argmax_beta": float(betas[int(acc.argmax() % acc.shape[1])]),
        "acc_argmax_sigma": float(sigmas[int(acc.argmax() // acc.shape[1])]),
        "acc_min": float(acc.min()),
        "out_png": out_png,
    }

    if run_part_c:
        sigma_fixed = 1.5
        betas_c, acc_c = part_c_beta_finescan(
            patterns,
            n_beta=20,
            beta_range=(1.0, 10.0),
            sigma_fixed=sigma_fixed,
            phase2_steps=phase2_steps,
            phase3_steps=phase3_steps,
        )
        out_c_png = os.path.join(OUT_DIR, "beta_finescan.png")
        plot_finescan(betas_c, acc_c, sigma_fixed, out_c_png)
        np.savez(
            os.path.join(OUT_DIR, "beta_finescan.npz"),
            betas=betas_c,
            accuracy=acc_c,
            sigma_fixed=sigma_fixed,
        )
        summary["part_c_out_png"] = out_c_png
        summary["part_c_acc_min"] = float(acc_c.min())
        summary["part_c_acc_max"] = float(acc_c.max())
        summary["part_c_acc_argmax_beta"] = float(betas_c[int(acc_c.argmax())])
    return summary


if __name__ == "__main__":
    summary = main()
    print("summary:", summary)
