"""Demo: kanji associative recall, Modern vs classical Hebb side-by-side.

Decision #3 + #5 + sub-decision (Aya + Yu + Aru, 2026-06-17): 8 kanji,
32x32 grayscale, half-occluded input. The 4-phase cycle runs once per kanji
per learning rule. Output grid: 8 rows x 4 columns
(original / occluded / Modern recall / Hebb recall).
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
# Use a Japanese-capable font so the row labels render. Fall through DejaVu Sans
# if none of the Windows fonts are available.
matplotlib.rcParams["font.family"] = [
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "Noto Sans JP",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ayaram import core  # noqa: E402
from ayaram.memory import HopfieldNetwork  # noqa: E402


KANJI_LABELS: tuple[str, ...] = ("Ren", "Ki", "Kou", "Sen", "Ka", "San", "Nichi", "Getsu")
KANJI_GLYPHS: tuple[str, ...] = ("人", "木", "口", "川", "火", "山", "日", "月")
KANJI_PATH = os.path.join(_ROOT, "data", "kanji_8_32x32.npy")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _half_occlude(pattern: np.ndarray, side: str = "right") -> np.ndarray:
    out = pattern.copy()
    if side == "right":
        out[:, 16:] = -1.0
    elif side == "left":
        out[:, :16] = -1.0
    elif side == "bottom":
        out[16:, :] = -1.0
    elif side == "top":
        out[:16, :] = -1.0
    else:
        raise ValueError(side)
    return out


def _recall_all(
    patterns_flat: torch.Tensor,
    occluded_flat: torch.Tensor,
    mode: str,
    *,
    beta: float,
    sigma_global: float,
    phase2_steps: int,
    phase3_steps: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Batched recall: all 8 kanji propagate through the cycle in parallel."""
    net = HopfieldNetwork(mode=mode, seed=seed).to(device)
    net.store_layer0(patterns_flat)
    cfg = core.CycleConfig(
        beta=beta,
        sigma_global=sigma_global,
        phase2_steps=phase2_steps,
        phase3_steps=phase3_steps,
    )
    n, d0 = occluded_flat.shape
    d1, d2 = net.layer_sizes[1], net.layer_sizes[2]
    state = core.CycleState(
        xi=[
            torch.zeros(n, d0, device=device),
            torch.zeros(n, d1, device=device),
            torch.zeros(n, d2, device=device),
        ]
    )
    g = torch.Generator(device=device).manual_seed(seed + 1)
    core.phase1_terrain(net, state, occluded_flat)
    core.phase2_fluctuation(net, state, cfg, generator=g)
    core.phase3_fixation(net, state, cfg)
    return state.xi[0]


def plot_grid(
    originals: np.ndarray,
    occluded: np.ndarray,
    modern: np.ndarray,
    hebb: np.ndarray,
    out_path: str,
    title: str = "kanji recall: Modern vs classical Hebb",
) -> None:
    n = originals.shape[0]
    fig, axes = plt.subplots(n, 4, figsize=(6, 1.6 * n))
    col_titles = ("original", "occluded", "Modern", "Hebb")
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t)
    for i in range(n):
        for j, arr in enumerate((originals[i], occluded[i], modern[i], hebb[i])):
            ax = axes[i, j]
            ax.imshow(arr, cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(
            f"{KANJI_GLYPHS[i]} ({KANJI_LABELS[i]})", rotation=0, labelpad=30, va="center"
        )
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved kanji grid -> {out_path}")


def cosine_per_kanji(out: np.ndarray, targets: np.ndarray) -> list[float]:
    sims = []
    for i in range(out.shape[0]):
        a = out[i].reshape(-1)
        b = targets[i].reshape(-1)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        sims.append(float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0)
    return sims


def main(
    *,
    beta: float = 5.0,
    sigma_global: float = 0.1,
    phase2_steps: int = 500,
    phase3_steps: int = 100,
    seed: int = 0,
) -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patterns = np.load(KANJI_PATH)
    n, h, w = patterns.shape
    assert (h, w) == (32, 32) and n == 8
    occluded = np.stack([_half_occlude(p, side="right") for p in patterns], axis=0)

    patterns_flat = torch.from_numpy(patterns.reshape(n, h * w)).to(torch.float32).to(dev)
    occluded_flat = torch.from_numpy(occluded.reshape(n, h * w)).to(torch.float32).to(dev)

    print(f"device: {dev}; beta={beta}, sigma_global={sigma_global}, "
          f"phase2={phase2_steps}, phase3={phase3_steps}")
    modern_out = _recall_all(
        patterns_flat,
        occluded_flat,
        mode="modern",
        beta=beta,
        sigma_global=sigma_global,
        phase2_steps=phase2_steps,
        phase3_steps=phase3_steps,
        device=dev,
        seed=seed,
    )
    hebb_out = _recall_all(
        patterns_flat,
        occluded_flat,
        mode="hebb",
        beta=beta,
        sigma_global=sigma_global,
        phase2_steps=phase2_steps,
        phase3_steps=phase3_steps,
        device=dev,
        seed=seed,
    )

    modern_np = modern_out.detach().cpu().numpy().reshape(n, h, w)
    hebb_np = hebb_out.detach().cpu().numpy().reshape(n, h, w)

    out_png = os.path.join(OUT_DIR, "kanji_memory.png")
    plot_grid(
        patterns,
        occluded,
        modern_np,
        hebb_np,
        out_png,
        title=(
            f"kanji recall (beta={beta}, sigma={sigma_global}, "
            f"phase2={phase2_steps})"
        ),
    )

    sims_modern = cosine_per_kanji(modern_np, patterns)
    sims_hebb = cosine_per_kanji(hebb_np, patterns)
    print("per-kanji cos sim:")
    for ch, m, h_ in zip(KANJI_GLYPHS, sims_modern, sims_hebb):
        print(f"  {ch}: Modern={m:.3f}  Hebb={h_:.3f}")
    print(
        f"means: Modern={np.mean(sims_modern):.3f}, "
        f"Hebb={np.mean(sims_hebb):.3f}"
    )
    return {
        "out_png": out_png,
        "modern_cos_mean": float(np.mean(sims_modern)),
        "hebb_cos_mean": float(np.mean(sims_hebb)),
        "modern_cos": sims_modern,
        "hebb_cos": sims_hebb,
    }


if __name__ == "__main__":
    main()
