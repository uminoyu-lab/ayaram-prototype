"""M5 demo: reverse recall (radical pattern at layer 1 -> kanji at layer 0).

Aru M5 sub-decision #3 (2026-06-17): try reverse recall in v0.1 using the
M4 stack (Option B encoding + spectral normalization + zero-centered
inter-layer Hebb). M3 reported reverse recall as effectively non-functional;
this demo tests whether the M4 fixes also help the backward direction.

For each of the 7 radicals in ``data/kanji_hierarchy_v15``:
  1. Build the layer-1 Option B encoding of "a kanji that contains exactly
     that single radical, count 1" -- i.e. an idealized atomic query.
  2. Inject into layer 1 (Phase 1 at ``layer_idx=1``), let layer 0 / 2
     be initialized at small random.
  3. Run the standard 4-phase cycle.
  4. Score the layer-0 readout against all 12 stored kanji bitmaps.

Output:
  demos/output/reverse_recall.png -- 7 radicals x top-3 kanji grid.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
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

from ayaram import core, encoding  # noqa: E402
from ayaram.memory import HopfieldNetwork  # noqa: E402
from data import kanji_hierarchy_v15 as KH_V15  # noqa: E402


KANJI_PATH = os.path.join(_ROOT, "data", "kanji_12_32x32_v15.npy")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
TOPK = 3


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a.norm()
    nb = b.norm()
    if na.item() == 0 or nb.item() == 0:
        return 0.0
    return float((a @ b / (na * nb)).item())


def _atomic_radical_query(
    radical: str,
    *,
    radicals: tuple[str, ...] = KH_V15.RADICALS,
    max_count: int = KH_V15.MAX_COUNT,
    dim: int = encoding.LAYER1_DIM,
) -> torch.Tensor:
    """The layer-1 Option B vector representing "this single radical, count 1"
    without committing to any specific kanji.

    Equivalent to a one-shot kanji whose only radical is ``radical`` with
    count 1: radical-presence dim active + count_unary[r][0] active.
    """
    if radical not in radicals:
        raise KeyError(f"unknown radical {radical!r}")
    R = len(radicals)
    p = torch.zeros(dim, dtype=torch.float32)
    r_idx = radicals.index(radical)
    p[r_idx] = 1.0
    p[R + r_idx * max_count + 0] = 1.0  # count >= 1
    return p


def run_reverse(
    *,
    beta: float = 5.0,
    sigma_global: float = 0.1,
    phase2_steps: int = 500,
    phase3_steps: int = 100,
    inter_layer_scale: float = 0.1,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bitmaps = np.load(KANJI_PATH)
    n, h, w = bitmaps.shape
    kanji = KH_V15.KANJI

    # Stored patterns + W_inter installed with the M4/M5 canonical settings.
    p0_flat = torch.from_numpy(bitmaps.reshape(n, h * w)).to(torch.float32).to(dev)
    p1 = encoding.encode_batch_radical_count_v15(
        kanji,
        kanji_radicals=KH_V15.KANJI_RADICALS,
        radicals=KH_V15.RADICALS,
        max_count=KH_V15.MAX_COUNT,
    ).to(dev)
    p2 = encoding.encode_batch_origin_v15(
        kanji,
        kanji_origin=KH_V15.KANJI_ORIGIN,
        origins=KH_V15.ORIGINS,
    ).to(dev)

    results: dict[str, list] = {}
    layer0_recall_per_radical: dict[str, np.ndarray] = {}
    cycle_summary: dict[str, dict] = {}

    for mode in ("modern", "hebb"):
        net = HopfieldNetwork(mode=mode, seed=seed).to(dev)
        net.learn(
            [p0_flat, p1, p2],
            normalize_inter="spectral",
            center_inter_inputs=True,
        )

        per_mode_ranks = []
        per_mode_bitmaps = []
        for radical in KH_V15.RADICALS:
            q1 = _atomic_radical_query(radical).to(dev)
            sizes = net.layer_sizes
            state = core.CycleState(
                xi=[torch.zeros(s, device=dev) for s in sizes]
            )
            # small random seed at layer 0 to break symmetry
            g0 = torch.Generator(device=dev).manual_seed(seed + hash(radical) % 10_000)
            state.xi[0] = 0.01 * torch.randn(sizes[0], device=dev, generator=g0)
            core.phase1_terrain(net, state, q1, layer_idx=1)
            cfg = core.CycleConfig(
                beta=beta,
                sigma_global=sigma_global,
                phase2_steps=phase2_steps,
                phase3_steps=phase3_steps,
                inter_layer_scale=inter_layer_scale,
            )
            g = torch.Generator(device=dev).manual_seed(seed + 1)
            core.phase2_fluctuation(net, state, cfg, generator=g)
            core.phase3_fixation(net, state, cfg)
            xi0 = state.xi[0]
            sims = [
                (k, _cosine(xi0, p0_flat[i]))
                for i, k in enumerate(kanji)
            ]
            sims.sort(key=lambda x: -x[1])
            per_mode_ranks.append({"radical": radical, "top": sims})
            per_mode_bitmaps.append(xi0.detach().cpu().numpy().reshape(h, w))

        results[mode] = per_mode_ranks
        layer0_recall_per_radical[mode] = np.stack(per_mode_bitmaps, axis=0)

        # Success metric: fraction of radicals whose top-K layer-0 picks
        # include at least one kanji that actually contains that radical.
        recovered = 0
        for entry in per_mode_ranks:
            r = entry["radical"]
            topk = [k for k, _ in entry["top"][:TOPK]]
            if any(r in KH_V15.KANJI_RADICALS[k] for k in topk):
                recovered += 1
        cycle_summary[mode] = {
            "n_radicals": len(KH_V15.RADICALS),
            "topk": TOPK,
            "topk_radical_hit_rate": recovered / len(KH_V15.RADICALS),
        }

    return {
        "results": results,
        "summary": cycle_summary,
        "bitmaps_per_radical": layer0_recall_per_radical,
        "kanji": kanji,
        "bitmaps": bitmaps,
    }


def plot_reverse_grid(reverse_out: dict, out_path: str) -> None:
    """Grid: rows = radicals, columns = [query | top-1 | top-2 | top-3] for
    each mode (Modern row block, then Hebb)."""
    modes = ("modern", "hebb")
    radicals = KH_V15.RADICALS
    n_rad = len(radicals)
    cols = 1 + TOPK
    fig, axes = plt.subplots(
        n_rad * len(modes), cols, figsize=(2.1 * cols, 1.55 * n_rad * len(modes))
    )

    # Pre-render the query as a synthetic layer-1 bar (no bitmap),
    # represented by the radical glyph in the first column.
    for m_i, mode in enumerate(modes):
        ranks = reverse_out["results"][mode]
        bitmaps_per_rad = reverse_out["bitmaps_per_radical"][mode]
        for r_i, entry in enumerate(ranks):
            row = m_i * n_rad + r_i
            r_label = entry["radical"]
            top = entry["top"][:TOPK]
            # column 0: radical glyph as a label + recalled layer-0 bitmap
            ax = axes[row, 0]
            ax.imshow(bitmaps_per_rad[r_i], cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{mode}\nquery={r_label}\nlayer-0 readout",
                fontsize=7,
            )
            for c, (k, s) in enumerate(top, start=1):
                ax = axes[row, c]
                k_idx = list(reverse_out["kanji"]).index(k)
                ax.imshow(reverse_out["bitmaps"][k_idx], cmap="gray", vmin=-1, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                contains = r_label in KH_V15.KANJI_RADICALS[k]
                marker = "OK" if contains else "no"
                ax.set_title(f"#{c}: {k}\ncos={s:.2f}  {marker}", fontsize=7)

    summary = reverse_out["summary"]
    fig.suptitle(
        "Reverse recall: atomic radical query at layer 1 -> kanji at layer 0  "
        f"(Modern top-{TOPK} radical-hit = "
        f"{summary['modern']['topk_radical_hit_rate']:.2f}; "
        f"Hebb top-{TOPK} = {summary['hebb']['topk_radical_hit_rate']:.2f})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    out = run_reverse()
    for mode, summary in out["summary"].items():
        print(
            f"{mode}: top-{summary['topk']} radical-hit "
            f"= {summary['topk_radical_hit_rate']:.3f} "
            f"({int(summary['topk_radical_hit_rate'] * summary['n_radicals'])}"
            f"/{summary['n_radicals']})"
        )
        for entry in out["results"][mode]:
            r = entry["radical"]
            top = entry["top"][:TOPK]
            hits = [
                "OK" if r in KH_V15.KANJI_RADICALS[k] else "no"
                for k, _ in top
            ]
            tops_str = ", ".join(
                f"{k}({s:.2f}, {h})" for (k, s), h in zip(top, hits)
            )
            print(f"  radical {r}: {tops_str}")
    plot_reverse_grid(out, os.path.join(OUT_DIR, "reverse_recall.png"))
    return out


if __name__ == "__main__":
    main()
