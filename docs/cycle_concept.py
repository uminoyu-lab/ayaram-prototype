"""Render the 4-phase cycle conceptual schematic for the v0.1 docs.

Produces ``docs/cycle_concept.png`` -- a single matplotlib figure that
shows Phase 1 / 2 / 3 / 4 as labelled blocks on a timeline, with the role
of each phase, the per-layer T_global value, and the dominant operator
during that phase. Used as a figure in the M5 evaluation report and
linked from ``docs/design_decisions.md``.
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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))


def render(out_path: str | None = None) -> str:
    out_path = out_path or os.path.join(_HERE, "cycle_concept.png")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.set_axis_off()

    phases = [
        (
            0.4, 2.8,
            "Phase 1 -- awake / terrain",
            "tab:blue",
            [
                r"$\xi_0 \leftarrow$ input bias",
                r"(M3+: optional learn=True)",
                r"$T_{\rm global}$ = T_AWAKE (low)",
                r"no fluctuation",
            ],
        ),
        (
            3.4, 5.8,
            "Phase 2 -- sleep / fluctuation",
            "tab:orange",
            [
                r"Langevin update on $\xi_l$",
                r"$\sigma_{\rm local}(l) = \sigma_g \cdot $layer_noise_ratio$(l)$",
                r"$T_{\rm global}$ = T_SLEEP (high)",
                r"~1000 steps (M1+ default)",
            ],
        ),
        (
            6.4, 8.8,
            "Phase 3 -- re-awake / fixation",
            "tab:green",
            [
                r"deterministic high-$\beta$ update",
                r"$\beta \leftarrow \beta \cdot 4$ (sharpens recall)",
                r"$T_{\rm global}$ = T_AWAKE",
                r"~100 steps",
            ],
        ),
        (
            9.4, 11.6,
            "Phase 4 -- readout",
            "tab:purple",
            [
                r"return $\xi_0$ (layer-0 state)",
                r"binarize via sign() for Ising",
                r"all-layers dict for hierarchical",
                r"(M3+)",
            ],
        ),
    ]

    for x0, x1, title, color, bullets in phases:
        rect = mpatches.FancyBboxPatch(
            (x0, 2.0),
            x1 - x0,
            2.8,
            boxstyle="round,pad=0.06,rounding_size=0.18",
            edgecolor=color,
            facecolor=color,
            alpha=0.13,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            (x0 + x1) / 2,
            4.55,
            title,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=color,
        )
        for i, txt in enumerate(bullets):
            ax.text(
                (x0 + x1) / 2,
                4.0 - 0.4 * i,
                txt,
                ha="center",
                va="top",
                fontsize=9,
            )

    # Arrow between phases
    for x0, x1, *_ in phases[:-1]:
        ax.annotate(
            "",
            xy=(x1 + 0.45, 3.4),
            xytext=(x1 + 0.05, 3.4),
            arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "gray"},
        )

    # Time axis
    ax.annotate(
        "",
        xy=(11.6, 1.3),
        xytext=(0.3, 1.3),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "black"},
    )
    ax.text(
        11.6, 1.0, "time (synchronous, all layers)",
        ha="right", va="top", fontsize=9, color="black",
    )

    # All-layer band
    ax.text(
        0.5,
        5.5,
        "All cells, all layers update synchronously every step  "
        r"(Decision #1: option C, $K_u$ per layer is fixed)",
        fontsize=10,
        color="black",
    )

    fig.suptitle(
        "Ayaram v0.1 -- 4-phase cycle",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    render(target)
