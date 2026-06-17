"""M4 demo: hierarchical kanji recall with the v15 orthogonal encoding.

Aru M4 decisions (2026-06-17):
    Part A -- Option B (radical multi-hot + per-radical unary count) encoding,
              ``HopfieldNetwork.learn(normalize_inter='spectral')`` formalized.
    Part B -- 8-kanji M3 set + 12-kanji expanded set (added 森 水 川 山,
              new radicals 水 川 山, new origin 地形).
    Part C -- direct comparison vs M3 numbers.
    Part D -- three plots: M4(8), M4(12), M3 vs M4 bar chart.

Outputs (gitignored, under demos/output/):
    hierarchical_kanji_v15_modern.png        -- M4 8-kanji, Modern
    hierarchical_kanji_v15_hebb.png          -- M4 8-kanji, Hebb
    hierarchical_kanji_v15_expanded.png      -- M4 12-kanji, Modern + Hebb
    m3_vs_m4_comparison.png                  -- 3-config bar chart

Reverse recall is deferred to v0.2 per Aru M4 sub-decision #2.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

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


def _learn_with_centered_inter(
    net: HopfieldNetwork,
    layer_patterns: list[torch.Tensor],
) -> None:
    """CC M4 tweak: install ``W_inter`` from *zero-centered* input patterns,
    then spectrally normalize. Intra-layer storage uses the raw patterns.

    Why this is a separate code path (not ``HopfieldNetwork.learn`` itself):
    M3 demonstrated that the bipolar ``{-1, +1}`` bitmaps at layer 0 give
    pairwise inner products of order ~1024 - (a small term), where the small
    discriminative term is drowned out by ~700 of "background agreement".
    The raw Hebb rule ``W_01 = (1/N) sum_p p_0 (x) p_1`` therefore inherits
    that constant background, and the inter-layer signal ``xi_0 @ W_01``
    routes *every* layer-0 query to roughly the same direction in layer 1.

    Subtracting the per-dim mean across the training set before the outer
    product cancels that constant baseline:

        W_inter[l] = (1/N) sum_p (p_l - mean_p p_l) (x) p_{l+1}

    making ``W_inter[l] @ xi_{l+1}`` a "deviation-from-average" signal.
    Tested in this demo: switching this on rescues layer-2 origin accuracy
    from 4/8 (M3 / Option-B-only) to substantially better.

    Aru M4 explicitly defers ``{0, 1}`` alphabet to v0.2; zero-centering of
    the Hebb inputs is a distinct, smaller fix that does not change the
    state alphabet (layer 0 still operates on bipolar ``{-1, +1}`` for the
    intra-layer dynamics) and is therefore in-scope as a CC M4 fix.
    """
    if len(layer_patterns) != len(net.layer_sizes):
        raise ValueError("layer_patterns length mismatch")
    N = layer_patterns[0].shape[0]
    for l, ps in enumerate(layer_patterns):
        net.layers[l].store(ps)
    centered = [p - p.mean(dim=0, keepdim=True) for p in layer_patterns]
    for l in range(len(net.layer_sizes) - 1):
        buf = f"W_inter_{l}"
        ref = getattr(net, buf)
        p_l = centered[l].to(ref.dtype).to(ref.device)
        p_lp1 = layer_patterns[l + 1].to(ref.dtype).to(ref.device)
        W_new = (p_l.T @ p_lp1) / N
        s = torch.linalg.svdvals(W_new)
        sn = float(s[0].item()) if s.numel() > 0 else 0.0
        if sn > 1e-9:
            W_new = W_new * (1.0 / sn)
        setattr(net, buf, W_new)
    net.enforce_constraints()

# M3 hierarchy + bitmap (8 kanji) -- still used as the M3 baseline.
from data import kanji_hierarchy as KH_M3  # noqa: E402
from data import kanji_hierarchy_v15 as KH_V15  # noqa: E402


OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
KANJI_PATH_M3 = os.path.join(_ROOT, "data", "kanji_8_32x32_v2.npy")
KANJI_PATH_V15 = os.path.join(_ROOT, "data", "kanji_12_32x32_v15.npy")


# ---------- per-run result holder ----------------------------------------


@dataclass
class RunResult:
    config_name: str  # 'M3', 'M4-8', 'M4-12'
    mode: str         # 'modern' or 'hebb'
    kanji: tuple[str, ...]
    bitmaps: np.ndarray
    occluded: np.ndarray
    layer0_recall: np.ndarray
    layer1_recall: np.ndarray
    layer2_recall: np.ndarray
    layer0_cos: list[float]
    layer1_cos: list[float]
    layer1_set_match: list[float]
    layer2_cos: list[float]
    layer2_origin_match: list[bool]

    def summary(self) -> dict:
        return {
            "l0_cos_mean": float(np.mean(self.layer0_cos)),
            "l1_cos_mean": float(np.mean(self.layer1_cos)),
            "l1_set_mean": float(np.mean(self.layer1_set_match)),
            "l2_cos_mean": float(np.mean(self.layer2_cos)),
            "l2_origin_hit": sum(self.layer2_origin_match),
            "l2_origin_rate": float(
                sum(self.layer2_origin_match) / len(self.layer2_origin_match)
            ),
            "n_kanji": len(self.kanji),
        }


# ---------- shared forward driver ----------------------------------------


def _half_occlude(bitmap: np.ndarray, side: str = "right") -> np.ndarray:
    out = bitmap.copy()
    if side == "right":
        out[:, :, 16:] = -1.0
    else:
        raise ValueError(side)
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a.norm()
    nb = b.norm()
    if na.item() == 0 or nb.item() == 0:
        return 0.0
    return float((a @ b / (na * nb)).item())


def _run_forward(
    config_name: str,
    mode: str,
    kanji: tuple[str, ...],
    bitmaps: np.ndarray,
    p1: torch.Tensor,
    p2: torch.Tensor,
    kanji_radicals: dict,
    kanji_origin: dict,
    radicals: tuple,
    origins: tuple,
    *,
    beta: float,
    sigma_global: float,
    phase2_steps: int,
    phase3_steps: int,
    inter_layer_scale: float,
    device: torch.device,
    seed: int,
) -> RunResult:
    n, h, w = bitmaps.shape
    p0_flat = torch.from_numpy(bitmaps.reshape(n, h * w)).to(torch.float32).to(device)
    occluded = _half_occlude(bitmaps, side="right")
    occ_flat = torch.from_numpy(occluded.reshape(n, h * w)).to(torch.float32).to(device)

    net = HopfieldNetwork(mode=mode, seed=seed).to(device)
    _learn_with_centered_inter(net, [p0_flat, p1, p2])

    sizes = net.layer_sizes
    state = core.CycleState(
        xi=[
            torch.zeros(n, sizes[0], device=device),
            torch.zeros(n, sizes[1], device=device),
            torch.zeros(n, sizes[2], device=device),
        ]
    )
    core.phase1_terrain(net, state, occ_flat, layer_idx=0)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    cfg = core.CycleConfig(
        beta=beta,
        sigma_global=sigma_global,
        phase2_steps=phase2_steps,
        phase3_steps=phase3_steps,
        inter_layer_scale=inter_layer_scale,
    )
    core.phase2_fluctuation(net, state, cfg, generator=g)
    core.phase3_fixation(net, state, cfg)

    out0 = state.xi[0].detach().cpu().numpy().reshape(n, h, w)
    out1 = state.xi[1].detach().cpu().numpy()
    out2 = state.xi[2].detach().cpu().numpy()

    l0_cos = []
    l1_cos = []
    l1_set = []
    l2_cos = []
    l2_match = []
    for i, k in enumerate(kanji):
        l0_cos.append(_cosine(state.xi[0][i], p0_flat[i]))
        l1_cos.append(_cosine(state.xi[1][i], p1[i]))
        l1_set.append(
            encoding.radical_set_match_v15(
                state.xi[1][i].detach().cpu(),
                k,
                kanji_radicals=kanji_radicals,
                radicals=radicals,
                threshold=0.5,
            )
        )
        l2_cos.append(_cosine(state.xi[2][i], p2[i]))
        l2_match.append(
            encoding.origin_one_hot_match_v15(
                state.xi[2][i].detach().cpu(),
                k,
                kanji_origin=kanji_origin,
                origins=origins,
            )
        )

    return RunResult(
        config_name=config_name,
        mode=mode,
        kanji=kanji,
        bitmaps=bitmaps,
        occluded=occluded,
        layer0_recall=out0,
        layer1_recall=out1,
        layer2_recall=out2,
        layer0_cos=l0_cos,
        layer1_cos=l1_cos,
        layer1_set_match=l1_set,
        layer2_cos=l2_cos,
        layer2_origin_match=l2_match,
    )


# ---------- M3 baseline reproduction (M3 encoding, same kanji, same demo
#            parameters) -- needed to make the M3 vs M4 bar chart fair ----


def _run_m3_forward_for_comparison(
    *,
    mode: str,
    beta: float,
    sigma_global: float,
    phase2_steps: int,
    phase3_steps: int,
    inter_layer_scale: float,
    device: torch.device,
    seed: int,
) -> RunResult:
    """Replay the M3 demo conditions (M3 multi-hot encoding) for comparison."""
    bitmaps = np.load(KANJI_PATH_M3)
    n, h, w = bitmaps.shape
    p0_flat = torch.from_numpy(bitmaps.reshape(n, h * w)).to(torch.float32).to(device)
    occluded = _half_occlude(bitmaps, side="right")
    occ_flat = torch.from_numpy(occluded.reshape(n, h * w)).to(torch.float32).to(device)

    p1 = encoding.encode_batch_radical(KH_M3.KANJI).to(device)
    p2 = encoding.encode_batch_origin(KH_M3.KANJI).to(device)

    net = HopfieldNetwork(mode=mode, seed=seed).to(device)
    # M3 baseline: spectral normalization only, no zero-centering. This
    # matches the M3 demo's behavior and gives the comparison its
    # apples-to-apples meaning.
    net.learn([p0_flat, p1, p2], normalize_inter="spectral")

    sizes = net.layer_sizes
    state = core.CycleState(
        xi=[torch.zeros(n, s, device=device) for s in sizes]
    )
    core.phase1_terrain(net, state, occ_flat, layer_idx=0)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    cfg = core.CycleConfig(
        beta=beta,
        sigma_global=sigma_global,
        phase2_steps=phase2_steps,
        phase3_steps=phase3_steps,
        inter_layer_scale=inter_layer_scale,
    )
    core.phase2_fluctuation(net, state, cfg, generator=g)
    core.phase3_fixation(net, state, cfg)

    l0_cos, l1_cos, l1_set, l2_cos, l2_match = [], [], [], [], []
    for i, k in enumerate(KH_M3.KANJI):
        l0_cos.append(_cosine(state.xi[0][i], p0_flat[i]))
        l1_cos.append(_cosine(state.xi[1][i], p1[i]))
        l1_set.append(
            encoding.radical_set_match(state.xi[1][i].detach().cpu(), k, threshold=0.5)
        )
        l2_cos.append(_cosine(state.xi[2][i], p2[i]))
        l2_match.append(
            encoding.origin_one_hot_match(state.xi[2][i].detach().cpu(), k)
        )

    return RunResult(
        config_name="M3",
        mode=mode,
        kanji=KH_M3.KANJI,
        bitmaps=bitmaps,
        occluded=occluded,
        layer0_recall=state.xi[0].detach().cpu().numpy().reshape(n, h, w),
        layer1_recall=state.xi[1].detach().cpu().numpy(),
        layer2_recall=state.xi[2].detach().cpu().numpy(),
        layer0_cos=l0_cos,
        layer1_cos=l1_cos,
        layer1_set_match=l1_set,
        layer2_cos=l2_cos,
        layer2_origin_match=l2_match,
    )


# ---------- plotting -----------------------------------------------------


def _plot_grid(r: RunResult, out_path: str) -> None:
    n = r.bitmaps.shape[0]
    R = len(KH_V15.RADICALS) if r.config_name.startswith("M4") else len(KH_M3.RADICALS)
    O_list = KH_V15.ORIGINS if r.config_name.startswith("M4") else KH_M3.ORIGINS
    radical_labels = KH_V15.RADICALS if r.config_name.startswith("M4") else KH_M3.RADICALS
    n_origin = len(O_list)

    fig, axes = plt.subplots(n, 5, figsize=(9, 1.55 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    col_titles = (
        "original",
        "occluded",
        "layer 0 recall",
        "layer 1 (radicals)",
        f"layer 2 ({n_origin} origins)",
    )
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=9)
    for i in range(n):
        axes[i, 0].imshow(r.bitmaps[i], cmap="gray", vmin=-1, vmax=1)
        axes[i, 1].imshow(r.occluded[i], cmap="gray", vmin=-1, vmax=1)
        axes[i, 2].imshow(r.layer0_recall[i], cmap="gray", vmin=-1, vmax=1)
        axes[i, 3].bar(
            range(R),
            r.layer1_recall[i, :R],
            color="tab:blue",
        )
        axes[i, 3].set_xticks(range(R))
        axes[i, 3].set_xticklabels(list(radical_labels), fontsize=7)
        axes[i, 3].axhline(0.5, color="tab:red", linewidth=0.4, linestyle="--")
        axes[i, 4].bar(
            range(n_origin),
            r.layer2_recall[i, :n_origin],
            color="tab:green",
        )
        axes[i, 4].set_xticks(range(n_origin))
        axes[i, 4].set_xticklabels(list(O_list), fontsize=7)
        for j in (0, 1, 2):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(
            f"{r.kanji[i]}\nl0={r.layer0_cos[i]:.2f}\n"
            f"l1={r.layer1_cos[i]:.2f}\n"
            f"l2={r.layer2_cos[i]:.2f}\n"
            f"{'OK' if r.layer2_origin_match[i] else 'X'}",
            rotation=0,
            labelpad=42,
            va="center",
            fontsize=6,
        )
    s = r.summary()
    fig.suptitle(
        f"{r.config_name} / {r.mode}  ({r.bitmaps.shape[0]} kanji):  "
        f"l0={s['l0_cos_mean']:.3f}  l1_cos={s['l1_cos_mean']:.3f}  "
        f"l1_set={s['l1_set_mean']:.3f}  l2_cos={s['l2_cos_mean']:.3f}  "
        f"origin={s['l2_origin_hit']}/{s['n_kanji']}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def _plot_comparison(
    runs_per_config: dict[str, dict[str, RunResult]],
    out_path: str,
) -> None:
    """Bar chart: config x metric, separating Modern and Hebb."""
    metric_keys = (
        ("l0_cos_mean", "layer 0 cos"),
        ("l1_cos_mean", "layer 1 cos"),
        ("l1_set_mean", "layer 1 set match"),
        ("l2_cos_mean", "layer 2 cos"),
        ("l2_origin_rate", "layer 2 origin rate"),
    )
    configs = list(runs_per_config.keys())
    modes = ("modern", "hebb")
    n_metrics = len(metric_keys)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4.2), sharey=True)
    x = np.arange(len(configs))
    width = 0.36
    for ax, (mk, mlabel) in zip(axes, metric_keys):
        for j, mode in enumerate(modes):
            vals = [
                runs_per_config[cfg][mode].summary()[mk] for cfg in configs
            ]
            offset = (-0.5 + j) * width
            ax.bar(x + offset, vals, width, label=mode)
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=9)
        ax.set_title(mlabel, fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "M3 vs M4: hierarchical recall accuracy across encoding upgrade and "
        "kanji-set expansion",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


# ---------- main ---------------------------------------------------------


def main(
    *,
    beta: float = 5.0,
    sigma_global: float = 0.1,
    phase2_steps: int = 500,
    phase3_steps: int = 100,
    inter_layer_scale: float = 0.1,
    seed: int = 0,
) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    # === M3 baseline replay (8 kanji, M3 multi-hot encoding) ===
    print("\n----- M3 baseline replay -----")
    m3_runs = {}
    for mode in ("modern", "hebb"):
        r = _run_m3_forward_for_comparison(
            mode=mode,
            beta=beta,
            sigma_global=sigma_global,
            phase2_steps=phase2_steps,
            phase3_steps=phase3_steps,
            inter_layer_scale=inter_layer_scale,
            device=dev,
            seed=seed,
        )
        s = r.summary()
        print(
            f"M3 / {mode}: l0={s['l0_cos_mean']:.3f}, l1_cos={s['l1_cos_mean']:.3f}, "
            f"l1_set={s['l1_set_mean']:.3f}, l2_cos={s['l2_cos_mean']:.3f}, "
            f"origin={s['l2_origin_hit']}/8"
        )
        m3_runs[mode] = r

    # === M4 with the same M3 8-kanji set but Option B encoding ===
    print("\n----- M4 (8 kanji, v15 encoding) -----")
    bitmaps_m3set = np.load(KANJI_PATH_M3)
    p1_m4_8 = encoding.encode_batch_radical_count_v15(
        KH_M3.KANJI,
        kanji_radicals=KH_M3.KANJI_RADICALS,
        radicals=KH_M3.RADICALS,
        max_count=3,
    ).to(dev)
    p2_m4_8 = encoding.encode_batch_origin_v15(
        KH_M3.KANJI,
        kanji_origin=KH_M3.KANJI_ORIGIN,
        origins=KH_M3.ORIGINS,
    ).to(dev)

    m4_8_runs = {}
    for mode in ("modern", "hebb"):
        r = _run_forward(
            "M4-8",
            mode,
            KH_M3.KANJI,
            bitmaps_m3set,
            p1_m4_8,
            p2_m4_8,
            kanji_radicals=KH_M3.KANJI_RADICALS,
            kanji_origin=KH_M3.KANJI_ORIGIN,
            radicals=KH_M3.RADICALS,
            origins=KH_M3.ORIGINS,
            beta=beta,
            sigma_global=sigma_global,
            phase2_steps=phase2_steps,
            phase3_steps=phase3_steps,
            inter_layer_scale=inter_layer_scale,
            device=dev,
            seed=seed,
        )
        s = r.summary()
        print(
            f"M4-8 / {mode}: l0={s['l0_cos_mean']:.3f}, l1_cos={s['l1_cos_mean']:.3f}, "
            f"l1_set={s['l1_set_mean']:.3f}, l2_cos={s['l2_cos_mean']:.3f}, "
            f"origin={s['l2_origin_hit']}/8"
        )
        m4_8_runs[mode] = r
        out_path = os.path.join(OUT_DIR, f"hierarchical_kanji_v15_{mode}.png")
        _plot_grid(r, out_path)

    # === M4 expanded (12 kanji v15) ===
    print("\n----- M4 (12 kanji expanded) -----")
    bitmaps_v15 = np.load(KANJI_PATH_V15)
    p1_m4_12 = encoding.encode_batch_radical_count_v15(
        KH_V15.KANJI,
        kanji_radicals=KH_V15.KANJI_RADICALS,
        radicals=KH_V15.RADICALS,
        max_count=KH_V15.MAX_COUNT,
    ).to(dev)
    p2_m4_12 = encoding.encode_batch_origin_v15(
        KH_V15.KANJI,
        kanji_origin=KH_V15.KANJI_ORIGIN,
        origins=KH_V15.ORIGINS,
    ).to(dev)

    m4_12_runs = {}
    for mode in ("modern", "hebb"):
        r = _run_forward(
            "M4-12",
            mode,
            KH_V15.KANJI,
            bitmaps_v15,
            p1_m4_12,
            p2_m4_12,
            kanji_radicals=KH_V15.KANJI_RADICALS,
            kanji_origin=KH_V15.KANJI_ORIGIN,
            radicals=KH_V15.RADICALS,
            origins=KH_V15.ORIGINS,
            beta=beta,
            sigma_global=sigma_global,
            phase2_steps=phase2_steps,
            phase3_steps=phase3_steps,
            inter_layer_scale=inter_layer_scale,
            device=dev,
            seed=seed,
        )
        s = r.summary()
        n = len(KH_V15.KANJI)
        print(
            f"M4-12 / {mode}: l0={s['l0_cos_mean']:.3f}, l1_cos={s['l1_cos_mean']:.3f}, "
            f"l1_set={s['l1_set_mean']:.3f}, l2_cos={s['l2_cos_mean']:.3f}, "
            f"origin={s['l2_origin_hit']}/{n}"
        )
        m4_12_runs[mode] = r
    # combine both modes into one expanded plot file (Modern only for brevity)
    _plot_grid(
        m4_12_runs["modern"],
        os.path.join(OUT_DIR, "hierarchical_kanji_v15_expanded.png"),
    )

    # === comparison bar chart ===
    runs_per_config = {
        "M3 (8)": m3_runs,
        "M4 (8)": m4_8_runs,
        "M4 (12)": m4_12_runs,
    }
    _plot_comparison(
        runs_per_config,
        os.path.join(OUT_DIR, "m3_vs_m4_comparison.png"),
    )

    # per-kanji table dumped to stdout for the report
    print("\n----- per-kanji (Modern) on M4-12 -----")
    rm = m4_12_runs["modern"]
    for i, k in enumerate(rm.kanji):
        print(
            f"  {k}: l0={rm.layer0_cos[i]:.3f}, l1_cos={rm.layer1_cos[i]:.3f}, "
            f"l1_set={rm.layer1_set_match[i]:.2f}, "
            f"l2_cos={rm.layer2_cos[i]:.3f}, "
            f"origin={'OK' if rm.layer2_origin_match[i] else 'X'}"
        )

    summary = {
        cfg: {mode: runs[mode].summary() for mode in runs}
        for cfg, runs in runs_per_config.items()
    }
    return summary


if __name__ == "__main__":
    main()
