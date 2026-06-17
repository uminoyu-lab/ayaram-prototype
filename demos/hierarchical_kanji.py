"""M3 demo: hierarchical kanji recall (kanji -> radical -> origin).

Decisions per Aru _to-cc-m3.md (Aya + Yu confirmed 2026-06-17):
    (b) hierarchical association is the M3 main target
    (alpha) inter-layer weights learn via Hebb-extension inside Phase 1

Three experiments:
    1. forward recall: half-occluded kanji bitmap at layer 0 -> 4-phase
       cycle -> read layer 0 / layer 1 (radicals) / layer 2 (origin)
    2. reverse recall: radical activation at layer 1 -> 4-phase cycle ->
       read layer 0 (which kanji come up?)
    3. per-layer dynamics: record state norms during Phase 2 / Phase 3
       for one representative kanji on each side (Modern / Hebb) to show
       the K_u-driven per-layer noise difference

Outputs:
    demos/output/hierarchical_kanji_modern.png   -- forward grid, Modern
    demos/output/hierarchical_kanji_hebb.png     -- forward grid, Hebb
    demos/output/hierarchical_dynamics.png       -- per-layer state norms
"""

from __future__ import annotations

import os
import sys
import time
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


def _normalize_inter_weights(net: HopfieldNetwork, target_spectral_norm: float = 1.0) -> None:
    """CC deviation from the strict Hebb-extension formula: rescale each
    inter-layer weight matrix to a common spectral norm.

    Why this is needed: the raw Hebb rule ``W_inter += outer(p_l, p_{l+1}) / N``
    leaves the inter-layer matrix scale dominated by the larger of the two
    layer norms. With layer-0 bitmaps having norm ~32 and layer-1 radical
    patterns having norm 1-3, ``W_01`` ends up with spectral norm ~30,
    whereas ``W_12`` (radicals -> one-hot origins) has spectral norm ~0.7.
    A single ``inter_layer_scale`` therefore cannot make both pairs talk at
    the same magnitude; in practice the layer-0/1 pair runs away while the
    layer-1/2 pair stays silent (the origin layer ends up locked to the
    majority class). Equalizing the spectral norms restores balance while
    preserving the Hebb-derived *direction* of the inter-layer coupling.

    This is a v0.1 pragmatic fix. v0.2 will revisit whether the Hebb learn
    rule should normalize patterns before the outer product (see README
    'v0.2 への宿題').
    """
    for l in range(len(net.layer_sizes) - 1):
        buf = f"W_inter_{l}"
        W = getattr(net, buf)
        s = torch.linalg.svdvals(W)
        sn = float(s[0].item()) if s.numel() > 0 else 0.0
        if sn > 1e-9:
            setattr(net, buf, W * (target_spectral_norm / sn))
from data.kanji_hierarchy import (  # noqa: E402
    KANJI,
    KANJI_ORIGIN,
    KANJI_RADICALS,
    ORIGINS,
    RADICALS,
)


KANJI_PATH = os.path.join(_ROOT, "data", "kanji_8_32x32_v2.npy")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


@dataclass
class ForwardResults:
    mode: str
    layer0_recall: np.ndarray  # (8, 32, 32)
    layer1_recall: np.ndarray  # (8, 256)
    layer2_recall: np.ndarray  # (8, 64)
    layer0_cos: list[float]
    layer1_radical_match: list[float]   # set match (threshold-based)
    layer1_cos: list[float]             # cosine vs target radical pattern
    layer2_origin_match: list[bool]
    layer2_cos: list[float]             # cosine vs target origin pattern


@dataclass
class ReverseResults:
    mode: str
    radical_inputs: list[str]
    layer0_recall: np.ndarray  # (n_radicals, 32, 32)
    layer1_recall: np.ndarray  # (n_radicals, 256)
    closest_kanji_by_l0: list[list[tuple[str, float]]]  # ranked, with cos sim


# ---------- forward (layer 0 occluded -> all layers) ----------------------


def _half_occlude_batch(bitmap: np.ndarray, side: str = "right") -> np.ndarray:
    """Mask half of each {-1, +1} (32, 32) pattern to the background -1."""
    out = bitmap.copy()
    if side == "right":
        out[:, :, 16:] = -1.0
    elif side == "left":
        out[:, :, :16] = -1.0
    elif side == "bottom":
        out[:, 16:, :] = -1.0
    elif side == "top":
        out[:, :16, :] = -1.0
    else:
        raise ValueError(side)
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a.norm()
    nb = b.norm()
    if na.item() == 0 or nb.item() == 0:
        return 0.0
    return float((a @ b / (na * nb)).item())


def run_forward(
    mode: str,
    bitmaps: np.ndarray,
    p1_target: torch.Tensor,
    p2_target: torch.Tensor,
    *,
    beta: float,
    sigma_global: float,
    phase2_steps: int,
    phase3_steps: int,
    inter_layer_scale: float,
    device: torch.device,
    seed: int = 0,
) -> ForwardResults:
    """All 8 kanji recall in one batched cycle."""
    n, h, w = bitmaps.shape
    occluded = _half_occlude_batch(bitmaps, side="right")
    p0_target_flat = torch.from_numpy(bitmaps.reshape(n, h * w)).to(torch.float32).to(device)
    occ_flat = torch.from_numpy(occluded.reshape(n, h * w)).to(torch.float32).to(device)

    net = HopfieldNetwork(mode=mode, seed=seed).to(device)
    net.learn([p0_target_flat, p1_target, p2_target])
    _normalize_inter_weights(net, target_spectral_norm=1.0)

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
    l1_match = []
    l1_cos_list = []
    l2_match = []
    l2_cos_list = []
    for i, k in enumerate(KANJI):
        l0_cos.append(_cosine(state.xi[0][i], p0_target_flat[i]))
        l1_match.append(
            encoding.radical_set_match(state.xi[1][i].detach().cpu(), k, threshold=0.5)
        )
        l1_cos_list.append(_cosine(state.xi[1][i], p1_target[i]))
        l2_match.append(
            encoding.origin_one_hot_match(state.xi[2][i].detach().cpu(), k)
        )
        l2_cos_list.append(_cosine(state.xi[2][i], p2_target[i]))

    return ForwardResults(
        mode=mode,
        layer0_recall=out0,
        layer1_recall=out1,
        layer2_recall=out2,
        layer0_cos=l0_cos,
        layer1_radical_match=l1_match,
        layer1_cos=l1_cos_list,
        layer2_origin_match=l2_match,
        layer2_cos=l2_cos_list,
    )


# ---------- reverse (radical at layer 1 -> what kanji come up?) -----------


def run_reverse(
    mode: str,
    bitmaps: np.ndarray,
    p1_target: torch.Tensor,
    p2_target: torch.Tensor,
    *,
    beta: float,
    sigma_global: float,
    phase2_steps: int,
    phase3_steps: int,
    inter_layer_scale: float,
    device: torch.device,
    seed: int = 0,
) -> ReverseResults:
    n, h, w = bitmaps.shape
    p0_target_flat = torch.from_numpy(bitmaps.reshape(n, h * w)).to(torch.float32).to(device)
    net = HopfieldNetwork(mode=mode, seed=seed).to(device)
    net.learn([p0_target_flat, p1_target, p2_target])
    _normalize_inter_weights(net, target_spectral_norm=1.0)

    radical_inputs = list(RADICALS)
    queries = torch.stack(
        [encoding.encode_radical(r).to(device) for r in radical_inputs]
    )  # (4, 256)

    sizes = net.layer_sizes
    state = core.CycleState(
        xi=[
            torch.zeros(len(radical_inputs), sizes[0], device=device),
            torch.zeros(len(radical_inputs), sizes[1], device=device),
            torch.zeros(len(radical_inputs), sizes[2], device=device),
        ]
    )
    core.phase1_terrain(net, state, queries, layer_idx=1)
    g = torch.Generator(device=device).manual_seed(seed + 7)
    cfg = core.CycleConfig(
        beta=beta,
        sigma_global=sigma_global,
        phase2_steps=phase2_steps,
        phase3_steps=phase3_steps,
        inter_layer_scale=inter_layer_scale,
    )
    core.phase2_fluctuation(net, state, cfg, generator=g)
    core.phase3_fixation(net, state, cfg)

    out0 = state.xi[0].detach().cpu().numpy().reshape(len(radical_inputs), h, w)
    out1 = state.xi[1].detach().cpu().numpy()

    # Rank stored kanji by cosine of layer-0 recall vs each target bitmap.
    closest = []
    for i in range(len(radical_inputs)):
        sims = []
        for j, k in enumerate(KANJI):
            sims.append((k, _cosine(state.xi[0][i], p0_target_flat[j])))
        sims.sort(key=lambda x: -x[1])
        closest.append(sims)
    return ReverseResults(
        mode=mode,
        radical_inputs=radical_inputs,
        layer0_recall=out0,
        layer1_recall=out1,
        closest_kanji_by_l0=closest,
    )


# ---------- per-layer dynamics (single kanji, record state norms) ---------


def run_dynamics(
    mode: str,
    bitmaps: np.ndarray,
    p1_target: torch.Tensor,
    p2_target: torch.Tensor,
    *,
    representative_idx: int,
    beta: float,
    sigma_global: float,
    phase2_steps: int,
    phase3_steps: int,
    inter_layer_scale: float,
    record_every: int = 10,
    device: torch.device,
    seed: int = 0,
) -> dict:
    """Run a single cycle with full Phase 2 / Phase 3 norm history."""
    n, h, w = bitmaps.shape
    occluded = _half_occlude_batch(bitmaps, side="right")
    occ_one = (
        torch.from_numpy(occluded[representative_idx].reshape(h * w))
        .to(torch.float32)
        .to(device)
    )
    p0_target_flat = torch.from_numpy(bitmaps.reshape(n, h * w)).to(torch.float32).to(device)
    net = HopfieldNetwork(mode=mode, seed=seed).to(device)
    net.learn([p0_target_flat, p1_target, p2_target])
    _normalize_inter_weights(net, target_spectral_norm=1.0)

    sizes = net.layer_sizes
    state = core.CycleState(xi=[torch.zeros(s, device=device) for s in sizes])
    core.phase1_terrain(net, state, occ_one, layer_idx=0)
    cfg = core.CycleConfig(
        beta=beta,
        sigma_global=sigma_global,
        phase2_steps=phase2_steps,
        phase3_steps=phase3_steps,
        inter_layer_scale=inter_layer_scale,
    )
    g = torch.Generator(device=device).manual_seed(seed + 13)

    history_steps = []
    history_norms: list[list[float]] = [[] for _ in sizes]
    history_phase: list[str] = []

    def record(step: int, phase_name: str) -> None:
        history_steps.append(step)
        history_phase.append(phase_name)
        for l in range(len(sizes)):
            history_norms[l].append(float(state.xi[l].norm().item()))

    record(0, "phase1")

    # Phase 2 with recording
    net.enforce_constraints()
    dt = cfg.dt
    keep = 1.0 - dt
    sqrt_2dt = (2.0 * dt) ** 0.5
    from ayaram.modes import layer_noise_ratio
    from ayaram.core import _inter_layer_signal
    for step in range(1, cfg.phase2_steps + 1):
        for l, layer in enumerate(net.layers):
            if not layer.has_patterns() and layer.mode == "modern":
                drift = torch.zeros_like(state.xi[l])
            else:
                drift = layer.step(state.xi[l], beta=cfg.beta)
            inter = cfg.inter_layer_scale * _inter_layer_signal(net, state.xi, l)
            sigma_l = cfg.sigma_global * layer_noise_ratio(l)
            eta = torch.randn(
                state.xi[l].shape,
                device=state.xi[l].device,
                dtype=state.xi[l].dtype,
                generator=g,
            )
            state.xi[l] = keep * state.xi[l] + dt * (drift + inter) + sigma_l * sqrt_2dt * eta
        if step % record_every == 0:
            record(step, "phase2")

    # Phase 3 with recording
    net.enforce_constraints()
    beta_hi = cfg.beta * cfg.phase3_beta_boost
    for step in range(1, cfg.phase3_steps + 1):
        for l, layer in enumerate(net.layers):
            if not layer.has_patterns() and layer.mode == "modern":
                drift = torch.zeros_like(state.xi[l])
            else:
                drift = layer.step(state.xi[l], beta=beta_hi)
            inter = cfg.inter_layer_scale * _inter_layer_signal(net, state.xi, l)
            state.xi[l] = keep * state.xi[l] + dt * (drift + inter)
        if step % record_every == 0:
            record(cfg.phase2_steps + step, "phase3")

    return {
        "mode": mode,
        "kanji": KANJI[representative_idx],
        "steps": np.asarray(history_steps),
        "norms": np.asarray(history_norms),  # (3, T)
        "phase_at_step": history_phase,
        "phase2_end": cfg.phase2_steps,
    }


# ---------- plotting ------------------------------------------------------


def plot_forward_grid(
    bitmaps: np.ndarray,
    fr: ForwardResults,
    out_path: str,
) -> None:
    n = bitmaps.shape[0]
    occluded = _half_occlude_batch(bitmaps, side="right")
    fig, axes = plt.subplots(n, 5, figsize=(8.5, 1.7 * n))
    col_titles = (
        "original",
        "occluded",
        "layer 0 recall",
        "layer 1 (radicals)",
        "layer 2 (origin)",
    )
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)
    for i in range(n):
        axes[i, 0].imshow(bitmaps[i], cmap="gray", vmin=-1, vmax=1)
        axes[i, 1].imshow(occluded[i], cmap="gray", vmin=-1, vmax=1)
        axes[i, 2].imshow(fr.layer0_recall[i], cmap="gray", vmin=-1, vmax=1)
        axes[i, 3].bar(
            range(len(RADICALS)),
            fr.layer1_recall[i, : len(RADICALS)],
            color="tab:blue",
        )
        axes[i, 3].set_xticks(range(len(RADICALS)))
        axes[i, 3].set_xticklabels(list(RADICALS), fontsize=8)
        axes[i, 3].axhline(0.5, color="tab:red", linewidth=0.5, linestyle="--")
        axes[i, 4].bar(
            range(len(ORIGINS)),
            fr.layer2_recall[i, : len(ORIGINS)],
            color="tab:green",
        )
        axes[i, 4].set_xticks(range(len(ORIGINS)))
        axes[i, 4].set_xticklabels(list(ORIGINS), fontsize=8)
        for j in (0, 1, 2):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(
            f"{KANJI[i]}\nl0 cos={fr.layer0_cos[i]:.2f}\n"
            f"l1 cos={fr.layer1_cos[i]:.2f}\n"
            f"l2 cos={fr.layer2_cos[i]:.2f}\n"
            f"orig={'OK' if fr.layer2_origin_match[i] else 'X'}",
            rotation=0,
            labelpad=42,
            va="center",
            fontsize=7,
        )
    fig.suptitle(
        f"Hierarchical recall ({fr.mode}): "
        f"l0 cos={np.mean(fr.layer0_cos):.3f}, "
        f"l1 cos={np.mean(fr.layer1_cos):.3f} "
        f"(set={np.mean(fr.layer1_radical_match):.2f}), "
        f"l2 cos={np.mean(fr.layer2_cos):.3f} "
        f"(orig {sum(fr.layer2_origin_match)}/{len(fr.layer2_origin_match)})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved forward ({fr.mode}) -> {out_path}")


def plot_dynamics(
    histories: dict[str, dict],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    layer_colors = ("tab:blue", "tab:orange", "tab:green")
    for ax, (mode, h) in zip(axes, histories.items()):
        for l in range(h["norms"].shape[0]):
            ax.plot(
                h["steps"],
                h["norms"][l],
                color=layer_colors[l],
                label=f"layer {l}",
                linewidth=1.4,
            )
        ax.axvline(
            h["phase2_end"],
            color="tab:gray",
            linestyle="--",
            linewidth=0.8,
            label="phase 2 -> 3",
        )
        ax.set_xlabel("step")
        ax.set_ylabel("|xi_l|")
        ax.set_title(f"{mode}: kanji = {h['kanji']}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Per-layer state norms over the 4-phase cycle "
        "(K_u barrier difference drives the per-layer noise floor)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved dynamics -> {out_path}")


# ---------- main ----------------------------------------------------------


def _report_reverse(rr: ReverseResults) -> None:
    print(f"\n[{rr.mode}] reverse recall (top 3 kanji per radical input):")
    for r, ranking in zip(rr.radical_inputs, rr.closest_kanji_by_l0):
        top3 = ", ".join(f"{k}({s:.2f})" for k, s in ranking[:3])
        print(f"  {r}: {top3}")


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
    bitmaps = np.load(KANJI_PATH)  # {-1, +1} (8, 32, 32)
    n = bitmaps.shape[0]
    print(f"device: {dev}; kanji: {KANJI}")

    p1 = encoding.encode_batch_radical(KANJI).to(dev)
    p2 = encoding.encode_batch_origin(KANJI).to(dev)

    t0 = time.time()
    summary: dict[str, dict] = {}
    histories: dict[str, dict] = {}

    representative_idx = KANJI.index("林")

    for mode in ("modern", "hebb"):
        print(f"\n----- {mode} -----")
        fr = run_forward(
            mode,
            bitmaps,
            p1,
            p2,
            beta=beta,
            sigma_global=sigma_global,
            phase2_steps=phase2_steps,
            phase3_steps=phase3_steps,
            inter_layer_scale=inter_layer_scale,
            device=dev,
            seed=seed,
        )
        print(
            f"forward: l0 cos = {np.mean(fr.layer0_cos):.3f}, "
            f"l1 cos = {np.mean(fr.layer1_cos):.3f} "
            f"(set match {np.mean(fr.layer1_radical_match):.3f}), "
            f"l2 cos = {np.mean(fr.layer2_cos):.3f} "
            f"(origin hit {sum(fr.layer2_origin_match)}/{n})"
        )
        for k, c0, c1, r_, c2, o in zip(
            KANJI,
            fr.layer0_cos,
            fr.layer1_cos,
            fr.layer1_radical_match,
            fr.layer2_cos,
            fr.layer2_origin_match,
        ):
            print(
                f"  {k}: l0_cos={c0:.3f}, l1_cos={c1:.3f} (set={r_:.2f}), "
                f"l2_cos={c2:.3f} (orig={'OK' if o else 'X'})"
            )
        out_path = os.path.join(OUT_DIR, f"hierarchical_kanji_{mode}.png")
        plot_forward_grid(bitmaps, fr, out_path)

        rr = run_reverse(
            mode,
            bitmaps,
            p1,
            p2,
            beta=beta,
            sigma_global=sigma_global,
            phase2_steps=phase2_steps,
            phase3_steps=phase3_steps,
            inter_layer_scale=inter_layer_scale,
            device=dev,
            seed=seed,
        )
        _report_reverse(rr)

        hist = run_dynamics(
            mode,
            bitmaps,
            p1,
            p2,
            representative_idx=representative_idx,
            beta=beta,
            sigma_global=sigma_global,
            phase2_steps=phase2_steps,
            phase3_steps=phase3_steps,
            inter_layer_scale=inter_layer_scale,
            record_every=10,
            device=dev,
            seed=seed,
        )
        histories[mode] = hist
        summary[mode] = {
            "forward_l0_cos_mean": float(np.mean(fr.layer0_cos)),
            "forward_l1_cos_mean": float(np.mean(fr.layer1_cos)),
            "forward_l1_radical_mean": float(np.mean(fr.layer1_radical_match)),
            "forward_l2_cos_mean": float(np.mean(fr.layer2_cos)),
            "forward_l2_origin_hits": sum(fr.layer2_origin_match),
            "forward_l0_cos": list(map(float, fr.layer0_cos)),
            "forward_l1_cos": list(map(float, fr.layer1_cos)),
            "forward_l1_radical": list(map(float, fr.layer1_radical_match)),
            "forward_l2_cos": list(map(float, fr.layer2_cos)),
            "forward_l2_origin": [bool(b) for b in fr.layer2_origin_match],
            "reverse_top1": [
                {"radical": r, "best": ranking[0][0], "cos": ranking[0][1]}
                for r, ranking in zip(rr.radical_inputs, rr.closest_kanji_by_l0)
            ],
            "reverse_full": [
                [(k, c) for k, c in ranking[:3]]
                for ranking in rr.closest_kanji_by_l0
            ],
            "out_png": out_path,
        }

    dyn_out = os.path.join(OUT_DIR, "hierarchical_dynamics.png")
    plot_dynamics(histories, dyn_out)
    summary["dynamics_out_png"] = dyn_out
    print(f"\ntotal elapsed: {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
