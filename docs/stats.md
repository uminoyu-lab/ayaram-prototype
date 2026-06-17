# Project Statistics — Ayaram Prototype v0.1

(Snapshot at M5 close, 2026-06-17.)

## Source code

| Area | files | LOC (incl. comments + docstrings) |
|---|---:|---:|
| `ayaram/` (package) | 7 | 1133 |
| `data/` (datasets + hierarchies) | 6 | 348 |
| `demos/` | 6 | 2001 |
| `tests/` | 10 | 1273 |
| **Total Python** | **29** | **5155** |

(Generated with `wc -l` over `.py` files excluding `.venv`, `.git`,
`__pycache__`, `.pytest_cache`.)

## Tests

- **97 tests, all green**, `uv run pytest` finishes in ~2 seconds on the
  RTX 5090 / CPython 3.12 setup.
- 10 test files:

  | file | tests | covered area |
  |---|---:|---|
  | `tests/test_smoke.py` | 3 | imports + package version string |
  | `tests/test_modes.py` | 10 | K_u barrier table, sigma_local, layer_noise_ratio |
  | `tests/test_learning.py` | 9 | Modern Hopfield update, Hebb weights / update, symmetrize, zero_diag |
  | `tests/test_memory.py` | 14 | HopfieldLayer + HopfieldNetwork, W = Wᵀ, Modern diag = 0 |
  | `tests/test_core.py` | 8 | 4-phase cycle semantics + invariants |
  | `tests/test_ising.py` | 12 | IsingProblem + MaxCutProblem (hand-checked 4-node graph) |
  | `tests/test_hierarchical.py` | 21 | hierarchy dicts, encoders, learn / recall / recall_from_layer |
  | `tests/test_encoding_v15.py` | 7 | Option B unary count, M3 colinearity resolved |
  | `tests/test_learn_spectral.py` | 6 | normalize_inter='spectral' / 'none', pair imbalance equalized |
  | `tests/test_learn_centering.py` | 7 | center_inter_inputs toggle, M4 layer-2 recovery, Hebb caveat |

## Dependencies

Direct (`pyproject.toml`):
- `torch>=2.7` (installed: **2.11.0+cu128** -- CUDA 12.8 wheel for RTX 5090 / Blackwell)
- `numpy>=2.0` (installed: **2.4.6**)
- `matplotlib>=3.9` (installed: **3.11.0**)
- `cairosvg>=2.7` (installed: **2.9.0**)
- (dev) `pytest>=8.0` (installed: **9.1.0**)

Transitive: 60 packages resolved in `uv.lock` (sympy, networkx, pillow,
fonttools, kiwisolver, contourpy, mpmath, jinja2, filelock, fsspec,
cairocffi, cffi, etc.).

## Runtime targets

- **Python**: 3.12 (pinned via `.python-version`); minimum supported 3.10.
- **GPU**: NVIDIA RTX 5090 (Blackwell, sm_120, 32 GiB) via PyTorch cu128
  wheel index. `torch.cuda.is_available() == True` and
  `torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 5090'`.
- **OS**: Windows 11 Pro (development), driver 596.21 / CUDA 13.2.
- **Package manager**: uv 0.11.15 (handles the cu128 torch index, the
  hatchling-built local install, and Python 3.12 itself).

## Demo runtimes (M5 snapshot, RTX 5090)

| demo | runtime | output |
|---|---|---|
| `attention_test.py` Part A | < 1 s | Theorem 3 max abs diff = 0.0 |
| `attention_test.py` Part B (20x20 β-σ map) | 413 s | `beta_sigma_map.png` + `.npz` |
| `attention_test.py` Part C (β finescan) | 22 s | `beta_finescan.png` |
| `kanji_memory.py` | 3 s | `kanji_memory.png` |
| `ising_solver.py` (N in {8,16,32} x 10 trials) | 10 s | `ising_maxcut.png` |
| `hierarchical_kanji.py` (M3 dynamics demo) | 3 s | 3 PNGs |
| `hierarchical_kanji_v15.py` (M3/M4/M4-12 comparison) | 4 s | 4 PNGs |
| `reverse_recall.py` (radical → kanji) | 4 s | `reverse_recall.png` |

## Git history

```
275b06e M5: center_inter_inputs API, reverse recall, l1_cos primary metric
d399dcd M4: orthogonal encoding + spectral normalize + 12-kanji expansion
db5abdf M3: hierarchical kanji recall (kanji -> radical -> origin)
7e23ff8 M2: MAX-CUT solver + Ising mapping + Part C beta finescan
0b47943 chore: untrack demos/output artifacts
4ffcd94 M1: 3-layer Hopfield, 4-phase cycle, demos, tests
254c4f3 M0: scaffold ayaram-prototype with v0.1 design decisions
```

(Plus the M5-polish commit that includes this file.)

Repository is local-only at this snapshot. `git push` is blocked by
project convention (license + public-repo decisions are pending Aya + Yu).
