"""Ayaram minimum prototype v0.1.

3-layer Hopfield network simulation in PyTorch.

Subpackages / modules:
    core      -- whole-cell-synchronous 4-phase cycle (decision #1)
    modes     -- aya-awake / aya-sleep switching, temperature T and noise sigma
    learning  -- Modern Hopfield continuous update (decision #2 primary) +
                 classical Hebb rule (decision #2 secondary, for comparison)
    memory    -- 3-layer Hopfield network with W = W^T symmetry enforcement
                 (decision #6)

All real implementation is deferred to M1. See README.md for the six design
decisions established by Aya + Yu on 2026-06-17.
"""

__version__ = "0.1.0a0"
