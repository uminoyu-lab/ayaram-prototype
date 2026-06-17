"""Ayaram minimum prototype v0.1.

3-layer Hopfield network simulation in PyTorch.

Subpackages / modules:
    core      -- whole-cell-synchronous 4-phase cycle (decision #1)
    modes     -- aya-awake / aya-sleep switching, temperature T and noise sigma
    learning  -- Modern Hopfield continuous update (decision #2 primary) +
                 classical Hebb rule (decision #2 secondary, for comparison)
    memory    -- 3-layer Hopfield network with W = W^T symmetry enforcement
                 (decision #6) + ``learn(normalize_inter, center_inter_inputs)``
                 (M4 / M5)
    ising     -- MAX-CUT problem mapping (M2)
    encoding  -- layer-1 radical + layer-2 origin encoders for the
                 hierarchical recall demo (M3 multi-hot + M4 Option B
                 orthogonal)

All milestones M0 -- M5 are complete. See README.md for the six design
decisions established by Aya + Yu on 2026-06-17 and the v0.2 homework list.
"""

__version__ = "0.1.0"
