"""Memory: 3-layer Hopfield network.

Design decision #3 (Aya + Yu, 2026-06-17): cell shape 32 x 32, layer widths
1024 / 256 / 64, starting with 8 stored kanji patterns.

Design decision #6 (Aya + Yu, 2026-06-17): physical constraints are
*symmetry only* for v0.1 -- W = W^T must be enforced every step. Full
physical realism (clipping, asymmetry of MTJ switching, etc.) is deferred to
v0.2. Symmetry-only keeps the Hopfield energy function well-defined and lets
us cleanly compare against Ramsauer 2020 (decision #4).

M1: implement HopfieldNet with three coupled layers, the W = W^T projection
applied at every update, and storage / retrieval methods used by
ayaram.core and the demos.
"""
