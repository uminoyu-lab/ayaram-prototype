"""Core: 4-phase cycle on the 3-layer MTJ array.

Design decision #1 (Aya + Yu, 2026-06-17): option C, whole-cell synchronous
switching. The three layers are an MLP-style hierarchy; the barrier difference
exists to enrich representation, not to provide per-layer time-scale
separation. v0.2 may move to time-scale separation (option A) without breaking
this interface.

Cell shape: 32 x 32 (decision #3). Layer widths: 1024 / 256 / 64 (decision #3).

The cycle is:
    Phase 1 (awake, terrain setup) -- map the problem onto layer-0 bias.
    Phase 2 (sleep, fluctuation)   -- lower barrier, apply Langevin / Gaussian
                                      noise so magnetization wanders the
                                      terrain stochastically.
    Phase 3 (re-awake, fixation)   -- raise barrier, freeze the state.
    Phase 4 (readout)              -- read the output layer.

M1: implement Phase 1-4 against ayaram.memory.HopfieldNet and
ayaram.modes.{Awake, Sleep}.
"""
