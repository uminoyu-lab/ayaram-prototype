"""Modes: aya-awake / aya-sleep switching.

Design decision #1 (Aya + Yu, 2026-06-17): whole-cell synchronous switching.
Every cell in every layer flips between awake (high barrier, stable) and sleep
(low barrier, fluctuating) at the same time. Per-layer barrier differences are
fixed parameters that shape representation; they are not the mechanism by
which the cycle progresses.

Design decision #4 (Aya + Yu, 2026-06-17): the sleep-mode noise strength sigma
is meant to correspond to the inverse-temperature beta of softmax attention
under Ramsauer 2020 Theorem 3. M1 must surface both as first-class knobs so
that demos/attention_test.py can sweep the sigma <-> beta map.

M1: implement Awake / Sleep mode objects with (temperature T, noise sigma)
parameters and a switch() method consumed by ayaram.core.
"""
