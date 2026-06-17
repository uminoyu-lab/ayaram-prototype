"""Tests for ayaram.modes."""

from __future__ import annotations

import math

import pytest

from ayaram import modes


def test_K_u_layers_monotone_ascending():
    assert modes.K_U_LAYERS[0] < modes.K_U_LAYERS[1] < modes.K_U_LAYERS[2]


def test_sigma_local_decreases_with_layer_at_fixed_T():
    T = 1.0
    sigmas = [modes.sigma_local(l, T) for l in range(3)]
    assert sigmas[0] > sigmas[1] > sigmas[2]


def test_sigma_local_scales_with_sqrt_T():
    T1, T2 = 1.0, 4.0
    s1 = modes.sigma_local(0, T1)
    s2 = modes.sigma_local(0, T2)
    assert math.isclose(s2 / s1, 2.0, rel_tol=1e-6)


def test_sigma_local_rejects_negative_T():
    with pytest.raises(ValueError):
        modes.sigma_local(0, -1e-9)


def test_sigma_local_rejects_bad_layer():
    with pytest.raises(IndexError):
        modes.sigma_local(3, 1.0)


def test_awake_quieter_than_sleep_at_every_layer():
    for l in range(3):
        assert modes.AWAKE.sigma(l) < modes.SLEEP.sigma(l)


def test_modes_constants():
    assert modes.AWAKE.name == "aya-awake"
    assert modes.SLEEP.name == "aya-sleep"
    assert modes.AWAKE.T_global < modes.SLEEP.T_global


def test_layer_noise_ratio_layer0_is_one():
    assert math.isclose(modes.layer_noise_ratio(0), 1.0, rel_tol=1e-9)


def test_layer_noise_ratio_decreases_with_layer():
    r = [modes.layer_noise_ratio(l) for l in range(3)]
    assert r[0] > r[1] > r[2] > 0


def test_layer_noise_ratio_matches_sigma_local_ratio():
    T = 0.7
    s0 = modes.sigma_local(0, T)
    for l in range(3):
        ratio = modes.sigma_local(l, T) / s0
        assert math.isclose(ratio, modes.layer_noise_ratio(l), rel_tol=1e-9)
