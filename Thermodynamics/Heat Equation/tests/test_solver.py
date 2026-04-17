import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from heat_equation.solver import solve, stability_timestep, initial_condition


def test_stability_criterion():
    alpha = 1e-2
    dx = 1.0 / 100
    dt = stability_timestep(dx, alpha)
    r = alpha * dt / dx**2
    assert r <= 0.5


def test_boundary_conditions():
    u, x, t = solve(u_left=10.0, u_right=0.0)
    assert np.all(u[:, 0] == 10.0)
    assert np.all(u[:, -1] == 0.0)


def test_output_shape():
    N = 50
    u, x, t = solve(N=N, tf=1.0)
    assert u.shape[1] == N + 1
    assert u.shape[0] == len(t)


def test_steady_state_convergence():
    L = 1.0
    u_left, u_right = 10.0, 0.0
    u, x, t = solve(L=L, tf=20.0, u_left=u_left, u_right=u_right)
    steady = u_left + (u_right - u_left) * x / L
    np.testing.assert_allclose(u[-1], steady, atol=0.1)


def test_mean_temperature_decreases():
    u, x, t = solve(u_left=0.0, u_right=0.0)
    means = u.mean(axis=1)
    assert means[-1] < means[0]


def test_custom_parameters():
    u, x, t = solve(L=2.0, N=50, alpha=5e-3, tf=5.0, u_left=20.0, u_right=5.0)
    assert u.shape[1] == 51
    assert np.all(np.isfinite(u))


def test_initial_condition_boundary():
    u0 = initial_condition(N=100, L=1.0, u_left=10.0, u_right=0.0)
    assert u0[0] == 10.0
    assert u0[-1] == 0.0