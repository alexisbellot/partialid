"""P3 tests: support_bound (Cor 7.7), margin soft independence, f-sensitivity."""

from __future__ import annotations

import pytest

from partialid import Assumptions, Query, Variables, bound
from partialid.convex_solve import cvxpy_available

RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


@pytest.fixture
def re1():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    return V, RE1_P, Q, a


def test_support_bound_cor77_refutes(re1):
    """Cor 7.7: k < |X| = 4 refutes fully-supported RE1."""
    V, P, Q, a = re1
    r = bound(V, P, Q, a & Assumptions.support_bound(3))
    assert not r.success


def test_support_bound_cor77_nonbinding(re1):
    """Cor 7.7: k ≥ 4 reproduces Manski."""
    V, P, Q, a = re1
    r0 = bound(V, P, Q, a)
    r4 = bound(V, P, Q, a & Assumptions.support_bound(4))
    assert r4.success, r4.message
    assert abs(r4.lower - r0.lower) < 1e-6
    assert abs(r4.upper - r0.upper) < 1e-6
    assert r4.method == "mip"


def test_margin_budget_front_door_anchor(re1):
    """Soft independence cookbook: |Q - Q_FD| ≤ κ (THEORY §8.8.3 numbers)."""
    V, P, Q, a = re1
    q_fd = 139 / 240  # ≈ 0.5792
    # κ=0 pins to front-door point
    r0 = bound(V, P, Q, a & Assumptions.margin_budget(0.0, s0=q_fd), method="lp")
    assert r0.success
    assert abs(r0.width) < 1e-6
    assert abs(r0.lower - q_fd) < 1e-6

    # small κ → width 2κ until hitting S=0 wall at κ=11/48
    kappa = 0.1
    r = bound(V, P, Q, a & Assumptions.margin_budget(kappa, s0=q_fd), method="lp")
    assert abs(r.width - 2 * kappa) < 1e-6

    # large κ → Manski
    r_big = bound(V, P, Q, a & Assumptions.margin_budget(1.0, s0=q_fd), method="lp")
    assert abs(r_big.lower - 0.35) < 1e-6
    assert abs(r_big.upper - 0.85) < 1e-6


@pytest.mark.skipif(not cvxpy_available(), reason="cvxpy not installed")
def test_f_sensitivity_rho0_point_id(re1):
    V, P, Q, a = re1
    r = bound(V, P, Q, a & Assumptions.f_sensitivity("X", 0.0, f="kl"))
    assert r.success, r.message
    assert abs(r.width) < 1e-3
    assert abs(r.lower - 0.70) < 1e-2


@pytest.mark.skipif(not cvxpy_available(), reason="cvxpy not installed")
def test_f_sensitivity_large_rho_manski(re1):
    V, P, Q, a = re1
    r = bound(V, P, Q, a & Assumptions.f_sensitivity("X", 10.0, f="tv"))
    assert r.success, r.message
    assert abs(r.lower - 0.35) < 1e-2
    assert abs(r.upper - 0.85) < 1e-2


@pytest.mark.skipif(not cvxpy_available(), reason="cvxpy not installed")
def test_f_sensitivity_monotone_in_rho(re1):
    V, P, Q, a = re1
    widths = []
    for rho in (0.0, 0.05, 0.2, 1.0):
        r = bound(V, P, Q, a & Assumptions.f_sensitivity("X", rho, f="tv"))
        assert r.success, r.message
        widths.append(r.width)
    assert widths[0] <= widths[1] + 1e-4
    assert widths[1] <= widths[2] + 1e-4
    assert widths[2] <= widths[3] + 1e-4
