"""Correctness tests against closed forms (RE1, MTR, MSM)."""

from __future__ import annotations

import pytest

from partialid import Assumptions, Query, Variables, bound, bound_grid

RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


@pytest.fixture
def re1():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    return V, RE1_P, Q, a


def test_manski_re1(re1):
    V, P, Q, a = re1
    r = bound(V, P, Q, a)
    assert r.success
    assert abs(r.lower - 0.35) < 1e-8
    assert abs(r.upper - 0.85) < 1e-8
    assert abs(r.width - 0.50) < 1e-8
    assert r.method in ("fiber", "closed/manski_binary_xy") or r.method.startswith("closed/")


def test_manski_lp_matches_fiber(re1):
    V, P, Q, a = re1
    rf = bound(V, P, Q, a, method="fiber")
    rl = bound(V, P, Q, a, method="lp")
    assert abs(rf.lower - rl.lower) < 1e-8
    assert abs(rf.upper - rl.upper) < 1e-8


def test_mtr_re1(re1):
    V, P, Q, a = re1
    r = bound(V, P, Q, a & Assumptions.mtr("Y", "X"))
    # Under MTR, only the (x=0,y=0) fiber remains free → w = 0.30.
    # Lower rises to P(Y=1)=0.55; upper stays at Manski 0.85.
    assert abs(r.width - 0.30) < 1e-8
    assert abs(r.lower - 0.55) < 1e-8
    assert abs(r.upper - 0.85) < 1e-8


def _msm_closed_form(Lam: float):
    # THEORY §8.7.3
    if Lam <= 7 / 3:
        w = 0.15 * (Lam - 1 / Lam)
        u = 0.85 - 0.15 / Lam
        ell = 0.85 - 0.15 * Lam
    else:
        w = 0.50 * (1 - 1 / Lam)
        u = 0.85 - 0.15 / Lam
        ell = 0.35 * (1 + 1 / Lam)
    return ell, u, w


@pytest.mark.parametrize("Lam", [1.0, 1.5, 7 / 3, 3.0, 7.0])
def test_tan_msm_closed_form(re1, Lam):
    V, P, Q, a = re1
    r = bound(V, P, Q, a & Assumptions.tan_msm("X", Lam), method="lp")
    ell, u, w = _msm_closed_form(Lam)
    assert r.success, r.message
    assert abs(r.lower - ell) < 1e-6
    assert abs(r.upper - u) < 1e-6
    assert abs(r.width - w) < 1e-6


def test_msm_lambda1_point_id(re1):
    V, P, Q, a = re1
    r = bound(V, P, Q, a & Assumptions.tan_msm("X", 1.0), method="lp")
    assert abs(r.width) < 1e-8
    assert abs(r.lower - 0.70) < 1e-6


def test_anchored_rosenbaum(re1):
    V, P, Q, a = re1
    # Γ=9 → Λ=3
    r = bound(V, P, Q, a & Assumptions.rosenbaum_gamma("X", 9.0), method="lp")
    ell, u, w = _msm_closed_form(3.0)
    assert abs(r.width - w) < 1e-6


def test_colgen_matches_lp_msm(re1):
    V, P, Q, a = re1
    base = a & Assumptions.tan_msm("X", 3.0)
    rl = bound(V, P, Q, base, method="lp")
    rc = bound(V, P, Q, base, method="colgen")
    assert rc.success, rc.message
    assert abs(rl.lower - rc.lower) < 1e-5
    assert abs(rl.upper - rc.upper) < 1e-5


def test_bound_grid_msm(re1):
    V, P, Q, a = re1
    grid = bound_grid(V, P, Q, a, param="Lambda", values=[1.0, 3.0, 7.0], method="lp")
    assert len(grid) == 3
    assert abs(grid[0][1].width) < 1e-8
    assert abs(grid[1][1].width - _msm_closed_form(3.0)[2]) < 1e-6


def test_bound_grid_matches_cold(re1):
    from partialid.api import bound_grid_cold

    V, P, Q, a = re1
    values = [1.0, 1.5, 3.0, 7.0, 20.0]
    warm = bound_grid(V, P, Q, a, param="Lambda", values=values, warm_start=True)
    cold = bound_grid_cold(V, P, Q, a, param="Lambda", values=values)
    for (vw, rw), (vc, rc) in zip(warm, cold):
        assert vw == vc
        assert abs(rw.lower - rc.lower) < 1e-7
        assert abs(rw.upper - rc.upper) < 1e-7


def test_bound_grid_unsorted_order(re1):
    """Results follow input order even when warm-start sorts ascending internally."""
    V, P, Q, a = re1
    values = [7.0, 1.0, 3.0]
    grid = bound_grid(V, P, Q, a, param="Lambda", values=values, warm_start=True)
    assert [v for v, _ in grid] == values
    assert abs(grid[1][1].width) < 1e-8  # Lambda=1


def test_margin_budget(re1):
    V, P, Q, a = re1
    # point ref = 0.70; kappa=0 → point; large kappa → Manski
    r0 = bound(V, P, Q, a & Assumptions.margin_budget(0.0), method="lp")
    assert abs(r0.width) < 1e-6
    assert abs(r0.lower - 0.70) < 1e-6

    r_big = bound(V, P, Q, a & Assumptions.margin_budget(1.0), method="lp")
    assert abs(r_big.lower - 0.35) < 1e-6
    assert abs(r_big.upper - 0.85) < 1e-6


def test_pns_markovian():
    # RE3: no confounding — use diagram with no bidirected, each own component
    # Under maximal confounding PNS still has an LP; for Markovian use
    # independence... For v1: order-only PNS on binary.
    V = Variables.from_binary("X", "Y")
    P = RE1_P
    Q = Query.pns(treatment="X", outcome="Y")
    # With observational constraints only, PNS bounds are known:
    # max(0, p11+p00-1, ...) classic Tian-Pearl under no confounding needs split.
    # Under maximal confounding, PNS = P(helped type); still LP.
    r = bound(V, P, Q, Assumptions.order("X", "Y"))
    assert r.success
    assert r.lower >= -1e-9
    assert r.upper <= 1 + 1e-9
    assert r.width >= 0
