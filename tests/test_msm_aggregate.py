"""MSM aggregate solver: margin closed form + transport LP + breakdown."""

from __future__ import annotations

import numpy as np
import pytest

from partialid import Assumptions, Query, Variables, bound, bound_path
from partialid.closed_forms import msm_closed_form_re1
from partialid.msm_aggregate import (
    msm_binary_xy_bounds,
    msm_binary_xy_breakdown,
    msm_binary_xy_breakpoints,
)

RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


@pytest.fixture
def re1():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    return V, RE1_P, Q


@pytest.mark.parametrize("Lam", [1.0, 1.5, 7 / 3, 3.0, 7.0, 20.0])
def test_msm_agg_matches_re1_analytic(re1, Lam):
    V, P, Q = re1
    a = Assumptions.order("X", "Y") & Assumptions.tan_msm("X", Lam)
    r = bound(V, P, Q, a, method="msm_agg")
    assert r.success
    assert r.method.startswith("msm_agg/")
    ell, u, w = msm_closed_form_re1(Lam)
    assert abs(r.lower - ell) < 1e-10
    assert abs(r.upper - u) < 1e-10
    assert abs(r.width - w) < 1e-10


def test_msm_agg_matches_lp_random_binary():
    rng = np.random.default_rng(0)
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    for _ in range(15):
        p = rng.dirichlet(np.ones(4))
        P = {(0, 0): float(p[0]), (0, 1): float(p[1]), (1, 0): float(p[2]), (1, 1): float(p[3])}
        e0 = P[(1, 0)] + P[(1, 1)]
        if not (0.05 < e0 < 0.95):
            continue
        for Lam in (1.0, 2.0, 5.0):
            a = Assumptions.order("X", "Y") & Assumptions.tan_msm("X", Lam)
            ra = bound(V, P, Q, a, method="msm_agg")
            rl = bound(V, P, Q, a, method="lp")
            assert ra.success and rl.success
            assert abs(ra.lower - rl.lower) + abs(ra.upper - rl.upper) < 1e-6


def test_msm_agg_card_y_matches_lp():
    rng = np.random.default_rng(2)
    k = 4
    V = Variables.from_mapping({"X": [0, 1], "Y": list(range(k))})
    cells = [(x, y) for x in (0, 1) for y in range(k)]
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=0)
    a = Assumptions.order("X", "Y") & Assumptions.tan_msm("X", 3.0)
    ra = bound(V, P, Q, a, method="msm_agg")
    rl = bound(V, P, Q, a, method="lp")
    assert ra.success and rl.success
    assert "transport" in ra.method or "msm_agg" in ra.method
    assert abs(ra.lower - rl.lower) + abs(ra.upper - rl.upper) < 1e-5


def test_msm_agg_cdep_matches_lp(re1):
    V, P, Q = re1
    a = Assumptions.order("X", "Y") & Assumptions.c_dependence("X", 0.15)
    ra = bound(V, P, Q, a, method="msm_agg")
    rl = bound(V, P, Q, a, method="lp")
    assert ra.success and rl.success
    assert abs(ra.lower - rl.lower) + abs(ra.upper - rl.upper) < 1e-5


def test_auto_routes_msm_agg_non_re1():
    """Non-RE1 table: auto should use msm_agg (closed RE1 pattern misses)."""
    P = {(0, 0): 0.1, (0, 1): 0.4, (1, 0): 0.2, (1, 1): 0.3}
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y") & Assumptions.tan_msm("X", 2.5)
    r = bound(V, P, Q, a, method="auto")
    assert r.success
    assert r.method.startswith("msm_agg/")


def test_breakpoints_re1_include_seven_thirds():
    bps = msm_binary_xy_breakpoints(RE1_P, lam_max=10.0)
    assert any(abs(b - 7 / 3) < 1e-8 for b in bps)


def test_breakdown_re1_vs_untreated():
    # THEORY §8.7.3: ℓ(Λ) hits P(Y=1|X=0)=0.40 at Λ=7
    star = msm_binary_xy_breakdown(RE1_P, null=0.40, which="lower")
    assert star is not None
    assert abs(star - 7.0) < 1e-4
    lo, _, _ = msm_binary_xy_bounds(RE1_P, star)
    assert abs(lo - 0.40) < 1e-4


def test_bound_path_breakdown_api(re1):
    V, P, Q = re1
    a = Assumptions.order("X", "Y")
    path = bound_path(V, P, Q, a, param="Lambda", theta_min=1.0, theta_max=20.0)
    assert path.closed_form
    star = path.breakdown(null=0.40, which="lower")
    assert star is not None
    assert abs(star - 7.0) < 1e-3


def test_bound_path_general_p_not_just_re1():
    P = {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.1, (1, 1): 0.4}
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    path = bound_path(
        V, P, Q, Assumptions.order("X", "Y"), param="Lambda", theta_min=1, theta_max=10
    )
    assert path.closed_form
    ell, u, _ = path.at(2.0)
    e2, u2, _ = msm_binary_xy_bounds(P, 2.0)
    assert abs(ell - e2) < 1e-12
    assert abs(u - u2) < 1e-12


def test_msm_agg_mtr_matches_lp_binary():
    V = Variables.from_binary("X", "Y")
    P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y") & Assumptions.mtr("Y", "X") & Assumptions.tan_msm("X", 3.0)
    ra = bound(V, P, Q, a, method="msm_agg")
    rl = bound(V, P, Q, a, method="lp")
    assert ra.success and rl.success
    assert "mtr" in ra.method or "MTR" in " ".join(ra.notes)
    assert abs(ra.lower - rl.lower) + abs(ra.upper - rl.upper) < 1e-6
    # MTR+MSM should be tighter than MSM alone
    a0 = Assumptions.order("X", "Y") & Assumptions.tan_msm("X", 3.0)
    r0 = bound(V, P, Q, a0, method="msm_agg")
    assert ra.width <= r0.width + 1e-9


def test_msm_agg_mtr_card_y_matches_lp():
    rng = np.random.default_rng(3)
    k = 3
    V = Variables.from_mapping({"X": [0, 1], "Y": list(range(k))})
    cells = [(x, y) for x in (0, 1) for y in range(k)]
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y") & Assumptions.mtr("Y", "X") & Assumptions.tan_msm("X", 2.0)
    for _ in range(40):
        p = rng.dirichlet(np.ones(len(cells)))
        P = {c: float(v) for c, v in zip(cells, p)}
        ra = bound(V, P, Q, a, method="msm_agg")
        rl = bound(V, P, Q, a, method="lp")
        if ra.success and rl.success:
            assert abs(ra.lower - rl.lower) + abs(ra.upper - rl.upper) < 1e-5
            return
        assert ra.success == rl.success  # both infeasible or both ok
    raise AssertionError("no feasible MTR+MSM draw in 40 trials")


def test_msm_agg_inert_chain_matches_lp():
    rng = np.random.default_rng(4)
    V = Variables.from_binary("W", "X", "Y")
    cells = list(V.cells())
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("W", "X", "Y") & Assumptions.tan_msm("X", 2.0)
    ra = bound(V, P, Q, a, method="msm_agg")
    rl = bound(V, P, Q, a, method="lp")
    assert ra.success and rl.success
    assert abs(ra.lower - rl.lower) + abs(ra.upper - rl.upper) < 1e-6


def test_iv_msm_tightens_and_msm_agg_matches_lp():
    from partialid.problems import sachs_binary_iv

    p = sachs_binary_iv(seed=2)
    r0 = bound(p.variables, p.data, p.query, p.assumptions, method="lp")
    a = p.assumptions & Assumptions.tan_msm("X", 3.0)
    rl = bound(p.variables, p.data, p.query, a, method="lp")
    ra = bound(p.variables, p.data, p.query, a, method="msm_agg")
    assert rl.success and ra.success
    assert rl.width < r0.width - 1e-6
    assert abs(ra.lower - rl.lower) + abs(ra.upper - rl.upper) < 1e-6
    assert "instrument" in ra.method or "msm_instrument" in ra.method


def test_iv_mtr_msm_matches_lp():
    from partialid.problems import sachs_binary_iv

    p = sachs_binary_iv(seed=2)
    # Conditional-on-Z MSM + MTR can refute tight Λ; Λ=10 is feasible for seed=2.
    a = p.assumptions & Assumptions.mtr("Y", "X") & Assumptions.tan_msm("X", 10.0)
    rl = bound(p.variables, p.data, p.query, a, method="lp")
    ra = bound(p.variables, p.data, p.query, a, method="msm_agg")
    assert rl.success and ra.success, (rl.message, ra.message)
    assert abs(ra.lower - rl.lower) + abs(ra.upper - rl.upper) < 1e-6
