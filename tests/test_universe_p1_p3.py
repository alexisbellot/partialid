"""P1–P3 correctness: PN/PS, conditional do, MTS, paths, audit, CI."""

from __future__ import annotations

import numpy as np
import pytest

from partialid import (
    Assumptions,
    Query,
    Variables,
    audit_assumptions,
    audit_bound,
    bound,
    bound_ci,
    bound_path,
)

RE1 = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


@pytest.fixture
def re1():
    V = Variables.from_binary("X", "Y")
    return V, RE1, Assumptions.order("X", "Y")


def test_pns_pn_ps_lp_consistent(re1):
    V, P, a = re1
    r_pns = bound(V, P, Query.pns(), a, method="lp")
    r_pn = bound(V, P, Query.pn(), a, method="lp")
    r_ps = bound(V, P, Query.ps(), a, method="lp")
    assert r_pns.success and r_pn.success and r_ps.success
    # Order-only: PNS ∈ [0, P00+P11]
    assert abs(r_pns.lower - 0.0) < 1e-8
    assert abs(r_pns.upper - 0.65) < 1e-8
    # PN/PS are probabilities given a positive cell
    assert 0.0 - 1e-9 <= r_pn.lower <= r_pn.upper <= 1.0 + 1e-9
    assert 0.0 - 1e-9 <= r_ps.lower <= r_ps.upper <= 1.0 + 1e-9


def test_pn_ps_match_fiber_when_eq_only(re1):
    V, P, a = re1
    for q in (Query.pn(), Query.ps(), Query.pns()):
        rf = bound(V, P, q, a, method="fiber")
        rl = bound(V, P, q, a, method="lp")
        assert abs(rf.lower - rl.lower) + abs(rf.upper - rl.upper) < 1e-8


def test_mts_tightens_manski(re1):
    V, P, a = re1
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r0 = bound(V, P, Q, a, method="lp")
    r1 = bound(V, P, Q, a & Assumptions.mts("X", "Y"), method="lp")
    assert r0.success and r1.success
    assert r1.width <= r0.width + 1e-9
    assert any("MTS" in n for n in r1.notes)


def test_mts_mtr_compose(re1):
    V, P, a = re1
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a2 = a & Assumptions.mtr("Y", "X") & Assumptions.mts("X", "Y")
    r = bound(V, P, Q, a2, method="lp")
    assert r.success
    assert r.width <= 0.30 + 1e-6  # at most MTR Manski


def test_conditional_interventional_matches_slice():
    rng = np.random.default_rng(1)
    V = Variables.from_binary("W", "X", "Y")
    cells = list(V.cells())
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    a = Assumptions.order("W", "X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1, condition={"W": 1})
    r = bound(V, P, Q, a, method="lp")
    assert r.success
    assert any("pretreatment condition" in n for n in r.notes)

    # Manual slice
    sliced = {}
    mass = 0.0
    for (w, x, y), v in P.items():
        if w == 1:
            mass += v
            sliced[(x, y)] = sliced.get((x, y), 0.0) + v
    sliced = {k: v / mass for k, v in sliced.items()}
    V2 = Variables.from_binary("X", "Y")
    r2 = bound(
        V2,
        sliced,
        Query.interventional(outcome="Y", do={"X": 1}, value=1),
        Assumptions.order("X", "Y"),
        method="lp",
    )
    assert abs(r.lower - r2.lower) + abs(r.upper - r2.upper) < 1e-6


def test_independence_outer_note():
    V = Variables.from_binary("X", "Y")
    P = dict(RE1)
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    # independence always forces merged-latent outer (sharp split not in v1)
    a = Assumptions.order("X", "Y") & Assumptions.independence(("X",), ("Y",))
    r = bound(V, P, Q, a, method="lp")
    assert r.success
    assert not r.tight
    assert "bounds_may_be_outer" in r.notes
    assert any("independence" in n.lower() for n in r.notes)


def test_bound_path_c_and_kappa_breakdown(re1):
    V, P, a = re1
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    path_c = bound_path(V, P, Q, a, param="c", theta_min=0.0, theta_max=0.45, n_grid=11)
    assert len(path_c.values) >= 2
    assert path_c.width[0] <= path_c.width[-1] + 1e-8  # more confounding → wider
    # breakdown should not raise for c
    _ = path_c.breakdown(null=float(path_c.lower[-1]), which="lower")

    path_k = bound_path(V, P, Q, a, param="kappa", theta_min=0.0, theta_max=0.4, n_grid=9)
    assert len(path_k.values) >= 2
    _ = path_k.breakdown(null=0.55, which="interval")


def test_iv_msm_path_monotone():
    from partialid.problems import binary_iv

    p = binary_iv(seed=0)
    path = bound_path(
        p.variables,
        p.data,
        p.query,
        p.assumptions,
        param="Lambda",
        theta_min=1.5,
        theta_max=8.0,
        n_grid=5,
        prefer_closed=False,
    )
    assert len(path.values) >= 2
    assert "repeated_bound" in (path.formula_id or "")
    ok = np.isfinite(path.width)
    assert ok.any()
    w = path.width[ok]
    assert np.all(np.diff(w) >= -1e-4)


def test_audit_report(re1):
    V, P, a = re1
    a2 = a & Assumptions.mtr("Y", "X") & Assumptions.tan_msm("X", 2.0)
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r = bound(V, P, Q, a2, method="lp")
    rep = audit_bound(r, query=Q, assumptions=a2)
    assert "MonotonicityAssumption" in rep.faces
    assert "TanMSMAssumption" in rep.budgets
    assert rep.method is not None
    assert rep.success
    base = audit_assumptions(a2)
    assert base.faces and base.budgets


def test_bound_ci_covers_point(re1):
    V, P, a = re1
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    ci = bound_ci(V, P, Q, a, n_boot=40, n=800, seed=0, level=0.9)
    assert ci.success
    assert ci.lower <= ci.point_lower + 1e-9
    assert ci.upper >= ci.point_upper - 1e-9
    assert ci.width >= (ci.point_upper - ci.point_lower) - 1e-9
