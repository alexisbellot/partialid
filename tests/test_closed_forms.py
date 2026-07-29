"""Closed-form Move layer."""

from __future__ import annotations

from partialid import Assumptions, Query, Variables, bound
from partialid.closed_forms import msm_closed_form_re1, try_closed_form
from partialid.problems import binary_iv

RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


def test_closed_manski_auto():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r = bound(V, RE1_P, Q, Assumptions.order("X", "Y"))
    assert r.method.startswith("closed/")
    assert abs(r.lower - 0.35) < 1e-8
    assert abs(r.upper - 0.85) < 1e-8


def test_closed_msm_re1():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y") & Assumptions.tan_msm("X", 3.0)
    r = bound(V, RE1_P, Q, a, method="closed")
    ell, u, w = msm_closed_form_re1(3.0)
    assert r.success
    assert abs(r.lower - ell) < 1e-8
    assert abs(r.upper - u) < 1e-8
    assert abs(r.width - w) < 1e-8


def test_closed_unavailable():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    # MTR has no dedicated closed export yet
    r = bound(V, RE1_P, Q, Assumptions.order("X", "Y") & Assumptions.mtr("Y", "X"), method="closed")
    assert not r.success


def test_try_closed_iv():
    p = binary_iv(seed=2)
    cfr = try_closed_form(p.variables, p.data, p.query, p.assumptions)
    assert cfr is not None
    r = bound(p.variables, p.data, p.query, p.assumptions, method="closed")
    assert r.success and r.method.startswith("closed/")
    assert abs(r.lower - cfr.lower) < 1e-8


def test_ate_manski_re1():
    V = Variables.from_binary("X", "Y")
    Q = Query.ate(outcome="Y", treatment="X", y=1, t1=1, t0=0)
    r = bound(V, RE1_P, Q, Assumptions.order("X", "Y"), method="lp")
    # ATE = P(Y=1|do X=1) - P(Y=1|do X=0)
    # Manski: do1 in [0.35,0.85], do0 in [0.20,0.70]; contrast bounds need joint LP
    assert r.success
    assert r.lower >= -1 - 1e-8 and r.upper <= 1 + 1e-8
    # Truth under max confounding freest ATE width: known RE1 ATE bounds
    # lower = 0.35 - 0.70 = -0.35, upper = 0.85 - 0.20 = 0.65
    assert abs(r.lower - (-0.35)) < 1e-6
    assert abs(r.upper - 0.65) < 1e-6
