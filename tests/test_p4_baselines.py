"""P4: hyperarc method + causaloptim-style formula oracles."""

from __future__ import annotations

import pytest

from partialid import Assumptions, Query, Variables, bound
from partialid.causaloptim_formulas import (
    RE1_P,
    balke_pearl_iv,
    manski_binary_xy,
    manski_exposure,
)
from partialid.problems import (
    sachs_binary_iv,
    sachs_ternary_exposure,
    shridharan_multi_treatment,
)


def test_hyperarc_matches_manski_re1():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    rf = bound(V, RE1_P, Q, a)
    rh = bound(V, RE1_P, Q, a, method="hyperarc")
    assert rh.success
    assert abs(rf.lower - rh.lower) < 1e-9
    assert abs(rf.upper - rh.upper) < 1e-9
    assert any("closed form" in n for n in rh.notes)


def test_hyperarc_multi_treatment_closed_form():
    p = shridharan_multi_treatment(2, 2)
    rf = bound(p.variables, p.data, p.query, p.assumptions)
    rh = bound(p.variables, p.data, p.query, p.assumptions, method="hyperarc")
    assert rh.success
    assert abs(rf.lower - rh.lower) < 1e-9
    assert abs(rf.upper - rh.upper) < 1e-9
    assert rh.notes and "closed form" in " ".join(rh.notes)


def test_hyperarc_iv_matches_instrument_lp():
    iv = sachs_binary_iv(seed=2)
    ri = bound(iv.variables, iv.data, iv.query, iv.assumptions)
    rh = bound(iv.variables, iv.data, iv.query, iv.assumptions, method="hyperarc")
    assert rh.success, rh.message
    assert abs(ri.lower - rh.lower) < 1e-8
    assert abs(ri.upper - rh.upper) < 1e-8
    # Pruning: fewer hyperarcs than raw response types
    assert any("raw_types≈16" in n for n in rh.notes)
    assert any("|H|=12" in n for n in rh.notes)


def test_manski_formula_oracle():
    lo, hi = manski_binary_xy(RE1_P)
    assert abs(lo - 0.35) < 1e-12
    assert abs(hi - 0.85) < 1e-12


def test_manski_ternary_matches_solver():
    p = sachs_ternary_exposure(seed=0)
    r = bound(p.variables, p.data, p.query, p.assumptions)
    lo, hi = manski_exposure(p.data, do_x=1, y=1)
    assert abs(r.lower - lo) < 1e-9
    assert abs(r.upper - hi) < 1e-9


@pytest.mark.parametrize("seed", [0, 2, 5, 9])
def test_balke_pearl_formula_matches_instrument(seed):
    iv = sachs_binary_iv(seed=seed)
    r = bound(iv.variables, iv.data, iv.query, iv.assumptions)
    lo, hi = balke_pearl_iv(iv.data)
    assert abs(r.lower - lo) < 1e-8
    assert abs(r.upper - hi) < 1e-8
