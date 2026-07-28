"""Track A3: budget-aware colgen pricing (A_ub'μ) matches dense LP on MSM."""

from __future__ import annotations

import numpy as np
import pytest

from partialid import Assumptions, Query, Variables, bound
from partialid.closed_forms import msm_closed_form_re1

RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


@pytest.fixture
def re1():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    return V, RE1_P, Q, Assumptions.order("X", "Y")


@pytest.mark.parametrize("Lam", [1.0, 1.5, 7 / 3, 3.0, 7.0, 20.0])
def test_colgen_msm_re1_matches_analytic(re1, Lam):
    V, P, Q, base = re1
    a = base & Assumptions.tan_msm("X", Lam)
    rc = bound(V, P, Q, a, method="colgen")
    assert rc.success, rc.message
    assert "budget_aware_pricing" in rc.notes
    ell, u, _ = msm_closed_form_re1(Lam)
    assert abs(rc.lower - ell) < 1e-6
    assert abs(rc.upper - u) < 1e-6


@pytest.mark.parametrize("k", [3, 5, 8])
def test_colgen_msm_card_y_matches_lp(k):
    rng = np.random.default_rng(k)
    V = Variables.from_mapping({"X": [0, 1], "Y": list(range(k))})
    cells = [(x, y) for x in (0, 1) for y in range(k)]
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=0)
    a = Assumptions.order("X", "Y") & Assumptions.tan_msm("X", 3.0)
    rl = bound(V, P, Q, a, method="lp")
    rc = bound(V, P, Q, a, method="colgen")
    assert rc.success, rc.message
    assert "budget_aware_pricing" in rc.notes
    assert abs(rl.lower - rc.lower) < 1e-5
    assert abs(rl.upper - rc.upper) < 1e-5


def test_colgen_cdep_matches_lp(re1):
    V, P, Q, base = re1
    a = base & Assumptions.c_dependence("X", 0.15)
    rl = bound(V, P, Q, a, method="lp")
    rc = bound(V, P, Q, a, method="colgen")
    assert rc.success, rc.message
    assert abs(rl.lower - rc.lower) < 1e-5
    assert abs(rl.upper - rc.upper) < 1e-5


def test_colgen_msm_grid_matches_lp(re1):
    V, P, Q, base = re1
    for Lam in np.linspace(1.0, 10.0, 9):
        a = base & Assumptions.tan_msm("X", float(Lam))
        rl = bound(V, P, Q, a, method="lp")
        rc = bound(V, P, Q, a, method="colgen")
        assert abs(rl.lower - rc.lower) + abs(rl.upper - rc.upper) < 1e-5
