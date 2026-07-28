"""Track A1–A2: dual LP policy + auto routing."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from partialid import Assumptions, Query, Variables, bound
from partialid.backends import prefer_colgen_auto, should_use_dual
from partialid.fiber_solve import is_fiber_separable
from partialid.formulate import LPData, formulate
from partialid.problems import (
    sachs_binary_iv,
    sachs_ternary_cardinalities,
    sachs_two_instruments_card,
)


def test_should_use_dual_two_iv():
    p = sachs_two_instruments_card(card_z=3, seed=15)
    lp = formulate(p.variables, p.data, p.query, p.assumptions)
    assert lp.A_ub is None
    assert should_use_dual(lp)
    assert not prefer_colgen_auto(lp)  # dual preferred over colgen


def test_should_use_dual_large_xy_eq_only():
    """Full-type (non-instrument) eq-only LP also qualifies — not IV-specific."""
    p = sachs_ternary_cardinalities(5, 2, seed=1)
    lp = formulate(p.variables, p.data, p.query, p.assumptions)
    assert is_fiber_separable(lp)
    assert lp.A_eq.shape[1] > 64
    assert should_use_dual(lp)


def test_auto_routes_dual_on_two_iv():
    p = sachs_two_instruments_card(card_z=3, seed=15)
    r = bound(p.variables, p.data, p.query, p.assumptions, method="auto")
    assert r.success
    assert r.method == "lp/dual"
    assert "dual_lp" in r.notes
    assert "auto_routed_dual" in r.notes
    r_lp = bound(p.variables, p.data, p.query, p.assumptions, method="lp")
    assert abs(r.lower - r_lp.lower) < 1e-8
    assert abs(r.upper - r_lp.upper) < 1e-8


def test_auto_prefers_fiber_or_closed_on_manski():
    V = Variables.from_binary("X", "Y")
    P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r = bound(V, P, Q, Assumptions.order("X", "Y"), method="auto")
    assert r.success
    assert r.method.startswith("closed") or r.method == "fiber"


def test_method_dual_force_small_iv():
    """force_dual works below the size gate (binary IV ~16 types)."""
    p = sachs_binary_iv(seed=2)
    lp = formulate(p.variables, p.data, p.query, p.assumptions)
    assert not should_use_dual(lp)  # below gate
    assert should_use_dual(lp, force=True)
    r = bound(p.variables, p.data, p.query, p.assumptions, method="dual")
    assert r.success
    assert "force_dual" in r.notes
    assert r.method == "lp/dual"
    r_lp = bound(p.variables, p.data, p.query, p.assumptions, method="lp")
    assert abs(r.lower - r_lp.lower) < 1e-8
    assert abs(r.upper - r_lp.upper) < 1e-8


def test_prefer_colgen_gate():
    """Synthetic: many vars but not enough vs 4*m → colgen gate."""

    class _TS:
        n_types = 5000

    A = sparse.eye(2000, 5000, format="csr")
    lp = LPData(
        c=np.zeros(5000),
        A_eq=A,
        b_eq=np.ones(2000) / 2000,
        A_ub=None,
        b_ub=None,
        type_space=_TS(),  # type: ignore[arg-type]
        allowed_types=np.ones(5000, dtype=bool),
        cells=[()],
        P_vec=np.ones(2000) / 2000,
    )
    assert not should_use_dual(lp)
    assert prefer_colgen_auto(lp)
