"""Unified query AST, CUT unnesting, and conditional ratios."""

from __future__ import annotations

import numpy as np

from partialid import (
    PO,
    Assumptions,
    Atom,
    CausalDiagram,
    LinComb,
    Lit,
    Query,
    Variables,
    bound,
)
from partialid.query_compile import (
    compile_objective,
    unnest_expr,
)
from partialid.types_canon import TypeSpace

RE1 = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


def test_sugar_matches_legacy_manski():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r = bound(V, RE1, Q, Assumptions.order("X", "Y"), method="auto")
    assert r.success
    assert abs(r.lower - 0.35) < 1e-8
    assert abs(r.upper - 0.85) < 1e-8


def test_ate_and_pns_via_expr():
    V = Variables.from_binary("X", "Y")
    a = Assumptions.order("X", "Y")
    r_ate = bound(V, RE1, Query.ate(), a, method="lp")
    assert r_ate.success
    r_pns = bound(V, RE1, Query.pns(), a, method="lp")
    assert r_pns.success
    assert r_pns.lower >= -1e-9


def test_pn_ps_fixed_denom():
    V = Variables.from_binary("X", "Y")
    a = Assumptions.order("X", "Y")
    r_pn = bound(V, RE1, Query.pn(), a, method="lp")
    r_ps = bound(V, RE1, Query.ps(), a, method="lp")
    assert r_pn.success and r_ps.success
    assert any("fixed denom" in n for n in r_pn.notes)


def test_cut_unnest_nde_style():
    """CUT: P(Y_{x', M_x}=y) = ∑_m P(Y_{x',m}=y, M_x=m) (NeurIPS 2021 Thm 4)."""
    V = Variables.from_binary("X", "M", "Y")
    g = CausalDiagram.from_edges(
        ["X", "M", "Y"],
        directed=[("X", "M"), ("X", "Y"), ("M", "Y")],
        bidirected=[("X", "Y")],
    )
    nested = PO("Y", {"X": 1, "M": PO("M", {"X": 0})})
    expr = LinComb.atom(Atom(Lit(nested, 1)))
    notes: list[str] = []
    out = unnest_expr(expr, V, g, notes)
    assert isinstance(out, LinComb)
    assert len(out.terms) == 2  # sum over M ∈ {0,1}
    assert any("CUT" in n for n in notes)
    # Each term is a joint unnested atom
    for _, atom in out.terms:
        assert all(
            (isinstance(lit.expr, str) or (isinstance(lit.expr, PO) and lit.expr.is_flat()))
            for lit in atom.lits
        )


def test_nested_query_bounds_run():
    V = Variables.from_binary("X", "M", "Y")
    # Uniform-ish P
    P = {}
    for x in (0, 1):
        for m in (0, 1):
            for y in (0, 1):
                P[(x, m, y)] = 0.125
    g = CausalDiagram.from_edges(
        ["X", "M", "Y"],
        directed=[("X", "M"), ("X", "Y"), ("M", "Y")],
        bidirected=[("X", "Y")],
    )
    nested = PO("Y", {"X": 1, "M": PO("M", {"X": 0})})
    Q = Query.from_expr(LinComb.atom(Atom(Lit(nested, 1))), outcome="Y", value=1)
    r = bound(V, P, Q, Assumptions.from_diagram(g), method="lp")
    assert r.success
    assert 0.0 - 1e-8 <= r.lower <= r.upper <= 1.0 + 1e-8
    assert any("CUT" in n for n in r.notes)


def test_compile_interventional_matches_typespace():
    V = Variables.from_binary("X", "Y")
    ts = TypeSpace.from_diagram(V, CausalDiagram.from_order(["X", "Y"]))
    allow = np.ones(ts.n_types, dtype=bool)
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    obj = compile_objective(ts, Q.expr, allow)
    assert obj.c.shape == (ts.n_types,)
    for t in range(ts.n_types):
        expected = ts.query_coefficient(t, "Y", 1, {"X": 1})
        assert abs(obj.c[t] - expected) < 1e-12


def test_conditional_factory_fractional_or_fixed():
    V = Variables.from_binary("X", "Y")
    # CF-conditioned: P(Y_1=1 | Y_0=0) — denom not observational → fractional
    numer = LinComb.atom(Atom(Lit(PO("Y", {"X": 1}), 1)))
    denom = LinComb.atom(Atom(Lit(PO("Y", {"X": 0}), 0)))
    Q = Query.conditional(numer, denom)
    r = bound(V, RE1, Q, Assumptions.order("X", "Y"), method="auto")
    assert r.success
    assert 0.0 - 1e-8 <= r.lower <= r.upper <= 1.0 + 1e-8
