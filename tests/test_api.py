"""API surface tests: diagrams, exclusions, c-dependence, composition."""

from __future__ import annotations

import pytest

from partialid import Assumptions, CausalDiagram, Query, Variables, bound

RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


def test_compose_and_describe():
    a = Assumptions.order("X", "Y") & Assumptions.mtr("Y", "X") & Assumptions.tan_msm("X", 2.0)
    assert len(a) == 3
    assert a.diagram() is not None
    assert len(a.describe()) == 3


def test_explicit_diagram_matches_order():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a_order = Assumptions.order("X", "Y")
    g = CausalDiagram.from_order(["X", "Y"])
    a_diag = Assumptions.from_diagram(g)
    r1 = bound(V, RE1_P, Q, a_order)
    r2 = bound(V, RE1_P, Q, a_diag)
    assert abs(r1.lower - r2.lower) < 1e-8
    assert abs(r1.upper - r2.upper) < 1e-8


def test_exclusion_shrinks_type_space():
    V = Variables.from_binary("X", "Z", "Y")
    # Uniform-ish table on 8 cells
    cells = V.cells(["X", "Z", "Y"])
    P = {c: 1 / 8 for c in cells}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a0 = Assumptions.order("X", "Z", "Y")
    a1 = a0 & Assumptions.exclusion("Y", "X")  # no direct X→Y
    r0 = bound(V, P, Q, a0)
    r1 = bound(V, P, Q, a1)
    # Exclusion alone need not shrink width for this query (THEORY remark),
    # but must remain feasible and valid.
    assert r0.success and r1.success
    assert r1.width <= r0.width + 1e-8


def test_c_dependence_interpolates():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    r0 = bound(V, RE1_P, Q, a & Assumptions.c_dependence("X", 0.0), method="lp")
    r1 = bound(V, RE1_P, Q, a & Assumptions.c_dependence("X", 1.0), method="lp")
    rm = bound(V, RE1_P, Q, a)
    assert abs(r0.width) < 1e-6  # e(r)=e0 ⇒ no confounding
    assert abs(r1.width - rm.width) < 1e-5  # c=1 covers [0,1]


def test_unanchored_rosenbaum_rejected():
    with pytest.raises(NotImplementedError):
        Assumptions.rosenbaum_gamma("X", 2.0, anchored=False)


def test_three_var_chain_manski():
    V = Variables.from_binary("X", "Z", "Y")
    # RE2-style: marginalize to same (X,Y) as RE1 by putting mass independent of Z pattern
    # Use P that matches RE1 on (X,Y) with Z split evenly
    P = {}
    base = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    for (x, y), p in base.items():
        P[(x, 0, y)] = p / 2
        P[(x, 1, y)] = p / 2
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r = bound(V, P, Q, Assumptions.order("X", "Z", "Y"))
    assert abs(r.width - 0.50) < 1e-6
    assert abs(r.lower - 0.35) < 1e-6
    assert abs(r.upper - 0.85) < 1e-6
