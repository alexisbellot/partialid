"""Coverage matrix for in-universe (structure × assumption × query) — P0."""

from __future__ import annotations

import pytest

from partialid import Assumptions, CausalDiagram, Query, Variables, bound
from partialid.problems import binary_iv

RE1 = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


def _xy():
    return Variables.from_binary("X", "Y"), dict(RE1)


@pytest.mark.parametrize(
    "name,assumptions,query_factory,expect_tight",
    [
        (
            "manski",
            lambda: Assumptions.order("X", "Y"),
            lambda: Query.interventional(outcome="Y", do={"X": 1}, value=1),
            True,
        ),
        (
            "mtr",
            lambda: Assumptions.order("X", "Y") & Assumptions.mtr("Y", "X"),
            lambda: Query.interventional(outcome="Y", do={"X": 1}, value=1),
            True,
        ),
        (
            "mts",
            lambda: Assumptions.order("X", "Y") & Assumptions.mts("X", "Y"),
            lambda: Query.interventional(outcome="Y", do={"X": 1}, value=1),
            True,
        ),
        (
            "msm",
            lambda: Assumptions.order("X", "Y") & Assumptions.tan_msm("X", 3.0),
            lambda: Query.interventional(outcome="Y", do={"X": 1}, value=1),
            True,
        ),
        (
            "mtr_msm",
            lambda: (
                Assumptions.order("X", "Y")
                & Assumptions.mtr("Y", "X")
                & Assumptions.tan_msm("X", 2.0)
            ),
            lambda: Query.interventional(outcome="Y", do={"X": 1}, value=1),
            True,
        ),
        ("pns", lambda: Assumptions.order("X", "Y"), lambda: Query.pns(), True),
        ("pn", lambda: Assumptions.order("X", "Y"), lambda: Query.pn(), True),
        ("ps", lambda: Assumptions.order("X", "Y"), lambda: Query.ps(), True),
        ("ate", lambda: Assumptions.order("X", "Y"), lambda: Query.ate(), True),
    ],
)
def test_coverage_order_only(name, assumptions, query_factory, expect_tight):
    V, P = _xy()
    r = bound(V, P, query_factory(), assumptions(), method="auto")
    assert r.success, (name, r.message)
    assert r.tight is expect_tight
    assert r.lower <= r.upper + 1e-9


def test_coverage_iv_and_iv_msm():
    p = binary_iv(seed=2)
    r0 = bound(p.variables, p.data, p.query, p.assumptions, method="auto")
    assert r0.success and r0.tight
    # Conditional LR-MSM on IV can refute small Λ; Λ=3 is feasible on seed=2
    a = p.assumptions & Assumptions.tan_msm("X", 3.0)
    r1 = bound(p.variables, p.data, p.query, a, method="auto")
    assert r1.success and r1.tight
    assert r1.width <= r0.width + 1e-6


def test_coverage_conditional():
    V = Variables.from_binary("W", "X", "Y")
    # product measure
    P = {}
    for w in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                P[(w, x, y)] = 0.125
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1, condition={"W": 0})
    r = bound(V, P, Q, Assumptions.order("W", "X", "Y"), method="auto")
    assert r.success and r.tight


def test_coverage_markov_id():
    V, P = _xy()
    g = CausalDiagram.from_edges(
        ["X", "Y"], directed=[("X", "Y")], bidirected=[]
    )
    r = bound(
        V,
        P,
        Query.interventional(outcome="Y", do={"X": 1}, value=1),
        Assumptions.from_diagram(g),
        method="auto",
    )
    assert r.success
    assert r.width < 1e-8
    assert r.method.startswith("id")
