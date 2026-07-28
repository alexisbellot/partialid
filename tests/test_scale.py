"""Scaling: matrix-free colgen + C-canonical parameterization."""

from __future__ import annotations

import numpy as np

from partialid import Assumptions, Query, Variables, bound
from partialid.column_gen import solve_bounds_colgen_lazy
from partialid.formulate import _normalize_P
from partialid.lazy_types import LazyTypeSpace, c_canonical_cardinality

RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


def test_matrix_free_colgen_matches_lp_msm():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y") & Assumptions.tan_msm("X", 3.0)
    rl = bound(V, RE1_P, Q, a, method="lp")
    rc = bound(V, RE1_P, Q, a, method="colgen")
    assert rc.success, rc.message
    assert abs(rl.lower - rc.lower) < 1e-5
    assert abs(rl.upper - rc.upper) < 1e-5
    assert any("matrix_free" in n for n in rc.notes)


def test_lazy_colgen_matches_manski():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    diagram = a.diagram()
    assert diagram is not None
    lazy = LazyTypeSpace.from_diagram(V, diagram)
    cells, P = _normalize_P(V, lazy.order, RE1_P)
    cg = solve_bounds_colgen_lazy(lazy, cells, P, Q)
    assert cg.success, cg.message
    assert abs(cg.lower - 0.35) < 1e-6
    assert abs(cg.upper - 0.85) < 1e-6


def test_c_canonical_matches_manski_re1():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    r = bound(V, RE1_P, Q, a, method="c_canonical")
    assert r.success
    assert abs(r.lower - 0.35) < 1e-8
    assert abs(r.upper - 0.85) < 1e-8
    assert r.method.startswith("c_canonical")
    assert any("d_U=" in n for n in r.notes)


def test_c_canonical_cardinality_triangle():
    V = Variables.from_binary("X", "Z", "Y")
    g = Assumptions.order("X", "Z", "Y").diagram()
    assert g is not None
    assert c_canonical_cardinality(V, g) == 8  # 2^3


def test_c_canonical_chain_matches_fiber():
    V = Variables.from_binary("X", "Z", "Y")
    rng = np.random.default_rng(0)
    cells = V.cells(["X", "Z", "Y"])
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Z", "Y")
    rf = bound(V, P, Q, a)  # fiber via inert prune
    rc = bound(V, P, Q, a, method="c_canonical")
    assert rc.success
    assert abs(rf.lower - rc.lower) < 1e-8
    assert abs(rf.upper - rc.upper) < 1e-8
