"""ID reduction + tightness warnings."""

from __future__ import annotations

import warnings

import numpy as np

from partialid import Assumptions, CausalDiagram, Query, Variables, bound
from partialid.identify import try_identify
from partialid.problems import sachs_binary_iv

RE1 = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}

RE2 = {
    (0, 0, 0): 0.20,
    (0, 0, 1): 0.05,
    (0, 1, 0): 0.10,
    (0, 1, 1): 0.15,
    (1, 0, 0): 0.05,
    (1, 0, 1): 0.10,
    (1, 1, 0): 0.10,
    (1, 1, 1): 0.25,
}


def test_order_only_skips_id_manski_tight():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r = bound(V, RE1, Q, Assumptions.order("X", "Y"))
    assert r.method in ("fiber", "closed/manski_binary_xy") or r.method.startswith("closed/")
    assert r.tight
    assert abs(r.lower - 0.35) < 1e-8
    assert abs(r.upper - 0.85) < 1e-8
    assert not r.warnings


def test_markov_g_formula_point_id():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    g = CausalDiagram.from_edges(["X", "Y"], directed=[("X", "Y")])
    a = Assumptions.from_diagram(g)
    r = bound(V, RE1, Q, a)
    assert r.method.startswith("id/")
    assert r.tight
    assert abs(r.lower - 0.70) < 1e-8
    assert abs(r.upper - 0.70) < 1e-8
    assert r.width == 0.0


def test_frontdoor_point_id():
    V = Variables.from_binary("X", "Z", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    # Markovian chain: g-formula = sum_z P(z|x)P(y|z) = 139/240 on RE2
    g = CausalDiagram.from_edges(["X", "Z", "Y"], directed=[("X", "Z"), ("Z", "Y")])
    a = Assumptions.from_diagram(g)
    r = bound(V, RE2, Q, a)
    assert r.method.startswith("id/")
    assert r.tight
    # Under X→Z→Y, ID yields P(y|x) (=0.70 on RE2). Truncated g-formula
    # with P(y|z) equals this only when Y⊥X|Z; RE2 violates that CI, so
    # the two plug-ins disagree. ID's expression is the Shpitser form.
    assert abs(r.lower - 0.70) < 1e-6
    assert abs(r.upper - r.lower) < 1e-10


def test_frontdoor_with_confounding():
    """Classic front-door: X→Z→Y with X↔Y (no direct X→Y)."""
    V = Variables.from_binary("X", "Z", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    g = CausalDiagram.from_edges(
        ["X", "Z", "Y"],
        directed=[("X", "Z"), ("Z", "Y")],
        bidirected=[("X", "Y")],
    )
    a = Assumptions.from_diagram(g)
    r = bound(V, RE2, Q, a)
    assert r.method.startswith("id/")
    assert r.tight
    assert abs(r.upper - r.lower) < 1e-10


def test_iv_not_point_id_instrument_tight():
    p = sachs_binary_iv(seed=2)
    r = bound(p.variables, p.data, p.query, p.assumptions)
    assert not r.method.startswith("id/")
    assert r.tight  # Balke–Pearl R-side LP or closed form is tight for IV
    blob = " ".join(r.notes)
    assert "instrument LP" in blob or "balke_pearl" in blob or r.method.startswith("closed/")


def test_isolated_node_degree1_tight():
    """Unused singleton is dropped by ancestral restrict; bow LP stays tight."""
    # X→Y confounded; isolated Z ⇒ not in An({X,Y}); c-degree-1, no merge
    V = Variables.from_binary("X", "Y", "Z")
    P = {
        (0, 0, 0): 0.15,
        (0, 0, 1): 0.15,
        (0, 1, 0): 0.10,
        (0, 1, 1): 0.10,
        (1, 0, 0): 0.075,
        (1, 0, 1): 0.075,
        (1, 1, 0): 0.175,
        (1, 1, 1): 0.175,
    }
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    g = CausalDiagram.from_edges(
        ["X", "Y", "Z"],
        directed=[("X", "Y")],
        bidirected=[("X", "Y")],
    )
    a = Assumptions.from_diagram(g)
    idr = try_identify(V, P, Q, a)
    assert not idr.identified
    r = bound(V, P, Q, a)
    assert r.tight
    assert "bounds_may_be_outer" not in r.notes
    assert "c-degree-1" in " ".join(r.notes) or "restricted to ancestral" in " ".join(r.notes)
    # Same as Manski on the X,Y margin: P(Y=1,X=1)=0.35, P(Y=1)=0.55
    assert abs(r.lower - 0.35) < 1e-8
    assert abs(r.upper - 0.85) < 1e-8


def test_two_free_districts_outer_or_fallback():
    """Large ≥2-free graph (no Sachs cut): size gate → outer LP."""
    V = Variables.from_binary("X", "U", "V", "Y")
    rng = np.random.default_rng(1)
    cells = V.cells(["X", "U", "V", "Y"])
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    g = CausalDiagram.from_edges(
        ["X", "U", "V", "Y"],
        directed=[("X", "Y"), ("U", "Y"), ("V", "Y"), ("X", "U")],
        bidirected=[("X", "Y"), ("U", "V")],
    )
    a = Assumptions.from_diagram(g)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = bound(V, P, Q, a)
    # 512×8 exceeds GPP size gate → merged outer
    assert not r.tight
    assert "bounds_may_be_outer" in r.notes
    assert r.warnings or any("may not be tight" in str(x.message) for x in w)


def test_prop84_prefers_instrument_lp_over_gpp():
    """Confounded WL with Sachs cut → Layer-1 LP, not bilinear GPP."""
    g = CausalDiagram.from_edges(
        ["X", "B", "A", "Y"],
        directed=[("X", "Y"), ("B", "A"), ("A", "Y")],
        bidirected=[("X", "Y"), ("A", "B")],
    )
    order = g.topological_order()
    V = Variables.from_binary(*order)
    from partialid.degree1 import analyze_degree1
    from partialid.gpp import build_gpp
    from partialid.instrument import detect_sachs_cut

    assert detect_sachs_cut(g, treatments=["X"], outcome="Y") is not None
    fac = analyze_degree1(g, treatments=["X"], outcome="Y")
    assert len(fac.free_districts) == 2
    cells = V.cells(order)
    P_dummy = {c: 1.0 / len(cells) for c in cells}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    gpp0 = build_gpp(V, P_dummy, Q, g, fac.free_districts)
    rng = np.random.default_rng(0)
    q1 = rng.dirichlet(np.ones(gpp0.A.shape[1]))
    q2 = rng.dirichlet(np.ones(gpp0.A.shape[2]))
    P = {c: float(p) for c, p in zip(gpp0.cells, np.einsum("cij,i,j->c", gpp0.A, q1, q2))}
    q_true = float(np.einsum("ij,i,j->", gpp0.c_obj, q1, q2))
    r = bound(V, P, Q, Assumptions.from_diagram(g))
    assert r.method == "lp"
    assert r.success and r.tight
    assert any("Prop. 8.4" in n or "Sachs" in n for n in r.notes)
    assert r.lower - 1e-5 <= q_true <= r.upper + 1e-5


def test_gpp_bilinear_tighter_than_outer():
    """K=2 free districts without Sachs cut: GPP contains truth and beats outer."""
    # X→A breaks Prop. 8.4; districts {X,Y}, {A,B}
    g = CausalDiagram.from_edges(
        ["X", "B", "A", "Y"],
        directed=[("X", "Y"), ("B", "A"), ("A", "Y"), ("X", "A")],
        bidirected=[("X", "Y"), ("A", "B")],
    )
    order = g.topological_order()
    V = Variables.from_binary(*order)

    from partialid.degree1 import analyze_degree1
    from partialid.formulate import formulate
    from partialid.gpp import build_gpp
    from partialid.instrument import detect_sachs_cut

    assert detect_sachs_cut(g, treatments=["X"], outcome="Y") is None
    fac = analyze_degree1(g, treatments=["X"], outcome="Y")
    assert len(fac.free_districts) == 2

    cells = V.cells(order)
    P_dummy = {c: 1.0 / len(cells) for c in cells}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    gpp0 = build_gpp(V, P_dummy, Q, g, fac.free_districts)
    rng = np.random.default_rng(0)
    q1 = rng.dirichlet(np.ones(gpp0.A.shape[1]))
    q2 = rng.dirichlet(np.ones(gpp0.A.shape[2]))
    P_vec = np.einsum("cij,i,j->c", gpp0.A, q1, q2)
    assert abs(P_vec.sum() - 1.0) < 1e-8
    P = {c: float(p) for c, p in zip(gpp0.cells, P_vec)}
    q_true = float(np.einsum("ij,i,j->", gpp0.c_obj, q1, q2))

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r_gpp = bound(V, P, Q, Assumptions.from_diagram(g))
    assert r_gpp.method == "gpp"
    assert r_gpp.success
    assert r_gpp.lower - 1e-5 <= q_true <= r_gpp.upper + 1e-5

    lp_outer = formulate(V, P, Q, Assumptions.from_diagram(g), prefer_gpp=False)
    assert lp_outer.outer_approximation
    from partialid.backends import solve_bounds_lp

    lo_r, hi_r = solve_bounds_lp(lp_outer)
    assert lo_r.success and hi_r.success
    assert r_gpp.lower >= lo_r.value - 1e-5
    assert r_gpp.upper <= hi_r.value + 1e-5
    assert (r_gpp.upper - r_gpp.lower) <= (hi_r.value - lo_r.value) + 1e-5


def test_napkin_identified_not_backdoor():
    """Classic napkin: Z1→X→Y, Z1→Z2→Y, X↔Z2 — ID, not plain backdoor."""
    V = Variables.from_binary("Z1", "Z2", "X", "Y")
    rng = np.random.default_rng(0)
    cells = V.cells(["Z1", "Z2", "X", "Y"])
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    g = CausalDiagram.from_edges(
        ["Z1", "Z2", "X", "Y"],
        directed=[("Z1", "X"), ("X", "Y"), ("Z1", "Z2"), ("Z2", "Y")],
        bidirected=[("X", "Z2")],
    )
    a = Assumptions.from_diagram(g)
    idr = try_identify(V, P, Q, a)
    assert idr.identified
    assert idr.criterion == "id"
    r = bound(V, P, Q, a)
    assert r.method == "id/id"
    assert r.tight
    assert abs(r.width) < 1e-12


def test_bow_not_identified():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    g = CausalDiagram.from_edges(["X", "Y"], directed=[("X", "Y")], bidirected=[("X", "Y")])
    idr = try_identify(V, RE1, Q, Assumptions.from_diagram(g))
    assert not idr.identified
