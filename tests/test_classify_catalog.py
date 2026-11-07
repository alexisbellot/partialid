"""Extensive Classify catalog (THEORY §7–§10) + formulate/bound agreement."""

from __future__ import annotations

import numpy as np
import pytest

from partialid import Assumptions, CausalDiagram, Query, Variables, bound
from partialid.formulate import formulate
from partialid.program_degree import ProgramKind, classify_program
from partialid.query_ast import PO, Atom, LinComb, Lit


def _softmax_P(V: Variables, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    keys = V.cells()
    logits = rng.normal(size=len(keys))
    w = np.exp(logits - logits.max())
    w = w / w.sum()
    return {k: float(v) for k, v in zip(keys, w)}


def _uniform_P(V: Variables) -> dict:
    cells = V.cells()
    return {c: 1.0 / len(cells) for c in cells}


def _Q_do(outcome: str = "Y", treatment: str = "X") -> Query:
    return Query.interventional(outcome=outcome, do={treatment: 1}, value=1)


def _free_sorted(prog) -> list[list[str]]:
    return sorted(sorted(d) for d in prog.free_districts)


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def g_bow():
    return CausalDiagram.from_edges(["X", "Y"], directed=[("X", "Y")], bidirected=[("X", "Y")])


def g_order_xy():
    return CausalDiagram.from_order(["X", "Y"])


def g_iv():
    return CausalDiagram.from_edges(
        ["Z", "X", "Y"],
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
    )


def g_extended_iv():
    return CausalDiagram.from_edges(
        ["X", "B", "A", "Y"],
        directed=[("X", "Y"), ("B", "A"), ("A", "Y")],
        bidirected=[("X", "Y"), ("A", "B")],
    )


def g_two_instruments():
    return CausalDiagram.from_edges(
        ["Z1", "Z2", "X", "Y"],
        directed=[("Z1", "X"), ("Z2", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
    )


def g_mediated():
    return CausalDiagram.from_edges(
        ["X", "W", "Y"],
        directed=[("X", "W"), ("W", "Y")],
        bidirected=[("X", "W")],
    )


def g_triple_hyperedge():
    return CausalDiagram.from_edges(
        ["X", "M", "Y"],
        directed=[("X", "M"), ("M", "Y")],
        bidirected=[("X", "M", "Y")],
    )


def g_markov():
    return CausalDiagram.from_edges(["X", "Y"], directed=[("X", "Y")], bidirected=[])


def g_pair_chain():
    return CausalDiagram.from_edges(
        ["X", "W", "Y"],
        directed=[("X", "W"), ("W", "Y")],
        bidirected=[("X", "W"), ("W", "Y")],
    )


def g_two_bows():
    return CausalDiagram.from_edges(
        ["X", "A", "B", "Y"],
        directed=[("X", "A"), ("A", "B"), ("B", "Y")],
        bidirected=[("X", "A"), ("B", "Y")],
    )


def g_cut_spoiled():
    """Extended-IV shape with a latent across the would-be cut."""
    return CausalDiagram.from_edges(
        ["X", "B", "A", "Y"],
        directed=[("X", "Y"), ("B", "A"), ("A", "Y")],
        bidirected=[("X", "Y"), ("A", "B"), ("B", "X")],
    )


def g_triangle():
    return CausalDiagram.from_edges(
        ["X", "Z", "Y"],
        directed=[("X", "Z"), ("X", "Y"), ("Z", "Y")],
        bidirected=[("X", "Z"), ("Z", "Y"), ("X", "Y")],
    )


def g_lambda3_chain():
    return CausalDiagram.from_edges(
        ["X", "W", "Y", "S"],
        directed=[("X", "W"), ("W", "Y"), ("Y", "S")],
        bidirected=[("X", "W"), ("W", "Y"), ("Y", "S")],
    )


def g_k3_factor():
    """Three free λ=1 districts under PNS (no cut)."""
    return CausalDiagram.from_edges(
        ["X", "A", "B", "C", "Y"],
        directed=[("X", "A"), ("A", "B"), ("B", "C"), ("C", "Y")],
        bidirected=[("X", "A"), ("B", "C")],
    )


# ---------------------------------------------------------------------------
# L*: LP / POINT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,graph,query,kind,degree,free,cut,in_H",
    [
        (
            "L1",
            g_bow(),
            _Q_do(),
            ProgramKind.LINEAR,
            1,
            [["X", "Y"]],
            None,
            True,
        ),
        (
            "L2",
            g_order_xy(),
            _Q_do(),
            ProgramKind.LINEAR,
            1,
            [["X", "Y"]],
            None,
            True,
        ),
        (
            "L3",
            g_iv(),
            _Q_do(),
            ProgramKind.LINEAR,
            1,
            [["X", "Y"]],
            (("Z",), ("X", "Y")),
            True,
        ),
        (
            "L4",
            g_extended_iv(),
            _Q_do(),
            ProgramKind.LINEAR,
            1,
            [["X", "Y"]],
            (("B", "A"), ("X", "Y")),
            True,
        ),
        (
            "L5",
            g_two_instruments(),
            _Q_do(),
            ProgramKind.LINEAR,
            1,
            [["X", "Y"]],
            (("Z1", "Z2"), ("X", "Y")),
            True,
        ),
        (
            "L6",
            g_mediated(),
            _Q_do(),
            ProgramKind.LINEAR,
            1,
            [["W", "X"]],
            None,
            True,
        ),
        (
            "L6ate",
            g_mediated(),
            Query.ate(outcome="Y", treatment="X"),
            ProgramKind.LINEAR,
            1,
            [["W", "X"]],
            None,
            True,
        ),
        (
            "L7",
            g_triple_hyperedge(),
            _Q_do(),
            ProgramKind.LINEAR,
            1,
            [["M", "X", "Y"]],
            None,
            True,
        ),
        (
            "L8",
            g_markov(),
            _Q_do(),
            ProgramKind.POINT,
            0,
            [],
            None,
            True,
        ),
        (
            "L9",
            g_markov(),
            Query.pns(treatment="X", outcome="Y"),
            ProgramKind.LINEAR,
            1,
            [["Y"]],
            None,
            True,
        ),
    ],
)
def test_catalog_L_classify(case_id, graph, query, kind, degree, free, cut, in_H):
    prog = classify_program(graph, query=query)
    assert prog.kind == kind, case_id
    assert prog.degree == degree, case_id
    assert _free_sorted(prog) == free, case_id
    assert prog.in_hypothesis_H is in_H, case_id
    if cut is None:
        assert prog.wl_wr_cut is None, case_id
    else:
        assert prog.wl_wr_cut is not None, case_id
        assert set(prog.wl_wr_cut[0]) == set(cut[0]), case_id
        assert set(prog.wl_wr_cut[1]) == set(cut[1]), case_id


def test_catalog_L_frontdoor_confounding_is_point_id():
    """Front-door with X↔Y: ID succeeds → classify POINT (not LINEAR)."""
    g = CausalDiagram.from_edges(
        ["X", "Z", "Y"],
        directed=[("X", "Z"), ("Z", "Y")],
        bidirected=[("X", "Y")],
    )
    prog = classify_program(g, query=_Q_do())
    assert prog.kind == ProgramKind.POINT
    assert prog.degree == 0
    assert any("ID" in n for n in prog.notes)


def test_catalog_L10_faces_do_not_change_kind():
    """MTR / exclusion faces present → still LINEAR (CompileAssump out of scope)."""
    g = g_bow()
    a = Assumptions.from_diagram(g) & Assumptions.mtr("Y", "X")
    prog = classify_program(g, query=_Q_do())
    assert prog.kind == ProgramKind.LINEAR
    V = Variables.from_binary("X", "Y")
    lp = formulate(V, _uniform_P(V), _Q_do(), a)
    assert lp.method_hint in ("lp", "fiber")
    assert lp.gpp is None


# ---------------------------------------------------------------------------
# B*: bilinear
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,graph,query,kind,degree,free",
    [
        (
            "B1",
            g_pair_chain(),
            _Q_do(),
            ProgramKind.LINEAR,
            2,
            [["W", "X", "Y"]],
        ),
        (
            "B2",
            g_mediated(),
            Query.pns(treatment="X", outcome="Y"),
            ProgramKind.BILINEAR_FACTOR,
            2,
            [["W", "X"], ["Y"]],
        ),
        (
            "B3",
            g_two_bows(),
            _Q_do(),
            ProgramKind.BILINEAR_FACTOR,
            2,
            [["A", "X"], ["B", "Y"]],
        ),
        (
            "B5",
            g_pair_chain(),
            Query.pns(treatment="X", outcome="Y"),
            ProgramKind.LINEAR,
            2,
            [["W", "X", "Y"]],
        ),
    ],
)
def test_catalog_B_classify(case_id, graph, query, kind, degree, free):
    prog = classify_program(graph, query=query)
    assert prog.kind == kind, case_id
    assert prog.degree == degree, case_id
    assert _free_sorted(prog) == free, case_id
    assert prog.in_hypothesis_H is True, case_id


def test_catalog_B4_cut_fails_not_linear():
    prog = classify_program(g_cut_spoiled(), query=_Q_do())
    assert prog.kind != ProgramKind.LINEAR
    assert prog.wl_wr_cut is None
    assert prog.degree >= 2


# ---------------------------------------------------------------------------
# P*: polynomial / out of (H)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,graph,query,degree,in_H",
    [
        ("P1", g_triangle(), _Q_do(), 3, False),
        (
            "P3",
            g_lambda3_chain(),
            Query.interventional(outcome="S", do={"X": 1}, value=1),
            3,
            False,
        ),
    ],
)
def test_catalog_P_classify(case_id, graph, query, degree, in_H):
    prog = classify_program(graph, query=query)
    assert prog.kind == ProgramKind.POLYNOMIAL, case_id
    assert prog.degree == degree, case_id
    assert prog.in_hypothesis_H is in_H, case_id


def test_catalog_P2_three_factor_districts():
    g = g_k3_factor()
    q = Query.pns(treatment="X", outcome="Y")
    prog = classify_program(g, query=q)
    assert prog.kind == ProgramKind.POLYNOMIAL
    assert prog.degree == 3
    assert len(prog.free_districts) == 3
    assert all(cd == 1 for cd in prog.c_degrees)


# ---------------------------------------------------------------------------
# Q*: query-sensitivity
# ---------------------------------------------------------------------------


def test_catalog_Q1_mediated_query_flip():
    g = g_mediated()
    assert classify_program(g, query=_Q_do()).kind == ProgramKind.LINEAR
    assert (
        classify_program(g, query=Query.pns(treatment="X", outcome="Y")).kind
        == ProgramKind.BILINEAR_FACTOR
    )
    assert (
        classify_program(g, query=Query.ate(outcome="Y", treatment="X")).kind == ProgramKind.LINEAR
    )


def test_catalog_Q2_iv_pn_stays_linear():
    g = g_iv()
    assert classify_program(g, query=_Q_do()).kind == ProgramKind.LINEAR
    assert (
        classify_program(g, query=Query.pn(treatment="X", outcome="Y")).kind == ProgramKind.LINEAR
    )


def g_additive_bows():
    """Two λ=1 bows on a chain: X→Y→Z→W with X↔Y and Z↔W."""
    return CausalDiagram.from_edges(
        ["X", "Y", "Z", "W"],
        directed=[("X", "Y"), ("Y", "Z"), ("Z", "W")],
        bidirected=[("X", "Y"), ("Z", "W")],
    )


def _Q_additive_yw() -> Query:
    expr = LinComb.atom(Atom(Lit(PO("Y", {"X": 1}), 1))) + LinComb.atom(
        Atom(Lit(PO("W", {"Z": 1}), 1))
    )
    return Query.from_expr(expr, outcome="Y", value=1)


def test_catalog_Q3_additive_two_bows_is_linear():
    """Each atom on one λ=1 district → leftover maps affine → LP (not bilinear)."""
    g = g_additive_bows()
    Q = _Q_additive_yw()
    prog = classify_program(g, query=Q)
    assert prog.kind == ProgramKind.LINEAR
    assert prog.degree == 2
    assert _free_sorted(prog) == [["W", "Z"], ["X", "Y"]]
    assert any("separable" in n for n in prog.notes)


def test_catalog_P_Yx_hits_parent_district_is_bilinear():
    """Syntactic {X,Y} is not enough: Y_x uses parent districts structurally."""
    g = CausalDiagram.from_edges(
        ["X", "U", "V", "Y"],
        directed=[("X", "Y"), ("U", "Y"), ("V", "Y"), ("X", "U")],
        bidirected=[("X", "Y"), ("U", "V")],
    )
    prog = classify_program(g, query=_Q_do())
    assert prog.kind == ProgramKind.BILINEAR_FACTOR
    assert prog.degree == 2


def test_catalog_Q3_conjunction_stays_bilinear():
    """One atom using both bows is coupled (not an LP)."""
    g = g_additive_bows()
    expr = LinComb.atom(Atom(Lit(PO("Y", {"X": 1}), 1), Lit(PO("W", {"Z": 1}), 1)))
    Q = Query.from_expr(expr, outcome="Y", value=1)
    prog = classify_program(g, query=Q)
    assert prog.kind == ProgramKind.BILINEAR_FACTOR
    assert prog.degree == 2


# ---------------------------------------------------------------------------
# Classify ↔ formulate / bound agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id,graph,query,expect_hint",
    [
        ("L4", g_extended_iv(), _Q_do(), "lp"),
        ("L3", g_iv(), _Q_do(), "lp"),
        ("B1", g_pair_chain(), _Q_do(), "lp"),
        ("B2", g_mediated(), Query.pns(treatment="X", outcome="Y"), "gpp"),
        ("B3", g_two_bows(), _Q_do(), "gpp"),
        ("Q3", g_additive_bows(), _Q_additive_yw(), "lp"),
    ],
)
def test_catalog_formulate_hint_matches_kind(case_id, graph, query, expect_hint):
    prog = classify_program(graph, query=query)
    V = Variables.from_binary(*graph.topological_order())
    P = _softmax_P(V, seed=hash(case_id) % 10_000)
    lp = formulate(V, P, query, Assumptions.from_diagram(graph))
    assert lp.method_hint == expect_hint, (
        f"{case_id}: classify={prog.kind} formulate={lp.method_hint}"
    )
    if expect_hint == "lp":
        assert lp.gpp is None
        assert not lp.outer_approximation


def test_catalog_additive_bows_stacked_lp_matches_sum():
    """Disconnected bows: stacked LP equals the sum of the two one-bow LPs."""
    g = CausalDiagram.from_edges(
        ["X", "Y", "Z", "W"],
        directed=[("X", "Y"), ("Z", "W")],
        bidirected=[("X", "Y"), ("Z", "W")],
    )
    Q = _Q_additive_yw()
    assert classify_program(g, query=Q).kind == ProgramKind.LINEAR
    V = Variables.from_binary(*g.topological_order())
    P = _softmax_P(V, seed=3)
    a = Assumptions.from_diagram(g)
    lp = formulate(V, P, Q, a)
    assert lp.method_hint == "lp"
    assert not lp.outer_approximation
    assert any("stacked" in n for n in lp.notes)
    r = bound(V, P, Q, a)
    assert r.success and r.tight
    assert r.method in ("lp", "dual")

    order = list(V.names)
    g_xy = CausalDiagram.from_edges(["X", "Y"], directed=[("X", "Y")], bidirected=[("X", "Y")])
    g_zw = CausalDiagram.from_edges(["Z", "W"], directed=[("Z", "W")], bidirected=[("Z", "W")])
    Vxy, Vzw = Variables.from_binary("X", "Y"), Variables.from_binary("Z", "W")
    Pxy: dict[tuple, float] = {k: 0.0 for k in Vxy.cells()}
    Pzw: dict[tuple, float] = {k: 0.0 for k in Vzw.cells()}
    for cell, p in P.items():
        d = dict(zip(order, cell))
        Pxy[(d["X"], d["Y"])] = Pxy.get((d["X"], d["Y"]), 0.0) + float(p)
        Pzw[(d["Z"], d["W"])] = Pzw.get((d["Z"], d["W"]), 0.0) + float(p)
    r_xy = bound(
        Vxy,
        Pxy,
        Query.interventional(outcome="Y", do={"X": 1}, value=1),
        Assumptions.from_diagram(g_xy),
    )
    r_zw = bound(
        Vzw,
        Pzw,
        Query.interventional(outcome="W", do={"Z": 1}, value=1),
        Assumptions.from_diagram(g_zw),
    )
    assert r_xy.tight and r_zw.tight
    assert abs(r.lower - (r_xy.lower + r_zw.lower)) < 1e-6
    assert abs(r.upper - (r_xy.upper + r_zw.upper)) < 1e-6


def test_catalog_extended_iv_no_gpp_split_brain():
    """Regression: confounded WL must classify LINEAR and formulate LP (not GPP)."""
    g = g_extended_iv()
    Q = _Q_do()
    prog = classify_program(g, query=Q)
    assert prog.kind == ProgramKind.LINEAR
    assert prog.degree == 1
    assert _free_sorted(prog) == [["X", "Y"]]
    V = Variables.from_binary(*g.topological_order())
    lp = formulate(V, _uniform_P(V), Q, Assumptions.from_diagram(g))
    assert lp.method_hint == "lp"
    assert lp.gpp is None


def test_catalog_pair_chain_bound_is_lp():
    g = g_pair_chain()
    V = Variables.from_binary(*g.topological_order())
    P = _softmax_P(V, seed=1)  # same seed as latent_hyperedges sharp test
    Q = _Q_do()
    prog = classify_program(g, query=Q)
    assert prog.kind == ProgramKind.LINEAR
    lp = formulate(V, P, Q, Assumptions.from_diagram(g))
    assert lp.method_hint == "lp"
    assert any("root-block lift" in n for n in lp.notes)
    r = bound(V, P, Q, Assumptions.from_diagram(g))
    assert r.method == "lp"
    assert r.success
    assert r.tight


def test_catalog_triangle_defaults_to_outer():
    """P1: classify POLYNOMIAL; default bound is outer (prefer_poly=False)."""
    g = g_triangle()
    V = Variables.from_binary(*g.topological_order())
    P = _softmax_P(V, seed=0)
    Q = _Q_do()
    prog = classify_program(g, query=Q)
    assert prog.kind == ProgramKind.POLYNOMIAL
    assert prog.in_hypothesis_H is False
    lp = formulate(V, P, Q, Assumptions.from_diagram(g))
    assert lp.gpp is None
    assert lp.outer_approximation
    assert any("prefer_poly=True" in n or "outer" in n for n in lp.notes)
    r = bound(V, P, Q, Assumptions.from_diagram(g))
    assert r.success
    assert r.tight is False
    assert r.method != "gpp/poly"


def test_catalog_P2_poly_path_opt_in():
    """Factor-polynomial K=3: prefer_poly=True → gpp/poly on realizable P."""
    g = g_k3_factor()
    Q = Query.pns(treatment="X", outcome="Y")
    prog = classify_program(g, query=Q)
    assert prog.kind == ProgramKind.POLYNOMIAL
    V = Variables.from_binary(*g.topological_order())
    # Default: outer
    lp_default = formulate(V, _uniform_P(V), Q, Assumptions.from_diagram(g))
    assert lp_default.outer_approximation
    assert lp_default.method_hint != "gpp/poly"
    # Opt-in: build tensors under a dummy P, then replace with a realizable law
    lp0 = formulate(V, _uniform_P(V), Q, Assumptions.from_diagram(g), prefer_poly=True)
    assert lp0.method_hint == "gpp/poly"
    poly = lp0.gpp
    rng = np.random.default_rng(0)
    q1 = rng.dirichlet(np.ones(poly.n1))
    q2 = rng.dirichlet(np.ones(poly.n2))
    q3 = rng.dirichlet(np.ones(poly.n3))
    P_vec = np.einsum("cijk,i,j,k->c", poly.A, q1, q2, q3)
    P = {poly.cells[i]: float(P_vec[i]) for i in range(len(poly.cells))}
    lp = formulate(V, P, Q, Assumptions.from_diagram(g), prefer_poly=True)
    assert lp.method_hint == "gpp/poly"
    r = bound(V, P, Q, Assumptions.from_diagram(g), prefer_poly=True)
    assert r.method == "gpp/poly"
    assert r.success
    assert np.isfinite(r.lower) and np.isfinite(r.upper)


def test_catalog_P4_prefer_poly_latent_falls_back_outer():
    """prefer_poly=True on latent λ≥3 → still outer (unsupported shape)."""
    g = g_triangle()
    V = Variables.from_binary(*g.topological_order())
    P = _softmax_P(V, seed=1)
    Q = _Q_do()
    lp = formulate(V, P, Q, Assumptions.from_diagram(g), prefer_poly=True, prefer_gpp=True)
    assert lp.outer_approximation
    assert any("poly" in n.lower() or "outer" in n.lower() for n in lp.notes)
    r = bound(V, P, Q, Assumptions.from_diagram(g), prefer_poly=True)
    assert r.tight is False
