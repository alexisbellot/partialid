"""Latent hyperedges, program-degree classification, and c-degree-2 bilinear."""

from __future__ import annotations

import numpy as np
import pytest

from partialid import Assumptions, CausalDiagram, Query, Variables, bound
from partialid.degree1 import analyze_degree1, c_degree_upper_bound
from partialid.graph import normalize_hyperedge
from partialid.program_degree import ProgramKind, classify_program
from partialid.query_ast import PO, Atom, LinComb, Lit
from partialid.query_footprint import extract_footprint_from_expr


def _softmax_P(V, seed=1):
    rng = np.random.default_rng(seed)
    keys = V.cells()
    logits = rng.normal(size=len(keys))
    w = np.exp(logits - logits.max())
    w = w / w.sum()
    return {k: float(v) for k, v in zip(keys, w)}


def test_normalize_hyperedge_sorts_and_rejects_singleton():
    assert normalize_hyperedge(("Y", "X", "M")) == ("M", "X", "Y")
    with pytest.raises(ValueError, match="≥2"):
        normalize_hyperedge(("X",))


def test_pair_chain_c_degree_2_vs_triple_hyperedge():
    g_pair = CausalDiagram.from_edges(
        ["X", "M", "Y"],
        directed=[("X", "M"), ("M", "Y")],
        bidirected=[("X", "M"), ("M", "Y")],
    )
    g_one = CausalDiagram.from_edges(
        ["X", "M", "Y"],
        directed=[("X", "M"), ("M", "Y")],
        bidirected=[("X", "M", "Y")],
    )
    assert g_pair.c_components() == [frozenset({"X", "M", "Y"})]
    assert g_one.c_components() == [frozenset({"X", "M", "Y"})]
    assert g_pair.c_degree(g_pair.nodes) == 2
    assert g_one.c_degree(g_one.nodes) == 1
    assert c_degree_upper_bound(g_pair, frozenset(g_pair.nodes)) == 2


def test_iv_remains_linear():
    """E2: Z→X→Y, X↔Y. Free block {X,Y} only → degree 1 LP."""
    g = CausalDiagram.from_edges(
        ["Z", "X", "Y"],
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
    )
    prog = classify_program(g, treatments=["X"], outcome="Y")
    assert prog.kind == ProgramKind.LINEAR
    assert prog.degree == 1
    assert prog.c_degrees == (1,)


def test_manski_order_is_full_hyperedge_linear():
    """E1/order: from_order ≡ one latent on all nodes → degree 1."""
    g = CausalDiagram.from_order(["Z", "X", "Y"])
    assert g.bidirected == (("X", "Y", "Z"),)
    assert "maximal_confounding" not in getattr(type(g), "__dataclass_fields__", {})
    prog = classify_program(g, treatments=["X"], outcome="Y")
    assert prog.kind == ProgramKind.LINEAR
    assert prog.degree == 1


def test_classifier_pair_chain_root_block_linear():
    """λ=2 with root private block X: sharp program is an LP, degree still 2."""
    g = CausalDiagram.from_edges(
        ["X", "M", "Y"],
        directed=[("X", "M"), ("M", "Y")],
        bidirected=[("X", "M"), ("M", "Y")],
    )
    prog = classify_program(g, treatments=["X"], outcome="Y")
    assert prog.kind == ProgramKind.LINEAR
    assert prog.degree == 2
    assert any("root private block" in n for n in prog.notes)


def test_classifier_triple_hyperedge_linear():
    g = CausalDiagram.from_edges(
        ["X", "M", "Y"],
        directed=[("X", "M"), ("M", "Y")],
        bidirected=[("X", "M", "Y")],
    )
    prog = classify_program(g, treatments=["X"], outcome="Y")
    assert prog.kind == ProgramKind.LINEAR
    fac = analyze_degree1(g, treatments=["X"], outcome="Y")
    assert fac.lp_reducible


def test_classifier_triangle_c_degree_3_polynomial():
    g = CausalDiagram.from_edges(
        ["X", "Z", "Y"],
        directed=[("X", "Z"), ("X", "Y"), ("Z", "Y")],
        bidirected=[("X", "Z"), ("Z", "Y"), ("X", "Y")],
    )
    assert g.c_degree(g.nodes) == 3
    prog = classify_program(g, treatments=["X"], outcome="Y")
    assert prog.kind == ProgramKind.POLYNOMIAL
    assert prog.degree >= 3


def test_sharp_bilinear_tighter_than_single_latent_lp():
    V = Variables.from_binary("X", "M", "Y")
    P = _softmax_P(V, seed=1)
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    g_pair = CausalDiagram.from_edges(
        ["X", "M", "Y"],
        directed=[("X", "M"), ("M", "Y")],
        bidirected=[("X", "M"), ("M", "Y")],
    )
    g_one = CausalDiagram.from_edges(
        ["X", "M", "Y"],
        directed=[("X", "M"), ("M", "Y")],
        bidirected=[("X", "M", "Y")],
    )
    r_bi = bound(V, P, Q, Assumptions.from_diagram(g_pair))
    r_lp = bound(V, P, Q, Assumptions.from_diagram(g_one))
    assert r_bi.method == "lp"
    assert r_bi.tight
    assert r_lp.tight
    # Two independent latents is a stricter model than one joint latent.
    assert r_bi.lower >= r_lp.lower - 1e-5
    assert r_bi.upper <= r_lp.upper + 1e-5
    assert r_bi.width < r_lp.width - 1e-4


def test_polynomial_c_degree_3_uses_outer():
    """λ=3 triangle: classify POLYNOMIAL; default bound is outer (tight=False)."""
    V = Variables.from_binary("X", "Z", "Y")
    P = _softmax_P(V, seed=0)
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    g = CausalDiagram.from_edges(
        ["X", "Z", "Y"],
        directed=[("X", "Z"), ("X", "Y"), ("Z", "Y")],
        bidirected=[("X", "Z"), ("Z", "Y"), ("X", "Y")],
    )
    prog = classify_program(g, query=Q)
    assert prog.kind == ProgramKind.POLYNOMIAL
    assert prog.in_hypothesis_H is False
    r = bound(V, P, Q, Assumptions.from_diagram(g))
    assert r.method != "gpp/poly"
    assert r.tight is False
    assert any("polynomial" in n or "outer" in n or "prefer_poly" in n for n in r.notes)


def _mediated_one_latent():
    """X→M→Y with only X↔M (E8)."""
    return CausalDiagram.from_edges(
        ["X", "M", "Y"],
        directed=[("X", "M"), ("M", "Y")],
        bidirected=[("X", "M")],
    )


def test_e8_interventional_linear_pns_bilinear_factor():
    """E8: same graph is LP for P(Y|do(X)) but bilinear/factor for PNS."""
    g = _mediated_one_latent()
    q_do = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    q_pns = Query.pns(treatment="X", outcome="Y")

    prog_do = classify_program(g, query=q_do)
    assert prog_do.kind == ProgramKind.LINEAR
    assert prog_do.degree == 1
    assert [sorted(d) for d in prog_do.free_districts] == [["M", "X"]]

    prog_pns = classify_program(g, query=q_pns)
    assert prog_pns.kind == ProgramKind.BILINEAR_FACTOR
    assert prog_pns.degree == 2
    free_sorted = sorted(sorted(d) for d in prog_pns.free_districts)
    assert free_sorted == [["M", "X"], ["Y"]]

    # ATE uses separate single-world atoms — Y stays absorbed
    prog_ate = classify_program(g, query=Query.ate(outcome="Y", treatment="X"))
    assert prog_ate.kind == ProgramKind.LINEAR


def _P_compatible_x_m_y(seed=2):
    """Observational law with Y ⊥ X | M (required by X→M→Y, X↔M)."""
    rng = np.random.default_rng(seed)
    # P(X,M)
    pm = rng.dirichlet(np.ones(4)).reshape(2, 2)
    # P(Y|M)
    py_m = np.stack([rng.dirichlet(np.ones(2)) for _ in range(2)], axis=0)
    P = {}
    for x in (0, 1):
        for m in (0, 1):
            for y in (0, 1):
                P[(x, m, y)] = float(pm[x, m] * py_m[m, y])
    return P


def test_e8_bound_routes_pns_to_gpp():
    V = Variables.from_binary("X", "M", "Y")
    P = _P_compatible_x_m_y(seed=2)
    g = _mediated_one_latent()
    A = Assumptions.from_diagram(g)

    r_do = bound(V, P, Query.interventional(outcome="Y", do={"X": 1}, value=1), A)
    assert r_do.success
    assert r_do.method in ("fiber", "lp", "id", "id/id") or r_do.method.startswith("id")
    assert r_do.tight
    assert r_do.method != "gpp"

    r_pns = bound(V, P, Query.pns(treatment="X", outcome="Y"), A)
    assert r_pns.success
    assert r_pns.method == "gpp"
    # Nested-LP attainables are returned; global BB certificate may be loose.
    assert 0.0 <= r_pns.lower + 1e-8
    assert r_pns.upper <= 1.0 + 1e-8
    assert r_pns.width >= -1e-8
    assert any("GPP bilinear" in n for n in r_pns.notes)


def test_e8_pn_ps_route_to_gpp_fixed_denom():
    """PN/PS are Cond but fixed observational denom → GPP (not merged outer)."""
    V = Variables.from_binary("X", "M", "Y")
    P = _P_compatible_x_m_y(seed=2)
    g = _mediated_one_latent()
    A = Assumptions.from_diagram(g)

    r_pn = bound(V, P, Query.pn(treatment="X", outcome="Y"), A)
    assert r_pn.success
    assert r_pn.method == "gpp"
    assert not any("merged-latent outer" in n for n in r_pn.notes)
    assert any("fixed observational denominator" in n for n in r_pn.notes)
    assert 0.0 <= r_pn.lower + 1e-8
    assert r_pn.upper <= 1.0 + 1e-8

    r_ps = bound(V, P, Query.ps(treatment="X", outcome="Y"), A)
    assert r_ps.success
    assert r_ps.method == "gpp"
    assert not any("merged-latent outer" in n for n in r_ps.notes)
    assert 0.0 <= r_ps.lower + 1e-8
    assert r_ps.upper <= 1.0 + 1e-8


def test_arbitrary_joint_counterfactual_footprint():
    """P(Y_z, W_x) multi-regime atom frees Y and W (no named factory)."""
    expr = LinComb.atom(Atom(Lit(PO("Y", {"Z": 1}), 1), Lit(PO("W", {"X": 0}), 1)))
    fp = extract_footprint_from_expr(expr)
    assert fp.cross_world_vars == frozenset({"Y", "W"})
    assert fp.intervention_vars == frozenset({"Z", "X"})

    # Single-world joint under one assignment is not cross-world
    expr_sw = LinComb.atom(Atom(Lit(PO("Y", {"X": 1}), 1), Lit(PO("M", {"X": 1}), 1)))
    fp_sw = extract_footprint_from_expr(expr_sw)
    assert fp_sw.cross_world_vars == frozenset()


def test_e9_arbitrary_conditional_classify_and_gpp():
    """P(Y_z, W_x | X=1, W=1) via Query.conditional → bilinear/factor GPP."""
    g = CausalDiagram.from_edges(
        ["X", "W", "Z", "Y"],
        directed=[("X", "W"), ("Z", "Y")],
        bidirected=[("X", "W"), ("Z", "Y")],
    )
    numer = Atom(Lit(PO("Y", {"Z": 1}), 1), Lit(PO("W", {"X": 0}), 1))
    given = Atom(Lit("X", 1), Lit("W", 1))
    Q = Query.conditional(numer, given, outcome="Y", value=1)

    prog = classify_program(g, query=Q)
    assert prog.kind == ProgramKind.BILINEAR_FACTOR
    assert prog.degree == 2
    free_sorted = sorted(sorted(d) for d in prog.free_districts)
    assert free_sorted == [["W", "X"], ["Y", "Z"]]

    # Compatible P: districts factor with CI across {X,W} ⊥ {Z,Y}.
    # Tuple keys follow diagram topological order (required by formulate).
    rng = np.random.default_rng(3)
    p_xw = rng.dirichlet(np.ones(4)).reshape(2, 2)
    p_zy = rng.dirichlet(np.ones(4)).reshape(2, 2)
    order = g.topological_order()
    P = {}
    for x in (0, 1):
        for w in (0, 1):
            for z in (0, 1):
                for y in (0, 1):
                    d = {"X": x, "W": w, "Z": z, "Y": y}
                    P[tuple(d[v] for v in order)] = float(p_xw[x, w] * p_zy[z, y])
    V = Variables.from_binary("X", "W", "Z", "Y")
    r = bound(V, P, Q, Assumptions.from_diagram(g))
    assert r.success
    assert r.method == "gpp"
    assert any("fixed observational denominator" in n for n in r.notes)
    assert not any("merged-latent outer" in n for n in r.notes)
    assert 0.0 <= r.lower + 1e-8
    assert r.upper <= 1.0 + 1e-8


def test_from_expr_does_not_inject_default_outcome_seed():
    """Arbitrary expr must not pull default outcome='Y' into the ancestral set."""
    g = CausalDiagram.from_edges(
        ["A", "B", "Y"],
        directed=[("A", "B")],
        bidirected=[("A", "B")],
    )
    # Query only involves A,B — default Query.outcome is still "Y"
    Q = Query.from_expr(
        LinComb.atom(Atom(Lit(PO("B", {"A": 1}), 1))),
        outcome="Y",  # unused default-style field
        value=1,
    )
    fac = analyze_degree1(g, query=Q)
    assert "Y" not in fac.ancestral
    assert fac.free_districts == (frozenset({"A", "B"}),)


RE2 = {
    (0, 0, 0): 0.20, (0, 0, 1): 0.05, (0, 1, 0): 0.10, (0, 1, 1): 0.15,
    (1, 0, 0): 0.05, (1, 0, 1): 0.10, (1, 1, 0): 0.10, (1, 1, 1): 0.25,
}


def test_pair_chain_lift_matches_known_intervals():
    """Pair chain on eq:re2: P(Y_1=1) is Manski; PNS is strictly tighter."""
    g = CausalDiagram.from_edges(
        ["X", "W", "Y"],
        directed=[("X", "W"), ("W", "Y")],
        bidirected=[("X", "W"), ("W", "Y")],
    )
    V = Variables.from_binary("X", "W", "Y")
    a = Assumptions.from_diagram(g)
    r = bound(V, RE2, Query.interventional(outcome="Y", do={"X": 1}, value=1), a)
    assert r.success and r.tight and r.method == "lp"
    assert abs(r.lower - 0.35) < 1e-6 and abs(r.upper - 0.85) < 1e-6
    pns = bound(V, RE2, Query.pns(treatment="X", outcome="Y"), a)
    assert pns.success and pns.tight
    assert abs(pns.lower - 0.0) < 1e-6 and abs(pns.upper - 0.60) < 1e-6


def test_confounded_instrument_matches_balke_pearl():
    V = Variables.from_binary("Z", "X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    g_iv = CausalDiagram.from_edges(
        ["Z", "X", "Y"],
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
    )
    g_zx = CausalDiagram.from_edges(
        ["Z", "X", "Y"],
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[("Z", "X"), ("X", "Y")],
    )
    r_iv = bound(V, RE2, Q, Assumptions.from_diagram(g_iv))
    r_zx = bound(V, RE2, Q, Assumptions.from_diagram(g_zx))
    assert r_iv.tight and r_zx.tight and r_zx.method == "lp"
    assert abs(r_iv.lower - r_zx.lower) < 1e-6
    assert abs(r_iv.upper - r_zx.upper) < 1e-6
    assert r_zx.lower > 0.45  # free joint on this table is [0.40, 0.80]


def test_descendant_sharing_a_latent_is_not_trimmed_away():
    """X→Y→D with {X,Y} and {X,D}: dropping D is a strictly loose outer bound."""
    g = CausalDiagram.from_edges(
        ["X", "Y", "D"],
        directed=[("X", "Y"), ("Y", "D")],
        bidirected=[("X", "Y"), ("X", "D")],
    )
    P = {
        (0, 0, 0): 0.046, (0, 0, 1): 0.010, (0, 1, 0): 0.047, (0, 1, 1): 0.498,
        (1, 0, 0): 0.103, (1, 0, 1): 0.279, (1, 1, 0): 0.010, (1, 1, 1): 0.007,
    }
    V = Variables.from_binary("X", "Y", "D")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    prog = classify_program(g, query=Q)
    assert any("D" in d for d in ("".join(sorted(s)) for s in prog.free_districts))
    # Neither private block is a root, so this is not the linear lift.
    assert prog.kind != ProgramKind.LINEAR
    r = bound(V, P, Q, Assumptions.from_diagram(g), prefer_gpp=False)
    # Manski on the (X,Y) margin is about [0.017, 0.618]. The full model is
    # strictly inside; an outer solver must not certify the Manski interval.
    assert r.success
    assert not (r.tight and r.lower < 0.018 and r.upper > 0.61)


def test_pinning_rejects_mediated_table_that_breaks_conditional_independence():
    g = CausalDiagram.from_edges(
        ["X", "W", "Y"],
        directed=[("X", "W"), ("W", "Y")],
        bidirected=[("X", "W")],
    )
    V = Variables.from_binary("X", "W", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r = bound(V, RE2, Q, Assumptions.from_diagram(g))
    assert r.success is False
    assert any("pinning" in n or "infeasible" in n for n in r.notes)
