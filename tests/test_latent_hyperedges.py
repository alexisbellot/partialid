"""Latent hyperedges, program-degree classification, and c-degree-2 bilinear."""

from __future__ import annotations

import numpy as np
import pytest

from partialid import Assumptions, CausalDiagram, Query, Variables, bound
from partialid.degree1 import analyze_degree1, c_degree_upper_bound
from partialid.graph import normalize_hyperedge
from partialid.program_degree import ProgramKind, classify_program


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


def test_classifier_pair_chain_bilinear_latent():
    g = CausalDiagram.from_edges(
        ["X", "M", "Y"],
        directed=[("X", "M"), ("M", "Y")],
        bidirected=[("X", "M"), ("M", "Y")],
    )
    prog = classify_program(g, treatments=["X"], outcome="Y")
    assert prog.kind == ProgramKind.BILINEAR_LATENT
    assert prog.degree == 2


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
    assert r_bi.method == "gpp/latent"
    assert r_bi.tight
    assert r_lp.tight
    # Two independent latents is a stricter model than one joint latent.
    assert r_bi.lower >= r_lp.lower - 1e-5
    assert r_bi.upper <= r_lp.upper + 1e-5
    assert r_bi.width < r_lp.width - 1e-4


def test_polynomial_c_degree_3_uses_outer():
    V = Variables.from_binary("X", "Z", "Y")
    P = _softmax_P(V, seed=0)
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    g = CausalDiagram.from_edges(
        ["X", "Z", "Y"],
        directed=[("X", "Z"), ("X", "Y"), ("Z", "Y")],
        bidirected=[("X", "Z"), ("Z", "Y"), ("X", "Y")],
    )
    r = bound(V, P, Q, Assumptions.from_diagram(g))
    assert r.tight is False
    assert any("polynomial" in n or "outer" in n for n in r.notes)


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
    assert r_do.method in ("fiber", "lp", "id/backdoor", "id/do_calc")
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
