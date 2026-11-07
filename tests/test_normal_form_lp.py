"""Realizability gate and the linear cases from the canonical normal form.

Dropped districts are part of the model. A table no response law can produce
must not come back as a tight interval. Factual factors, a stacked λ=2 block,
contained hyperedges, and the sufficient fiber lemma are linear programs.
"""

from __future__ import annotations

from partialid import Assumptions, CausalDiagram, Query, Variables, bound
from partialid.program_degree import ProgramKind, classify_program
from partialid.query_ast import PO, Atom, LinComb, Lit


def _joint(g: CausalDiagram, weight):
    order = g.topological_order()
    V = Variables.from_binary(*order)
    cells = list(V.cells())
    raw = [float(weight(dict(zip(order, c)))) for c in cells]
    total = sum(raw)
    return V, {c: w / total for c, w in zip(cells, raw)}


def _uniform(g: CausalDiagram):
    return _joint(g, lambda _d: 1.0)


def _do(outcome: str, treatment: str, t: int = 1) -> Query:
    return Query.interventional(outcome=outcome, do={treatment: t}, value=1)


def test_bow_plus_unrealizable_chain_is_not_tight():
    g = CausalDiagram.from_edges(
        ["A", "B", "X", "W", "Y"],
        directed=[("A", "B"), ("X", "W"), ("W", "Y")],
        bidirected=[("A", "B"), ("X", "W"), ("W", "Y")],
    )
    V, P = _joint(g, lambda d: 1.0 if d["W"] == 0 and d["X"] == d["Y"] else 0.0)
    r = bound(V, P, _do("B", "A", 0), Assumptions.from_diagram(g))
    assert r.success is False
    assert any("no response law" in n for n in r.notes)


def test_bow_plus_uniform_chain_stays_tight():
    g = CausalDiagram.from_edges(
        ["A", "B", "X", "W", "Y"],
        directed=[("A", "B"), ("X", "W"), ("W", "Y")],
        bidirected=[("A", "B"), ("X", "W"), ("W", "Y")],
    )
    V, P = _uniform(g)
    r = bound(V, P, _do("B", "A", 0), Assumptions.from_diagram(g))
    assert r.success and r.tight


def test_directed_cut_drops_unrealizable_chain():
    g = CausalDiagram.from_edges(
        ["U", "V", "W", "X", "Y"],
        directed=[("U", "V"), ("V", "W"), ("W", "X"), ("X", "Y")],
        bidirected=[("U", "V"), ("V", "W"), ("X", "Y")],
    )
    V, P = _joint(g, lambda d: 1.0 if d["V"] == 0 and d["U"] == d["W"] else 0.0)
    r = bound(V, P, _do("Y", "X"), Assumptions.from_diagram(g))
    assert r.success is False
    assert any("no response law" in n for n in r.notes)


def test_directed_cut_uniform_chain_stays_tight():
    g = CausalDiagram.from_edges(
        ["U", "V", "W", "X", "Y"],
        directed=[("U", "V"), ("V", "W"), ("W", "X"), ("X", "Y")],
        bidirected=[("U", "V"), ("V", "W"), ("X", "Y")],
    )
    V, P = _uniform(g)
    r = bound(V, P, _do("Y", "X"), Assumptions.from_diagram(g))
    assert r.success and r.tight


def test_factual_factor_on_two_bows_is_a_tight_lp():
    g = CausalDiagram.from_edges(
        ["X", "Y", "Z", "W"],
        directed=[("X", "Y"), ("Z", "W")],
        bidirected=[("X", "Y"), ("Z", "W")],
    )
    expr = LinComb.atom(Atom(Lit("Y", 1), Lit(PO("W", {"Z": 1}), 1)))
    Q = Query.from_expr(expr, outcome="W", value=1)
    prog = classify_program(g, query=Q)
    assert prog.kind == ProgramKind.LINEAR
    V, P = _uniform(g)
    r = bound(V, P, Q, Assumptions.from_diagram(g))
    assert r.success and r.tight
    bow = CausalDiagram.from_edges(["Z", "W"], directed=[("Z", "W")], bidirected=[("Z", "W")])
    Vb, Pb = _uniform(bow)
    rb = bound(Vb, Pb, _do("W", "Z"), Assumptions.from_diagram(bow))
    assert rb.tight
    assert abs(r.lower - 0.5 * rb.lower) < 1e-6
    assert abs(r.upper - 0.5 * rb.upper) < 1e-6


def test_sum_of_root_private_and_bow_is_a_stacked_lp():
    g = CausalDiagram.from_edges(
        ["X", "W", "Y", "A", "B"],
        directed=[("X", "W"), ("W", "Y"), ("A", "B")],
        bidirected=[("X", "W"), ("W", "Y"), ("A", "B")],
    )
    expr = LinComb.atom(Atom(Lit(PO("Y", {"X": 1}), 1))) + LinComb.atom(
        Atom(Lit(PO("B", {"A": 0}), 1))
    )
    Q = Query.from_expr(expr, outcome="Y", value=1)
    prog = classify_program(g, query=Q)
    assert prog.kind == ProgramKind.LINEAR
    assert prog.degree == 3
    V, P = _uniform(g)
    r = bound(V, P, Q, Assumptions.from_diagram(g))
    assert r.success and r.tight
    chain = CausalDiagram.from_edges(
        ["X", "W", "Y"],
        directed=[("X", "W"), ("W", "Y")],
        bidirected=[("X", "W"), ("W", "Y")],
    )
    bow = CausalDiagram.from_edges(["A", "B"], directed=[("A", "B")], bidirected=[("A", "B")])
    Vc, Pc = _uniform(chain)
    Vb, Pb = _uniform(bow)
    rc = bound(Vc, Pc, _do("Y", "X"), Assumptions.from_diagram(chain))
    rb = bound(Vb, Pb, _do("B", "A", 0), Assumptions.from_diagram(bow))
    assert rc.tight and rb.tight
    assert abs(r.lower - (rc.lower + rb.lower)) < 1e-6
    assert abs(r.upper - (rc.upper + rb.upper)) < 1e-6


def test_contained_hyperedge_counts_as_one():
    g = CausalDiagram.from_edges(
        ["X", "Y", "Z"],
        directed=[("X", "Y"), ("Y", "Z")],
        bidirected=[("Y", "Z"), ("X", "Y", "Z")],
    )
    assert g.c_degree(["X", "Y", "Z"]) == 2
    Q = Query.interventional(outcome="Z", do={"X": 0}, value=1)
    prog = classify_program(g, query=Q)
    assert prog.kind == ProgramKind.LINEAR
    assert prog.degree == 1
    assert any("contained" in n for n in prog.notes)


def test_factual_instrument_remargins():
    g = CausalDiagram.from_edges(
        ["Z", "X", "Y"],
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
    )
    V, P = _uniform(g)
    Q = Query.from_expr(LinComb.atom(Atom(Lit("Z", 1))), outcome="Z", value=1)
    assert classify_program(g, query=Q).kind == ProgramKind.POINT
    r = bound(V, P, Q, Assumptions.from_diagram(g))
    assert r.success and r.tight
    assert abs(r.lower - 0.5) < 1e-6
    assert abs(r.upper - 0.5) < 1e-6


def test_unrealizable_instrument_factor_is_infeasible_not_a_crash():
    g = CausalDiagram.from_edges(
        ["Z", "X", "Y"],
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
    )
    V, P = _joint(g, lambda d: 1.0 if d["X"] == 1 and d["Y"] == 1 - d["Z"] else 0.0)
    Q = Query.from_expr(LinComb.atom(Atom(Lit("Z", 1))), outcome="Z", value=1)
    r = bound(V, P, Q, Assumptions.from_diagram(g))
    assert r.success is False


def test_star_is_a_tight_lp_and_does_not_ignore_root_independence():
    g = CausalDiagram.from_edges(
        ["Z1", "Z2", "Z3", "W"],
        directed=[("Z1", "W"), ("Z2", "W"), ("Z3", "W")],
        bidirected=[("Z1", "W"), ("Z2", "W"), ("Z3", "W")],
    )
    Q = Query.interventional(outcome="W", do={"Z1": 1, "Z2": 1, "Z3": 1}, value=1)
    prog = classify_program(g, query=Q)
    assert prog.kind == ProgramKind.LINEAR
    assert prog.degree == 3

    def mechanism(d):
        # Z1 = U, and W is 0 when U is 0 and equals Z2 when U is 1.
        if d["Z1"] == 0:
            return 1.0 if d["W"] == 0 else 0.0
        return 1.0 if d["W"] == d["Z2"] else 0.0

    V, P = _joint(g, mechanism)
    r = bound(V, P, Q, Assumptions.from_diagram(g))
    assert r.success and r.tight
    # Sharp interval. The observational conditional P(W=1 | Z=1) is the upper
    # endpoint, not the identified value.
    assert abs(r.lower - 0.5) < 1e-6
    assert abs(r.upper - 1.0) < 1e-6

    Vd, Pd = _joint(g, lambda d: 0.0 if d["Z2"] != d["Z3"] else 1.0)
    rd = bound(Vd, Pd, Q, Assumptions.from_diagram(g))
    assert rd.success is False


def test_mixing_step_counterexample_is_not_linear():
    g = CausalDiagram.from_edges(
        ["Z", "W", "A", "D"],
        directed=[("Z", "W"), ("Z", "A"), ("Z", "D")],
        bidirected=[("W", "A"), ("A", "D", "Z")],
    )
    prog = classify_program(g, query=_do("W", "Z", 0))
    assert prog.kind != ProgramKind.LINEAR
