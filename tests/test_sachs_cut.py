"""Sachs / Prop. 8.4 cut completeness vs GPP routing."""

from __future__ import annotations

from partialid import Assumptions, CausalDiagram, Query, Variables
from partialid.formulate import formulate
from partialid.instrument import detect_sachs_cut


def test_prop84_confounded_wl_is_lp_not_gpp():
    g = CausalDiagram.from_edges(
        ["X", "B", "A", "Y"],
        directed=[("X", "Y"), ("B", "A"), ("A", "Y")],
        bidirected=[("X", "Y"), ("A", "B")],
    )
    cut = detect_sachs_cut(g, treatments=["X"], outcome="Y")
    assert cut is not None
    assert set(cut[1]) == {"X", "Y"}
    V = Variables.from_binary(*g.topological_order())
    # uniform feasible enough for formulate routing
    cells = V.cells(g.topological_order())
    P = {c: 1.0 / len(cells) for c in cells}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    lp = formulate(V, P, Q, Assumptions.from_diagram(g))
    assert lp.method_hint == "lp"
    assert lp.gpp is None
    assert any("Prop. 8.4" in n or "Sachs" in n for n in lp.notes)


def test_true_gpp_has_no_sachs_cut():
    g = CausalDiagram.from_edges(
        ["X", "B", "A", "Y"],
        directed=[("X", "Y"), ("B", "A"), ("A", "Y"), ("X", "A")],
        bidirected=[("X", "Y"), ("A", "B")],
    )
    assert detect_sachs_cut(g, treatments=["X"], outcome="Y") is None


def test_isolated_z_not_sachs_wl():
    g = CausalDiagram.from_edges(
        ["X", "Y", "Z"],
        directed=[("X", "Y")],
        bidirected=[("X", "Y")],
    )
    assert detect_sachs_cut(g, treatments=["X"], outcome="Y") is None


def test_classic_iv_sachs():
    g = CausalDiagram.from_edges(
        ["Z", "X", "Y"],
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
    )
    cut = detect_sachs_cut(g, treatments=["X"], outcome="Y")
    assert cut == (("Z",), ("X", "Y"))
