"""Extra unit coverage for public API modules (Phase 3 quality bar)."""

from __future__ import annotations

import pytest

from partialid import (
    Assumptions,
    Query,
    Variables,
    from_counts,
    from_records,
)


def test_variables_validation_and_helpers():
    with pytest.raises(ValueError, match="unique"):
        Variables(names=("X", "X"), domains={"X": (0, 1)})
    with pytest.raises(ValueError, match="missing domain"):
        Variables(names=("X",), domains={})
    with pytest.raises(ValueError, match="empty domain"):
        Variables(names=("X",), domains={"X": ()})
    with pytest.raises(ValueError, match="observed unknowns"):
        Variables.from_binary("X", "Y", observed=["Z"])

    V = Variables.from_mapping({"X": (0, 1), "Y": ("a", "b")})
    assert V.size("Y") == 2
    assert V.index("Y") == 1
    assert "X" in V and len(V) == 2
    assert V.cell_index((1, "b")) == 3
    with pytest.raises(KeyError):
        V.cell_index((9, "b"))
    R = V.restrict(["Y"])
    assert R.names == ("Y",)


def test_from_counts_error_paths_and_exact_probs():
    with pytest.raises(ValueError, match="non-empty"):
        from_counts((), {})
    with pytest.raises(ValueError, match="empty"):
        from_counts(("X",), {})
    with pytest.raises(ValueError, match="length"):
        from_counts(("X", "Y"), {(0,): 1})
    with pytest.raises(ValueError, match="negative"):
        from_counts(("X",), {(0,): -1})
    with pytest.raises(ValueError, match="domains missing"):
        from_counts(("X",), {(0,): 1}, domains={"Y": (0,)})
    data = from_counts(("X",), {(0,): 0.4, (1,): 0.6}, normalize=False)
    assert abs(sum(data.P.values()) - 1.0) < 1e-12
    # near-1 mass renormalizes quietly
    near = from_counts(("X",), {(0,): 0.5000000001, (1,): 0.5})
    assert abs(sum(near.P.values()) - 1.0) < 1e-12


def test_from_records_errors():
    with pytest.raises(ValueError, match="non-empty"):
        from_records((), [])
    with pytest.raises(ValueError, match="empty"):
        from_records(("X",), [])
    with pytest.raises(ValueError, match="row 0"):
        from_records(("X", "Y"), [(0,)])


def test_query_factories_describe():
    assert "do(X=1)" in Query.interventional(outcome="Y", do={"X": 1}, value=1).describe()
    assert "ATE" in Query.ate().describe()
    assert "PNS" in Query.pns().describe()
    assert "PN" in Query.pn().describe()
    assert "PS" in Query.ps().describe()
    from partialid import Atom, Lit

    q = Query.conditional(Atom(Lit("Y", 1)), Atom(Lit("X", 1)))
    assert "cond" in q.kind or "P" in q.describe()


def test_assumption_validation_errors():
    with pytest.raises(ValueError, match="direction"):
        Assumptions.mtr("Y", "X", direction="sideways")
    with pytest.raises(ValueError, match="direction"):
        Assumptions.mts("X", "Y", direction="sideways")
    with pytest.raises(ValueError, match="c-dependence"):
        Assumptions.c_dependence("X", -0.1)
    with pytest.raises(ValueError, match="Gamma"):
        Assumptions.rosenbaum_gamma("X", 0.5)
    with pytest.raises(NotImplementedError):
        Assumptions.rosenbaum_gamma("X", 2.0, anchored=False)
    with pytest.raises(ValueError, match="kappa"):
        Assumptions.margin_budget(-1)
    with pytest.raises(ValueError, match="rho"):
        Assumptions.f_sensitivity("X", -1)
    with pytest.raises(ValueError, match="f must"):
        Assumptions.f_sensitivity("X", 0.1, f="weird")
    with pytest.raises(ValueError, match="support"):
        Assumptions.support_bound(0)
    with pytest.raises(ValueError, match="duplicate"):
        Assumptions.manski("X", "X")
    with pytest.raises(ValueError, match="three distinct"):
        Assumptions.iv("Z", "Z", "Y")


def test_assumption_set_diagram_rules():
    a = Assumptions.order("X", "Y") & Assumptions.diagram(["X", "Y"], directed=[("X", "Y")])
    with pytest.raises(ValueError, match="either a diagram or an order"):
        a.diagram()
    with pytest.raises(ValueError, match="at most one Order"):
        (Assumptions.order("X", "Y") & Assumptions.order("Y", "X")).diagram()


def test_query_expr_from_kind_and_conditional():
    q = Query(kind="interventional", outcome="Y", value=1, intervention={"X": 1}, expr=None)
    assert q.expr is not None
    assert "do(X=1)" in q.describe()
    q2 = Query(
        kind="ate",
        outcome="Y",
        value=1,
        intervention={},
        meta={"treatment": "X", "y": 1, "t1": 1, "t0": 0},
        expr=None,
    )
    assert q2.expr is not None
    for kind in ("pns", "pn", "ps"):
        qk = Query(
            kind=kind,
            outcome="Y",
            value=1,
            intervention={},
            meta={"treatment": "X"},
            expr=None,
        )
        assert qk.expr is not None
    from partialid import Atom, Lit

    qc = Query.conditional(Atom(Lit("Y", 1)), Atom(Lit("X", 1)))
    assert qc.kind == "cond"


def test_from_dataframe_error_paths():
    try:
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"pandas unavailable: {e}")
    from partialid import from_dataframe

    with pytest.raises(TypeError):
        from_dataframe({"X": [0]}, columns=["X"])
    with pytest.raises(ValueError, match="missing columns"):
        from_dataframe(pd.DataFrame({"X": [0]}), columns=["Y"])
    with pytest.raises(ValueError, match="no rows"):
        from_dataframe(pd.DataFrame({"X": []}), columns=["X"])
    with pytest.raises(ValueError, match="missing values"):
        from_dataframe(pd.DataFrame({"X": [0, None]}), columns=["X"])


def test_variables_restrict_observed():
    V = Variables.from_binary("X", "Y", "U", observed=["X", "Y"])
    assert V.observed_names() == ("X", "Y")
    R = V.restrict(["X", "U"])
    assert R.names == ("X", "U")
