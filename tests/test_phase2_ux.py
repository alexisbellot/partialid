"""Phase 2 practitioner UX: data ingress, recipes, summaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from partialid import (
    Assumptions,
    Query,
    Variables,
    bound,
    bound_path,
    from_counts,
    from_records,
)


def test_from_counts_normalizes_and_infers_domains():
    data = from_counts(("X", "Y"), {(0, 0): 3, (0, 1): 2, (1, 0): 1, (1, 1): 4})
    assert abs(sum(data.P.values()) - 1.0) < 1e-12
    assert data.variables.domains["X"] == (0, 1)
    assert data.notes and "normalized" in data.notes[0]


def test_from_counts_rejects_bad_mass_without_normalize():
    with pytest.raises(ValueError, match="normalize=True"):
        from_counts(("X",), {(0,): 2.0}, normalize=False)


def test_from_records_empirical():
    data = from_records(("X", "Y"), [(0, 0), (0, 1), (1, 1), (1, 1)])
    assert data.P[(1, 1)] == pytest.approx(0.5)
    assert abs(sum(data.P.values()) - 1.0) < 1e-12


def test_from_dataframe_optional():
    try:
        import pandas as pd
    except Exception as e:  # noqa: BLE001 — optional extra / ABI skew
        pytest.skip(f"pandas unavailable: {e}")
    from partialid import from_dataframe

    df = pd.DataFrame({"X": [0, 1, 1], "Y": [0, 0, 1]})
    data = from_dataframe(df, columns=["X", "Y"])
    assert data.n_obs == 3
    assert data.P[(1, 0)] == pytest.approx(1 / 3)


def test_recipes_manski_msm_iv():
    a = Assumptions.manski("X", "Y")
    assert len(a) == 1
    b = Assumptions.sensitivity_msm("X", "Y", 3.0)
    assert len(b) == 2
    iv = Assumptions.iv("Z", "X", "Y", monotone=True)
    assert len(iv) == 2
    assert len(Assumptions.iv("Z", "X", "Y")) == 1
    with pytest.raises(ValueError, match="Lambda"):
        Assumptions.sensitivity_msm("X", "Y", 0.5)


def test_bound_result_summary_and_to_dict():
    V = Variables.from_binary("X", "Y")
    P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r = bound(V, P, Q, Assumptions.sensitivity_msm("X", "Y", 3.0))
    text = r.summary(Q)
    assert "interval:" in text and "tight:" in text
    d = r.to_dict()
    assert d["width"] == pytest.approx(r.width)
    assert abs(r.width - 1 / 3) < 1e-3


def test_sensitivity_path_one_liner():
    V = Variables.from_binary("X", "Y")
    P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    path = bound_path(
        V,
        P,
        Q,
        Assumptions.manski("X", "Y"),
        param="Lambda",
        theta_min=1.0,
        theta_max=5.0,
    )
    lo, hi, w = path.at(3.0)
    assert w == pytest.approx(1 / 3, rel=1e-3)
    assert "sensitivity path" in path.summary(at=3.0)


def test_notebook_csv_bundle():
    csv = Path(__file__).resolve().parents[1] / "notebooks" / "data" / "job_training.csv"
    assert csv.is_file()
    import pandas as pd

    from partialid import from_dataframe

    data = from_dataframe(pd.read_csv(csv), columns=["treated", "employed"])
    Q = Query.interventional(outcome="employed", do={"treated": 1}, value=1)
    r = bound(
        data.variables,
        data.P,
        Q,
        Assumptions.order("treated", "employed"),
    )
    assert r.success and r.width > 0
