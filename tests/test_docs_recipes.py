"""CI checks for docs recipes and showcase notebooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from partialid import (
    Assumptions,
    CausalDiagram,
    Query,
    Variables,
    bound,
    bound_path,
    from_counts,
)

ROOT = Path(__file__).resolve().parents[1]


def test_recipe_manski():
    data = from_counts(
        ("X", "Y"),
        {(0, 0): 30, (0, 1): 20, (1, 0): 15, (1, 1): 35},
    )
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r = bound(data.variables, data.P, Q, Assumptions.manski("X", "Y"))
    assert r.success and r.width > 0


def test_recipe_msm_path():
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
    _, _, w = path.at(3.0)
    assert abs(w - 1 / 3) < 1e-3


def test_recipe_iv_monotone():
    from partialid.problems import binary_iv

    p = binary_iv(seed=2)
    r = bound(p.variables, p.data, p.query, Assumptions.iv("Z", "X", "Y"))
    assert r.success and r.tight
    # no-defiers can be infeasible for this SCM sample — document that path
    r_m = bound(
        p.variables,
        p.data,
        p.query,
        Assumptions.iv("Z", "X", "Y", monotone=True),
    )
    assert r_m.success is False


def test_recipe_margin_budget():
    V = Variables.from_binary("X", "Y")
    P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r = bound(V, P, Q, Assumptions.manski("X", "Y") & Assumptions.margin_budget(0.05))
    assert r.success and r.tight


def test_recipe_measurement_error():
    V = Variables.from_binary("X", "Y", "Y2", observed=["X", "Y2"])
    g = CausalDiagram.from_edges(
        ["X", "Y", "Y2"],
        directed=[("X", "Y"), ("Y", "Y2")],
        bidirected=[("Y", "Y2")],
        maximal_confounding=False,
    )
    P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    Q = Query.ate(outcome="Y", treatment="X", y=1, t1=1, t0=0)
    a = Assumptions.from_diagram(g) & Assumptions.mtr("Y2", "Y")
    r = bound(V, P, Q, a)
    assert r.success and r.tight


def _exec_notebook(path: Path) -> None:
    """Execute code cells in an ``.ipynb`` without requiring Jupyter."""
    import json

    nb = json.loads(path.read_text())
    ns: dict = {"__name__": "__main__", "__file__": str(path)}
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if not src.strip():
            continue
        try:
            exec(compile(src, f"{path.name}:cell{i}", "exec"), ns)
        except Exception as exc:  # pragma: no cover - surfaced in pytest
            raise AssertionError(f"{path.name} cell {i} failed: {exc}") from exc


@pytest.mark.parametrize(
    "name",
    sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb")),
)
def test_notebook_runs(name: str):
    """All Jupyter notebooks under notebooks/ must execute cleanly."""
    _exec_notebook(ROOT / "notebooks" / name)

@pytest.mark.skipif(
    __import__("importlib").util.find_spec("mkdocs") is None,
    reason="mkdocs not installed",
)
def test_mkdocs_build(tmp_path):
    import subprocess
    import sys

    site = tmp_path / "site"
    subprocess.check_call(
        [sys.executable, "-m", "mkdocs", "build", "-d", str(site), "-q"],
        cwd=ROOT,
    )
    assert (site / "index.html").is_file()
