"""partialid — partial identification for discrete structural causal models.

Quick start
-----------
>>> from partialid import Variables, Query, Assumptions, bound
>>> V = Variables.from_binary("X", "Y")
>>> P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
>>> Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
>>> a = Assumptions.order("X", "Y") & Assumptions.tan_msm(treatment="X", Lambda=3.0)
>>> r = bound(V, P, Q, a)
>>> round(r.width, 3)
0.333
"""

from .api import BoundResult, bound, bound_grid
from .assumptions import Assumptions, AssumptionSet
from .audit import AuditReport, audit_assumptions, audit_bound
from .closed_forms import ClosedFormResult, try_closed_form
from .data import ObservationalData, from_counts, from_dataframe, from_records
from .graph import CausalDiagram
from .inference import BoundCIResult, bound_ci
from .parametric import BoundPathResult, bound_path
from .problems import Problem, all_literature_problems
from .query import Query
from .query_ast import PO, Atom, Cond, LinComb, Lit
from .variables import Variables

__all__ = [
    "Variables",
    "CausalDiagram",
    "Query",
    "PO",
    "Lit",
    "Atom",
    "LinComb",
    "Cond",
    "Assumptions",
    "AssumptionSet",
    "ObservationalData",
    "from_counts",
    "from_records",
    "from_dataframe",
    "bound",
    "bound_grid",
    "bound_path",
    "bound_ci",
    "BoundResult",
    "BoundPathResult",
    "BoundCIResult",
    "AuditReport",
    "audit_assumptions",
    "audit_bound",
    "ClosedFormResult",
    "try_closed_form",
    "Problem",
    "all_literature_problems",
]


def _package_version() -> str:
    """Single source of truth: the installed distribution version (pyproject)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return "0.0.0+local"
    try:
        return version("partialid")
    except PackageNotFoundError:  # pragma: no cover
        return "0.0.0+local"


__version__ = _package_version()
