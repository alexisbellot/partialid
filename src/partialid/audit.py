"""Assumption / solve audit reports (P3 workflow layer)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .api import BoundResult
from .assumptions import (
    AssumptionSet,
    CDependenceAssumption,
    DiagramAssumption,
    ExclusionAssumption,
    FSensitivityAssumption,
    IndependenceAssumption,
    MarginBudgetAssumption,
    MonotonicityAssumption,
    MTSAssumption,
    NoDefiersAssumption,
    OrderAssumption,
    RosenbaumAssumption,
    SupportBoundAssumption,
    TanMSMAssumption,
    as_assumption_set,
)
from .query import Query


@dataclass
class AuditReport:
    """Structured summary of assumptions, query, and bound result."""

    query: str
    atoms: list[dict[str, Any]] = field(default_factory=list)
    faces: list[str] = field(default_factory=list)
    budgets: list[str] = field(default_factory=list)
    structure: list[str] = field(default_factory=list)
    method: str | None = None
    tight: bool | None = None
    success: bool | None = None
    lower: float | None = None
    upper: float | None = None
    width: float | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outer: bool = False
    cuts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        bits = [
            f"query={self.query!r}",
            f"method={self.method!r}",
            f"tight={self.tight}",
            f"faces={self.faces}",
            f"budgets={self.budgets}",
        ]
        if self.lower is not None:
            bits.append(f"interval=[{self.lower:.4g},{self.upper:.4g}]")
        return "AuditReport(" + ", ".join(bits) + ")"


def _atom_dict(atom) -> dict[str, Any]:
    d = {"type": type(atom).__name__}
    for k, v in getattr(atom, "__dict__", {}).items():
        if k.startswith("_"):
            continue
        if hasattr(v, "nodes"):  # CausalDiagram
            d[k] = {
                "nodes": list(v.nodes),
                "directed": list(v.directed),
                "bidirected": list(v.bidirected),
            }
        else:
            d[k] = v
    return d


def audit_assumptions(assumptions: AssumptionSet | None) -> AuditReport:
    """Classify assumption atoms into structure / faces / budgets."""
    a = as_assumption_set(assumptions)
    report = AuditReport(query="")
    for atom in a.atoms:
        report.atoms.append(_atom_dict(atom))
        name = type(atom).__name__
        if isinstance(atom, (OrderAssumption, DiagramAssumption, IndependenceAssumption)):
            report.structure.append(name)
        elif isinstance(
            atom,
            (
                MonotonicityAssumption,
                MTSAssumption,
                NoDefiersAssumption,
                ExclusionAssumption,
                SupportBoundAssumption,
            ),
        ):
            report.faces.append(name)
        elif isinstance(
            atom,
            (
                TanMSMAssumption,
                CDependenceAssumption,
                RosenbaumAssumption,
                MarginBudgetAssumption,
                FSensitivityAssumption,
            ),
        ):
            report.budgets.append(name)
        else:
            report.structure.append(name)
    return report


def audit_bound(
    result: BoundResult,
    *,
    query: Query | None = None,
    assumptions: AssumptionSet | None = None,
) -> AuditReport:
    """Combine assumption taxonomy with a ``BoundResult``."""
    base = audit_assumptions(assumptions) if assumptions is not None else AuditReport(query="")
    if query is not None:
        base.query = query.describe()
    base.method = result.method
    base.tight = result.tight
    base.success = result.success
    base.lower = float(result.lower) if result.success else None
    base.upper = float(result.upper) if result.success else None
    base.width = float(result.width) if result.success else None
    base.notes = list(result.notes)
    base.warnings = list(result.warnings)
    base.outer = (not result.tight) or any("outer" in n for n in result.notes)
    for n in result.notes:
        if "Sachs" in n or "inert prune" in n or "instrument LP" in n or "c-degree" in n:
            base.cuts.append(n)
    return base
