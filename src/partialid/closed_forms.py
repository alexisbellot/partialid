"""Analytic / closed-form bounds (THEORY Moves) without calling HiGHS.

``try_closed_form`` returns a result when a recognized pattern matches;
otherwise ``None`` so callers fall through to LP / fiber / GPP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .assumptions import (
    AssumptionSet,
    DiagramAssumption,
    NoDefiersAssumption,
    OrderAssumption,
    RosenbaumAssumption,
    TanMSMAssumption,
    as_assumption_set,
)
from .causaloptim_formulas import balke_pearl_iv, manski_binary_xy
from .graph import CausalDiagram
from .query import Query
from .variables import Variables

_RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


@dataclass
class ClosedFormResult:
    lower: float
    upper: float
    formula_id: str
    notes: list[str] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.upper - self.lower


def _xy_binary_p(variables: Variables, data: Mapping) -> dict[tuple[int, int], float] | None:
    if list(variables.names) != ["X", "Y"]:
        return None
    if variables.domains.get("X") != (0, 1) or variables.domains.get("Y") != (0, 1):
        return None
    out: dict[tuple[int, int], float] = {}
    for key, val in data.items():
        if isinstance(key, dict):
            cell = (int(key["X"]), int(key["Y"]))
        else:
            cell = (int(key[0]), int(key[1]))
        out[cell] = out.get(cell, 0.0) + float(val)
    s = sum(out.values())
    if s <= 0:
        return None
    return {k: v / s for k, v in out.items()}


def _matches_re1_table(P: Mapping[tuple, float], atol: float = 1e-9) -> bool:
    for cell, v in _RE1_P.items():
        if abs(float(P.get(cell, 0.0)) - v) > atol:
            return False
    return abs(sum(P.values()) - 1.0) < 1e-8


def msm_closed_form_re1(Lam: float) -> tuple[float, float, float]:
    """THEORY §8.7.3 Tan MSM width on the canonical RE1 table."""
    if Lam <= 7 / 3:
        w = 0.15 * (Lam - 1 / Lam)
        u = 0.85 - 0.15 / Lam
        ell = 0.85 - 0.15 * Lam
    else:
        w = 0.50 * (1 - 1 / Lam)
        u = 0.85 - 0.15 / Lam
        ell = 0.35 * (1 + 1 / Lam)
    return float(ell), float(u), float(w)


def _is_order_only_xy(assumptions: AssumptionSet) -> bool:
    """True only for pure order-only XY (no faces, budgets, or support)."""
    if assumptions.of_type(DiagramAssumption):
        return False
    orders = assumptions.of_type(OrderAssumption)
    if len(orders) != 1:
        return False
    if tuple(orders[0].order) != ("X", "Y"):
        return False
    # Exactly one atom: the order assumption
    return len(assumptions.atoms) == 1


def _classic_binary_iv(diagram: CausalDiagram | None) -> bool:
    if diagram is None or set(diagram.nodes) != {"Z", "X", "Y"}:
        return False
    di = set(diagram.directed)
    bi = {frozenset(e) for e in diagram.bidirected}
    return di == {("Z", "X"), ("X", "Y")} and bi == {frozenset(("X", "Y"))}


def try_closed_form(
    variables: Variables,
    data: Mapping,
    query: Query,
    assumptions: AssumptionSet | None = None,
) -> ClosedFormResult | None:
    """Return analytic bounds when a known pattern matches, else None."""
    assumptions = as_assumption_set(assumptions)
    if query.kind not in ("interventional", "ate"):
        return None

    msm = list(assumptions.of_type(TanMSMAssumption))
    ros = list(assumptions.of_type(RosenbaumAssumption))
    if (msm or ros) and query.kind == "interventional":
        if query.outcome == "Y" and dict(query.intervention) == {"X": 1} and query.value == 1:
            Pxy = _xy_binary_p(variables, data)
            if Pxy is not None and _matches_re1_table(Pxy):
                if msm and not ros:
                    Lam = float(msm[0].Lambda)
                    src = f"tan_msm_re1_L{Lam:g}"
                elif ros and not msm:
                    Lam = float(ros[0].Gamma) ** 0.5
                    src = f"anchored_rosenbaum_re1_G{ros[0].Gamma:g}"
                else:
                    return None
                extra = [
                    a
                    for a in assumptions.atoms
                    if not isinstance(
                        a,
                        (
                            OrderAssumption,
                            TanMSMAssumption,
                            RosenbaumAssumption,
                        ),
                    )
                ]
                if extra:
                    return None
                ell, u, _w = msm_closed_form_re1(Lam)
                return ClosedFormResult(
                    lower=ell,
                    upper=u,
                    formula_id=src,
                    notes=[f"closed_form:{src}", "THEORY §8.7.3"],
                )

    if query.kind == "interventional" and _classic_binary_iv(assumptions.diagram()):
        if (
            query.outcome == "Y"
            and dict(query.intervention) == {"X": 1}
            and query.value == 1
            and list(variables.names) == ["Z", "X", "Y"]
        ):
            if assumptions.of_type(NoDefiersAssumption):
                return None
            if any(not isinstance(a, DiagramAssumption) for a in assumptions.atoms):
                return None
            P_iv: dict[tuple, float] = {}
            for key, val in data.items():
                if isinstance(key, dict):
                    cell = (key["Z"], key["X"], key["Y"])
                else:
                    cell = tuple(key)
                P_iv[cell] = P_iv.get(cell, 0.0) + float(val)
            lo, hi = balke_pearl_iv(P_iv)
            if not (lo == lo and hi == hi):
                return None
            return ClosedFormResult(
                lower=float(lo),
                upper=float(hi),
                formula_id="balke_pearl_iv",
                notes=["closed_form:balke_pearl_iv", "causaloptim_formulas"],
            )

    if query.kind == "interventional" and _is_order_only_xy(assumptions):
        if (
            query.outcome == "Y"
            and len(query.intervention) == 1
            and "X" in query.intervention
            and query.value in (0, 1)
        ):
            Pxy = _xy_binary_p(variables, data)
            if Pxy is None:
                return None
            do_x = int(query.intervention["X"])
            y = int(query.value)
            lo, hi = manski_binary_xy(Pxy, do_x=do_x, y=y)
            return ClosedFormResult(
                lower=lo,
                upper=hi,
                formula_id="manski_binary_xy",
                notes=["closed_form:manski_binary_xy", "Move 1 / natural bounds"],
            )

    return None
