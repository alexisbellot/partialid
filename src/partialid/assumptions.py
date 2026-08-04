"""Assumption taxonomy — the public face of the library.

Assumptions compose with ``&`` (meet / intersection of constraint sets):

>>> a = Assumptions.order("X", "Y") & Assumptions.mtr("Y", "X") & Assumptions.tan_msm("X", 3)

Every named factory maps onto the two-axis taxonomy (face constraints vs
sensitivity budgets).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .graph import CausalDiagram

# ---------------------------------------------------------------------------
# Atomic assumption records (data only; semantics live in formulate/)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderAssumption:
    """Topological order under one latent on all listed nodes (Manski)."""

    order: tuple[str, ...]


@dataclass(frozen=True)
class DiagramAssumption:
    diagram: CausalDiagram


@dataclass(frozen=True)
class ExclusionAssumption:
    """Variable ``child`` does not depend on ``parent`` (missing directed edge)."""

    child: str
    parent: str


@dataclass(frozen=True)
class MonotonicityAssumption:
    """Response of ``outcome`` is monotone nondecreasing in ``treatment``."""

    outcome: str
    treatment: str
    direction: str = "increasing"  # or "decreasing"


@dataclass(frozen=True)
class MTSAssumption:
    """Monotone treatment selection (Manski–Pepper).

    For each treatment level ``x``, ``E[Y_x | X=1] ≥ E[Y_x | X=0]``
    (or the reverse if ``direction=\"decreasing\"``). Linear in type masses
    when treatment is a binary root.
    """

    treatment: str
    outcome: str
    direction: str = "increasing"


@dataclass(frozen=True)
class NoDefiersAssumption:
    """Compliance monotonicity: no defiers for instrument → treatment."""

    treatment: str
    instrument: str


@dataclass(frozen=True)
class TanMSMAssumption:
    """Anchored Tan MSM via relative-propensity likelihood ratios.

    Constrains
    ``Λ⁻¹ ≤ [π(x|r,pa)/π(x'|r,pa)] / [π₀(x|pa)/π₀(x'|pa)] ≤ Λ``.
    Binary treatment recovers classical odds MSM; with parents ``Pa`` of the
    treatment the budget is conditional on ``pa``.
    """

    treatment: str
    Lambda: float
    anchored: bool = True


@dataclass(frozen=True)
class CDependenceAssumption:
    """Masten–Poirier c-dependence: ‖π(·|r,pa) − π₀(·|pa)‖_∞ ≤ c (linear).

    Binary treatment recovers ``|e(r) − e₀| ≤ c``; conditional on parents when
    the diagram gives ``Pa(treatment)``.
    """

    treatment: str
    c: float


@dataclass(frozen=True)
class RosenbaumAssumption:
    """Rosenbaum Γ. If ``anchored=True``, reduces to Tan MSM with Λ=√Γ."""

    treatment: str
    Gamma: float
    anchored: bool = True


@dataclass(frozen=True)
class MarginBudgetAssumption:
    """Soft independence: |S - S0| ≤ κ on the identifying margin of the query.

    ``reference`` is a named hard pole: currently ``\"point\"`` (S0 from
    no-confounding / g-formula when available) or a float S0 directly via
    ``s0``.
    """

    kappa: float
    s0: float | None = None
    reference: str = "point"


@dataclass(frozen=True)
class FSensitivityAssumption:
    """Anchored f-sensitivity: Σ D_f(π(·|r,pa) ‖ π₀(·|pa)) ≤ ρ (convex).

    Uses the same conditional / categorical propensity as Tan MSM. ``f`` selects
    the generator: ``\"kl\"``, ``\"chi2\"``, or ``\"tv\"`` (LP-representable).
    Binary root recovers Jin–Ren–Candès.
    """

    treatment: str
    rho: float
    f: str = "kl"  # "kl" | "chi2" | "tv"


@dataclass(frozen=True)
class SupportBoundAssumption:
    """At most ``k`` latent types have positive mass (union of faces; MIP)."""

    k: int


@dataclass(frozen=True)
class IndependenceAssumption:
    """Hard independence / c-split between groups.

    Sharp factorization of the product program is not implemented in v1: formulate
    forces a merged-latent **outer** LP (``tight=False``). Prefer ID / diagram
    structure that already separates c-components when you need sharp bounds.
    """

    groups: tuple[tuple[str, ...], ...]


AssumptionAtom = (
    OrderAssumption
    | DiagramAssumption
    | ExclusionAssumption
    | MonotonicityAssumption
    | MTSAssumption
    | NoDefiersAssumption
    | TanMSMAssumption
    | CDependenceAssumption
    | RosenbaumAssumption
    | MarginBudgetAssumption
    | FSensitivityAssumption
    | SupportBoundAssumption
    | IndependenceAssumption
)


@dataclass(frozen=True)
class AssumptionSet:
    """Composable set of atomic assumptions."""

    atoms: tuple[AssumptionAtom, ...] = ()

    def __and__(self, other: AssumptionSet | AssumptionAtom) -> AssumptionSet:
        if isinstance(other, AssumptionSet):
            return AssumptionSet(self.atoms + other.atoms)
        return AssumptionSet(self.atoms + (other,))

    def __rand__(self, other: AssumptionAtom) -> AssumptionSet:
        return AssumptionSet((other,) + self.atoms)

    def __iter__(self):
        return iter(self.atoms)

    def __len__(self) -> int:
        return len(self.atoms)

    def of_type(self, cls: type) -> tuple:
        return tuple(a for a in self.atoms if isinstance(a, cls))

    def diagram(self) -> CausalDiagram | None:
        """Resolve structural diagram from order / explicit diagram atoms."""
        diags = self.of_type(DiagramAssumption)
        orders = self.of_type(OrderAssumption)
        if diags and orders:
            raise ValueError("specify either a diagram or an order, not both")
        if len(diags) > 1:
            raise ValueError("at most one DiagramAssumption allowed")
        if len(orders) > 1:
            raise ValueError("at most one OrderAssumption allowed")
        if diags:
            return diags[0].diagram
        if orders:
            return CausalDiagram.from_order(orders[0].order)
        return None

    def describe(self) -> list[str]:
        out = []
        for a in self.atoms:
            out.append(f"{type(a).__name__}: {a}")
        return out


class Assumptions:
    """Factory namespace for named literature assumptions.

    Compose with ``&`` (intersection of constraint sets). See the docs
    Assumptions catalog for the mapping onto common PI / sensitivity models.
    """

    @staticmethod
    def empty() -> AssumptionSet:
        return AssumptionSet()

    @staticmethod
    def order(*variables: str) -> AssumptionSet:
        """Known topological order with one latent on all variables (Manski).

        Encoded as ``bidirected=[tuple(sorted(variables))]``. For a Markovian
        order (no confounding), use ``diagram(..., bidirected=[])``.
        """
        return AssumptionSet((OrderAssumption(order=tuple(variables)),))

    @staticmethod
    def diagram(
        nodes: Sequence[str],
        directed: Iterable[tuple[str, str]] = (),
        bidirected: Iterable[Sequence[str]] = (),
    ) -> AssumptionSet:
        g = CausalDiagram.from_edges(nodes, directed, bidirected)
        return AssumptionSet((DiagramAssumption(diagram=g),))

    @staticmethod
    def from_diagram(diagram: CausalDiagram) -> AssumptionSet:
        return AssumptionSet((DiagramAssumption(diagram=diagram),))

    @staticmethod
    def exclusion(child: str, parent: str) -> AssumptionSet:
        """Missing directed edge: ``child`` does not depend on ``parent``."""
        return AssumptionSet((ExclusionAssumption(child=child, parent=parent),))

    @staticmethod
    def mtr(
        outcome: str,
        treatment: str,
        *,
        direction: str = "increasing",
    ) -> AssumptionSet:
        """Monotone treatment response."""
        if direction not in ("increasing", "decreasing"):
            raise ValueError("direction must be 'increasing' or 'decreasing'")
        return AssumptionSet(
            (MonotonicityAssumption(outcome=outcome, treatment=treatment, direction=direction),)
        )

    @staticmethod
    def mts(
        treatment: str,
        outcome: str,
        *,
        direction: str = "increasing",
    ) -> AssumptionSet:
        """Monotone treatment selection (Manski–Pepper)."""
        if direction not in ("increasing", "decreasing"):
            raise ValueError("direction must be 'increasing' or 'decreasing'")
        return AssumptionSet(
            (MTSAssumption(treatment=treatment, outcome=outcome, direction=direction),)
        )

    @staticmethod
    def no_defiers(treatment: str, instrument: str) -> AssumptionSet:
        return AssumptionSet((NoDefiersAssumption(treatment=treatment, instrument=instrument),))

    @staticmethod
    def tan_msm(
        treatment: str,
        Lambda: float,
        *,
        anchored: bool = True,
    ) -> AssumptionSet:
        """Relative-propensity LR MSM (classical Tan odds when treatment is binary)."""
        if Lambda < 1:
            raise ValueError("Tan MSM requires Lambda >= 1")
        if not anchored:
            raise NotImplementedError(
                "Unanchored Tan MSM is bilinear (not LP). Use anchored=True or a future NLP path."
            )
        return AssumptionSet(
            (TanMSMAssumption(treatment=treatment, Lambda=float(Lambda), anchored=True),)
        )

    @staticmethod
    def c_dependence(treatment: str, c: float) -> AssumptionSet:
        """ℓ∞ c-dependence on the (conditional) treatment propensity."""
        if c < 0:
            raise ValueError("c-dependence requires c >= 0")
        return AssumptionSet((CDependenceAssumption(treatment=treatment, c=float(c)),))

    @staticmethod
    def rosenbaum_gamma(
        treatment: str,
        Gamma: float,
        *,
        anchored: bool = True,
    ) -> AssumptionSet:
        if Gamma < 1:
            raise ValueError("Rosenbaum Gamma requires Gamma >= 1")
        if not anchored:
            raise NotImplementedError(
                "Unanchored Rosenbaum Γ is bilinear (not LP). "
                "Use anchored=True (Λ=√Γ Tan MSM) or a future NLP path."
            )
        return AssumptionSet(
            (RosenbaumAssumption(treatment=treatment, Gamma=float(Gamma), anchored=True),)
        )

    @staticmethod
    def margin_budget(
        kappa: float,
        *,
        s0: float | None = None,
        reference: str = "point",
    ) -> AssumptionSet:
        if kappa < 0:
            raise ValueError("kappa must be >= 0")
        return AssumptionSet(
            (MarginBudgetAssumption(kappa=float(kappa), s0=s0, reference=reference),)
        )

    @staticmethod
    def f_sensitivity(
        treatment: str,
        rho: float,
        *,
        f: str = "kl",
    ) -> AssumptionSet:
        """Anchored f-sensitivity on π(·|r,pa) vs π₀(·|pa) (convex; categorical / conditional)."""
        if rho < 0:
            raise ValueError("f-sensitivity requires rho >= 0")
        if f not in ("kl", "chi2", "tv"):
            raise ValueError("f must be 'kl', 'chi2', or 'tv'")
        return AssumptionSet((FSensitivityAssumption(treatment=treatment, rho=float(rho), f=f),))

    @staticmethod
    def support_bound(k: int) -> AssumptionSet:
        if k < 1:
            raise ValueError("support bound k must be >= 1")
        return AssumptionSet((SupportBoundAssumption(k=k),))

    @staticmethod
    def independence(*groups: Sequence[str]) -> AssumptionSet:
        """Hard c-split / independence between groups of variables.

        v1 returns a certified **outer** approximation (``tight=False``); the
        sharp product factorization is not solved.
        """
        if len(groups) < 2:
            raise ValueError("independence requires at least two groups")
        return AssumptionSet((IndependenceAssumption(groups=tuple(tuple(g) for g in groups)),))

    # ------------------------------------------------------------------
    # Practitioner recipes (thin wrappers over atomic factories)
    # ------------------------------------------------------------------

    @staticmethod
    def manski(
        treatment: str,
        outcome: str,
        *covariates: str,
    ) -> AssumptionSet:
        """Order-only Manski regime: maximal confounding along ``treatment → … → outcome``.

        Equivalent to ``Assumptions.order(treatment, *covariates, outcome)``.
        """
        order = (treatment, *covariates, outcome)
        if len(set(order)) != len(order):
            raise ValueError(f"manski order has duplicate names: {order}")
        return Assumptions.order(*order)

    @staticmethod
    def sensitivity_msm(
        treatment: str,
        outcome: str,
        Lambda: float,
        *covariates: str,
    ) -> AssumptionSet:
        """Manski order plus anchored Tan MSM with budget ``Lambda`` (≥ 1).

        One-liner for the common sensitivity workflow::

            Assumptions.sensitivity_msm("X", "Y", Lambda=3.0)
        """
        return Assumptions.manski(treatment, outcome, *covariates) & Assumptions.tan_msm(
            treatment, Lambda
        )

    @staticmethod
    def iv(
        instrument: str,
        treatment: str,
        outcome: str,
        *,
        monotone: bool = False,
    ) -> AssumptionSet:
        """Classic binary IV diagram ``Z→X→Y`` with latent confounding ``X↔Y``.

        Parameters
        ----------
        monotone :
            If True, add compliance monotonicity (``no_defiers``). Default False
            because many IV laws are infeasible under no-defiers; opt in when
            you claim monotonicity.
        """
        names = (instrument, treatment, outcome)
        if len(set(names)) != 3:
            raise ValueError(
                f"iv requires three distinct names, got instrument={instrument!r}, "
                f"treatment={treatment!r}, outcome={outcome!r}"
            )
        g = CausalDiagram.from_edges(
            nodes=names,
            directed=((instrument, treatment), (treatment, outcome)),
            bidirected=((treatment, outcome),),
        )
        a = Assumptions.from_diagram(g)
        if monotone:
            a = a & Assumptions.no_defiers(treatment, instrument)
        return a


def as_assumption_set(obj: AssumptionSet | AssumptionAtom | None) -> AssumptionSet:
    if obj is None:
        return AssumptionSet()
    if isinstance(obj, AssumptionSet):
        return obj
    return AssumptionSet((obj,))
