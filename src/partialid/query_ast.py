"""Unified linear / fractional query AST.

Every numeric query is eventually a real linear combination of *atomic*
events (unnested response-type atoms (Prop. 3 / Cor. 1)), optionally as a ratio for
conditionals (Correa, Lee & Bareinboim, NeurIPS 2021, §4).

Nested potential outcomes are accepted here as syntax; the compiler
unnests them via the Counterfactual Unnesting Theorem (CUT; NeurIPS 2021
Thm 4) before building the LP objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence


@dataclass(frozen=True)
class PO:
    """Potential outcome ``var`` under an assignment (values may nest ``PO``)."""

    var: str
    assignment: Mapping[str, object]  # value: constant | PO

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignment", dict(self.assignment))

    def is_flat(self) -> bool:
        return all(not isinstance(v, PO) for v in self.assignment.values())

    def describe(self) -> str:
        if not self.assignment:
            return self.var
        parts = []
        for k, v in self.assignment.items():
            if isinstance(v, PO):
                parts.append(f"{k}={v.describe()}")
            else:
                parts.append(f"{k}={v}")
        return f"{self.var}({', '.join(parts)})"


@dataclass(frozen=True)
class Lit:
    """Equality literal: factual ``var=value`` or ``PO(...)=value``."""

    expr: str | PO
    value: object

    def describe(self) -> str:
        if isinstance(self.expr, PO):
            return f"{self.expr.describe()}={self.value}"
        return f"{self.expr}={self.value}"


@dataclass(frozen=True)
class Atom:
    """Conjunction of literals (one atomic probability event)."""

    lits: tuple[Lit, ...]

    def __init__(self, *lits: Lit) -> None:
        object.__setattr__(self, "lits", tuple(lits))

    def describe(self) -> str:
        if not self.lits:
            return "1"
        return " ∧ ".join(lit.describe() for lit in self.lits)


@dataclass(frozen=True)
class LinComb:
    """Real linear combination of atoms: ``sum coeff_i * P(atom_i)``."""

    terms: tuple[tuple[float, Atom], ...]

    def __init__(self, terms: Sequence[tuple[float, Atom]] | None = None) -> None:
        object.__setattr__(self, "terms", tuple(terms or ()))

    @classmethod
    def atom(cls, atom: Atom, coeff: float = 1.0) -> LinComb:
        return cls([(coeff, atom)])

    def __add__(self, other: LinComb) -> LinComb:
        return LinComb(self.terms + other.terms)

    def __sub__(self, other: LinComb) -> LinComb:
        return self + (-1.0) * other

    def __rmul__(self, scalar: float) -> LinComb:
        return LinComb([(float(scalar) * c, a) for c, a in self.terms])

    def __mul__(self, scalar: float) -> LinComb:
        return float(scalar) * self

    def describe(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for c, a in self.terms:
            ad = f"P({a.describe()})"
            if abs(c - 1.0) < 1e-15:
                parts.append(ad)
            elif abs(c + 1.0) < 1e-15:
                parts.append(f"-{ad}")
            else:
                parts.append(f"{c:g}*{ad}")
        out = parts[0]
        for p in parts[1:]:
            if p.startswith("-"):
                out += f" - {p[1:]}"
            else:
                out += f" + {p}"
        return out

    def iter_atoms(self) -> Iterator[Atom]:
        for _, a in self.terms:
            yield a


@dataclass(frozen=True)
class Cond:
    """Conditional ``P(numer | denom) = P(numer ∧ denom) / P(denom)``.

    Justification: Correa, Lee & Bareinboim (NeurIPS 2021) §4 reduce
    conditional counterfactuals to ratios of marginals. When ``denom`` is
    an observational event with known mass under ``P``, the ratio is
    fixed-denominator linear (as in Tian–Pearl PN/PS). Otherwise the
    objective is linear-fractional in the type distribution ``q``.
    """

    numer: LinComb
    denom: LinComb

    def describe(self) -> str:
        return f"({self.numer.describe()}) / ({self.denom.describe()})"


Expr = LinComb | Cond


def conjunct_atoms(a: Atom, b: Atom) -> Atom:
    """Conjunction of two atoms (merge literals)."""
    return Atom(*(a.lits + b.lits))
