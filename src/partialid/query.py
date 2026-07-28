"""Causal queries — unified AST with named factories as sugar.

All queries compile to :class:`~partialid.query_ast.LinComb` or
:class:`~partialid.query_ast.Cond` (see :mod:`partialid.query_compile`).
Named constructors remain for ergonomics and pattern-matching fast paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .query_ast import PO, Atom, Cond, Expr, LinComb, Lit


def _interventional_expr(outcome: str, do: Mapping[str, object], value: object) -> LinComb:
    return LinComb.atom(Atom(Lit(PO(outcome, dict(do)), value)))


def _ate_expr(outcome: str, treatment: str, y: object, t1: object, t0: object) -> LinComb:
    return _interventional_expr(outcome, {treatment: t1}, y) - _interventional_expr(
        outcome, {treatment: t0}, y
    )


def _pns_expr(treatment: str, outcome: str) -> LinComb:
    return LinComb.atom(
        Atom(
            Lit(PO(outcome, {treatment: 1}), 1),
            Lit(PO(outcome, {treatment: 0}), 0),
        )
    )


def _pn_expr(treatment: str, outcome: str) -> Cond:
    # P(Y_0=0 | X=1, Y=1) = P(Y_0=0, X=1, Y=1) / P(X=1, Y=1)
    numer = LinComb.atom(
        Atom(
            Lit(PO(outcome, {treatment: 0}), 0),
            Lit(treatment, 1),
            Lit(outcome, 1),
        )
    )
    denom = LinComb.atom(Atom(Lit(treatment, 1), Lit(outcome, 1)))
    return Cond(numer, denom)


def _ps_expr(treatment: str, outcome: str) -> Cond:
    numer = LinComb.atom(
        Atom(
            Lit(PO(outcome, {treatment: 1}), 1),
            Lit(treatment, 0),
            Lit(outcome, 0),
        )
    )
    denom = LinComb.atom(Atom(Lit(treatment, 0), Lit(outcome, 0)))
    return Cond(numer, denom)


@dataclass(frozen=True)
class Query:
    """A scalar causal query.

    Prefer building via factories or :meth:`from_expr`. The fields
    ``kind`` / ``outcome`` / ``value`` / ``intervention`` / ``meta`` are
    retained for solver fast-path pattern matching; ``expr`` is the
    source of truth for the LP objective compiler.
    """

    kind: str
    outcome: str
    value: object
    intervention: Mapping[str, object]
    meta: Mapping[str, object] | None = None
    expr: Expr | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "intervention", dict(self.intervention))
        if self.meta is not None:
            object.__setattr__(self, "meta", dict(self.meta))
        if self.expr is None:
            object.__setattr__(self, "expr", self._expr_from_kind())

    def _expr_from_kind(self) -> Expr:
        if self.kind == "interventional":
            return _interventional_expr(self.outcome, self.intervention, self.value)
        if self.kind == "ate":
            m = self.meta or {}
            return _ate_expr(
                self.outcome,
                str(m.get("treatment", "X")),
                m.get("y", self.value),
                m.get("t1", 1),
                m.get("t0", 0),
            )
        if self.kind == "pns":
            return _pns_expr(str((self.meta or {}).get("treatment", "X")), self.outcome)
        if self.kind == "pn":
            return _pn_expr(str((self.meta or {}).get("treatment", "X")), self.outcome)
        if self.kind == "ps":
            return _ps_expr(str((self.meta or {}).get("treatment", "X")), self.outcome)
        if self.kind in ("lincomb", "cond", "expr"):
            raise ValueError("kind requires explicit expr=")
        raise NotImplementedError(f"query kind {self.kind!r}")

    @classmethod
    def from_expr(
        cls,
        expr: Expr,
        *,
        outcome: str = "Y",
        value: object = 1,
        intervention: Mapping[str, object] | None = None,
        meta: Mapping[str, object] | None = None,
        tag: str | None = None,
    ) -> Query:
        """General entry: any :class:`LinComb` or :class:`Cond`."""
        kind = tag or ("cond" if isinstance(expr, Cond) else "lincomb")
        return cls(
            kind=kind,
            outcome=outcome,
            value=value,
            intervention=dict(intervention or {}),
            meta=meta,
            expr=expr,
        )

    @classmethod
    def interventional(
        cls,
        *,
        outcome: str,
        do: Mapping[str, object],
        value: object,
        condition: Mapping[str, object] | None = None,
    ) -> Query:
        """P(outcome=value | do(...), condition=...).

        ``condition`` on pretreatment covariates is applied by slicing /
        renormalizing ``P`` before formulate (see ``condition_slice``).
        """
        meta = {"condition": dict(condition)} if condition else None
        expr = _interventional_expr(outcome, do, value)
        return cls(
            kind="interventional",
            outcome=outcome,
            value=value,
            intervention=dict(do),
            meta=meta,
            expr=expr,
        )

    @classmethod
    def ate(
        cls,
        *,
        outcome: str = "Y",
        treatment: str = "X",
        y: object = 1,
        t1: object = 1,
        t0: object = 0,
    ) -> Query:
        """P(Y=y|do(t1)) − P(Y=y|do(t0))."""
        expr = _ate_expr(outcome, treatment, y, t1, t0)
        return cls(
            kind="ate",
            outcome=outcome,
            value=y,
            intervention={treatment: t1},
            meta={"treatment": treatment, "t1": t1, "t0": t0, "y": y},
            expr=expr,
        )

    @classmethod
    def pns(cls, *, treatment: str = "X", outcome: str = "Y") -> Query:
        """P(Y_{1}=1, Y_{0}=0) for binary treatment/outcome."""
        return cls(
            kind="pns",
            outcome=outcome,
            value=1,
            intervention={treatment: 1},
            meta={"treatment": treatment, "y1": 1, "y0": 0},
            expr=_pns_expr(treatment, outcome),
        )

    @classmethod
    def pn(cls, *, treatment: str = "X", outcome: str = "Y") -> Query:
        """Probability of necessity: P(Y_{0}=0 | X=1, Y=1) (binary)."""
        return cls(
            kind="pn",
            outcome=outcome,
            value=1,
            intervention={treatment: 1},
            meta={"treatment": treatment},
            expr=_pn_expr(treatment, outcome),
        )

    @classmethod
    def ps(cls, *, treatment: str = "X", outcome: str = "Y") -> Query:
        """Probability of sufficiency: P(Y_{1}=1 | X=0, Y=0) (binary)."""
        return cls(
            kind="ps",
            outcome=outcome,
            value=1,
            intervention={treatment: 0},
            meta={"treatment": treatment},
            expr=_ps_expr(treatment, outcome),
        )

    @classmethod
    def conditional(
        cls,
        numer: LinComb | Atom,
        given: LinComb | Atom,
        *,
        outcome: str = "Y",
        value: object = 1,
    ) -> Query:
        """P(numer | given) as a ratio of marginals (NeurIPS 2021 §4)."""
        n = numer if isinstance(numer, LinComb) else LinComb.atom(numer)
        d = given if isinstance(given, LinComb) else LinComb.atom(given)
        return cls.from_expr(Cond(n, d), outcome=outcome, value=value, tag="cond")

    def describe(self) -> str:
        """Short human-readable label for logs / ``BoundResult.summary``."""
        if self.kind == "interventional":
            do = ", ".join(f"{k}={v}" for k, v in self.intervention.items())
            base = f"P({self.outcome}={self.value} | do({do}))"
            cond = (self.meta or {}).get("condition")
            if isinstance(cond, Mapping) and cond:
                c = ", ".join(f"{k}={v}" for k, v in cond.items())
                return f"{base}, {c}"
            return base
        if self.kind == "ate":
            m = self.meta or {}
            t = m.get("treatment", "X")
            return (
                f"ATE: P({self.outcome}={m.get('y', self.value)} | do({t}={m.get('t1', 1)})) "
                f"- P({self.outcome}={m.get('y', self.value)} | do({t}={m.get('t0', 0)}))"
            )
        if self.kind == "pns":
            t = (self.meta or {}).get("treatment", "X")
            return f"PNS({t}→{self.outcome})"
        if self.kind == "pn":
            return "PN (probability of necessity)"
        if self.kind == "ps":
            return "PS (probability of sufficiency)"
        if self.expr is not None:
            return self.expr.describe()
        return self.kind
