"""Compile unified query AST → LP / linear-fractional objectives.

Preprocessing pipeline (each step is cited):

1. **Legacy sugar → AST** — named factories build :class:`LinComb` / :class:`Cond`.
2. **CUT unnesting** — Correa, Lee & Bareinboim (NeurIPS 2021) Theorem 4
   (Counterfactual Unnesting Theorem): any nested counterfactual
   ``P(Y_{Ẑ,X̂}=y)`` equals ``∑_x P(Y_{Ẑ,x}=y, X̂=x)`` under the theorem's
   ancestral conditions. We expand nested ``PO`` values into a sum of
   *unnested* joint atoms before building ``c``.
3. **Conditional → ratio** — same paper §4: ``P(A|B)=P(A∧B)/P(B)``.
   Observational denominators with known mass under ``P`` yield a
   *fixed-denominator* linear objective (Tian–Pearl PN/PS). Otherwise we
   emit a linear-fractional pair ``(c_num, c_den)``.
4. **Unnested atom → type coefficient** — unnested response-type atoms (Prop. 3):
   an atomic (unnested) query is ``αᵀ q`` on response types.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .graph import CausalDiagram
from .query_ast import PO, Atom, Cond, Expr, LinComb, Lit, conjunct_atoms
from .types_canon import TypeSpace, evaluate_world
from .variables import Variables


@dataclass
class LinearObjective:
    c: np.ndarray
    notes: list[str] = field(default_factory=list)


@dataclass
class FractionalObjective:
    """Maximize/minimize ``(c_numᵀ q) / (c_denᵀ q)`` with ``c_denᵀ q > 0``."""

    c_num: np.ndarray
    c_den: np.ndarray
    notes: list[str] = field(default_factory=list)


Objective = LinearObjective | FractionalObjective


# ---------------------------------------------------------------------------
# CUT unnesting
# ---------------------------------------------------------------------------


def _ancestors_in_mutilated(diagram: CausalDiagram, y: str, intervened: set[str]) -> set[str]:
    """Ancestors of ``y`` in ``G`` with outgoing edges from ``intervened`` removed.

    Used to check CUT's side condition ``X ⊆ An(Y)_{G_Z}``.
    """
    # Build parent map excluding edges out of intervened nodes (do(Z) mutilation)
    parents: dict[str, list[str]] = {n: [] for n in diagram.nodes}
    for a, b in diagram.directed:
        if a in intervened:
            continue
        parents[b].append(a)
    seen: set[str] = set()
    stack = [y]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(parents.get(n, []))
    return seen


def _find_nested_slot(po: PO) -> tuple[str, PO] | None:
    """Return one ``(parent_key, nested_PO)`` slot if ``po`` is nested."""
    for k, v in po.assignment.items():
        if isinstance(v, PO):
            return k, v
    return None


def unnest_po_event(
    po: PO,
    value: object,
    variables: Variables,
    diagram: CausalDiagram,
    notes: list[str],
) -> LinComb:
    """Expand ``P(po = value)`` via CUT until all potential outcomes are flat.

    Implements NeurIPS 2021 Theorem 4 recursively: for a nested assignment
    ``Y(..., X=X̂_do, ...)`` we sum over the domain of the nested variable.
    """
    slot = _find_nested_slot(po)
    if slot is None:
        return LinComb.atom(Atom(Lit(po, value)))

    parent_key, nested = slot
    # CUT: P(Y_{Ẑ, X̂}=y) = ∑_x P(Y_{Ẑ,x}=y, X̂=x)
    # Here nested is X̂ = PO(X, do_X); we sum over x ∈ domain(X) (= domain of nested.var)
    x_var = nested.var
    if x_var not in variables.domains:
        raise ValueError(f"CUT: unknown nested variable {x_var!r}")

    # Ancestral check: X ⊆ An(Y)_{G_Z} where Ẑ are the other (flat) interventions
    flat_intervened = {k for k, v in po.assignment.items() if not isinstance(v, PO)}
    # Also treat parent_key as being set by natural intervention of x_var
    an = _ancestors_in_mutilated(diagram, po.var, flat_intervened)
    if x_var not in an and x_var != po.var:
        # Soft note: still expand (response-type path eval would also work);
        # warn that CUT's stated side-condition may fail.
        notes.append(
            f"CUT expansion of {po.describe()}: {x_var} not in An({po.var}) "
            f"under G with do({sorted(flat_intervened)}); expanding anyway "
            f"(NeurIPS 2021 Thm 4 side-condition)"
        )

    terms: list[tuple[float, Atom]] = []
    for x_val in variables.domains[x_var]:
        # Outer PO with nested slot replaced by constant x_val
        new_assign = dict(po.assignment)
        new_assign[parent_key] = x_val
        outer_flat_or_nested = PO(po.var, new_assign)
        # Joint: (Y_{...x...}=y) ∧ (X̂ = x_val)
        left = unnest_po_event(outer_flat_or_nested, value, variables, diagram, notes)
        right_po = nested  # still may nest further
        right = unnest_po_event(right_po, x_val, variables, diagram, notes)
        # Distribute product of lincombs (all coeffs 1 from CUT sum)
        for c1, a1 in left.terms:
            for c2, a2 in right.terms:
                terms.append((c1 * c2, conjunct_atoms(a1, a2)))

    notes.append(
        f"CUT (NeurIPS 2021 Thm 4): unnested {po.describe()}={value} "
        f"by summing over {x_var}∈{list(variables.domains[x_var])}"
    )
    return LinComb(terms)


def unnest_atom(
    atom: Atom,
    variables: Variables,
    diagram: CausalDiagram,
    notes: list[str],
) -> LinComb:
    """Unnest every nested PO literal inside an atom; return a LinComb."""
    # Start with the unit event and multiply (conjunct) expansions of each lit
    acc = LinComb.atom(Atom())  # empty conjunction = sure event
    for lit in atom.lits:
        if isinstance(lit.expr, str):
            piece = LinComb.atom(Atom(lit))
        elif isinstance(lit.expr, PO):
            if lit.expr.is_flat():
                piece = LinComb.atom(Atom(lit))
            else:
                piece = unnest_po_event(lit.expr, lit.value, variables, diagram, notes)
        else:
            raise TypeError(f"bad literal expr {lit.expr!r}")
        # Conjunct distribute
        new_terms: list[tuple[float, Atom]] = []
        for c1, a1 in acc.terms:
            for c2, a2 in piece.terms:
                new_terms.append((c1 * c2, conjunct_atoms(a1, a2)))
        acc = LinComb(new_terms)
    return acc


def unnest_expr(
    expr: Expr,
    variables: Variables,
    diagram: CausalDiagram,
    notes: list[str] | None = None,
) -> Expr:
    """Apply CUT to all nested POs in a LinComb or Cond."""
    notes = notes if notes is not None else []
    if isinstance(expr, Cond):
        return Cond(
            unnest_expr(expr.numer, variables, diagram, notes),  # type: ignore[arg-type]
            unnest_expr(expr.denom, variables, diagram, notes),  # type: ignore[arg-type]
        )
    assert isinstance(expr, LinComb)
    terms: list[tuple[float, Atom]] = []
    for coeff, atom in expr.terms:
        expanded = unnest_atom(atom, variables, diagram, notes)
        for c, a in expanded.terms:
            terms.append((coeff * c, a))
    return LinComb(terms)


# ---------------------------------------------------------------------------
# Atom evaluation on response types
# ---------------------------------------------------------------------------


def _eval_flat_po(
    ts: TypeSpace,
    type_idx: int,
    po: PO,
) -> object:
    if not po.is_flat():
        raise ValueError("internal: expected flat PO after CUT")
    world = evaluate_world(
        ts.order,
        ts.response_fns,
        ts.joint_types[type_idx],
        ts.diagram,
        po.assignment,
    )
    return world[po.var]


def atom_holds(ts: TypeSpace, type_idx: int, atom: Atom) -> bool:
    """Whether an *unnested* atom holds for a response type (Prop. 3)."""
    for lit in atom.lits:
        if isinstance(lit.expr, str):
            world = evaluate_world(
                ts.order,
                ts.response_fns,
                ts.joint_types[type_idx],
                ts.diagram,
                None,
            )
            if world[lit.expr] != lit.value:
                return False
        elif isinstance(lit.expr, PO):
            if _eval_flat_po(ts, type_idx, lit.expr) != lit.value:
                return False
        else:
            return False
    return True


def atom_holds_via_eval(
    evaluate: Callable[[Mapping[str, object] | None], Mapping[str, object]],
    atom: Atom,
) -> bool:
    """Whether an *unnested* atom holds under a multi-world evaluator.

    ``evaluate(intervention)`` returns the endogenous assignment under that
    intervention (``None`` = observational). Used by factorized GPP builders
    that are not a single :class:`TypeSpace`.
    """
    for lit in atom.lits:
        if isinstance(lit.expr, str):
            if evaluate(None)[lit.expr] != lit.value:
                return False
        elif isinstance(lit.expr, PO):
            if not lit.expr.is_flat():
                raise ValueError("internal: expected flat PO after CUT")
            if evaluate(dict(lit.expr.assignment))[lit.expr.var] != lit.value:
                return False
        else:
            return False
    return True


def observational_lincomb_mass(
    denom: LinComb,
    order: Sequence[str],
    cells: Sequence[tuple],
    P_vec: np.ndarray,
) -> float | None:
    """Mass of a purely observational LinComb under ``P`` (single-atom denom).

    Used for Tian–Pearl fixed-denominator linearization without a TypeSpace.
    """
    if len(denom.terms) != 1:
        return None
    coeff, atom = denom.terms[0]
    if any(isinstance(lit.expr, PO) for lit in atom.lits):
        return None
    idx = {n: i for i, n in enumerate(order)}
    mass = 0.0
    for cell, p in zip(cells, P_vec):
        ok = True
        for lit in atom.lits:
            assert isinstance(lit.expr, str)
            if lit.expr not in idx or cell[idx[lit.expr]] != lit.value:
                ok = False
                break
        if ok:
            mass += float(p)
    return float(coeff) * mass


def query_coeff_via_eval(
    expr: Expr,
    evaluate: Callable[[Mapping[str, object] | None], Mapping[str, object]],
    *,
    variables: Variables,
    diagram: CausalDiagram,
    order: Sequence[str] | None = None,
    cells: Sequence[tuple] | None = None,
    P_vec: np.ndarray | None = None,
) -> float:
    """Objective coefficient of a query for one factorized response type.

    Supports :class:`LinComb` and :class:`Cond` with a *fixed observational*
    denominator (PN / PS). Truly fractional conditionals raise
    ``NotImplementedError`` (no Charnes–Cooper on the GPP path yet).
    """
    notes: list[str] = []
    unnested = unnest_expr(expr, variables, diagram, notes)
    if isinstance(unnested, LinComb):
        total = 0.0
        for coeff, atom in unnested.terms:
            if atom_holds_via_eval(evaluate, atom):
                total += float(coeff)
        return total

    assert isinstance(unnested, Cond)
    if order is None or cells is None or P_vec is None:
        raise NotImplementedError(
            "GPP Cond queries require observational (order, cells, P) "
            "for fixed-denominator linearization"
        )
    dmass = observational_lincomb_mass(unnested.denom, order, cells, P_vec)
    if dmass is None:
        raise NotImplementedError(
            "GPP does not support truly fractional Cond "
            "(denominator not a fixed observational mass)"
        )
    if dmass <= 1e-15:
        raise ValueError("conditional undefined: P(denom)=0")
    numer = unnested.numer
    if not _numer_implies_denom(unnested.numer, unnested.denom):
        numer = _conjunct_lincombs(unnested.numer, unnested.denom)
    total = 0.0
    for coeff, atom in numer.terms:
        if atom_holds_via_eval(evaluate, atom):
            total += float(coeff)
    return total / dmass


def lincomb_coeff_via_eval(
    expr: Expr,
    evaluate: Callable[[Mapping[str, object] | None], Mapping[str, object]],
    *,
    variables: Variables,
    diagram: CausalDiagram,
    order: Sequence[str] | None = None,
    cells: Sequence[tuple] | None = None,
    P_vec: np.ndarray | None = None,
) -> float:
    """Alias of :func:`query_coeff_via_eval` (kept for call-site clarity)."""
    return query_coeff_via_eval(
        expr,
        evaluate,
        variables=variables,
        diagram=diagram,
        order=order,
        cells=cells,
        P_vec=P_vec,
    )


def gpp_expr_supported(
    expr: Expr,
    *,
    variables: Variables,
    diagram: CausalDiagram,
    order: Sequence[str],
    cells: Sequence[tuple],
    P_vec: np.ndarray,
) -> bool:
    """True if ``expr`` can be compiled to a linear GPP objective."""
    notes: list[str] = []
    unnested = unnest_expr(expr, variables, diagram, notes)
    if isinstance(unnested, LinComb):
        return True
    if isinstance(unnested, Cond):
        dmass = observational_lincomb_mass(unnested.denom, order, cells, P_vec)
        return dmass is not None and dmass > 1e-15
    return False


def _lincomb_coefficients(
    ts: TypeSpace,
    lin: LinComb,
    allow: np.ndarray,
) -> np.ndarray:
    c = np.zeros(ts.n_types, dtype=float)
    for coeff, atom in lin.terms:
        for t in range(ts.n_types):
            if not allow[t]:
                continue
            if atom_holds(ts, t, atom):
                c[t] += float(coeff)
    return c


def _observational_atom_mass(
    atom: Atom,
    ts: TypeSpace,
    cells: Sequence[tuple],
    P_vec: np.ndarray,
) -> float | None:
    """If ``atom`` is purely observational (no PO lits), return its mass under P.

    Used for fixed-denominator linearization of PN/PS-style conditionals.
    """
    if any(isinstance(lit.expr, PO) for lit in atom.lits):
        return None
    obs = ts.variables.observed_names(ts.order)
    idx = {n: i for i, n in enumerate(obs)}
    mass = 0.0
    for cell, p in zip(cells, P_vec):
        ok = True
        for lit in atom.lits:
            assert isinstance(lit.expr, str)
            if lit.expr not in idx:
                return None
            if cell[idx[lit.expr]] != lit.value:
                ok = False
                break
        if ok:
            mass += float(p)
    return mass


def _try_fixed_denominator(
    denom: LinComb,
    ts: TypeSpace,
    cells: Sequence[tuple],
    P_vec: np.ndarray,
) -> float | None:
    """Return known observational mass if denom is a single obs atom (or scale)."""
    if len(denom.terms) != 1:
        return None
    coeff, atom = denom.terms[0]
    mass = _observational_atom_mass(atom, ts, cells, P_vec)
    if mass is None:
        return None
    return float(coeff) * mass


def compile_objective(
    ts: TypeSpace,
    expr: Expr,
    allow: np.ndarray,
    *,
    cells: Sequence[tuple] | None = None,
    P_vec: np.ndarray | None = None,
    variables: Variables | None = None,
    diagram: CausalDiagram | None = None,
) -> Objective:
    """Compile AST → :class:`LinearObjective` or :class:`FractionalObjective`."""
    notes: list[str] = []
    variables = variables or ts.variables
    diagram = diagram or ts.diagram

    unnested = unnest_expr(expr, variables, diagram, notes)

    if isinstance(unnested, LinComb):
        c = _lincomb_coefficients(ts, unnested, allow)
        return LinearObjective(c=c, notes=notes)

    assert isinstance(unnested, Cond)
    # Prefer fixed-denominator when denom mass is known from P
    if cells is not None and P_vec is not None:
        dmass = _try_fixed_denominator(unnested.denom, ts, cells, P_vec)
        if dmass is not None:
            if dmass <= 1e-15:
                raise ValueError("conditional undefined: P(denom)=0")
            # numer should be P(numer ∧ denom); if user passed Cond(numer, denom)
            # with numer already the joint (PN sugar), use numer directly.
            # Standard form: P(A|B)=P(A∧B)/P(B). Our Cond stores numer as the
            # *numerator event of the joint form* when sugar builds
            # Cond(joint_atom_as_lincomb, denom) OR Cond(A, B) meaning A|B.
            # Convention in this package: Cond(numer, denom) means
            # P(numer | denom) = P(numer ∧ denom) / P(denom) when numer and
            # denom are disjoint events; but PN sugar builds numer already as
            # the joint indicator on the fiber. Detect: if numer atoms already
            # include all denom observational lits, treat numer as joint.
            c_num = _lincomb_coefficients(ts, unnested.numer, allow)
            # If numer does not already encode the joint, conjunct with denom.
            if not _numer_implies_denom(unnested.numer, unnested.denom):
                joint = _conjunct_lincombs(unnested.numer, unnested.denom)
                c_num = _lincomb_coefficients(ts, joint, allow)
            notes.append(
                f"conditional: fixed denom mass={dmass:.6g} from observational P "
                f"(NeurIPS 2021 §4 / Tian–Pearl PN-PS linearization)"
            )
            return LinearObjective(c=c_num / dmass, notes=notes)

    c_num_lc = unnested.numer
    if not _numer_implies_denom(unnested.numer, unnested.denom):
        c_num_lc = _conjunct_lincombs(unnested.numer, unnested.denom)
    c_num = _lincomb_coefficients(ts, c_num_lc, allow)
    c_den = _lincomb_coefficients(ts, unnested.denom, allow)
    notes.append(
        "conditional: linear-fractional objective (c_numᵀq)/(c_denᵀq); "
        "NeurIPS 2021 §4 ratio of marginals"
    )
    return FractionalObjective(c_num=c_num, c_den=c_den, notes=notes)


def _numer_implies_denom(numer: LinComb, denom: LinComb) -> bool:
    """True if every numer atom already contains all denom observational lits."""
    if len(denom.terms) != 1:
        return False
    _, d_atom = denom.terms[0]
    d_lits = {(lit.expr, lit.value) for lit in d_atom.lits if isinstance(lit.expr, str)}
    if not d_lits:
        return False
    for _, a in numer.terms:
        a_lits = {(lit.expr, lit.value) for lit in a.lits if isinstance(lit.expr, str)}
        if not d_lits.issubset(a_lits):
            return False
    return True


def _conjunct_lincombs(a: LinComb, b: LinComb) -> LinComb:
    terms: list[tuple[float, Atom]] = []
    for c1, a1 in a.terms:
        for c2, a2 in b.terms:
            terms.append((c1 * c2, conjunct_atoms(a1, a2)))
    return LinComb(terms)
