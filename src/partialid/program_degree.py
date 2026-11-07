"""Classify diagrams by partial-ID program degree (THEORY_PROGRAM_DEGREE §6).

Sufficient leftover-LP certificate (not claimed complete). Under
ADMG / hyperedge product / face-or-LIN assumptions / LIN-or-FIXDEN
queries, ``Classify`` is:

1. ancestral trim + footprint free blocks (C1: single-world unconfounded
   districts are not free unless cross-world);
2. directed-cut C2 loop (Lemma 3.3 / Prop. 8.4 WL–WR): drop free districts
   wholly on the I-side;
3. if Shpitser–Pearl ID succeeds for an interventional (or ATE) query,
   ``F ← ∅`` → POINT;
4. leftover maps are linear → LINEAR, even if ``K≥2`` (stacked LP). A
   district is an LP block when it is unconfounded or ``λ=1``, when it is
   ``λ=2`` with a root private block, or when the sufficient fiber lemma
   applies. An atom may name several blocks when every block but one is a
   factual factor whose variables are not downstream of the remaining block;
5. else ``degree = ∑ λ`` → BILINEAR_* / POLYNOMIAL.

Worked examples (query ``Q = P(Y=y | do(X=x))`` unless noted)
--------------------------------------------------------------

**E1. Bow / Manski.** Directed ``X→Y``, hyperedge ``[("X","Y")]``.
**Linear** (degree 1).

**E2. IV.** ``Z→X→Y``, hyperedge ``[("X","Y")]``. Free ``{X,Y}`` only
(``Z`` absorbed). **Linear**.

**E3. Pair chain.** ``X→M→Y``, hyperedges ``[("X","M"),("M","Y")]``.
**Linear** (degree 2): one private block is the root ``X``, so the latent
independence is linear in the joint response-type law.

**E4. Triple hyperedge.** One latent on ``{X,M,Y}``. **Linear**.

**E5. Two separate confounded pairs**, one atom using both counterfactuals
(e.g. ``P(Y_x, W_z)``). **Bilinear/factor**. Additive ``P(Y_x)+P(W_z)``, and a
product ``P(Y=y, W_z)`` whose factual factor is not downstream of the other
bow, are **Linear** (stacked LP, the factual factor a scale).

**E6. Triangle.** Three pairwise hyperedges. **Polynomial** (degree 3),
out of (H) ``λ≤2``.

**E8. Mediated + one latent.** Interventional → **Linear**; PNS →
**Bilinear/factor**.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .degree1 import (
    analyze_degree1,
    free_block_degree,
    induced_subgraph,
)
from .graph import CausalDiagram
from .query import Query


class ProgramKind(str, Enum):
    LINEAR = "linear"
    BILINEAR_FACTOR = "bilinear/factor"
    BILINEAR_LATENT = "bilinear/latent"
    POLYNOMIAL = "polynomial"
    POINT = "point"


@dataclass(frozen=True)
class ProgramDegree:
    kind: ProgramKind
    degree: int
    free_districts: tuple[frozenset[str], ...]
    """Post-elimination (post-C2) surviving free districts."""

    c_degrees: tuple[int, ...]
    notes: tuple[str, ...]
    wl_wr_cut: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    in_hypothesis_H: bool = True
    pre_cut_free_districts: tuple[frozenset[str], ...] = ()

    @property
    def is_bilinear(self) -> bool:
        return self.kind in (ProgramKind.BILINEAR_FACTOR, ProgramKind.BILINEAR_LATENT)

    @property
    def is_linear(self) -> bool:
        return self.kind == ProgramKind.LINEAR


def _must_vars(
    *,
    treatments: Sequence[str] | None,
    outcome: str | None,
    query: Query | None,
) -> set[str]:
    must: set[str] = set()
    if outcome is not None:
        must.add(outcome)
    if treatments:
        must.update(treatments)
    if query is not None:
        if query.outcome:
            must.add(query.outcome)
        must.update(query.intervention.keys())
        if query.kind == "ate":
            must.add(str((query.meta or {}).get("treatment", "X")))
        if query.kind in ("pns", "pn", "ps"):
            must.add(str((query.meta or {}).get("treatment", "X")))
            must.add(str((query.meta or {}).get("outcome", query.outcome)))
    return must


def _apply_directed_cuts(
    g_an: CausalDiagram,
    free: list[frozenset[str]],
    *,
    treatments: Sequence[str] | None,
    outcome: str | None,
    notes: list[str],
) -> tuple[list[frozenset[str]], tuple[tuple[str, ...], tuple[str, ...]] | None]:
    """C2 loop: drop free districts wholly in I when a WL–WR cut exists."""
    from .instrument import detect_instrument_split, detect_wl_wr_cut

    recorded: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    # Resolve treatments/outcome for cut detection
    t_list = list(treatments) if treatments else None
    out = outcome

    while True:
        if not free:
            break
        split = detect_wl_wr_cut(g_an, treatments=t_list, outcome=out)
        if split is None:
            split = detect_instrument_split(g_an)
        if split is None:
            break
        wl, wr = split
        wr_s, wl_s = set(wr), set(wl)
        # Only drop districts entirely on the I-side
        drop = [d for d in free if d <= wl_s and not (d & wr_s)]
        if not drop:
            # Cut exists but no free I-side district (e.g. classic IV: Z not free)
            if recorded is None:
                recorded = (wl, wr)
                notes.append(
                    f"C2 directed cut WL={list(wl)} WR={list(wr)} (no free I-side district to drop)"
                )
            break
        for d in drop:
            notes.append(f"C2 drop free district {sorted(d)} via cut WL={list(wl)} WR={list(wr)}")
        free = [d for d in free if d not in drop]
        recorded = (wl, wr)
        # Continue: another cut may apply on remaining structure
    return free, recorded


def _ancestors_after_do(
    diagram: CausalDiagram, seeds: Sequence[str], intervened: Sequence[str]
) -> set[str]:
    """Directed ancestors of ``seeds`` in ``G`` with arrows into ``intervened`` removed."""
    inter = set(intervened)
    an = set(seeds)
    changed = True
    while changed:
        changed = False
        for a, b in diagram.directed:
            if b in inter:
                continue
            if b in an and a not in an:
                an.add(a)
                changed = True
    return an


def _atom_mentioned_vars(atom: object) -> set[str]:
    """Variables named in an AST atom (factual or nested PO)."""
    from .query_ast import PO
    from .query_footprint import _walk_po_vars

    mentioned: set[str] = set()
    for lit in getattr(atom, "lits", ()):
        expr = lit.expr
        if isinstance(expr, str):
            mentioned.add(expr)
        elif isinstance(expr, PO):
            _walk_po_vars(expr, mentioned, set())
    return mentioned


def _po_structural_vars(diagram: CausalDiagram, po: object) -> set[str]:
    """Nodes that enter a potential outcome after ``do`` on its assignment."""
    from .query_ast import PO

    assert isinstance(po, PO)
    mentioned: set[str] = set()

    def rec(p: PO) -> None:
        inter: list[str] = []
        for k, v in p.assignment.items():
            inter.append(k)
            if isinstance(v, PO):
                rec(v)
        mentioned.update(_ancestors_after_do(diagram, (p.var,), inter))
        mentioned.update(inter)

    rec(po)
    return mentioned


def _atom_structural_vars(atom: object, diagram: CausalDiagram | None) -> set[str]:
    """Structural footprint of an atom (mutilated ancestors), else AST names."""
    from .query_ast import PO

    if diagram is None:
        return _atom_mentioned_vars(atom)
    mentioned: set[str] = set()
    for lit in getattr(atom, "lits", ()):
        expr = lit.expr
        if isinstance(expr, str):
            mentioned.add(expr)
        elif isinstance(expr, PO):
            mentioned |= _po_structural_vars(diagram, expr)
    return mentioned


def _atom_free_districts(
    mentioned: set[str], free: Sequence[frozenset[str]]
) -> list[frozenset[str]]:
    return [d for d in free if mentioned & d]


def _observed_set(diagram: CausalDiagram, variables: object | None) -> set[str]:
    if variables is None:
        return set(diagram.nodes)
    obs = getattr(variables, "observed", None)
    if obs is None:
        return set(getattr(variables, "names", diagram.nodes))
    return set(obs)


def _district_downstream(diagram: CausalDiagram, district: frozenset[str], target: str) -> bool:
    """True when ``target`` is in ``district`` or a directed descendant of it."""
    if target in district:
        return True
    seen: set[str] = set()
    stack = [v for v in district]
    children: dict[str, list[str]] = {}
    for a, b in diagram.directed:
        children.setdefault(a, []).append(b)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for c in children.get(n, ()):
            if c == target:
                return True
            stack.append(c)
    return False


def district_is_lp_block(diagram: CausalDiagram, district: frozenset[str]) -> bool:
    """True when this district's fiber is a polytope the solver can build.

    Unconfounded or one hyperedge, two hyperedges with a root private block,
    or the sufficient fiber lemma (including a star of instruments into one
    confounded variable).
    """
    from .degree1 import root_private_sides
    from .graphical_lp import sufficient_hyperedge

    hes = diagram.latent_hyperedges_in(district)
    if len(hes) <= 1:
        return True
    if len(hes) == 2 and root_private_sides(diagram, district) is not None:
        return True
    return sufficient_hyperedge(diagram, district) is not None


def split_atom(
    atom: object,
    diagram: CausalDiagram,
    free: Sequence[frozenset[str]],
    observed: set[str],
) -> tuple[str, frozenset[str] | None]:
    """How one atom sits on the leftover districts.

    ``("const", None)`` — every literal is factual. The probability is an
    additive constant read from ``P``.
    ``("block", district)`` — one counterfactual district, and every factual
    literal is either outside the free districts or in a district that does
    not lie downstream of it. Those factual literals scale that block.
    ``("cross", None)`` — two counterfactual districts, including a parent
    district that still affects the potential outcome, or a factual factor
    downstream of the counterfactual district. A factual literal in the
    same district stays inside the event.
    """
    from .query_ast import PO

    free_of = {v: d for d in free for v in d}
    cf: list[frozenset[str]] = []
    factual_districts: set[frozenset[str]] = set()
    for lit in getattr(atom, "lits", ()):
        expr = lit.expr
        if isinstance(expr, str):
            if expr not in observed:
                return ("cross", None)
            d = free_of.get(expr)
            if d is not None:
                factual_districts.add(d)
            continue
        if not isinstance(expr, PO):
            return ("cross", None)
        # Ancestors in the mutilated graph, not only the names in the atom.
        # A parent district that still affects the potential outcome is part
        # of the event, even when the atom never writes its name.
        named = _po_structural_vars(diagram, expr)
        touched: list[frozenset[str]] = []
        for v in named:
            d = free_of.get(v)
            if d is not None and d not in touched:
                touched.append(d)
        if len(touched) != 1:
            return ("cross", None)
        cf.append(touched[0])
    unique_cf: list[frozenset[str]] = []
    for d in cf:
        if d not in unique_cf:
            unique_cf.append(d)
    if len(unique_cf) > 1:
        return ("cross", None)
    if not unique_cf:
        if any(not isinstance(lit.expr, str) for lit in getattr(atom, "lits", ())):
            return ("cross", None)
        return ("const", None)
    district = unique_cf[0]
    # A factual literal in the same district stays inside the event
    # (PN/PS). A factual literal in another district is a scale, unless
    # that district sits downstream of this one.
    for d in factual_districts:
        if d == district:
            continue
        for v in d:
            if _district_downstream(diagram, district, v):
                return ("cross", None)
    return ("block", district)


def leftover_maps_are_linear(
    free: Sequence[frozenset[str]],
    degrees: Sequence[int],
    *,
    query: Query | None,
    treatments: Sequence[str] | None,
    outcome: str | None,
    diagram: CausalDiagram | None = None,
    variables: object | None = None,
) -> bool:
    """True when leftover maps are affine in the leftover type masses.

    Every leftover district must be an LP block (see ``district_is_lp_block``).
    Each atom is then either a factual constant or a scale times one block.
    A product of two counterfactual events on different blocks is not affine.
    """
    if not free:
        return True
    if diagram is None or any(
        not district_is_lp_block(diagram, d) for d in free
    ):
        # ``degrees`` still gates the no-diagram fallback used by older callers.
        if diagram is not None or any(int(deg) != 1 for deg in degrees):
            return False

    def _fallback_vars() -> set[str]:
        mentioned = set(treatments or ())
        if outcome:
            mentioned.add(outcome)
        if diagram is not None and outcome:
            mentioned |= _ancestors_after_do(diagram, (outcome,), list(treatments or ()))
        return mentioned

    if query is None or query.expr is None or diagram is None:
        return len(_atom_free_districts(_fallback_vars(), free)) <= 1

    from .query_ast import Cond, LinComb
    from .query_footprint import _denom_is_observational

    expr = query.expr
    atoms: list[object] = []
    if isinstance(expr, Cond):
        atoms.extend(expr.numer.iter_atoms())
        if not _denom_is_observational(expr.denom):
            atoms.extend(expr.denom.iter_atoms())
    elif isinstance(expr, LinComb):
        atoms.extend(expr.iter_atoms())
    else:
        return len(_atom_free_districts(_fallback_vars(), free)) <= 1

    observed = _observed_set(diagram, variables)
    for atom in atoms:
        role, _district = split_atom(atom, diagram, free, observed)
        if role == "cross":
            return False
    return True


def classify_program(
    diagram: CausalDiagram,
    *,
    treatments: Sequence[str] | None = None,
    outcome: str | None = None,
    query: Query | None = None,
    variables: object | None = None,
    has_wl_wr_cut: bool = False,  # deprecated; cut is detected internally
) -> ProgramDegree:
    """Classify by post-elimination ∑ free-block degrees (THEORY_PROGRAM_DEGREE §6).

    Pass ``query=`` for any AST query. ``treatments`` + ``outcome`` remain
    supported for interventional classification. ``has_wl_wr_cut`` is ignored
    (kept for API compatibility); directed cuts are applied automatically.
    """
    del has_wl_wr_cut  # cut detection is internal (C2)

    from .graphical_lp import without_contained_hyperedges

    reduced = without_contained_hyperedges(diagram)
    contained_note: str | None = None
    if reduced is not diagram:
        diagram = reduced
        contained_note = "contained hyperedges deleted before counting λ"

    if query is not None:
        fac = analyze_degree1(
            diagram,
            treatments=treatments,
            outcome=outcome,
            query=query,
            variables=variables,
        )
        treatments = (
            list(treatments) if treatments is not None else sorted(query.intervention.keys())
        )
        if query.kind == "ate":
            treatments = [str((query.meta or {}).get("treatment", "X"))]
        outcome = outcome if outcome is not None else query.outcome
    else:
        if treatments is None or outcome is None:
            raise TypeError("classify_program requires treatments+outcome, or query=")
        fac = analyze_degree1(diagram, treatments=treatments, outcome=outcome)

    notes = list(fac.notes)
    if contained_note is not None:
        notes.append(contained_note)
    g_an = induced_subgraph(diagram, fac.ancestral)
    pre_cut = tuple(fac.free_districts)
    free_list = list(pre_cut)

    # Hypothesis (H): λ ≤ 2 on every ancestral district
    max_lambda = 0
    for d in fac.districts:
        cd = g_an.c_degree(d) if d else 0
        max_lambda = max(max_lambda, cd)
    in_H = max_lambda <= 2
    if not in_H:
        notes.append(
            f"out of (H): ancestral max λ={max_lambda} > 2 "
            "(hyperedge product with λ≤2 required for Thm 5.5)"
        )

    free_list, cut = _apply_directed_cuts(
        g_an,
        free_list,
        treatments=treatments,
        outcome=outcome,
        notes=notes,
    )

    # THEORY_PROGRAM_DEGREE §6 step 9: if ID(Q) then F ← ∅ → POINT
    # Skip Manski full-latent (never point-ID for do()) and counterfactuals.
    from .id_algorithm import is_identified
    from .identify import is_manski_full_latent

    id_eligible = (
        treatments is not None
        and outcome is not None
        and not is_manski_full_latent(g_an)
        and (query is None or query.kind in ("interventional", "ate"))
    )
    # Classical ID reads functionals of P on the outcome; if the outcome (or a
    # treatment) is latent endogenous, those functionals are not available from
    # the observational table alone (measurement-error graphs).
    if id_eligible and variables is not None:
        obs = getattr(variables, "observed", None)
        if obs is not None:
            need = set(treatments) | {outcome}
            if not need <= set(obs):
                id_eligible = False
                notes.append(
                    f"skip ID→POINT: query vars {sorted(need - set(obs))} are latent endogenous"
                )
    if id_eligible and is_identified(g_an, list(treatments), [outcome]):
        return ProgramDegree(
            kind=ProgramKind.POINT,
            degree=0,
            free_districts=(),
            c_degrees=(),
            notes=tuple(notes + ["program: point (ID)"]),
            wl_wr_cut=cut,
            in_hypothesis_H=in_H,
            pre_cut_free_districts=pre_cut,
        )

    free = tuple(free_list)
    degrees = tuple(free_block_degree(g_an, d) for d in free)

    if not free:
        return ProgramDegree(
            kind=ProgramKind.POINT,
            degree=0,
            free_districts=(),
            c_degrees=(),
            notes=tuple(notes + ["program: point (no free district after elimination)"]),
            wl_wr_cut=cut,
            in_hypothesis_H=in_H,
            pre_cut_free_districts=pre_cut,
        )

    degree = int(sum(degrees))
    k = len(free)
    uncoupled = leftover_maps_are_linear(
        free,
        degrees,
        query=query,
        treatments=treatments,
        outcome=outcome,
        diagram=g_an,
        variables=variables,
    )

    if uncoupled:
        tag = "linear"
        if any(int(d) >= 3 for d in degrees):
            tag = "linear (sufficient fiber lemma)"
        elif any(int(d) == 2 for d in degrees):
            tag = "linear (λ=2 root private block; independence is linear)"
        if k >= 2:
            tag = f"linear (separable K={k}; stacked LP)"
        elif cut is not None:
            tag += " (after C2 directed cut)" if pre_cut != free else " (WL–WR cut present)"
        return ProgramDegree(
            kind=ProgramKind.LINEAR,
            degree=degree if degree > 0 else 1,
            free_districts=free,
            c_degrees=degrees,
            notes=tuple(notes + [f"program: {tag} (∑ free-block degrees={degree}, K={k})"]),
            wl_wr_cut=cut,
            in_hypothesis_H=in_H,
            pre_cut_free_districts=pre_cut,
        )

    if degree == 2:
        if k == 1 and degrees[0] == 2:
            kind = ProgramKind.BILINEAR_LATENT
            tag = "bilinear/latent (one district, two hyperedges)"
        elif k == 2 and all(d == 1 for d in degrees):
            kind = ProgramKind.BILINEAR_FACTOR
            tag = "bilinear/factor (K=2 degree-1 districts)"
        else:
            kind = ProgramKind.POLYNOMIAL
            tag = f"polynomial degree 2 (atypical shape degrees={degrees})"
        return ProgramDegree(
            kind=kind,
            degree=2,
            free_districts=free,
            c_degrees=degrees,
            notes=tuple(notes + [f"program: {tag}"]),
            wl_wr_cut=cut,
            in_hypothesis_H=in_H and kind != ProgramKind.POLYNOMIAL,
            pre_cut_free_districts=pre_cut,
        )

    return ProgramDegree(
        kind=ProgramKind.POLYNOMIAL,
        degree=degree,
        free_districts=free,
        c_degrees=degrees,
        notes=tuple(
            notes
            + [f"program: polynomial degree {degree} (∑ free-block degrees, K={k}, d={degrees})"]
        ),
        wl_wr_cut=cut,
        in_hypothesis_H=False,
        pre_cut_free_districts=pre_cut,
    )
