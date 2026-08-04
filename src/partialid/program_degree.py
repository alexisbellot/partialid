"""Classify diagrams by partial-ID program degree (THEORY_PROGRAM_DEGREE §6).

Under hypothesis (H) — ADMG, hyperedge product, face/LIN assumptions,
LIN/FIXDEN queries — ``Classify`` is:

1. ancestral trim + footprint free blocks (C1: single-world unconfounded
   districts are not free unless cross-world);
2. directed-cut C2 loop (Lemma 3.3 / Prop. 8.4 WL–WR): drop free districts
   wholly on the I-side;
3. if Shpitser–Pearl ID succeeds for an interventional (or ATE) query,
   ``F ← ∅`` → POINT;
4. else ``degree = ∑ λ`` over surviving free blocks → LINEAR / BILINEAR_* /
   POLYNOMIAL.

Worked examples (query ``Q = P(Y=y | do(X=x))`` unless noted)
--------------------------------------------------------------

**E1. Bow / Manski.** Directed ``X→Y``, hyperedge ``[("X","Y")]``.
**Linear** (degree 1).

**E2. IV.** ``Z→X→Y``, hyperedge ``[("X","Y")]``. Free ``{X,Y}`` only
(``Z`` absorbed). **Linear**.

**E3. Pair chain.** ``X→M→Y``, hyperedges ``[("X","M"),("M","Y")]``.
**Bilinear/latent** (degree 2).

**E4. Triple hyperedge.** One latent on ``{X,M,Y}``. **Linear**.

**E5. Two separate confounded pairs** (both free, no cut). **Bilinear/factor**.

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
                    f"C2 directed cut WL={list(wl)} WR={list(wr)} "
                    "(no free I-side district to drop)"
                )
            break
        for d in drop:
            notes.append(
                f"C2 drop free district {sorted(d)} via cut "
                f"WL={list(wl)} WR={list(wr)}"
            )
        free = [d for d in free if d not in drop]
        recorded = (wl, wr)
        # Continue: another cut may apply on remaining structure
    return free, recorded


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

    if query is not None:
        fac = analyze_degree1(
            diagram,
            treatments=treatments,
            outcome=outcome,
            query=query,
            variables=variables,
        )
        treatments = (
            list(treatments)
            if treatments is not None
            else sorted(query.intervention.keys())
        )
        if query.kind == "ate":
            treatments = [str((query.meta or {}).get("treatment", "X"))]
        outcome = outcome if outcome is not None else query.outcome
    else:
        if treatments is None or outcome is None:
            raise TypeError("classify_program requires treatments+outcome, or query=")
        fac = analyze_degree1(diagram, treatments=treatments, outcome=outcome)

    notes = list(fac.notes)
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
                    "skip ID→POINT: query vars "
                    f"{sorted(need - set(obs))} are latent endogenous"
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

    if degree <= 1:
        tag = "linear"
        if cut is not None:
            tag += " (after C2 directed cut)" if pre_cut != free else " (WL–WR cut present)"
        return ProgramDegree(
            kind=ProgramKind.LINEAR,
            degree=1,
            free_districts=free,
            c_degrees=degrees,
            notes=tuple(
                notes + [f"program: {tag} (∑ free-block degrees={degree}, K={k})"]
            ),
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
            + [
                f"program: polynomial degree {degree} "
                f"(∑ free-block degrees, K={k}, d={degrees})"
            ]
        ),
        wl_wr_cut=cut,
        in_hypothesis_H=False,
        pre_cut_free_districts=pre_cut,
    )
