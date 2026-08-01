"""Classify diagrams by partial-ID program degree from the explicit GPP.

Worked examples (query ``Q = P(Y=y | do(X=x))`` unless noted)
--------------------------------------------------------------

**E1. Bow / Manski.** Directed ``X→Y``, hyperedge ``[("X","Y")]``.
One district ``{X,Y}``, one latent ``U``, response types ``(R_X,R_Y)``.
Program: ``min/max cᵀq  s.t. Aq = P``, ``q ∈ Δ``. **Linear** (degree 1).

**E2. IV.** ``Z→X→Y``, hyperedge ``[("X","Y")]``.
Districts ``{Z}``, ``{X,Y}``. Free confounded block only ``{X,Y}`` (``Z`` is
point-identified / WL). Same LP on R-side types (Prop. 8.4). **Linear**.

**E3. Two overlapping pairs.** ``X→M→Y``, hyperedges ``[("X","M"),("M","Y")]``.
One district ``{X,M,Y}``, latents ``U₁⊥U₂``.
``P(v)=∑_{u₁,u₂} p₁(u₁)p₂(u₂) A_{v,u₁u₂}``, same for ``Q``. **Bilinear**
(degree 2, latent).

**E4. One triple latent.** ``X→M→Y``, hyperedge ``[("X","M","Y")]``.
One district, one ``U``. Joint type LP as in E1 with 32 binary types. **Linear**.

**E5. Two separate confounded pairs.** e.g. hyperedges ``[("X","Y"),("U","V")]``
with both districts ancestral and free.
``P,Q`` multilinear in ``(q^{XY}, q^{UV})``. **Bilinear** (degree 2, factor).

**E6. Triangle.** Three pairwise hyperedges on ``{X,Z,Y}``.
One district, ``U₁⊥U₂⊥U₃``. Forms cubic in ``(p₁,p₂,p₃)``. **Polynomial**
(degree 3).

**E7. Three free c-degree-1 districts.** Product of three ``q`` blocks.
**Polynomial** (degree 3).

**E8. Mediated + one latent, PNS.** ``X→M→Y``, hyperedge ``[("X","M")]``.
Interventional ``P(Y|do(X))``: free ``{X,M}`` only → **Linear**.
PNS ``P(Y₁=1,Y₀=0)``: free ``{X,M}`` and cross-world ``{Y}`` → **Bilinear**
(factor).

Rule (extracted from E1–E8)
---------------------------
Let free districts be confounded ancestral blocks plus cross-world
unconfounded blocks (see :mod:`partialid.query_footprint`). Let ``d_C`` be
the # hyperedges in ``C``, or ``1`` if ``C`` is cross-world with no hyperedge.
Then

    degree = ∑_C d_C

(each factor is one free simplex in the GPP). Map degree → kind:

- 0 → point (Markovian / ID)
- 1 → linear
- 2 → bilinear (latent if one district with ``d=2``; factor if two ``d=1``)
- ≥3 → polynomial
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .degree1 import analyze_degree1, free_block_degree
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
    c_degrees: tuple[int, ...]
    notes: tuple[str, ...]

    @property
    def is_bilinear(self) -> bool:
        return self.kind in (ProgramKind.BILINEAR_FACTOR, ProgramKind.BILINEAR_LATENT)

    @property
    def is_linear(self) -> bool:
        return self.kind == ProgramKind.LINEAR


def classify_program(
    diagram: CausalDiagram,
    *,
    treatments: Sequence[str] | None = None,
    outcome: str | None = None,
    query: Query | None = None,
    has_wl_wr_cut: bool = False,
) -> ProgramDegree:
    """Classify by ∑ free-block degrees (see module doc).

    Pass ``query=`` for counterfactual / general AST queries; ``treatments`` +
    ``outcome`` remain supported for interventional classification.
    """
    if query is not None:
        fac = analyze_degree1(diagram, treatments=treatments, outcome=outcome, query=query)
    else:
        if treatments is None or outcome is None:
            raise TypeError("classify_program requires treatments+outcome, or query=")
        fac = analyze_degree1(diagram, treatments=treatments, outcome=outcome)

    free = fac.free_districts
    degrees = tuple(free_block_degree(diagram, d) for d in free)
    notes = list(fac.notes)

    if not free:
        return ProgramDegree(
            kind=ProgramKind.POINT,
            degree=0,
            free_districts=(),
            c_degrees=(),
            notes=tuple(notes + ["program: point (no free district)"]),
        )

    # WL–WR / instrument cut absorbs exo blocks → one free simplex remains
    if has_wl_wr_cut and all(d <= 1 for d in degrees):
        return ProgramDegree(
            kind=ProgramKind.LINEAR,
            degree=1,
            free_districts=free,
            c_degrees=degrees,
            notes=tuple(notes + ["program: linear (WL–WR / instrument cut)"]),
        )

    degree = int(sum(degrees))
    k = len(free)

    if degree <= 1:
        return ProgramDegree(
            kind=ProgramKind.LINEAR,
            degree=1,
            free_districts=free,
            c_degrees=degrees,
            notes=tuple(
                notes + [f"program: linear (∑ free-block degrees={degree}, K={k})"]
            ),
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
    )
