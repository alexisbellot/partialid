"""Lazy response-type space: no full joint enumeration.

Used by matrix-free column generation. Columns are keys
``tuple[int, ...]`` (one response-fn index per variable in order).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import numpy as np

from .graph import CausalDiagram
from .types_canon import _all_functions, evaluate_world
from .variables import Variables

TypeKey = tuple[int, ...]


@dataclass
class LazyTypeSpace:
    """Response functions only; joint types are generated on demand."""

    variables: Variables
    diagram: CausalDiagram
    order: tuple[str, ...]
    response_fns: dict[str, list[dict]]
    fn_counts: tuple[int, ...]

    @classmethod
    def from_diagram(cls, variables: Variables, diagram: CausalDiagram) -> LazyTypeSpace:
        order = diagram.topological_order()
        response_fns: dict[str, list[dict]] = {}
        for v in order:
            parents = list(diagram.parents(v))
            domain_in = variables.cells(parents) if parents else []
            response_fns[v] = _all_functions(domain_in, variables.domains[v])
        fn_counts = tuple(len(response_fns[v]) for v in order)
        return cls(
            variables=variables,
            diagram=diagram,
            order=order,
            response_fns=response_fns,
            fn_counts=fn_counts,
        )

    @property
    def n_types_est(self) -> int:
        n = 1
        for c in self.fn_counts:
            n *= c
        return int(n)

    def observational_cell(self, key: TypeKey) -> tuple:
        world = evaluate_world(self.order, self.response_fns, key, self.diagram, None)
        return tuple(world[v] for v in self.order)

    def query_coefficient(
        self,
        key: TypeKey,
        outcome: str,
        value: object,
        intervention: Mapping[str, object],
    ) -> float:
        world = evaluate_world(self.order, self.response_fns, key, self.diagram, intervention)
        return 1.0 if world[outcome] == value else 0.0

    def iter_fiber(self, cell: tuple, *, max_list: int | None = None) -> Iterator[TypeKey]:
        """Yield type keys consistent with observational ``cell`` (may be exponential)."""
        # Backtrack over response-fn indices constrained by the cell
        target = dict(zip(self.order, cell))
        partial: list[int | None] = [None] * len(self.order)

        def ok_prefix(depth: int) -> bool:
            # Evaluate determined variables up to depth-1; check against target
            ids = []
            for i in range(depth):
                if partial[i] is None:
                    return True
                ids.append(partial[i])
            # Build world for assigned prefix only — need full key for evaluate_world
            # Fill unassigned with 0 temporarily and only trust assigned nodes
            trial = [partial[i] if partial[i] is not None else 0 for i in range(len(self.order))]
            world: dict[str, object] = {}
            for i, v in enumerate(self.order):
                if i >= depth:
                    break
                parents = self.diagram.parents(v)
                if any(p not in world for p in parents):
                    # parent not yet in prefix — skip check
                    if parents and max(self.order.index(p) for p in parents) >= depth:
                        continue
                key = tuple(world[p] for p in parents) if parents else ()
                world[v] = self.response_fns[v][partial[i]][key]  # type: ignore[index]
                if world[v] != target[v]:
                    return False
            return True

        count = 0

        def rec(i: int):
            nonlocal count
            if max_list is not None and count >= max_list:
                return
            if i == len(self.order):
                key = tuple(partial)  # type: ignore[arg-type]
                if self.observational_cell(key) == cell:
                    count += 1
                    yield key
                return
            for fi in range(self.fn_counts[i]):
                partial[i] = fi
                if ok_prefix(i + 1):
                    yield from rec(i + 1)
                partial[i] = None

        yield from rec(0)

    def one_fiber_member(self, cell: tuple) -> TypeKey | None:
        for key in self.iter_fiber(cell, max_list=1):
            return key
        return None

    def fiber_extremes_for_query(
        self,
        cell: tuple,
        outcome: str,
        value: object,
        intervention: Mapping[str, object],
        *,
        max_list: int | None = 512,
    ) -> tuple[TypeKey | None, TypeKey | None, float, float]:
        """Return (key_min_c, key_max_c, c_min, c_max) in the fiber.

        ``max_list`` caps enumeration (None = full fiber). For huge fibers
        this is a heuristic seed for column generation, not an exact Move-1.
        """
        k_lo = k_hi = None
        c_lo, c_hi = 1.0, 0.0
        found = False
        for key in self.iter_fiber(cell, max_list=max_list):
            c = self.query_coefficient(key, outcome, value, intervention)
            if not found or c < c_lo:
                c_lo, k_lo = c, key
            if not found or c > c_hi:
                c_hi, k_hi = c, key
            found = True
        if not found:
            return None, None, float("nan"), float("nan")
        return k_lo, k_hi, c_lo, c_hi

    def price_best_keys(
        self,
        cells: Sequence[tuple],
        P_vec: np.ndarray,
        eq_duals: np.ndarray,
        *,
        outcome: str,
        value: object,
        intervention: Mapping[str, object],
        maximize: bool,
        active: set[TypeKey],
        tol: float,
        max_per_fiber: int = 64,
        ub_contrib: Mapping[TypeKey, float] | None = None,
    ) -> TypeKey | None:
        """Matrix-free pricing: scan fibers only (bounded per fiber), pick best rc < 0.

        Equality-only reduced cost: ``rc = c - π_cell``. Optional ``ub_contrib``
        adds Track A3 budget duals ``(A_ub'μ)_key`` when the caller materializes
        them (lazy MSM master is not wired yet — use materialized ``colgen``).
        """
        best_key = None
        best_rc = 0.0
        for i, cell in enumerate(cells):
            if P_vec[i] <= 0 and abs(eq_duals[i]) < 1e-15:
                # still may need columns for zero cells
                pass
            pi = float(eq_duals[i])
            n_seen = 0
            for key in self.iter_fiber(cell):
                if key in active:
                    continue
                c = self.query_coefficient(key, outcome, value, intervention)
                # reduced cost for min c'q : rc = c - π_cell (A column is e_cell)
                rc = c - pi
                if ub_contrib is not None:
                    rc -= float(ub_contrib.get(key, 0.0))
                if maximize:
                    rc = -rc
                if rc < -tol and (best_key is None or rc < best_rc):
                    best_key = key
                    best_rc = rc
                n_seen += 1
                if n_seen >= max_per_fiber:
                    break
        return best_key


def c_canonical_cardinality(variables: Variables, diagram: CausalDiagram) -> int:
    """|Ω_Pa(C)| for the single-component case C = all nodes (Thm 2 / NeurIPS'22)."""
    n = 1
    for v in diagram.topological_order():
        n *= variables.size(v)
    return int(n)
