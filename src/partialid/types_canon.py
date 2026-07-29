"""Canonical response-function types (Balke–Pearl / Zhang)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

from .graph import CausalDiagram
from .variables import Variables


def _all_functions(domain_in: Sequence, domain_out: Sequence) -> list[dict]:
    """All maps from parent-configurations → outcome value."""
    out_vals = list(domain_out)
    if not domain_in:
        return [{(): v} for v in out_vals]
    keys = list(domain_in)
    return [
        {keys[i]: assignment[i] for i in range(len(keys))}
        for assignment in product(out_vals, repeat=len(keys))
    ]


def evaluate_world(
    order: Sequence[str],
    response_fns: dict[str, list[dict]],
    type_ids: Sequence[int],
    diagram: CausalDiagram,
    intervention: Mapping[str, object] | None = None,
) -> dict[str, object]:
    intervention = dict(intervention or {})
    values: dict[str, object] = {}
    for v, fi in zip(order, type_ids):
        if v in intervention:
            values[v] = intervention[v]
            continue
        parents = diagram.parents(v)
        key = tuple(values[p] for p in parents) if parents else ()
        values[v] = response_fns[v][fi][key]
    return values


@dataclass
class TypeSpace:
    """Joint response types for a single c-component (Prop. 0.2 linear reduction)."""

    variables: Variables
    diagram: CausalDiagram
    order: tuple[str, ...]
    response_fns: dict[str, list[dict]]
    joint_types: list[tuple[int, ...]]
    fiber: dict[tuple, list[int]]
    n_types: int
    type_index: dict[tuple[int, ...], int]

    @classmethod
    def from_diagram(cls, variables: Variables, diagram: CausalDiagram) -> TypeSpace:
        order = diagram.topological_order()
        for n in order:
            if n not in variables:
                raise ValueError(f"diagram node {n!r} not in Variables")

        response_fns: dict[str, list[dict]] = {}
        for v in order:
            parents = list(diagram.parents(v))
            domain_in = variables.cells(parents) if parents else []
            response_fns[v] = _all_functions(domain_in, variables.domains[v])

        fn_counts = [len(response_fns[v]) for v in order]
        joint_types = list(product(*[range(c) for c in fn_counts]))
        type_index = {jt: i for i, jt in enumerate(joint_types)}

        # Observational fibers are indexed by *observed* coordinates only
        # (latent endogenous marginalized; latent endogenous / measurement error).
        obs_order = variables.observed_names(order)
        fiber: dict[tuple, list[int]] = {cell: [] for cell in variables.cells(obs_order)}
        for t_idx, type_ids in enumerate(joint_types):
            world = evaluate_world(order, response_fns, type_ids, diagram, None)
            cell = tuple(world[v] for v in obs_order)
            fiber[cell].append(t_idx)

        return cls(
            variables=variables,
            diagram=diagram,
            order=order,
            response_fns=response_fns,
            joint_types=joint_types,
            fiber=fiber,
            n_types=len(joint_types),
            type_index=type_index,
        )

    def evaluate_intervention(
        self, type_idx: int, intervention: Mapping[str, object]
    ) -> dict[str, object]:
        return evaluate_world(
            self.order,
            self.response_fns,
            self.joint_types[type_idx],
            self.diagram,
            intervention,
        )

    def observational_cell(self, type_idx: int) -> tuple:
        world = evaluate_world(
            self.order,
            self.response_fns,
            self.joint_types[type_idx],
            self.diagram,
            None,
        )
        return tuple(world[v] for v in self.order)

    def query_coefficient(
        self,
        type_idx: int,
        outcome: str,
        value: object,
        intervention: Mapping[str, object],
    ) -> float:
        world = self.evaluate_intervention(type_idx, intervention)
        return 1.0 if world[outcome] == value else 0.0

    def pns_coefficient(self, type_idx: int, treatment: str, outcome: str) -> float:
        y1 = self.evaluate_intervention(type_idx, {treatment: 1})[outcome]
        y0 = self.evaluate_intervention(type_idx, {treatment: 0})[outcome]
        return 1.0 if (y1 == 1 and y0 == 0) else 0.0

    def pn_numerator_coefficient(self, type_idx: int, treatment: str, outcome: str) -> float:
        """Indicator for types in the (X=1,Y=1) fiber with Y_0=0 (PN numerator)."""
        cell = self.observational_cell(type_idx)
        if cell[self.order.index(treatment)] != 1 or cell[self.order.index(outcome)] != 1:
            return 0.0
        y0 = self.evaluate_intervention(type_idx, {treatment: 0})[outcome]
        return 1.0 if y0 == 0 else 0.0

    def ps_numerator_coefficient(self, type_idx: int, treatment: str, outcome: str) -> float:
        """Indicator for types in the (X=0,Y=0) fiber with Y_1=1 (PS numerator)."""
        cell = self.observational_cell(type_idx)
        if cell[self.order.index(treatment)] != 0 or cell[self.order.index(outcome)] != 0:
            return 0.0
        y1 = self.evaluate_intervention(type_idx, {treatment: 1})[outcome]
        return 1.0 if y1 == 1 else 0.0

    def outcome_response_index(self, type_idx: int, outcome: str) -> int:
        return self.joint_types[type_idx][self.order.index(outcome)]

    def treatment_value_if_root(self, type_idx: int, treatment: str) -> object:
        if self.diagram.parents(treatment):
            raise ValueError("treatment is not a root in the diagram")
        fi = self.outcome_response_index(type_idx, treatment)
        return self.response_fns[treatment][fi][()]

    def response_on(self, type_idx: int, var: str, parent_values: Mapping[str, object]) -> object:
        fi = self.outcome_response_index(type_idx, var)
        parents = self.diagram.parents(var)
        key = tuple(parent_values[p] for p in parents) if parents else ()
        return self.response_fns[var][fi][key]
