"""Within-district c-degree-2 bilinear program (two independent latent hyperedges).

For a single free district with exactly two latent hyperedges e1, e2, the sharp
canonical program is bilinear in (p^(1), p^(2)). We build tensors (A, c) and
reuse the nested-LP / spatial-BB solver from ``gpp.py``.

Finite parameterization (sufficient)
------------------------------------
Partition endogenous nodes into only1 = e1\\e2, only2 = e2\\e1, shared = e1∩e2.

- U2 indexes ordinary response types for variables in only2 (size n2 = ∏ |R_V|).
  The U2 index also labels the “other latent” coordinate for shared variables.
- U1 indexes ordinary response types for only1, times — for each shared V — an
  arbitrary table Ω_U2 × Ω_pa(V) → Ω_V.

Observational / interventional maps are then bilinear forms in (p1, p2).
"""

from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

import numpy as np

from .gpp import (
    GPPData,
    _aggregate_i_rows,
    _aggregate_j_columns,
    _to_sparse,
    gpp_size_ok,
)
from .graph import CausalDiagram
from .query import Query
from .types_canon import _all_functions
from .variables import Variables

# Slightly larger than district-factor GPP: U1 tables can hit 512 on binary chains.
_LATENT_MAX_BLOCK = 512
_LATENT_MAX_PRODUCT = 10_000


def latent_bilinear_size_ok(n1: int, n2: int) -> bool:
    if n1 <= 0 or n2 <= 0:
        return False
    if max(n1, n2) > _LATENT_MAX_BLOCK:
        return False
    return n1 * n2 <= _LATENT_MAX_PRODUCT


def _resp_fns(variables: Variables, diagram: CausalDiagram, order: Sequence[str]):
    response_fns: dict[str, list[dict]] = {}
    for v in order:
        parents = list(diagram.parents(v))
        domain_in = variables.cells(parents) if parents else []
        response_fns[v] = _all_functions(domain_in, variables.domains[v])
    return response_fns


def _encode_u1_states(
    only1: tuple[str, ...],
    shared: tuple[str, ...],
    response_fns: dict[str, list[dict]],
    variables: Variables,
    diagram: CausalDiagram,
    n2: int,
) -> list[dict]:
    """Each U1 state: only1 response indices + shared tables [u2][pa] -> value."""
    only1_counts = [len(response_fns[v]) for v in only1] or [1]
    only1_types = list(product(*[range(c) for c in only1_counts])) if only1 else [()]

    shared_meta: list[tuple[str, list, list]] = []
    for v in shared:
        parents = list(diagram.parents(v))
        pa_cells = list(variables.cells(parents)) if parents else [()]
        dom = list(variables.domains[v])
        shared_meta.append((v, pa_cells, dom))

    # table size per shared var: n2 * n_pa * choices encoded as product of dom sizes
    # We enumerate all maps (u2, pa) -> value as product over positions of dom.
    shared_maps: list[list] = []
    for v, pa_cells, dom in shared_meta:
        n_pos = n2 * len(pa_cells)
        maps = list(product(dom, repeat=n_pos)) if n_pos else [()]
        shared_maps.append(maps)

    if not shared_maps:
        shared_combos = [()]
    else:
        shared_combos = list(product(*shared_maps))

    states: list[dict] = []
    for otype in only1_types:
        for smaps in shared_combos:
            tables: dict[str, dict[tuple, object]] = {}
            for si, (v, pa_cells, _dom) in enumerate(shared_meta):
                flat = smaps[si] if shared_meta else ()
                tab: dict[tuple, object] = {}
                idx = 0
                for u2 in range(n2):
                    for pa in pa_cells:
                        tab[(u2, pa)] = flat[idx]
                        idx += 1
                tables[v] = tab
            states.append({"only1": otype, "shared_tables": tables})
    return states


def build_latent_bilinear(
    variables: Variables,
    data: Mapping,
    query: Query,
    diagram: CausalDiagram,
    district: frozenset[str],
    *,
    notes: Sequence[str] | None = None,
) -> GPPData:
    """Build GPPData tensors for a c-degree-2 district (two latent hyperedges).

    Supports any :class:`~partialid.query_ast.LinComb` query (interventional,
    ATE, PNS, …). Fractional Cond queries are not supported here.
    """
    if query.expr is None:
        raise ValueError("latent bilinear requires query.expr")

    edges = diagram.latent_hyperedges_in(district)
    if len(edges) != 2:
        raise ValueError(f"latent bilinear expects exactly 2 hyperedges; got {len(edges)}")
    if set(diagram.nodes) != set(district):
        raise ValueError("latent bilinear expects diagram restricted to the free district")

    e1, e2 = edges[0], edges[1]
    s1, s2 = set(e1), set(e2)
    only1 = tuple(v for v in diagram.topological_order() if v in s1 - s2)
    only2 = tuple(v for v in diagram.topological_order() if v in s2 - s1)
    shared = tuple(v for v in diagram.topological_order() if v in s1 & s2)
    if not shared and (not only1 or not only2):
        raise ValueError("degenerate hyperedge overlap for latent bilinear")

    order = diagram.topological_order()
    response_fns = _resp_fns(variables, diagram, order)

    # U2: response types for only2 variables (index also keys shared tables)
    if only2:
        counts2 = [len(response_fns[v]) for v in only2]
        u2_types = list(product(*[range(c) for c in counts2]))
    else:
        # Shared-only overlap with empty only2: use a unit index
        u2_types = [()]
    n2 = len(u2_types)
    u1_states = _encode_u1_states(only1, shared, response_fns, variables, diagram, n2)
    n1 = len(u1_states)
    if not latent_bilinear_size_ok(n1, n2):
        raise ValueError(f"latent bilinear too large: ({n1}, {n2})")
    if not gpp_size_ok([min(n1, n2), min(n1, n2)]) and max(n1, n2) > _LATENT_MAX_BLOCK:
        raise ValueError(f"latent bilinear exceeds block gate: ({n1}, {n2})")

    from .formulate import _normalize_P
    from .query_compile import lincomb_coeff_via_eval

    cells, P = _normalize_P(variables, order, data)
    cell_index = {cell: k for k, cell in enumerate(cells)}
    A = np.zeros((len(cells), n1, n2), dtype=float)
    c_obj = np.zeros((n1, n2), dtype=float)

    only1_pos = {v: i for i, v in enumerate(only1)}
    only2_pos = {v: i for i, v in enumerate(only2)}

    def evaluate(i: int, j: int, interv: Mapping[str, object] | None) -> dict[str, object]:
        interv = dict(interv or {})
        st = u1_states[i]
        o2 = u2_types[j]
        values: dict[str, object] = {}
        for v in order:
            if v in interv:
                values[v] = interv[v]
                continue
            parents = diagram.parents(v)
            pa = tuple(values[p] for p in parents) if parents else ()
            if v in only1_pos:
                fi = st["only1"][only1_pos[v]]
                values[v] = response_fns[v][fi][pa]
            elif v in only2_pos:
                fi = o2[only2_pos[v]]
                values[v] = response_fns[v][fi][pa]
            elif v in shared:
                values[v] = st["shared_tables"][v][(j, pa)]
            else:
                # Unconfounded singleton inside district (should be rare)
                # Use first response as exogenous noise absorbed into U1 index 0
                values[v] = response_fns[v][0][pa]
        return values

    for i in range(n1):
        for j in range(n2):

            def _eval(
                interv: Mapping[str, object] | None,
                _i: int = i,
                _j: int = j,
            ) -> dict[str, object]:
                return evaluate(_i, _j, interv)

            world = _eval(None)
            cell = tuple(world[v] for v in order)
            A[cell_index[cell], i, j] += 1.0
            c_obj[i, j] = lincomb_coeff_via_eval(
                query.expr, _eval, variables=variables, diagram=diagram
            )

    n_raw_i, n_raw_j = A.shape[1], A.shape[2]
    A, c_obj, n_agg_i = _aggregate_i_rows(A, c_obj)
    A, c_obj, n_agg_j = _aggregate_j_columns(A, c_obj)
    if A.shape[2] > A.shape[1]:
        A = np.transpose(A, (0, 2, 1))
        c_obj = c_obj.T
        only1, only2 = only2, only1
    n1, n2 = A.shape[1], A.shape[2]
    sp = _to_sparse(A, c_obj)

    note_list = list(notes or [])
    note_list.append(
        f"latent bilinear c-degree-2 hyperedges {list(edges[0])}/{list(edges[1])} "
        f"types=({n1},{n2}) only1={list(only1)} only2={list(only2)} shared={list(shared)}"
    )
    if n_agg_i < n_raw_i or n_agg_j < n_raw_j:
        note_list.append(
            f"latent bilinear aggregated types: raw=({n_raw_i},{n_raw_j}) → ({n1},{n2})"
        )
    note_list.append(f"GPP sparse nnz_A={sp['a_val'].size} nnz_c={sp['c_val'].size}")

    # Stub district metadata for GPPData compatibility
    free = (frozenset(only1) | frozenset(shared), frozenset(only2) | frozenset(shared))
    return GPPData(
        variables=variables,
        diagram=diagram,
        order=order,
        free_districts=free,
        district_orders=(only1 + shared, only2 + shared),
        district_types=([tuple()] * n1, [tuple()] * n2),
        A=A,
        c_obj=c_obj,
        cells=cells,
        P_vec=P,
        response_fns=response_fns,
        notes=note_list,
        certified_eligible=True,
        n_raw_j=n_raw_j,
        **sp,
    )
