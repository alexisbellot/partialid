"""Size-gated nested multilinear solver for program degree ≥ 3.

Supports:
- **Factor-polynomial:** K=3 free λ=1 districts → 3-way tensors (A, c);
  recover feasible q3 via ALS / L-BFGS, then nest the K=2 GPP solver.
- **Latent-polynomial:** one free district with λ≥3 — not built as a sharp
  tensor here (support explodes); callers should fall back to an explicit
  merged-latent outer LP with ``tight=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from .gpp import (
    GPPBounds,
    GPPData,
    _aggregate_i_rows,
    _aggregate_j_columns,
    _to_sparse,
    gpp_size_ok,
    solve_bounds_gpp,
)
from .graph import CausalDiagram
from .query import Query
from .types_canon import _all_functions, evaluate_world
from .variables import Variables

_POLY_MAX_BLOCK = 64
_POLY_MAX_PRODUCT = 4_000
_POLY_MAX_SEEDS = 8


@dataclass
class PolyGPPData:
    """Factorized degree-3 program over three free districts."""

    variables: Variables
    diagram: CausalDiagram
    order: tuple[str, ...]
    free_districts: tuple[frozenset[str], ...]
    A: np.ndarray  # (n_cells, n1, n2, n3)
    c_obj: np.ndarray  # (n1, n2, n3)
    cells: list[tuple]
    P_vec: np.ndarray
    notes: list[str] = field(default_factory=list)
    n1: int = 0
    n2: int = 0
    n3: int = 0


def poly_size_ok(n_types: Sequence[int]) -> bool:
    if len(n_types) < 3:
        return False
    if any(n > _POLY_MAX_BLOCK for n in n_types):
        return False
    prod = 1
    for n in n_types:
        prod *= int(n)
        if prod > _POLY_MAX_PRODUCT:
            return False
    return True


def build_poly_factor_gpp(
    variables: Variables,
    data: Mapping,
    query: Query,
    diagram: CausalDiagram,
    free_districts: Sequence[frozenset[str]],
    *,
    notes: Sequence[str] | None = None,
) -> PolyGPPData:
    """Assemble (A, c) for K=3 free c-degree-1 districts."""
    if query.expr is None:
        raise ValueError("poly GPP requires query.expr")
    free = tuple(frozenset(d) for d in free_districts)
    if len(free) != 3:
        raise NotImplementedError(
            f"poly factor GPP supports exactly 3 free districts; got {len(free)}"
        )
    covered = set().union(*free)
    if covered != set(diagram.nodes):
        missing = set(diagram.nodes) - covered
        raise ValueError(
            f"poly GPP expects every ancestral node in a free district; missing {sorted(missing)}"
        )

    order = diagram.topological_order()
    response_fns: dict[str, list[dict]] = {}
    for v in order:
        parents = list(diagram.parents(v))
        domain_in = variables.cells(parents) if parents else []
        response_fns[v] = _all_functions(domain_in, variables.domains[v])

    d_orders = tuple(tuple(v for v in order if v in d) for d in free)
    d_types: list[list[tuple[int, ...]]] = []
    for dord in d_orders:
        counts = [len(response_fns[v]) for v in dord]
        d_types.append(list(product(*[range(c) for c in counts])))
    n1, n2, n3 = len(d_types[0]), len(d_types[1]), len(d_types[2])
    if not poly_size_ok([n1, n2, n3]):
        raise ValueError(f"poly GPP too large: blocks ({n1}, {n2}, {n3})")

    loc: dict[str, tuple[int, int]] = {}
    for di, dord in enumerate(d_orders):
        for j, v in enumerate(dord):
            loc[v] = (di, j)

    def full_type_ids(i: int, j: int, k: int) -> list[int]:
        local = [d_types[0][i], d_types[1][j], d_types[2][k]]
        out: list[int] = []
        for v in order:
            di, pos = loc[v]
            out.append(local[di][pos])
        return out

    from .formulate import _normalize_P
    from .query_compile import query_coeff_via_eval

    cells, P = _normalize_P(variables, order, data)
    n_cells = len(cells)
    A = np.zeros((n_cells, n1, n2, n3), dtype=float)
    c_obj = np.zeros((n1, n2, n3), dtype=float)
    cell_index = {cell: idx for idx, cell in enumerate(cells)}
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                tids = full_type_ids(i, j, k)

                def _eval(
                    interv: Mapping[str, object] | None,
                    _tids: list[int] = tids,
                ) -> dict[str, object]:
                    return evaluate_world(order, response_fns, _tids, diagram, interv)

                world = _eval(None)
                cell = tuple(world[v] for v in order)
                A[cell_index[cell], i, j, k] += 1.0
                c_obj[i, j, k] = query_coeff_via_eval(
                    query.expr,
                    _eval,
                    variables=variables,
                    diagram=diagram,
                    order=order,
                    cells=cells,
                    P_vec=P,
                )

    # Put smallest mode last for nesting over q3
    dims = [n1, n2, n3]
    order_idx = sorted(range(3), key=lambda t: dims[t], reverse=True)
    A = np.transpose(A, (0, order_idx[0] + 1, order_idx[1] + 1, order_idx[2] + 1))
    c_obj = np.transpose(c_obj, (order_idx[0], order_idx[1], order_idx[2]))
    free = (free[order_idx[0]], free[order_idx[1]], free[order_idx[2]])
    n1, n2, n3 = A.shape[1], A.shape[2], A.shape[3]

    note_list = list(notes or [])
    note_list.append(
        f"poly factor GPP K=3 blocks {[sorted(d) for d in free]} types=({n1},{n2},{n3})"
    )
    return PolyGPPData(
        variables=variables,
        diagram=diagram,
        order=order,
        free_districts=free,
        A=A,
        c_obj=c_obj,
        cells=cells,
        P_vec=P,
        notes=note_list,
        n1=n1,
        n2=n2,
        n3=n3,
    )


def _project_simplex_ls(A_mat: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    n = A_mat.shape[1]
    res = linprog(
        np.zeros(n),
        A_eq=np.vstack([A_mat, np.ones((1, n))]),
        b_eq=np.append(b, 1.0),
        bounds=[(0.0, None)] * n,
        method="highs",
    )
    if res.success:
        q = np.asarray(res.x, dtype=float)
        s = float(q.sum())
        return q / s if s > 0 else None
    # Soft ℓ1 residual
    from scipy.optimize import lsq_linear

    # Parameterize last coord via 1-sum; solve NNLS on free vars approx
    if n == 1:
        return np.ones(1)
    A_free = A_mat[:, :-1] - A_mat[:, [-1]]
    b_free = b - A_mat[:, -1]
    ls = lsq_linear(A_free, b_free, bounds=(0.0, 1.0))
    if not ls.success:
        return None
    q_free = np.clip(ls.x, 0.0, None)
    q_last = 1.0 - float(q_free.sum())
    if q_last < -1e-8:
        return None
    q = np.append(q_free, max(q_last, 0.0))
    s = float(q.sum())
    return q / s if s > 0 else None


def _contract_to_gpp(poly: PolyGPPData, q3: np.ndarray) -> GPPData:
    q3 = np.asarray(q3, dtype=float)
    s = float(q3.sum())
    if s <= 0:
        raise ValueError("q3 mass zero")
    q3 = q3 / s
    A2 = np.tensordot(poly.A, q3, axes=([3], [0]))
    c2 = np.tensordot(poly.c_obj, q3, axes=([2], [0]))
    A2, c2, _ = _aggregate_i_rows(A2, c2)
    A2, c2, _ = _aggregate_j_columns(A2, c2)
    if A2.shape[2] > A2.shape[1]:
        A2 = np.transpose(A2, (0, 2, 1))
        c2 = c2.T
        free = (poly.free_districts[1], poly.free_districts[0])
    else:
        free = (poly.free_districts[0], poly.free_districts[1])
    n1, n2 = A2.shape[1], A2.shape[2]
    if not gpp_size_ok([n1, n2]):
        raise ValueError(f"contracted GPP too large: ({n1}, {n2})")
    sp = _to_sparse(A2, c2)
    return GPPData(
        variables=poly.variables,
        diagram=poly.diagram,
        order=poly.order,
        free_districts=free,
        district_orders=((), ()),
        district_types=([tuple()] * n1, [tuple()] * n2),
        A=A2,
        c_obj=c2,
        cells=poly.cells,
        P_vec=poly.P_vec,
        response_fns={},
        notes=list(poly.notes) + [f"poly nest slice n3={poly.n3}"],
        certified_eligible=True,
        n_raw_j=n2,
        **sp,
    )


def _slice_feasible(poly: PolyGPPData, q3: np.ndarray) -> bool:
    try:
        gpp = _contract_to_gpp(poly, q3)
    except Exception:
        return False
    from .gpp import _find_feasible_q2, _lp_given_q2

    rng = np.random.default_rng(1)
    for j in range(min(gpp.n2, 8)):
        q2 = np.zeros(gpp.n2)
        q2[j] = 1.0
        _, _, ok, _ = _lp_given_q2(gpp, q2, maximize=True)
        if ok:
            return True
    return bool(_find_feasible_q2(gpp, rng, n_starts=6))


def _find_feasible_q3(poly: PolyGPPData, rng: np.random.Generator) -> list[np.ndarray]:
    """Recover feasible outer masses via 3-way ALS + residual minimization."""
    A, P = poly.A, poly.P_vec
    n1, n2, n3 = poly.n1, poly.n2, poly.n3
    found: list[np.ndarray] = []

    for _ in range(32):
        q1 = rng.dirichlet(np.ones(n1))
        q2 = rng.dirichlet(np.ones(n2))
        q3 = rng.dirichlet(np.ones(n3))
        for _it in range(50):
            A1 = np.einsum("cijk,j,k->ci", A, q2, q3)
            q1n = _project_simplex_ls(A1, P)
            if q1n is None:
                break
            q1 = q1n
            A2 = np.einsum("cijk,i,k->cj", A, q1, q3)
            q2n = _project_simplex_ls(A2, P)
            if q2n is None:
                break
            q2 = q2n
            A3 = np.einsum("cijk,i,j->ck", A, q1, q2)
            q3n = _project_simplex_ls(A3, P)
            if q3n is None:
                break
            q3 = q3n
            pred = np.einsum("cijk,i,j,k->c", A, q1, q2, q3)
            if float(np.sum((pred - P) ** 2)) <= 1e-12:
                break
        pred = np.einsum("cijk,i,j,k->c", A, q1, q2, q3)
        if float(np.sum((pred - P) ** 2)) > 1e-8:
            continue
        if _slice_feasible(poly, q3):
            found.append(q3 / float(q3.sum()))
        if len(found) >= _POLY_MAX_SEEDS:
            break

    if found:
        return found

    # L-BFGS on (q1,q2,q3) softmax parameters
    from scipy.optimize import minimize

    def pack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, b = n1, n1 + n2
        e1 = np.exp(theta[:a] - np.max(theta[:a]))
        e2 = np.exp(theta[a:b] - np.max(theta[a:b]))
        e3 = np.exp(theta[b:] - np.max(theta[b:]))
        return e1 / e1.sum(), e2 / e2.sum(), e3 / e3.sum()

    def residual(theta: np.ndarray) -> float:
        q1, q2, q3 = pack(theta)
        pred = np.einsum("cijk,i,j,k->c", A, q1, q2, q3)
        return float(np.sum((pred - P) ** 2))

    for _ in range(24):
        theta0 = rng.normal(size=n1 + n2 + n3)
        res = minimize(
            residual,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": 300, "ftol": 1e-14},
        )
        if residual(res.x) > 1e-10:
            continue
        _, _, q3 = pack(res.x)
        if _slice_feasible(poly, q3):
            found.append(q3 / float(q3.sum()))
        if len(found) >= _POLY_MAX_SEEDS:
            break
    return found


def solve_bounds_poly(poly: PolyGPPData) -> GPPBounds:
    """Nest over q3; solve bilinear GPP on each feasible outer mass."""
    notes = list(poly.notes)
    rng = np.random.default_rng(0)
    seeds = _find_feasible_q3(poly, rng)
    notes.append(f"poly feasible q3 seeds={len(seeds)}")
    if not seeds:
        return GPPBounds(
            lower=float("nan"),
            upper=float("nan"),
            success=False,
            message="poly nest: no feasible q3",
            notes=notes,
            certified=False,
        )

    # One local perturbation per seed (keeps cost ~O(#seeds) GPP solves)
    candidates = list(seeds)
    for q in seeds[:3]:
        noise = rng.dirichlet(np.ones(poly.n3))
        candidates.append(0.9 * q + 0.1 * noise)

    best_lo = float("inf")
    best_hi = float("-inf")
    any_ok = False
    q3_lo = q3_hi = None
    all_certified = True
    for q3 in candidates:
        try:
            gpp = _contract_to_gpp(poly, q3)
        except Exception:
            continue
        gb = solve_bounds_gpp(gpp)
        if not gb.success:
            continue
        any_ok = True
        all_certified = all_certified and gb.certified
        if gb.lower < best_lo:
            best_lo = gb.lower
            q3_lo = np.asarray(q3, dtype=float)
        if gb.upper > best_hi:
            best_hi = gb.upper
            q3_hi = np.asarray(q3, dtype=float)

    if not any_ok:
        return GPPBounds(
            lower=float("nan"),
            upper=float("nan"),
            success=False,
            message="poly nest: seeds but contracted GPP failed",
            notes=notes,
            certified=False,
        )
    notes.append(f"poly nested GPP over n3={poly.n3}; bounds=[{best_lo:.6g}, {best_hi:.6g}]")
    return GPPBounds(
        lower=float(best_lo),
        upper=float(best_hi),
        success=True,
        message="ok",
        q_blocks_lower=[q3_lo] if q3_lo is not None else None,
        q_blocks_upper=[q3_hi] if q3_hi is not None else None,
        notes=notes,
        certified=bool(all_certified and poly.n3 <= 4),
    )
