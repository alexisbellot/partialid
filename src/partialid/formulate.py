"""Build the partial-identification LP from types + assumptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse

from .assumptions import (
    AssumptionSet,
    CDependenceAssumption,
    ExclusionAssumption,
    FSensitivityAssumption,
    IndependenceAssumption,
    MarginBudgetAssumption,
    MonotonicityAssumption,
    MTSAssumption,
    NoDefiersAssumption,
    RosenbaumAssumption,
    SupportBoundAssumption,
    TanMSMAssumption,
    as_assumption_set,
)
from .graph import CausalDiagram, full_latent_hyperedge
from .query import Query
from .types_canon import TypeSpace
from .variables import Variables


def _is_manski_full_latent(diagram: CausalDiagram) -> bool:
    """True when one hyperedge covers every node (order-only Manski encoding)."""
    return (
        len(diagram.nodes) >= 2
        and len(diagram.bidirected) == 1
        and set(diagram.bidirected[0]) == set(diagram.nodes)
    )


def _merged_latent_diagram(diagram: CausalDiagram) -> CausalDiagram:
    """Outer approximation: collapse to one latent on all nodes."""
    return CausalDiagram(
        nodes=diagram.nodes,
        directed=diagram.directed,
        bidirected=full_latent_hyperedge(diagram.nodes),
    )


@dataclass
class LPData:
    """Numeric LP: min/max c'q s.t. A_eq q = b_eq, A_ub q <= b_ub, q >= 0.

    When ``c_den`` is set, the objective is linear-fractional
    ``(cᵀq)/(c_denᵀq)`` (conditional queries; NeurIPS 2021 §4).
    """

    c: np.ndarray
    A_eq: sparse.csr_matrix
    b_eq: np.ndarray
    A_ub: sparse.csr_matrix | None
    b_ub: np.ndarray | None
    type_space: TypeSpace
    allowed_types: np.ndarray  # boolean mask over all types (after hard faces)
    cells: list[tuple]
    P_vec: np.ndarray
    method_hint: str = "lp"  # "lp" | "fiber" | "mip" | "convex" | "gpp" | "gpp/latent" | "gpp/poly" | "fractional"
    notes: list[str] = field(default_factory=list)
    support_k: int | None = None
    f_sensitivity: FSensitivityAssumption | None = None
    e0: float | None = None
    # Conditional propensity π0[pa][x] for anchored budgets / f-sensitivity
    pi0: dict | None = None
    c_den: np.ndarray | None = None  # fractional denom coefficients
    # True when the LP is a valid outer approximation (may be loose)
    outer_approximation: bool = False
    # Optional factorized GPP / poly (method_hint startswith "gpp")
    gpp: object | None = None


def _normalize_P(
    variables: Variables,
    order: Sequence[str],
    data: Mapping,
) -> tuple[list[tuple], np.ndarray]:
    cells = variables.cells(order)
    P = np.zeros(len(cells), dtype=float)
    # accept dict keyed by tuples in `order`, or by name-dict
    for key, val in data.items():
        if isinstance(key, dict):
            cell = tuple(key[v] for v in order)
        else:
            cell = tuple(key)
            if len(cell) != len(order):
                raise ValueError(f"data key {key} has length {len(cell)}, expected {len(order)}")
        idx = cells.index(cell)
        P[idx] = float(val)
    s = P.sum()
    if s <= 0:
        raise ValueError("data probabilities must sum to a positive value")
    if abs(s - 1.0) > 1e-8:
        P = P / s
    return cells, P


def _remargin_data(
    variables: Variables,
    data: Mapping,
    keep: Sequence[str],
) -> dict[tuple, float]:
    """Marginalize a joint table onto ``keep``.

    Tuple keys are read in observed-variable order (declaration / topo order
    among observed names). Latent endogenous nodes do not index ``data``.
    """
    src = variables.names
    if set(keep) - set(src):
        raise ValueError(f"keep {list(keep)} not subset of {list(src)}")
    obs_src = variables.observed_names(src)
    cells, P = _normalize_P(variables, obs_src, data)
    # Project onto the observed coordinates that remain in ``keep``.
    obs_keep = tuple(n for n in keep if n in obs_src)
    obs_src_idx = {n: i for i, n in enumerate(obs_src)}
    marg: dict[tuple, float] = {}
    for cell, p in zip(cells, P):
        key = tuple(cell[obs_src_idx[n]] for n in obs_keep)
        marg[key] = marg.get(key, 0.0) + float(p)
    return marg


def _exo_are_markovian_children(
    diagram: CausalDiagram,
    free: frozenset[str],
    exo: Sequence[str],
) -> bool:
    """True if every exo node is an unconfounded descendant of ``free`` (not WL)."""
    free_s, exo_s = set(free), set(exo)
    if not exo_s:
        return False
    for v in exo_s:
        # Must not appear in any latent hyperedge
        if any(v in he for he in diagram.bidirected):
            return False
        for p in diagram.parents(v):
            if p not in free_s and p not in exo_s:
                return False
    # No directed edges from exo back into free (exo are children / post)
    for a, b in diagram.directed:
        if a in exo_s and b in free_s:
            return False
    return True


def _empirical_kernel(
    variables: Variables,
    order: Sequence[str],
    data: Mapping,
    child: str,
    parents: Sequence[str],
) -> dict[tuple, dict[object, float]]:
    """Estimate P(child | parents) from the observational table."""
    cells, P = _normalize_P(variables, order, data)
    idx = {n: i for i, n in enumerate(order)}
    if child not in idx:
        raise ValueError(f"{child!r} not in observational order")
    for p in parents:
        if p not in idx:
            raise ValueError(f"parent {p!r} not in observational order")
    counts: dict[tuple, dict[object, float]] = {}
    for cell, pmass in zip(cells, P):
        pa = tuple(cell[idx[p]] for p in parents)
        bucket = counts.setdefault(pa, {})
        bucket[cell[idx[child]]] = bucket.get(cell[idx[child]], 0.0) + float(pmass)
    out: dict[tuple, dict[object, float]] = {}
    for pa, bucket in counts.items():
        s = sum(bucket.values())
        if s <= 0:
            continue
        out[pa] = {k: v / s for k, v in bucket.items()}
    return out


def _outcome_mass_with_kernels(
    world: Mapping[str, object],
    exo_order: Sequence[str],
    kernels: Mapping[str, dict[tuple, dict[object, float]]],
    parent_of: Mapping[str, tuple[str, ...]],
    outcome: str,
    value: object,
) -> float:
    """P(outcome=value | free-world) after chaining Markovian exo kernels."""
    # Values already known from the free district
    known = dict(world)
    if outcome in known:
        return 1.0 if known[outcome] == value else 0.0

    # Forward: if every exo before outcome is a Dirac given parents, fold in;
    # otherwise enumerate product of supports along exo_order up to outcome.
    pending = [v for v in exo_order if v not in known]
    # Recursive expectation
    def _expect(i: int) -> float:
        if outcome in known:
            return 1.0 if known[outcome] == value else 0.0
        if i >= len(pending):
            return 1.0 if known.get(outcome) == value else 0.0
        v = pending[i]
        parents = parent_of[v]
        pa = tuple(known[p] for p in parents)
        dist = kernels[v].get(pa)
        if dist is None:
            return 0.0
        if v == outcome:
            return float(dist.get(value, 0.0))
        total = 0.0
        for val, pr in dist.items():
            if pr <= 0:
                continue
            known[v] = val
            total += pr * _expect(i + 1)
            del known[v]
        return total

    return _expect(0)


def _apply_exclusions(diagram: CausalDiagram, assumptions: AssumptionSet) -> CausalDiagram:
    """Remove directed edges forbidden by ExclusionAssumption."""
    drop = {(a.parent, a.child) for a in assumptions.of_type(ExclusionAssumption)}
    if not drop:
        return diagram
    directed = tuple(e for e in diagram.directed if e not in drop)
    return CausalDiagram(
        nodes=diagram.nodes,
        directed=directed,
        bidirected=diagram.bidirected,
    )


def _hard_face_mask(ts: TypeSpace, assumptions: AssumptionSet) -> np.ndarray:
    """Boolean mask: True = type allowed by hard face assumptions."""
    allow = _mono_mask(ts, assumptions)

    for nd in assumptions.of_type(NoDefiersAssumption):
        mono_set = AssumptionSet(
            (
                MonotonicityAssumption(
                    outcome=nd.treatment,
                    treatment=nd.instrument,
                    direction="increasing",
                ),
            )
        )
        allow &= _mono_mask(ts, mono_set)

    return allow


def _mono_mask(ts: TypeSpace, assumptions: AssumptionSet) -> np.ndarray:
    """Mask from monotonicity atoms."""
    n = ts.n_types
    allow = np.ones(n, dtype=bool)
    for m in assumptions.of_type(MonotonicityAssumption):
        for t in range(n):
            if not allow[t]:
                continue
            treatment = m.treatment
            outcome = m.outcome
            parents = list(ts.diagram.parents(outcome))
            if treatment not in parents:
                continue
            other_parents = [p for p in parents if p != treatment]
            other_cells = ts.variables.cells(other_parents) if other_parents else [()]
            fi = ts.outcome_response_index(t, outcome)
            fn = ts.response_fns[outcome][fi]
            ok = True
            t_domain = list(ts.variables.domains[treatment])
            for oc in other_cells:
                oc = tuple(oc) if other_parents else ()
                vals = []
                for tv in t_domain:
                    full = []
                    oi = 0
                    for p in parents:
                        if p == treatment:
                            full.append(tv)
                        else:
                            full.append(oc[oi])
                            oi += 1
                    vals.append(fn[tuple(full)])
                for a, b in zip(vals, vals[1:]):
                    if m.direction == "increasing" and not (a <= b):
                        ok = False
                        break
                    if m.direction == "decreasing" and not (a >= b):
                        ok = False
                        break
                if not ok:
                    break
            allow[t] = ok
    return allow


def _build_objective(
    ts: TypeSpace,
    query: Query,
    allow: np.ndarray,
    *,
    cells: list[tuple] | None = None,
    P_vec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Compile ``query.expr`` → ``(c, c_den|None, notes)``.

    Uses :mod:`partialid.query_compile` (CUT unnesting + Prop. 3
    coefficients; conditionals via NeurIPS 2021 §4 ratios).
    """
    from .query_compile import FractionalObjective, LinearObjective, compile_objective

    if query.expr is None:
        raise ValueError("query.expr is required for objective compilation")
    obj = compile_objective(
        ts,
        query.expr,
        allow,
        cells=cells,
        P_vec=P_vec,
        variables=ts.variables,
        diagram=ts.diagram,
    )
    if isinstance(obj, LinearObjective):
        return obj.c, None, list(obj.notes)
    assert isinstance(obj, FractionalObjective)
    return obj.c_num, obj.c_den, list(obj.notes)


def _mts_rows(
    ts: TypeSpace,
    allow: np.ndarray,
    cells: list[tuple],
    P_vec: np.ndarray,
    treatment: str,
    outcome: str,
    direction: str,
) -> tuple[list[np.ndarray], list[float]]:
    """MTS: E[Y_x|X=1] ≥ E[Y_x|X=0] (or reverse) for each binary x."""
    if tuple(ts.variables.domains[treatment]) != (0, 1):
        raise NotImplementedError("MTS currently requires binary treatment 0/1")
    if ts.diagram.parents(treatment):
        raise NotImplementedError("MTS currently requires root treatment")
    # cells are in observed-variable order (see formulate)
    obs = ts.variables.observed_names(ts.order)
    t_pos = list(obs).index(treatment)
    px1 = float(sum(p for cell, p in zip(cells, P_vec) if cell[t_pos] == 1))
    px0 = float(sum(p for cell, p in zip(cells, P_vec) if cell[t_pos] == 0))
    if not (px0 > 1e-15 and px1 > 1e-15):
        raise ValueError("MTS requires 0 < P(X=0), P(X=1)")

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    sign = 1.0 if direction == "increasing" else -1.0
    for x in (0, 1):
        # E[Y_x|X=1] - E[Y_x|X=0] ≥ 0  ⇒  -sign * (·) ≤ 0
        r = np.zeros(ts.n_types)
        for t in range(ts.n_types):
            if not allow[t]:
                continue
            xv = ts.treatment_value_if_root(t, treatment)
            yx = ts.evaluate_intervention(t, {treatment: x})[outcome]
            ind = 1.0 if yx == 1 else 0.0
            if xv == 1:
                r[t] += ind / px1
            elif xv == 0:
                r[t] -= ind / px0
        rows.append(-sign * r)
        rhs.append(0.0)
    return rows, rhs


def _inert_prune_order_only(
    variables: Variables,
    data: Mapping,
    query: Query,
    diagram: CausalDiagram,
    notes: list[str],
) -> tuple[Variables, Mapping, CausalDiagram]:
    """Drop variables inert for interventional / ATE queries under order-only / max confounding.

    THEORY Move 2: for Q=P(Y|do(X)) under order-only, only X and Y matter.
    """
    if query.kind not in ("interventional", "ate"):
        return variables, data, diagram
    # Only Manski / full-latent order diagrams are inert-prunable.
    if not _is_manski_full_latent(diagram):
        return variables, data, diagram

    if query.kind == "ate":
        treat = (query.meta or {}).get("treatment", "X")
        keep = {treat, query.outcome}
    else:
        keep = set(query.intervention) | {query.outcome}
    # Latent endogenous: never drop observed proxies (latent endogenous / measurement error); they carry P.
    if variables.observed is not None and set(variables.observed) != set(variables.names):
        keep |= set(variables.observed)
    if keep.issubset(diagram.nodes) and set(diagram.nodes) - keep:
        dropped = [n for n in diagram.nodes if n not in keep]
        # preserve relative order
        new_order = tuple(n for n in diagram.topological_order() if n in keep)
        notes.append(f"inert prune dropped {dropped}; kept {list(new_order)}")
        V2 = variables.restrict(new_order)
        # marginalize data over *observed* coordinates only
        old_order = diagram.topological_order()
        obs_old = variables.observed_names(old_order)
        cells, P = _normalize_P(variables, obs_old, data)
        # New observed key: projection of obs_old onto observed vars in new_order
        obs_new = V2.observed_names(new_order)
        obs_old_idx = {n: i for i, n in enumerate(obs_old)}
        marg: dict[tuple, float] = {}
        for cell, p in zip(cells, P):
            key = tuple(cell[obs_old_idx[n]] for n in obs_new)
            marg[key] = marg.get(key, 0.0) + p
        g2 = CausalDiagram.from_order(new_order)
        return V2, marg, g2
    return variables, data, diagram


def formulate(
    variables: Variables,
    data: Mapping,
    query: Query,
    assumptions: AssumptionSet | None = None,
    *,
    optimize: bool = True,
    prefer_gpp: bool = True,
    prefer_poly: bool = False,
) -> LPData:
    """Build the partial-identification LP (linear reduction or instrument R-side).

    Parameters
    ----------
    optimize :
        If True (default), use inert-variable pruning and instrument R-side LPs.
        If False, use a *naive* baseline: full joint type expansion under maximal
        confounding (no inert prune, no instrument factorization).
    prefer_gpp :
        When classify yields bilinear factor/latent and size gates pass, build a
        GPP instead of the merged-latent outer LP. Set False to force the outer LP.
    prefer_poly :
        When classify yields POLYNOMIAL (deg≥3), try the nested multilinear
        ``gpp/poly`` solver. Default False: use the merged-latent outer LP
        (``tight=False``). Opt in with ``prefer_poly=True``.
    """
    from .instrument import formulate_instrument

    assumptions = as_assumption_set(assumptions)
    notes: list[str] = []
    if not optimize:
        notes.append("naive: no inert prune, no instrument R-side LP")

    diagram = assumptions.diagram()
    if diagram is None:
        diagram = CausalDiagram.from_order(variables.names)
        notes.append("defaulted to declaration order with one full latent (Manski)")

    diagram = _apply_exclusions(diagram, assumptions)

    # Inert prune first; WL–WR / instrument LP comes from post-cut classify below
    # (single cut authority — THEORY_PROGRAM_DEGREE §6 C2).
    if optimize:
        variables, data, diagram = _inert_prune_order_only(
            variables, data, query, diagram, notes
        )

    outer = False
    # Hard independence: v1 does not factorize the product program. Force a
    # merged-latent outer so we never claim sharp bounds under an unimplemented atom.
    if assumptions.of_type(IndependenceAssumption):
        notes.append(
            "independence atom: hard c-split not fully factorized; "
            "using merged-latent outer LP (valid outer approximation)."
        )
        outer = True
        diagram = _merged_latent_diagram(diagram)

    # Program-degree router: dispatch solely from post-cut classify
    # (THEORY_PROGRAM_DEGREE §6).
    if optimize and query.expr is not None:
        from .degree1 import analyze_degree1, induced_subgraph
        from .program_degree import ProgramKind, classify_program
        from .query_ast import Cond as _Cond
        from .query_ast import LinComb as _LinComb
        from .query_compile import gpp_expr_supported

        fac = analyze_degree1(
            diagram,
            treatments=list(query.intervention.keys()),
            outcome=query.outcome,
            query=query,
            variables=variables,
        )
        notes.extend(list(fac.notes))
        prog = classify_program(
            diagram,
            treatments=list(query.intervention.keys()),
            outcome=query.outcome,
            query=query,
            variables=variables,
        )
        notes.extend(
            [
                n
                for n in prog.notes
                if n.startswith("program:")
                or n.startswith("C2 ")
                or n.startswith("out of (H)")
            ]
        )

        def _gpp_query_ok(diag: CausalDiagram, vars_: Variables, data_: Mapping) -> bool:
            if isinstance(query.expr, _LinComb):
                return True
            if not isinstance(query.expr, _Cond):
                return False
            ord_ = diag.topological_order()
            cells_p, p_vec = _normalize_P(vars_, ord_, data_)
            return gpp_expr_supported(
                query.expr,
                variables=vars_,
                diagram=diag,
                order=ord_,
                cells=cells_p,
                P_vec=p_vec,
            )

        def _gpp_stub(gpp_obj, hint: str = "gpp") -> LPData:
            v0 = diagram.topological_order()[0]
            stub_g = CausalDiagram(nodes=(v0,), directed=(), bidirected=())
            stub_ts = TypeSpace.from_diagram(variables.restrict((v0,)), stub_g)
            n = stub_ts.n_types
            return LPData(
                c=np.zeros(n),
                A_eq=sparse.csr_matrix((0, n)),
                b_eq=np.zeros(0),
                A_ub=None,
                b_ub=None,
                type_space=stub_ts,
                allowed_types=np.ones(n, dtype=bool),
                cells=[],
                P_vec=np.zeros(0),
                method_hint=hint,
                notes=list(gpp_obj.notes),
                outer_approximation=False,
                gpp=gpp_obj,
            )

        def _merged_outer(reason: str) -> None:
            nonlocal outer, diagram
            notes.append(reason)
            outer = True
            diagram = _merged_latent_diagram(diagram)

        def _restrict_ancestral() -> None:
            nonlocal variables, data, diagram
            an_order = tuple(
                n for n in diagram.topological_order() if n in fac.ancestral
            )
            if set(an_order) != set(diagram.nodes):
                data = _remargin_data(variables, data, an_order)
                variables = variables.restrict(an_order)
                diagram = induced_subgraph(diagram, an_order)
                notes.append(f"restricted to ancestral {list(an_order)}")

        free = prog.free_districts

        if prog.kind == ProgramKind.LINEAR:
            # Prefer recorded C2 cut → instrument LP (same as early Prop. 8.4 path)
            if (
                prog.wl_wr_cut is not None
                and query.kind in ("interventional", "ate")
            ):
                _restrict_ancestral()
                instruments, wr = prog.wl_wr_cut
                notes.append(
                    f"Prop. 8.4 WL–WR cut (classify C2): "
                    f"WL={list(instruments)}, WR={list(wr)}"
                )
                lp = formulate_instrument(
                    variables, data, query, diagram, instruments, wr, assumptions
                )
                lp.notes = notes + lp.notes
                lp.outer_approximation = False
                return lp

            if len(free) == 1:
                D = free[0]
                r_side = tuple(n for n in diagram.topological_order() if n in D)
                an_order = tuple(
                    n for n in diagram.topological_order() if n in fac.ancestral
                )
                exo = tuple(
                    n for n in an_order if n not in D
                )

                if set(an_order) != set(diagram.nodes):
                    data = _remargin_data(variables, data, an_order)
                    variables = variables.restrict(an_order)
                    diagram = induced_subgraph(diagram, an_order)
                    notes.append(f"restricted to ancestral {list(an_order)}")

                if exo:
                    if query.kind in ("interventional", "ate") and _exo_are_markovian_children(
                        diagram, frozenset(D), exo
                    ):
                        from .types_canon import evaluate_world

                        an_obs = variables.observed_names(an_order)
                        parent_of = {v: tuple(diagram.parents(v)) for v in exo}
                        kernels = {
                            v: _empirical_kernel(
                                variables, an_obs, data, v, parent_of[v]
                            )
                            for v in exo
                        }
                        data_r = _remargin_data(variables, data, r_side)
                        variables_r = variables.restrict(r_side)
                        diagram_r = induced_subgraph(diagram, r_side)
                        notes.append(
                            f"c-degree-1 LP on free district {list(r_side)}; "
                            f"absorbed Markovian exo {list(exo)} into objective"
                        )
                        ts = TypeSpace.from_diagram(variables_r, diagram_r)
                        cells, P = _normalize_P(variables_r, ts.order, data_r)
                        allow = _hard_face_mask(ts, assumptions)
                        c = np.zeros(ts.n_types, dtype=float)
                        if query.kind == "ate":
                            meta = query.meta or {}
                            treatment = str(meta.get("treatment", "X"))
                            t1, t0 = meta.get("t1", 1), meta.get("t0", 0)
                            yv = meta.get("y", query.value)
                            for t in range(ts.n_types):
                                if not allow[t]:
                                    continue
                                w1 = evaluate_world(
                                    ts.order,
                                    ts.response_fns,
                                    ts.joint_types[t],
                                    ts.diagram,
                                    {treatment: t1},
                                )
                                w0 = evaluate_world(
                                    ts.order,
                                    ts.response_fns,
                                    ts.joint_types[t],
                                    ts.diagram,
                                    {treatment: t0},
                                )
                                c[t] = _outcome_mass_with_kernels(
                                    w1, exo, kernels, parent_of, query.outcome, yv
                                ) - _outcome_mass_with_kernels(
                                    w0, exo, kernels, parent_of, query.outcome, yv
                                )
                        else:
                            interv = dict(query.intervention)
                            for t in range(ts.n_types):
                                if not allow[t]:
                                    continue
                                w = evaluate_world(
                                    ts.order,
                                    ts.response_fns,
                                    ts.joint_types[t],
                                    ts.diagram,
                                    interv,
                                )
                                c[t] = _outcome_mass_with_kernels(
                                    w,
                                    exo,
                                    kernels,
                                    parent_of,
                                    query.outcome,
                                    query.value,
                                )
                        n = ts.n_types
                        eq_rows = []
                        eq_rhs = []
                        for i, cell in enumerate(cells):
                            row = np.zeros(n)
                            for ti in ts.fiber[cell]:
                                if allow[ti]:
                                    row[ti] = 1.0
                            eq_rows.append(row)
                            eq_rhs.append(P[i])
                        return LPData(
                            c=c,
                            A_eq=sparse.csr_matrix(np.vstack(eq_rows)),
                            b_eq=np.asarray(eq_rhs, dtype=float),
                            A_ub=None,
                            b_ub=None,
                            type_space=ts,
                            allowed_types=allow,
                            cells=cells,
                            P_vec=P,
                            method_hint="fiber",
                            notes=notes,
                            outer_approximation=False,
                        )

                    notes.append(
                        f"LINEAR free district {list(r_side)} with exo {list(exo)}; "
                        f"no absorption path for query kind {query.kind!r} — "
                        "using ancestral TypeSpace (may merge districts)."
                    )
                    # Fall through to TypeSpace on ancestral diagram.
                else:
                    if set(an_order) != set(r_side):
                        data = _remargin_data(variables, data, r_side)
                    variables = variables.restrict(r_side)
                    diagram = induced_subgraph(diagram, r_side)
                    notes.append(f"c-degree-1 single-block LP on {list(r_side)}")
                    # fall through with restricted diagram

        elif prog.kind == ProgramKind.POINT:
            an_order = tuple(
                n for n in diagram.topological_order() if n in fac.ancestral
            )
            if set(an_order) != set(diagram.nodes):
                diagram = induced_subgraph(diagram, an_order)
                notes.append(
                    "POINT: no free district; restricted ancestral "
                    f"{list(an_order)} (no merge)"
                )

        elif prog.kind == ProgramKind.BILINEAR_LATENT and prefer_gpp:
            from .latent_bilinear import build_latent_bilinear

            _restrict_ancestral()
            D = free[0]
            d_order = tuple(n for n in diagram.topological_order() if n in D)
            if set(d_order) != set(diagram.nodes):
                data = _remargin_data(variables, data, d_order)
                variables = variables.restrict(d_order)
                diagram = induced_subgraph(diagram, d_order)
            try:
                if not _gpp_query_ok(diagram, variables, data):
                    raise ValueError(
                        "latent bilinear requires LinComb or fixed-denom Cond"
                    )
                gpp = build_latent_bilinear(
                    variables, data, query, diagram, frozenset(D), notes=notes
                )
                return _gpp_stub(gpp, hint="gpp/latent")
            except (ValueError, NotImplementedError, RuntimeError) as exc:
                _merged_outer(
                    f"latent bilinear unavailable ({exc}); "
                    "using merged-latent outer LP (valid, may be conservative)."
                )

        elif prog.kind == ProgramKind.BILINEAR_FACTOR and prefer_gpp:
            from .gpp import build_gpp, gpp_size_ok
            from .types_canon import _all_functions as _afns

            _restrict_ancestral()
            exo = tuple(
                n
                for n in diagram.topological_order()
                if not any(n in d for d in free)
            )
            try:
                if exo:
                    raise ValueError("exogenous nodes with ≥2 free districts")
                if len(free) != 2:
                    raise ValueError(f"bilinear/factor expects K=2; got {len(free)}")
                order_peek = diagram.topological_order()
                counts = []
                for d in free:
                    dord = tuple(v for v in order_peek if v in d)
                    n = 1
                    for v in dord:
                        parents = list(diagram.parents(v))
                        din = variables.cells(parents) if parents else []
                        n *= len(_afns(din, variables.domains[v]))
                    counts.append(n)
                if not gpp_size_ok(counts):
                    raise ValueError(f"GPP too large: {counts}")
                if not _gpp_query_ok(diagram, variables, data):
                    raise ValueError(
                        "GPP requires LinComb or fixed-denom Cond (PN/PS)"
                    )
                gpp = build_gpp(
                    variables, data, query, diagram, free, notes=notes
                )
                if isinstance(query.expr, _Cond):
                    gpp.notes = list(gpp.notes) + [
                        "GPP Cond: fixed observational denominator "
                        "(Tian–Pearl PN/PS linearization)"
                    ]
                return _gpp_stub(gpp, hint="gpp")
            except (ValueError, NotImplementedError, RuntimeError) as exc:
                _merged_outer(
                    f"bilinear/factor GPP unavailable ({exc}); "
                    "using merged-latent outer LP (valid, may be conservative)."
                )

        elif prog.kind == ProgramKind.POLYNOMIAL:
            _restrict_ancestral()
            poly_built = False
            if prefer_poly and prefer_gpp:
                try:
                    if not _gpp_query_ok(diagram, variables, data):
                        raise ValueError(
                            "poly GPP requires LinComb or fixed-denom Cond"
                        )
                    # Factor-polynomial: K=3 free λ=1 districts
                    if (
                        len(free) == 3
                        and all(cd == 1 for cd in prog.c_degrees)
                    ):
                        from .poly_gpp import build_poly_factor_gpp

                        exo = tuple(
                            n
                            for n in diagram.topological_order()
                            if not any(n in d for d in free)
                        )
                        if exo:
                            raise ValueError(
                                "poly factor GPP: exogenous nodes present"
                            )
                        poly = build_poly_factor_gpp(
                            variables, data, query, diagram, free, notes=notes
                        )
                        poly_built = True
                        return _gpp_stub(poly, hint="gpp/poly")
                    # Latent-polynomial (λ≥3): size gate almost always fails;
                    # leave an explicit note and fall through to outer.
                    raise ValueError(
                        f"poly solver: unsupported shape K={len(free)} "
                        f"c_degrees={prog.c_degrees} (latent λ≥3 not size-gated)"
                    )
                except (ValueError, NotImplementedError, RuntimeError) as exc:
                    notes.append(
                        f"poly GPP unavailable ({exc}); "
                        "explicit outer fallback (tight=False)."
                    )
            elif not prefer_poly:
                notes.append(
                    "polynomial (deg≥3): outer LP by default; "
                    "pass prefer_poly=True for nested multilinear gpp/poly"
                )
            if not poly_built:
                _merged_outer(
                    f"program_kind=polynomial degree={prog.degree}; "
                    "using merged-latent outer LP (valid, may be conservative; "
                    "not the sharp Classify program)."
                )

        else:
            _merged_outer(
                f"program_kind={prog.kind.value} degree={prog.degree}; "
                "using merged-latent outer LP (valid, may be conservative)."
            )

    elif len(diagram.c_components()) > 1 and not assumptions.of_type(MarginBudgetAssumption):
        notes.append(
            f"diagram has {len(diagram.c_components())} c-components; "
            "using merged-latent outer LP (valid, may be conservative)."
        )
        outer = True
        diagram = _merged_latent_diagram(diagram)
    elif not optimize and not _is_manski_full_latent(diagram):
        # Naive baseline: force one big type block
        notes.append("naive: forcing one full-latent type expansion")
        outer = True
        diagram = _merged_latent_diagram(diagram)

    ts = TypeSpace.from_diagram(variables, diagram)
    # Observational law is over observed coordinates only (latent endogenous
    # marginalized; latent endogenous / measurement error measurement-error graphs).
    obs_order = variables.observed_names(ts.order)
    if len(obs_order) != len(ts.order):
        notes.append(
            f"latent endogenous {[n for n in ts.order if n not in obs_order]}; "
            f"P indexed by observed {list(obs_order)} (latent endogenous / measurement error)"
        )
    cells, P = _normalize_P(variables, obs_order, data)
    allow = _hard_face_mask(ts, assumptions)
    # Map variable name → index in observational cells (may omit latents)
    obs_index = {n: i for i, n in enumerate(obs_order)}

    c, c_den, obj_notes = _build_objective(ts, query, allow, cells=cells, P_vec=P)
    notes.extend(obj_notes)

    # Equality: A q = P over allowed types only (forbidden stay at 0 via bounds)
    n = ts.n_types
    eq_rows = []
    eq_rhs = []
    for i, cell in enumerate(cells):
        row = np.zeros(n)
        for t in ts.fiber[cell]:
            if allow[t]:
                row[t] = 1.0
        eq_rows.append(row)
        eq_rhs.append(P[i])

    A_eq = sparse.csr_matrix(np.vstack(eq_rows)) if eq_rows else sparse.csr_matrix((0, n))
    b_eq = np.asarray(eq_rhs, dtype=float)

    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []
    e0: float | None = None
    pi0: dict | None = None
    f_sens_atom: FSensitivityAssumption | None = None

    # Observed propensity for sensitivity models (conditional on Pa(X) when present)
    sens = (
        list(assumptions.of_type(TanMSMAssumption))
        + list(assumptions.of_type(CDependenceAssumption))
        + list(assumptions.of_type(RosenbaumAssumption))
        + list(assumptions.of_type(FSensitivityAssumption))
    )
    if sens:
        from .sensitivity_budgets import (
            build_pi0_for_treatment,
            cdep_linf_rows,
            msm_lr_rows,
        )

        treatment = sens[0].treatment
        t_idx = obs_index[treatment]
        # Marginal P(X=1) kept for templates / binary f-sensitivity fallback
        e0 = float(sum(P[i] for i, cell in enumerate(cells) if cell[t_idx] == 1))
        pi0 = build_pi0_for_treatment(ts, cells, P, treatment)
        parents = list(ts.diagram.parents(treatment))
        cond_tag = f", conditional on {parents}" if parents else ""

        for a in assumptions.of_type(TanMSMAssumption):
            if not a.anchored:
                raise NotImplementedError("unanchored Tan MSM")
            rows, rhs = msm_lr_rows(ts, allow, a.treatment, a.Lambda, pi0)
            ub_rows.extend(rows)
            ub_rhs.extend(rhs)
            notes.append(f"Tan MSM Λ={a.Lambda} (LR relative propensity{cond_tag})")

        for a in assumptions.of_type(RosenbaumAssumption):
            Lam = float(np.sqrt(a.Gamma))
            rows, rhs = msm_lr_rows(ts, allow, a.treatment, Lam, pi0)
            ub_rows.extend(rows)
            ub_rhs.extend(rhs)
            notes.append(f"anchored Rosenbaum Γ={a.Gamma} → MSM Λ={Lam} (LR{cond_tag})")

        for a in assumptions.of_type(CDependenceAssumption):
            rows, rhs = cdep_linf_rows(ts, allow, a.treatment, a.c, pi0)
            ub_rows.extend(rows)
            ub_rhs.extend(rhs)
            notes.append(f"c-dependence c={a.c} (ℓ∞{cond_tag})")

        f_atoms = list(assumptions.of_type(FSensitivityAssumption))
        if len(f_atoms) > 1:
            raise ValueError("at most one f_sensitivity atom")
        if f_atoms:
            f_sens_atom = f_atoms[0]
            notes.append(f"f-sensitivity {f_sens_atom.f} ρ={f_sens_atom.rho} [convex]")

    # Manski–Pepper MTS (linear inequalities on counterfactual means)
    for a in assumptions.of_type(MTSAssumption):
        rows, rhs = _mts_rows(ts, allow, cells, P, a.treatment, a.outcome, a.direction)
        ub_rows.extend(rows)
        ub_rhs.extend(rhs)
        notes.append(f"MTS {a.direction} on {a.treatment}→{a.outcome}")

    # Margin budget on the query value around a reference
    for a in assumptions.of_type(MarginBudgetAssumption):
        if a.s0 is not None:
            q0 = a.s0
        elif a.reference == "point":
            q0 = _point_reference(variables, ts, cells, P, query)
        else:
            raise ValueError(f"unknown margin reference {a.reference!r}")
        ub_rows.append(c.copy())
        ub_rhs.append(q0 + a.kappa)
        ub_rows.append(-c.copy())
        ub_rhs.append(-q0 + a.kappa)
        notes.append(f"margin budget κ={a.kappa} around {q0}")

    if ub_rows:
        A_ub = sparse.csr_matrix(np.vstack(ub_rows))
        b_ub = np.asarray(ub_rhs, dtype=float)
    else:
        A_ub, b_ub = None, None

    support_atoms = list(assumptions.of_type(SupportBoundAssumption))
    support_k = support_atoms[0].k if support_atoms else None
    if len(support_atoms) > 1:
        raise ValueError("at most one support_bound atom")
    if support_k is not None:
        notes.append(f"support_bound k={support_k} [MIP]")

    if f_sens_atom is not None and support_k is not None:
        raise NotImplementedError("combine f_sensitivity with support_bound later")

    if f_sens_atom is not None:
        method = "convex"
    elif support_k is not None:
        method = "mip"
    elif c_den is not None:
        method = "fractional"
    elif A_ub is None:
        method = "fiber"
    else:
        method = "lp"

    return LPData(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        type_space=ts,
        allowed_types=allow,
        cells=cells,
        P_vec=P,
        method_hint=method,
        notes=notes,
        support_k=support_k,
        f_sensitivity=f_sens_atom,
        e0=e0,
        pi0=pi0,
        c_den=c_den,
        outer_approximation=outer,
        gpp=None,
    )


def _point_reference(
    variables: Variables,
    ts: TypeSpace,
    cells: list[tuple],
    P: np.ndarray,
    query: Query,
) -> float:
    """No-confounding reference for interventional queries: P(Y=y | X=x)."""
    if query.kind != "interventional":
        raise ValueError("point reference currently for interventional queries only")
    if len(query.intervention) != 1:
        raise ValueError("point reference expects a single intervention target")
    ((treat, tval),) = query.intervention.items()
    # P(outcome=value | treat=tval)
    num = 0.0
    den = 0.0
    obs = ts.variables.observed_names(ts.order)
    o_idx = list(obs).index(query.outcome)
    t_idx = list(obs).index(treat)
    for i, cell in enumerate(cells):
        if cell[t_idx] == tval:
            den += P[i]
            if cell[o_idx] == query.value:
                num += P[i]
    if den <= 0:
        raise ValueError("cannot form point reference: zero propensity")
    return num / den
