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
from .graph import CausalDiagram
from .query import Query
from .types_canon import TypeSpace
from .variables import Variables


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
    method_hint: str = "lp"  # "lp" | "fiber" | "mip" | "convex" | "gpp" | "fractional"
    notes: list[str] = field(default_factory=list)
    support_k: int | None = None
    f_sensitivity: FSensitivityAssumption | None = None
    e0: float | None = None
    # Conditional propensity π0[pa][x] for anchored budgets / f-sensitivity
    pi0: dict | None = None
    c_den: np.ndarray | None = None  # fractional denom coefficients
    # True when the LP is a valid outer approximation (may be loose)
    outer_approximation: bool = False
    # Optional factorized bilinear GPP (method_hint="gpp")
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

    Tuple keys are read in ``variables.names`` order (declaration order).
    """
    src = variables.names
    if set(keep) - set(src):
        raise ValueError(f"keep {list(keep)} not subset of {list(src)}")
    cells, P = _normalize_P(variables, src, data)
    keep_idx = [src.index(n) for n in keep]
    marg: dict[tuple, float] = {}
    for cell, p in zip(cells, P):
        key = tuple(cell[i] for i in keep_idx)
        marg[key] = marg.get(key, 0.0) + float(p)
    return marg


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
        maximal_confounding=diagram.maximal_confounding,
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

    Uses :mod:`partialid.query_compile` (CUT unnesting + Sachs Prop. 3
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
    if not diagram.maximal_confounding and diagram.bidirected:
        return variables, data, diagram

    if query.kind == "ate":
        treat = (query.meta or {}).get("treatment", "X")
        keep = {treat, query.outcome}
    else:
        keep = set(query.intervention) | {query.outcome}
    # Latent endogenous: never drop observed proxies (Sachs §6.3); they carry P.
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
        g2 = CausalDiagram.from_order(new_order, maximal_confounding=True)
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
) -> LPData:
    """Build the partial-identification LP (linear reduction or instrument R-side).

    Parameters
    ----------
    optimize :
        If True (default), use inert-variable pruning and instrument R-side LPs.
        If False, use a *naive* baseline: full joint type expansion under maximal
        confounding (no inert prune, no instrument factorization).
    prefer_gpp :
        When ≥2 free confounded districts fit the size gate, build a bilinear GPP
        instead of the merged-latent outer LP. Set False to force the outer LP.
    """
    from .instrument import detect_instrument_split, detect_sachs_cut, formulate_instrument

    assumptions = as_assumption_set(assumptions)
    notes: list[str] = []
    if not optimize:
        notes.append("naive: no inert prune, no instrument R-side LP")

    diagram = assumptions.diagram()
    if diagram is None:
        diagram = CausalDiagram.from_order(variables.names, maximal_confounding=True)
        notes.append("defaulted to declaration order with maximal confounding")

    diagram = _apply_exclusions(diagram, assumptions)

    # Prop. 8.4 / Sachs WL–WR cut → tight R-side LP (before merging / GPP)
    if optimize:
        split = None
        treatments = list(query.intervention.keys())
        if query.kind == "ate":
            treatments = [(query.meta or {}).get("treatment", "X")]
        if query.kind in ("interventional", "ate"):
            split = detect_sachs_cut(
                diagram,
                treatments=treatments,
                outcome=query.outcome,
            )
        if split is None and query.kind in ("interventional", "ate"):
            split = detect_instrument_split(diagram)
        if split is not None and query.kind in ("interventional", "ate"):
            instruments, r_side = split
            notes.append(f"Prop. 8.4 / Sachs cut: WL={list(instruments)}, WR={list(r_side)}")
            lp = formulate_instrument(
                variables, data, query, diagram, instruments, r_side, assumptions
            )
            lp.notes = notes + lp.notes
            lp.outer_approximation = False
            return lp

        # Inert-variable pruning for order-only interventional queries
        variables, data, diagram = _inert_prune_order_only(variables, data, query, diagram, notes)

    outer = False
    if optimize and query.kind == "interventional":
        from .degree1 import analyze_degree1, induced_subgraph

        fac = analyze_degree1(
            diagram,
            treatments=list(query.intervention.keys()),
            outcome=query.outcome,
        )
        notes.extend(list(fac.notes))

        if fac.lp_reducible and len(fac.free_districts) == 1:
            D = fac.free_districts[0]
            r_side = tuple(n for n in diagram.topological_order() if n in D)
            an_order = tuple(n for n in diagram.topological_order() if n in fac.ancestral)
            exo = fac.exogenous

            if set(an_order) != set(diagram.nodes):
                data = _remargin_data(variables, data, an_order)
                variables = variables.restrict(an_order)
                diagram = induced_subgraph(diagram, an_order)
                notes.append(f"restricted to ancestral {list(an_order)}")

            if exo:
                notes.append(
                    f"c-degree-1 LP on free district {list(r_side)} (exogenous {list(exo)})"
                )
                lp = formulate_instrument(
                    variables,
                    data,
                    query,
                    diagram,
                    exo,
                    r_side,
                    assumptions,
                )
                lp.notes = notes + lp.notes
                lp.outer_approximation = False
                return lp
            # Single confounded block, no exogenous: TypeSpace on D only
            if set(an_order) != set(r_side):
                data = _remargin_data(variables, data, r_side)
            variables = variables.restrict(r_side)
            diagram = induced_subgraph(diagram, r_side)
            notes.append(f"c-degree-1 single-block LP on {list(r_side)}")
            # fall through with restricted diagram

        elif fac.lp_reducible and len(fac.free_districts) == 0:
            # Markovian ancestral — should have been ID'd; keep diagram, no merge
            an_order = tuple(n for n in diagram.topological_order() if n in fac.ancestral)
            if set(an_order) != set(diagram.nodes):
                diagram = induced_subgraph(diagram, an_order)
                notes.append(
                    "c-degree-1: no free district; restricted ancestral "
                    f"{list(an_order)} (no merge)"
                )

        elif len(fac.free_districts) >= 2:
            # Prop. 8.4 cut on the ancestral graph, else Zhang GPP (K=2)
            from .gpp import build_gpp, gpp_size_ok

            an_order = tuple(n for n in diagram.topological_order() if n in fac.ancestral)
            if set(an_order) != set(diagram.nodes):
                data = _remargin_data(variables, data, an_order)
                variables = variables.restrict(an_order)
                diagram = induced_subgraph(diagram, an_order)
                notes.append(f"restricted to ancestral {list(an_order)}")

            split_an = detect_sachs_cut(
                diagram,
                treatments=list(query.intervention.keys()),
                outcome=query.outcome,
            )
            if split_an is not None:
                instruments, r_side = split_an
                notes.append(
                    f"Prop. 8.4 / Sachs cut (ancestral): WL={list(instruments)}, WR={list(r_side)}"
                )
                lp = formulate_instrument(
                    variables, data, query, diagram, instruments, r_side, assumptions
                )
                lp.notes = notes + lp.notes
                lp.outer_approximation = False
                return lp

            free = fac.free_districts
            exo = tuple(n for n in diagram.topological_order() if not any(n in d for d in free))
            try:
                if exo:
                    raise ValueError("exogenous nodes with ≥2 free districts")
                from .types_canon import _all_functions as _afns

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
                if prefer_gpp and len(free) == 2 and gpp_size_ok(counts):
                    gpp = build_gpp(
                        variables,
                        data,
                        query,
                        diagram,
                        free,
                        notes=notes,
                    )
                    v0 = diagram.topological_order()[0]
                    stub_g = CausalDiagram(
                        nodes=(v0,),
                        directed=(),
                        bidirected=(),
                        maximal_confounding=True,
                    )
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
                        method_hint="gpp",
                        notes=list(gpp.notes),
                        outer_approximation=False,
                        gpp=gpp,
                    )
            except Exception as exc:
                notes.append(f"GPP unavailable ({exc}); falling back to outer LP")

            notes.append(
                f"diagram has {len(diagram.c_components())} c-components; "
                "using merged-latent outer LP (valid, may be conservative)."
            )
            outer = True
            diagram = CausalDiagram(
                nodes=diagram.nodes,
                directed=diagram.directed,
                bidirected=(),
                maximal_confounding=True,
            )
        else:
            # lp_reducible False with odd shape — merged outer
            notes.append(
                f"diagram has {len(diagram.c_components())} c-components; "
                "using merged-latent outer LP (valid, may be conservative)."
            )
            outer = True
            diagram = CausalDiagram(
                nodes=diagram.nodes,
                directed=diagram.directed,
                bidirected=(),
                maximal_confounding=True,
            )

    elif len(diagram.c_components()) > 1 and not assumptions.of_type(MarginBudgetAssumption):
        notes.append(
            f"diagram has {len(diagram.c_components())} c-components; "
            "using merged-latent outer LP (valid, may be conservative)."
        )
        outer = True
        diagram = CausalDiagram(
            nodes=diagram.nodes,
            directed=diagram.directed,
            bidirected=(),
            maximal_confounding=True,
        )
    elif not optimize and not diagram.maximal_confounding:
        # Naive baseline: force one big type block
        notes.append("naive: forcing maximal confounding type expansion")
        outer = True
        diagram = CausalDiagram(
            nodes=diagram.nodes,
            directed=diagram.directed,
            bidirected=(),
            maximal_confounding=True,
        )

    ts = TypeSpace.from_diagram(variables, diagram)
    # Observational law is over observed coordinates only (latent endogenous
    # marginalized; Sachs et al. 2022 §6.3 measurement-error graphs).
    obs_order = variables.observed_names(ts.order)
    if len(obs_order) != len(ts.order):
        notes.append(
            f"latent endogenous {[n for n in ts.order if n not in obs_order]}; "
            f"P indexed by observed {list(obs_order)} (Sachs 2022 §6.3)"
        )
    cells, P = _normalize_P(variables, obs_order, data)
    allow = _hard_face_mask(ts, assumptions)
    # Map variable name → index in observational cells (may omit latents)
    obs_index = {n: i for i, n in enumerate(obs_order)}

    c, c_den, obj_notes = _build_objective(ts, query, allow, cells=cells, P_vec=P)
    notes.extend(obj_notes)

    if assumptions.of_type(IndependenceAssumption):
        if outer:
            notes.append(
                "independence atom present: merged-latent path is a certified "
                "outer unless ID/Sachs already applied (hard split not fully "
                "factorized in v1)"
            )
        else:
            notes.append(
                "independence atom noted; tight factorization only when ID/Sachs "
                "reduces the problem"
            )

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
