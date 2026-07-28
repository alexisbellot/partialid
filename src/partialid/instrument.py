"""Instrument / R-side linear programs (Balke–Pearl, Sachs Conditions 1–4).

Optimizations (P1):
- array-backed response tables (no per-call dict lookups)
- single pass type → stratum incidence (COO), not cell×type Python scans
- column aggregation: merge types with identical (A-column, objective)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse

from .assumptions import (
    AssumptionSet,
    CDependenceAssumption,
    MonotonicityAssumption,
    NoDefiersAssumption,
    RosenbaumAssumption,
    TanMSMAssumption,
    as_assumption_set,
)
from .graph import CausalDiagram
from .query import Query
from .variables import Variables


def _sensitivity_atoms(assumptions: AssumptionSet | None):
    """Return (msm, ros, cdep) lists; empty if none."""
    a = as_assumption_set(assumptions)
    return (
        list(a.of_type(TanMSMAssumption)),
        list(a.of_type(RosenbaumAssumption)),
        list(a.of_type(CDependenceAssumption)),
    )


def _instrument_cell_for_pa(
    ts: "InstrumentTypeSpace",
    parents: Sequence[str],
    pa: tuple,
) -> tuple:
    """Build an instrument cell realizing parent config ``pa`` (dummies elsewhere)."""
    pa_map = {p: pa[i] for i, p in enumerate(parents)}
    vals = []
    for name in ts.instruments:
        if name in pa_map:
            vals.append(pa_map[name])
        else:
            vals.append(ts.domain_list[name][0])
    return tuple(vals)


def _instrument_budget_rows(
    ts: "InstrumentTypeSpace",
    allow: np.ndarray,
    treatment: str,
    *,
    kind: str,
    value: float,
    pi0: Mapping[tuple, Mapping[object, float]],
    parents: Sequence[str],
) -> tuple[list[np.ndarray], list[float]]:
    """Anchored Tan MSM / c-dep on R-side types, conditional on Pa(treatment)."""
    from itertools import product as _product

    if treatment not in ts.r_side:
        raise ValueError(f"treatment {treatment!r} not on R-side {ts.r_side}")
    t_pos = ts.r_side.index(treatment)
    domain_x = list(ts.domain_list[treatment])
    groups: dict[tuple, list[int]] = {}
    for t in range(ts.n_types):
        if not allow[t]:
            continue
        key = tuple(fi for i, fi in enumerate(ts.joint_types[t]) if i != t_pos)
        groups.setdefault(key, []).append(t)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    n = ts.n_types
    eps = 1e-15

    for members in groups.values():
        for pa, pi_pa in pi0.items():
            ic = _instrument_cell_for_pa(ts, parents, pa)
            by_x: dict[object, list[int]] = {x: [] for x in domain_x}
            for t in members:
                r_idx = ts.eval_r_indices(t, ic, None)
                x = domain_x[r_idx[t_pos]]
                by_x[x].append(t)

            if kind == "msm":
                Lam = float(value)
                if Lam < 1.0 - 1e-15:
                    raise ValueError("Tan MSM requires Lambda >= 1")
                levels = [x for x in domain_x if pi_pa.get(x, 0.0) > eps and by_x[x]]
                if len(levels) < 2:
                    continue
                for x, xp in _product(levels, repeat=2):
                    if x == xp:
                        continue
                    row = np.zeros(n)
                    for t in by_x[x]:
                        row[t] += pi_pa[xp]
                    for t in by_x[xp]:
                        row[t] -= Lam * pi_pa[x]
                    if np.any(np.abs(row) > eps):
                        rows.append(row)
                        rhs.append(0.0)
            elif kind == "c":
                c = float(value)
                for x in domain_x:
                    lo = pi_pa.get(x, 0.0) - c
                    hi = pi_pa.get(x, 0.0) + c
                    r_hi = np.zeros(n)
                    for t in members:
                        r_hi[t] -= hi
                    for t in by_x[x]:
                        r_hi[t] += 1.0
                    if np.any(np.abs(r_hi) > eps):
                        rows.append(r_hi)
                        rhs.append(0.0)
                    r_lo = np.zeros(n)
                    for t in members:
                        r_lo[t] += lo
                    for t in by_x[x]:
                        r_lo[t] -= 1.0
                    if np.any(np.abs(r_lo) > eps):
                        rows.append(r_lo)
                        rhs.append(0.0)
            else:
                raise ValueError(f"unknown budget kind {kind!r}")
    return rows, rhs


def detect_instrument_split(
    diagram: CausalDiagram,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return (instruments, r_side) if diagram looks like Sachs WL/WR, else None.

    Classic rule: unconfounded nodes = WL, bidirected nodes = WR.
    See also :func:`detect_sachs_cut` for confounded-WL cuts (Prop. 8.4).
    """
    if diagram.maximal_confounding:
        return None
    bi_nodes = set()
    for a, b in diagram.bidirected:
        bi_nodes.add(a)
        bi_nodes.add(b)
    instruments = tuple(n for n in diagram.nodes if n not in bi_nodes)
    r_side = tuple(n for n in diagram.nodes if n in bi_nodes)
    if not instruments or not r_side:
        return None
    if not _valid_sachs_cut(diagram, instruments, r_side):
        return None
    has_edge_into_r = any(a in instruments and b in r_side for a, b in diagram.directed)
    if not has_edge_into_r:
        return None
    return instruments, r_side


def _valid_sachs_cut(
    diagram: CausalDiagram,
    wl: Sequence[str],
    wr: Sequence[str],
) -> bool:
    """Sachs Conditions 1–2: directed cut WL→WR only; no bidirected edge across.

    Also requires at least one directed WL→WR edge (rules out inert WL nodes).
    """
    wl_s, wr_s = set(wl), set(wr)
    if not wl_s or not wr_s:
        return False
    if wl_s | wr_s != set(diagram.nodes) or wl_s & wr_s:
        return False
    for a, b in diagram.bidirected:
        if (a in wl_s) ^ (b in wl_s):
            return False
    for a, b in diagram.directed:
        if a in wr_s and b in wl_s:
            return False
    if not any(a in wl_s and b in wr_s for a, b in diagram.directed):
        return False
    return True


def detect_sachs_cut(
    diagram: CausalDiagram,
    *,
    treatments: Sequence[str] | None = None,
    outcome: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Prop. 8.4 / Sachs Prop. 2 cut: (WL, WR) with linear R-side LP.

    Unlike :func:`detect_instrument_split`, WL may itself be confounded.
    Prefers the valid cut whose WR is smallest (fewest nodes), among cuts that
    contain ``outcome`` and all ``treatments`` in WR when provided.
    """
    if diagram.maximal_confounding:
        return None
    nodes = list(diagram.topological_order())
    must: set[str] = set()
    if outcome is not None:
        must.add(outcome)
    if treatments:
        must.update(treatments)
    if must - set(nodes):
        return None

    candidates: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    # 1) Classic unconfounded-WL split
    classic = detect_instrument_split(diagram)
    if classic is not None:
        candidates.append(classic)

    # 2) WR = each confounded district that contains all of `must`
    comps = diagram.c_components()
    for d in comps:
        if must and not must.issubset(d):
            continue
        wr = tuple(n for n in nodes if n in d)
        wl = tuple(n for n in nodes if n not in d)
        if _valid_sachs_cut(diagram, wl, wr):
            candidates.append((wl, wr))

    # 3) WR = union of all districts that intersect must
    if must:
        wr_set: set[str] = set()
        for d in comps:
            if d & must:
                wr_set |= set(d)
        wr_set |= must
        wr = tuple(n for n in nodes if n in wr_set)
        wl = tuple(n for n in nodes if n not in wr_set)
        if wl and _valid_sachs_cut(diagram, wl, wr):
            candidates.append((wl, wr))

    # Dedup and score: prefer smaller WR, then smaller type-ish proxy (|WR|)
    best = None
    best_key = None
    seen: set[tuple[str, ...]] = set()
    for wl, wr in candidates:
        key = (wr, wl)
        if wr in seen:
            continue
        seen.add(wr)
        if must and not must.issubset(wr):
            continue
        score = (len(wr), len(wl))
        if best is None or score < best_key:
            best = (wl, wr)
            best_key = score
    return best


def _parent_config_index(
    parents: Sequence[str],
    values: Mapping[str, object],
    variables: Variables,
) -> int:
    """Mixed-radix index of a parent configuration."""
    if not parents:
        return 0
    idx = 0
    stride = 1
    # little-endian over parents order
    for p in reversed(parents):
        dom = variables.domains[p]
        idx += dom.index(values[p]) * stride
        stride *= len(dom)
    return idx


def _encode_functions(
    domain_in: list,
    domain_out: Sequence,
) -> np.ndarray:
    """Stack all response functions as (n_fn, n_in) int16 table of outcome indices."""
    out_vals = list(domain_out)
    out_index = {v: i for i, v in enumerate(out_vals)}
    if not domain_in:
        # constants
        return np.array([[out_index[v]] for v in out_vals], dtype=np.int16)
    keys = list(domain_in)
    rows = []
    for assignment in product(out_vals, repeat=len(keys)):
        rows.append([out_index[assignment[i]] for i in range(len(keys))])
    return np.asarray(rows, dtype=np.int16)


@dataclass
class InstrumentTypeSpace:
    """Response types on the R-side with array-backed evaluation."""

    variables: Variables
    diagram: CausalDiagram
    instruments: tuple[str, ...]
    r_side: tuple[str, ...]
    order: tuple[str, ...]
    # per R-side variable: (n_fn, n_parent_configs) outcome **indices**
    fn_tables: dict[str, np.ndarray]
    # parent names per R-side var
    parents: dict[str, tuple[str, ...]]
    # domains as tuples for index↔value
    domain_list: dict[str, tuple]
    joint_types: list[tuple[int, ...]]
    n_types: int
    type_index: dict[tuple[int, ...], int]
    i_cells: list[tuple]

    @classmethod
    def build(
        cls,
        variables: Variables,
        diagram: CausalDiagram,
        instruments: Sequence[str],
        r_side: Sequence[str],
    ) -> InstrumentTypeSpace:
        full_order = diagram.topological_order()
        instruments = tuple(i for i in full_order if i in instruments)
        r_side = tuple(v for v in full_order if v in r_side)
        order = instruments + r_side
        domain_list = {n: tuple(variables.domains[n]) for n in order}

        parents: dict[str, tuple[str, ...]] = {}
        fn_tables: dict[str, np.ndarray] = {}
        for v in r_side:
            pa = tuple(diagram.parents(v))
            parents[v] = pa
            domain_in = variables.cells(list(pa)) if pa else []
            fn_tables[v] = _encode_functions(domain_in, variables.domains[v])

        fn_counts = [fn_tables[v].shape[0] for v in r_side]
        joint_types = list(product(*[range(c) for c in fn_counts]))
        type_index = {jt: i for i, jt in enumerate(joint_types)}
        i_cells = variables.cells(list(instruments))
        return cls(
            variables=variables,
            diagram=diagram,
            instruments=instruments,
            r_side=r_side,
            order=order,
            fn_tables=fn_tables,
            parents=parents,
            domain_list=domain_list,
            joint_types=joint_types,
            n_types=len(joint_types),
            type_index=type_index,
            i_cells=i_cells,
        )

    def eval_r_indices(
        self,
        type_idx: int,
        inst_cell: tuple,
        intervention: Mapping[str, object] | None = None,
    ) -> tuple[int, ...]:
        """Return R-side outcome **indices** under instrument cell / optional do().

        Interventions may target instruments (parents in WL) or R-side nodes.
        When ``do(X)`` sets an instrument parent, that override replaces the
        observational instrument cell for descendant evaluation (needed for
        ATE/interventional queries with treatment in WL, e.g. Sachs §6.3).
        """
        intervention = intervention or {}
        # map name -> current value index
        val_idx: dict[str, int] = {}
        for k, name in enumerate(self.instruments):
            if name in intervention:
                val_idx[name] = self.domain_list[name].index(intervention[name])
            else:
                val_idx[name] = self.domain_list[name].index(inst_cell[k])
        type_ids = self.joint_types[type_idx]
        out: list[int] = []
        for v, fi in zip(self.r_side, type_ids):
            if v in intervention:
                vi = self.domain_list[v].index(intervention[v])
                val_idx[v] = vi
                out.append(vi)
                continue
            pa = self.parents[v]
            if not pa:
                pcfg = 0
            else:
                # mixed radix over parent indices
                pcfg = 0
                stride = 1
                for p in reversed(pa):
                    pcfg += val_idx[p] * stride
                    stride *= len(self.domain_list[p])
            vi = int(self.fn_tables[v][fi, pcfg])
            val_idx[v] = vi
            out.append(vi)
        return tuple(out)

    def query_coeff(
        self,
        type_idx: int,
        outcome: str,
        value: object,
        intervention: Mapping[str, object],
    ) -> float:
        # Instrument cell is irrelevant when every instrument in ``intervention``
        # is overridden; still pass a dummy for any unset instrument parents.
        dummy = tuple(self.domain_list[i][0] for i in self.instruments)
        r_idx = self.eval_r_indices(type_idx, dummy, intervention)
        opos = self.r_side.index(outcome)
        return 1.0 if self.domain_list[outcome][r_idx[opos]] == value else 0.0


def _face_mask(
    ts: InstrumentTypeSpace,
    diagram: CausalDiagram,
    assumptions: AssumptionSet | None,
) -> np.ndarray:
    n = ts.n_types
    allow = np.ones(n, dtype=bool)
    if assumptions is None:
        return allow

    def mono_ok(var: str, treatment: str, direction: str, type_idx: int) -> bool:
        if var not in ts.r_side or treatment not in ts.parents[var]:
            return True
        vpos = ts.r_side.index(var)
        fi = ts.joint_types[type_idx][vpos]
        table = ts.fn_tables[var]
        pa = list(ts.parents[var])
        t_i = pa.index(treatment)
        other = [p for j, p in enumerate(pa) if j != t_i]
        # iterate other parent configs
        other_ranges = [range(len(ts.domain_list[p])) for p in other]
        t_card = len(ts.domain_list[treatment])
        if not other:
            other_iter: list = [()]
        else:
            other_iter = list(product(*other_ranges))
        for oc in other_iter:
            vals = []
            for tv in range(t_card):
                full = []
                oi = 0
                for j, p in enumerate(pa):
                    if j == t_i:
                        full.append(tv)
                    else:
                        full.append(oc[oi])
                        oi += 1
                pcfg = 0
                stride = 1
                for j in range(len(pa) - 1, -1, -1):
                    pcfg += full[j] * stride
                    stride *= len(ts.domain_list[pa[j]])
                vals.append(int(table[fi, pcfg]))
            for a, b in zip(vals, vals[1:]):
                if direction == "increasing" and a > b:
                    return False
                if direction == "decreasing" and a < b:
                    return False
        return True

    for nd in assumptions.of_type(NoDefiersAssumption):
        for t in range(n):
            if allow[t] and not mono_ok(nd.treatment, nd.instrument, "increasing", t):
                allow[t] = False
    for m in assumptions.of_type(MonotonicityAssumption):
        for t in range(n):
            if allow[t] and not mono_ok(m.outcome, m.treatment, m.direction, t):
                allow[t] = False
    return allow


def formulate_instrument(
    variables: Variables,
    data: Mapping,
    query: Query,
    diagram: CausalDiagram,
    instruments: Sequence[str],
    r_side: Sequence[str],
    assumptions: AssumptionSet | None = None,
):
    """Build Balke–Pearl / Sachs R-side LP (sparse + aggregated)."""
    from .formulate import LPData, _normalize_P

    ts = InstrumentTypeSpace.build(variables, diagram, instruments, r_side)
    notes = [f"instrument LP: I={list(instruments)}, R={list(r_side)}, raw_types={ts.n_types}"]

    # Observational law may omit latent endogenous nodes in R (Sachs §6.3).
    # Constraints are then on P(obs_R | I), with latent R coords marginalized.
    obs_r = tuple(v for v in ts.r_side if variables.is_observed(v))
    latent_r = tuple(v for v in ts.r_side if not variables.is_observed(v))
    if latent_r:
        notes.append(
            f"latent on R-side {list(latent_r)}; constraints use obs_R={list(obs_r)} "
            f"(Sachs 2022 §6.3)"
        )
    obs_ir = tuple(instruments) + obs_r

    topo = diagram.topological_order()
    obs_topo = variables.observed_names(topo)
    cells_topo, P_topo = _normalize_P(variables, obs_topo, data)
    data_ir: dict[tuple, float] = {}
    for cell, p in zip(cells_topo, P_topo):
        d = dict(zip(obs_topo, cell))
        data_ir[tuple(d[v] for v in obs_ir)] = data_ir.get(
            tuple(d[v] for v in obs_ir), 0.0
        ) + float(p)
    cells, P = _normalize_P(variables, obs_ir, data_ir)

    i_idx = list(range(len(instruments)))
    n_i = len(ts.i_cells)
    # Flat index over *observed* R configs (latent coords marginalized)
    n_obs_r_cards = [len(ts.domain_list[v]) for v in obs_r]

    def obs_r_tuple_to_flat(r_idx_tuple: tuple[int, ...]) -> int:
        flat = 0
        stride = 1
        for j in range(len(r_idx_tuple) - 1, -1, -1):
            flat += r_idx_tuple[j] * stride
            stride *= n_obs_r_cards[j]
        return flat

    def full_r_to_obs_flat(r_idx_full: tuple[int, ...]) -> int:
        """Project full R index tuple onto observed R coordinates."""
        obs_idx = tuple(r_idx_full[k] for k, v in enumerate(ts.r_side) if v in obs_r)
        return obs_r_tuple_to_flat(obs_idx)

    # P(i) and conditional targets P(obs_r|i) keyed by (i_flat, obs_r_flat)
    P_i = np.zeros(n_i)
    i_cell_to_flat = {ic: k for k, ic in enumerate(ts.i_cells)}
    for cell, p in zip(cells, P):
        ic = tuple(cell[j] for j in i_idx)
        P_i[i_cell_to_flat[ic]] += p

    row_ids: dict[tuple[int, int], int] = {}
    b_list: list[float] = []

    # Build RHS from data cells over obs_ir
    for cell, p in zip(cells, P):
        ic = tuple(cell[j] for j in i_idx)
        i_flat = i_cell_to_flat[ic]
        pi = P_i[i_flat]
        if pi <= 1e-15:
            continue
        r_vals = cell[len(instruments) :]
        r_idx_t = tuple(ts.domain_list[obs_r[k]].index(r_vals[k]) for k in range(len(obs_r)))
        r_flat = obs_r_tuple_to_flat(r_idx_t)
        key = (i_flat, r_flat)
        if key not in row_ids:
            row_ids[key] = len(b_list)
            b_list.append(p / pi)
        else:
            b_list[row_ids[key]] += p / pi

    n_obs_rows = len(b_list)
    sum_row = n_obs_rows  # last row
    n_rows = n_obs_rows + 1

    assumptions = as_assumption_set(assumptions)
    allow = _face_mask(ts, diagram, assumptions)
    if query.kind not in ("interventional", "ate"):
        raise NotImplementedError("instrument LP currently supports interventional and ate queries")

    msm_l, ros_l, cdep_l = _sensitivity_atoms(assumptions)
    keep_raw = bool(msm_l or ros_l or cdep_l)
    if keep_raw:
        notes.append("no column aggregation (sensitivity budgets on R-side)")

    # Conditional propensity π0(x|Pa) for anchored budgets
    e0: float | None = None
    treat_for_sens: str | None = None
    pi0: dict[tuple, dict[object, float]] | None = None
    sens_parents: list[str] = []
    if keep_raw:
        from .sensitivity_budgets import observed_conditional_propensity

        treat_for_sens = (msm_l or ros_l or cdep_l)[0].treatment
        if treat_for_sens not in ts.order:
            raise ValueError(f"sensitivity treatment {treat_for_sens!r} not in instrument order")
        t_ord = ts.order.index(treat_for_sens)
        e0 = float(sum(p for cell, p in zip(cells, P) if cell[t_ord] == 1))
        sens_parents = list(diagram.parents(treat_for_sens))
        pi0 = observed_conditional_propensity(
            ts.order,
            cells,
            P,
            treat_for_sens,
            sens_parents,
            list(ts.domain_list[treat_for_sens]),
        )

    # --- Pass over types: build column signatures and COO contributions ---
    sig_to_col: dict[tuple, int] = {}
    col_c: list[float] = []
    coo_rows: list[int] = []
    coo_cols: list[int] = []
    coo_data: list[float] = []
    raw_col_types: list[int] = []  # type index per column (raw path)

    raw_allowed = int(allow.sum())
    for t in range(ts.n_types):
        if not allow[t]:
            continue
        if query.kind == "ate":
            meta = query.meta or {}
            treatment = meta.get("treatment", "X")
            t1 = meta.get("t1", 1)
            t0 = meta.get("t0", 0)
            obj = ts.query_coeff(t, query.outcome, query.value, {treatment: t1}) - ts.query_coeff(
                t, query.outcome, query.value, {treatment: t0}
            )
        else:
            obj = ts.query_coeff(t, query.outcome, query.value, query.intervention)
        hits: list[int] = []
        for i_flat, ic in enumerate(ts.i_cells):
            if P_i[i_flat] <= 1e-15:
                continue
            r_idx_t = ts.eval_r_indices(t, ic, None)
            r_flat = full_r_to_obs_flat(r_idx_t)
            key = (i_flat, r_flat)
            rid = row_ids.get(key)
            if rid is not None:
                hits.append(rid)
        hits_t = tuple(sorted(hits))
        if keep_raw:
            col = len(col_c)
            raw_col_types.append(t)
            col_c.append(obj)
        else:
            sig = (obj, hits_t)
            if sig in sig_to_col:
                continue
            col = len(col_c)
            sig_to_col[sig] = col
            col_c.append(obj)
        for rid in hits_t:
            coo_rows.append(rid)
            coo_cols.append(col)
            coo_data.append(1.0)
        coo_rows.append(sum_row)
        coo_cols.append(col)
        coo_data.append(1.0)

    n_cols = len(col_c)
    if keep_raw:
        notes.append(f"raw_cols={n_cols} (allowed raw types={raw_allowed})")
    else:
        notes.append(f"aggregated_cols={n_cols} (from {raw_allowed} allowed raw types)")

    A_eq = sparse.csr_matrix((coo_data, (coo_rows, coo_cols)), shape=(n_rows, n_cols), dtype=float)
    b_eq = np.asarray(b_list + [1.0], dtype=float)
    c = np.asarray(col_c, dtype=float)

    A_ub = None
    b_ub = None
    if keep_raw and treat_for_sens is not None and pi0 is not None:
        # Build dense rows on full type space, then restrict to kept columns
        type_to_col = {t: j for j, t in enumerate(raw_col_types)}
        ub_rows: list[np.ndarray] = []
        ub_rhs: list[float] = []
        cond_tag = f", conditional on {sens_parents}" if sens_parents else ""
        for a in msm_l:
            if a.treatment != treat_for_sens:
                raise NotImplementedError("multiple distinct MSM treatments")
            if not a.anchored:
                raise NotImplementedError("unanchored Tan MSM on instrument LP")
            rows, rhs = _instrument_budget_rows(
                ts,
                allow,
                a.treatment,
                kind="msm",
                value=a.Lambda,
                pi0=pi0,
                parents=sens_parents,
            )
            ub_rows.extend(rows)
            ub_rhs.extend(rhs)
            notes.append(f"Tan MSM Λ={a.Lambda} (R-side LR{cond_tag}, marg e0={e0:.4g})")
        for a in ros_l:
            if a.treatment != treat_for_sens:
                raise NotImplementedError("multiple distinct Rosenbaum treatments")
            if not a.anchored:
                raise NotImplementedError("unanchored Rosenbaum on instrument LP")
            Lam = float(np.sqrt(a.Gamma))
            rows, rhs = _instrument_budget_rows(
                ts,
                allow,
                a.treatment,
                kind="msm",
                value=Lam,
                pi0=pi0,
                parents=sens_parents,
            )
            ub_rows.extend(rows)
            ub_rhs.extend(rhs)
            notes.append(f"anchored Rosenbaum Γ={a.Gamma} → MSM Λ={Lam:g} (LR{cond_tag})")
        for a in cdep_l:
            if a.treatment != treat_for_sens:
                raise NotImplementedError("multiple distinct c-dep treatments")
            rows, rhs = _instrument_budget_rows(
                ts,
                allow,
                a.treatment,
                kind="c",
                value=a.c,
                pi0=pi0,
                parents=sens_parents,
            )
            ub_rows.extend(rows)
            ub_rhs.extend(rhs)
            notes.append(f"c-dependence c={a.c} (R-side ℓ∞{cond_tag})")

        if ub_rows:
            A_full = np.vstack(ub_rows)
            A_cols = np.zeros((len(ub_rows), n_cols), dtype=float)
            for t, j in type_to_col.items():
                A_cols[:, j] = A_full[:, t]
            A_ub = sparse.csr_matrix(A_cols)
            b_ub = np.asarray(ub_rhs, dtype=float)

    import types

    if keep_raw:
        # Compact type space view: columns ↔ allowed raw types in order
        shim = types.SimpleNamespace(
            n_types=n_cols,
            fiber={(): list(range(n_cols))},
            joint_types=[ts.joint_types[t] for t in raw_col_types],
            order=ts.r_side,
            diagram=diagram,
            response_fns={},
            variables=variables,
            type_index={},
            raw_types=ts.n_types,
            aggregated=False,
            instrument_ts=ts,
            raw_col_types=raw_col_types,
        )
        allow_out = np.ones(n_cols, dtype=bool)
        ts_out = shim
    else:
        shim = types.SimpleNamespace(
            n_types=n_cols,
            fiber={(): list(range(n_cols))},
            joint_types=list(range(n_cols)),
            order=ts.r_side,
            diagram=diagram,
            response_fns={},
            variables=variables,
            type_index={},
            raw_types=ts.n_types,
            aggregated=True,
        )
        allow_out = np.ones(n_cols, dtype=bool)
        ts_out = shim

    return LPData(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        type_space=ts_out,  # type: ignore[arg-type]
        allowed_types=allow_out,
        cells=[()],
        P_vec=np.array([1.0]),
        method_hint="lp",
        notes=notes,
        e0=e0,
    )
