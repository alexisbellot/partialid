"""Batch sensitivity grids: reuse type space / A_eq; update only budget rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from scipy import sparse

from .assumptions import AssumptionSet, as_assumption_set
from .formulate import LPData, _point_reference, formulate
from .sensitivity_budgets import (
    build_pi0_for_treatment,
    cdep_linf_rows,
    msm_lr_rows,
    rest_type_groups,
)
from .types_canon import TypeSpace
from .variables import Variables


@dataclass
class MSMGroup:
    """Types sharing an R_{-X} key, partitioned by treatment level (root binary)."""

    ones: np.ndarray  # type indices with treatment=1
    zeros: np.ndarray
    members: np.ndarray | None = None  # all members (multi-valued / conditional)

    def all_members(self) -> np.ndarray:
        if self.members is not None:
            return self.members
        return np.concatenate([self.ones, self.zeros])


def _msm_groups(ts: TypeSpace, allow: np.ndarray, treatment: str) -> list[MSMGroup]:
    """Build partner groups for column generation (rest-response partitions)."""
    parents = list(ts.diagram.parents(treatment))
    groups_raw = rest_type_groups(ts, allow, treatment)
    groups: list[MSMGroup] = []
    domain_x = list(ts.variables.domains[treatment])
    # For partner seeding we only need the rest partition; treatment level
    # split uses a representative parent config (empty or first).
    pa = () if not parents else next(iter(ts.variables.cells(parents)))
    pa_map = {p: pa[i] for i, p in enumerate(parents)}
    for members in groups_raw.values():
        by_x: dict[object, list[int]] = {x: [] for x in domain_x}
        for t in members:
            x = ts.response_on(t, treatment, pa_map)
            by_x[x].append(t)
        mem = np.asarray(members, dtype=int)
        if set(domain_x) == {0, 1} and by_x.get(0) and by_x.get(1):
            groups.append(
                MSMGroup(
                    ones=np.asarray(by_x[1], dtype=int),
                    zeros=np.asarray(by_x[0], dtype=int),
                    members=mem,
                )
            )
        elif len(members) >= 2:
            groups.append(
                MSMGroup(
                    ones=np.asarray([], dtype=int),
                    zeros=np.asarray([], dtype=int),
                    members=mem,
                )
            )
    return groups


def _msm_aub_from_groups(
    n: int,
    groups: list[MSMGroup],
    Lambda: float,
    theta0: float,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Binary-root Tan MSM rows from cached ones/zeros groups (template fast path)."""
    rows = []
    cols = []
    data = []
    rhs = []
    r = 0
    for g in groups:
        if len(g.ones) == 0 or len(g.zeros) == 0:
            continue
        for t in g.ones:
            rows.append(r)
            cols.append(int(t))
            data.append(1.0)
        for t in g.zeros:
            rows.append(r)
            cols.append(int(t))
            data.append(-Lambda * theta0)
        rhs.append(0.0)
        r += 1
        for t in g.zeros:
            rows.append(r)
            cols.append(int(t))
            data.append(1.0)
        for t in g.ones:
            rows.append(r)
            cols.append(int(t))
            data.append(-Lambda / theta0)
        rhs.append(0.0)
        r += 1
    A_ub = sparse.csr_matrix((data, (rows, cols)), shape=(r, n), dtype=float)
    return A_ub, np.asarray(rhs, dtype=float)


def _cdep_aub_from_groups(
    n: int,
    groups: list[MSMGroup],
    c: float,
    e0: float,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows, cols, data, rhs = [], [], [], []
    r = 0
    lo = e0 - c
    hi = e0 + c
    for g in groups:
        if len(g.ones) == 0 or len(g.zeros) == 0:
            continue
        for t in g.zeros:
            rows.append(r)
            cols.append(int(t))
            data.append(-hi)
        for t in g.ones:
            rows.append(r)
            cols.append(int(t))
            data.append(1.0 - hi)
        rhs.append(0.0)
        r += 1
        for t in g.zeros:
            rows.append(r)
            cols.append(int(t))
            data.append(lo)
        for t in g.ones:
            rows.append(r)
            cols.append(int(t))
            data.append(lo - 1.0)
        rhs.append(0.0)
        r += 1
    A_ub = sparse.csr_matrix((data, (rows, cols)), shape=(r, n), dtype=float)
    return A_ub, np.asarray(rhs, dtype=float)


@dataclass
class SensitivityTemplate:
    """Cached LP pieces for budget sweeps (MSM / c-dep / kappa)."""

    base_lp: LPData  # without sensitivity A_ub
    treatment: str
    e0: float
    theta0: float
    groups: list[MSMGroup]
    q0_margin: float | None  # for kappa
    notes: list[str]
    pi0: dict[tuple, dict[object, float]] | None = None
    use_general_budget: bool = False

    @classmethod
    def build(
        cls,
        variables: Variables,
        data: Mapping,
        query,
        base_assumptions: AssumptionSet | None,
        *,
        treatment: str = "X",
        optimize: bool = True,
    ) -> SensitivityTemplate:
        base = as_assumption_set(base_assumptions)
        # Formulate without sensitivity atoms
        lp = formulate(variables, data, query, base, optimize=optimize)
        if lp.A_ub is not None:
            # strip any existing ub from unrelated constraints — shouldn't happen
            pass
        ts = lp.type_space
        if not isinstance(ts, TypeSpace):
            raise ValueError("sensitivity grids require a standard TypeSpace (not instrument LP)")
        obs = ts.variables.observed_names(ts.order)
        t_idx = list(obs).index(treatment)
        e0 = float(sum(lp.P_vec[i] for i, cell in enumerate(lp.cells) if cell[t_idx] == 1))
        if not (0 < e0 < 1):
            raise ValueError(f"e0={e0} must lie in (0,1) for sensitivity grids")
        theta0 = e0 / (1.0 - e0)
        groups = _msm_groups(ts, lp.allowed_types, treatment)
        pi0 = build_pi0_for_treatment(ts, lp.cells, lp.P_vec, treatment)
        parents = list(ts.diagram.parents(treatment))
        domain_x = list(ts.variables.domains[treatment])
        use_general = bool(parents) or set(domain_x) != {0, 1}
        try:
            q0 = _point_reference(variables, ts, lp.cells, lp.P_vec, query)
        except Exception:
            q0 = None
        return cls(
            base_lp=lp,
            treatment=treatment,
            e0=e0,
            theta0=theta0,
            groups=groups,
            q0_margin=q0,
            notes=list(lp.notes)
            + [
                f"sensitivity template e0={e0:.4g}, n_groups={len(groups)}"
                + (f", conditional on {parents}" if parents else "")
            ],
            pi0=pi0,
            use_general_budget=use_general,
        )

    def lp_for(
        self,
        param: Literal["Lambda", "Gamma", "c", "kappa"],
        value: float,
    ) -> LPData:
        n = len(self.base_lp.c)
        notes = list(self.notes)
        ts = self.base_lp.type_space
        assert isinstance(ts, TypeSpace)
        if param == "Lambda":
            Lam = float(value)
            if self.use_general_budget and self.pi0 is not None:
                rows, rhs = msm_lr_rows(
                    ts, self.base_lp.allowed_types, self.treatment, Lam, self.pi0
                )
                A_ub = sparse.csr_matrix(np.vstack(rows)) if rows else sparse.csr_matrix((0, n))
                b_ub = np.asarray(rhs, dtype=float)
            else:
                A_ub, b_ub = _msm_aub_from_groups(n, self.groups, Lam, self.theta0)
            notes.append(f"Tan MSM Λ={Lam} [template]")
        elif param == "Gamma":
            Lam = float(np.sqrt(value))
            if self.use_general_budget and self.pi0 is not None:
                rows, rhs = msm_lr_rows(
                    ts, self.base_lp.allowed_types, self.treatment, Lam, self.pi0
                )
                A_ub = sparse.csr_matrix(np.vstack(rows)) if rows else sparse.csr_matrix((0, n))
                b_ub = np.asarray(rhs, dtype=float)
            else:
                A_ub, b_ub = _msm_aub_from_groups(n, self.groups, Lam, self.theta0)
            notes.append(f"anchored Rosenbaum Γ={value} → MSM Λ={Lam} [template]")
        elif param == "c":
            if self.use_general_budget and self.pi0 is not None:
                rows, rhs = cdep_linf_rows(
                    ts, self.base_lp.allowed_types, self.treatment, float(value), self.pi0
                )
                A_ub = sparse.csr_matrix(np.vstack(rows)) if rows else sparse.csr_matrix((0, n))
                b_ub = np.asarray(rhs, dtype=float)
            else:
                A_ub, b_ub = _cdep_aub_from_groups(n, self.groups, float(value), self.e0)
            notes.append(f"c-dependence c={value} [template]")
        elif param == "kappa":
            if self.q0_margin is None:
                raise ValueError("margin budget requires a point reference")
            q0 = self.q0_margin
            kappa = float(value)
            A_ub = sparse.csr_matrix(np.vstack([self.base_lp.c, -self.base_lp.c]), dtype=float)
            b_ub = np.array([q0 + kappa, -q0 + kappa], dtype=float)
            notes.append(f"margin budget κ={kappa} around {q0} [template]")
        else:
            raise ValueError(f"unknown param {param!r}")

        return LPData(
            c=self.base_lp.c,
            A_eq=self.base_lp.A_eq,
            b_eq=self.base_lp.b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            type_space=self.base_lp.type_space,
            allowed_types=self.base_lp.allowed_types,
            cells=self.base_lp.cells,
            P_vec=self.base_lp.P_vec,
            method_hint="lp",
            notes=notes,
            e0=self.e0,
            pi0=self.pi0,
        )
