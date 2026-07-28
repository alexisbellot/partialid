"""Anchored sensitivity budgets on discrete treatment propensities.

Default MSM uses **likelihood-ratio bounds on relative propensities**
(Tan 2006 generalized to categorical treatments):

    Λ⁻¹ ≤ [π(x|r,pa) / π(x'|r,pa)] / [π₀(x|pa) / π₀(x'|pa)] ≤ Λ

for all treatment levels x, x' with positive baseline mass.  When treatment is
binary this is exactly classical Tan odds MSM
(q₁ ≤ Λ θ₀ q₀ and q₀ ≤ (Λ/θ₀) q₁ with θ₀ = e₀/(1-e₀)).

**c-dependence** uses the simplest categorical lift of Masten–Poirier:

    ‖π(·|r,pa) − π₀(·|pa)‖_∞ ≤ c

which reduces to |e(r)−e₀| ≤ c for binary treatment.

When the diagram gives parents Pa of the treatment, both budgets are
**conditional**: π₀(·|pa) = P(X|Pa=pa) and constraints are applied per
parent configuration pa.
"""

from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

import numpy as np

from .types_canon import TypeSpace
from .variables import Variables

_EPS = 1e-15


def parent_configs(variables: Variables, parents: Sequence[str]) -> list[tuple]:
    if not parents:
        return [()]
    return variables.cells(list(parents))


def observed_conditional_propensity(
    order: Sequence[str],
    cells: Sequence[tuple],
    P: np.ndarray,
    treatment: str,
    parents: Sequence[str],
    domain_x: Sequence,
) -> dict[tuple, dict[object, float]]:
    """Return ``π0[pa][x] = P(X=x | Pa=pa)`` for strata with positive mass."""
    t_idx = list(order).index(treatment)
    p_idxs = [list(order).index(p) for p in parents]
    mass_pa: dict[tuple, float] = {}
    mass_pa_x: dict[tuple, dict[object, float]] = {}
    for cell, p in zip(cells, P):
        pa = tuple(cell[i] for i in p_idxs)
        x = cell[t_idx]
        mass_pa[pa] = mass_pa.get(pa, 0.0) + float(p)
        bucket = mass_pa_x.setdefault(pa, {})
        bucket[x] = bucket.get(x, 0.0) + float(p)

    out: dict[tuple, dict[object, float]] = {}
    for pa, m in mass_pa.items():
        if m <= _EPS:
            continue
        out[pa] = {x: mass_pa_x[pa].get(x, 0.0) / m for x in domain_x}
    return out


def rest_type_groups(
    ts: TypeSpace,
    allow: np.ndarray,
    treatment: str,
) -> dict[tuple, list[int]]:
    """Partition types by response indices of every variable except ``treatment``."""
    t_pos = ts.order.index(treatment)
    groups: dict[tuple, list[int]] = {}
    for t in range(ts.n_types):
        if not allow[t]:
            continue
        key = tuple(fi for i, fi in enumerate(ts.joint_types[t]) if i != t_pos)
        groups.setdefault(key, []).append(t)
    return groups


def _treatment_value(
    ts: TypeSpace,
    type_idx: int,
    treatment: str,
    parents: Sequence[str],
    pa: tuple,
) -> object:
    pa_map = {p: pa[i] for i, p in enumerate(parents)}
    return ts.response_on(type_idx, treatment, pa_map)


def msm_lr_rows(
    ts: TypeSpace,
    allow: np.ndarray,
    treatment: str,
    Lambda: float,
    pi0: Mapping[tuple, Mapping[object, float]],
) -> tuple[list[np.ndarray], list[float]]:
    """Relative-propensity LR inequalities (Tan when binary)."""
    if Lambda < 1.0 - 1e-15:
        raise ValueError("Tan MSM requires Lambda >= 1")
    parents = list(ts.diagram.parents(treatment))
    domain_x = list(ts.variables.domains[treatment])
    groups = rest_type_groups(ts, allow, treatment)
    n = ts.n_types
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for members in groups.values():
        for pa, pi_pa in pi0.items():
            by_x: dict[object, list[int]] = {x: [] for x in domain_x}
            for t in members:
                x = _treatment_value(ts, t, treatment, parents, pa)
                by_x[x].append(t)
            # Pairwise relative LR for levels with positive baseline mass.
            levels = [x for x in domain_x if pi_pa.get(x, 0.0) > _EPS and by_x[x]]
            if len(levels) < 2:
                continue
            for x, xp in product(levels, repeat=2):
                if x == xp:
                    continue
                # q(x) * π0(x') - Λ * π0(x) * q(x') <= 0
                row = np.zeros(n)
                for t in by_x[x]:
                    row[t] += pi_pa[xp]
                for t in by_x[xp]:
                    row[t] -= Lambda * pi_pa[x]
                if np.any(np.abs(row) > _EPS):
                    rows.append(row)
                    rhs.append(0.0)
    return rows, rhs


def cdep_linf_rows(
    ts: TypeSpace,
    allow: np.ndarray,
    treatment: str,
    c: float,
    pi0: Mapping[tuple, Mapping[object, float]],
) -> tuple[list[np.ndarray], list[float]]:
    """ℓ∞ c-dependence: max_x |π(x|r,pa) − π₀(x|pa)| ≤ c."""
    if c < 0:
        raise ValueError("c-dependence requires c >= 0")
    parents = list(ts.diagram.parents(treatment))
    domain_x = list(ts.variables.domains[treatment])
    groups = rest_type_groups(ts, allow, treatment)
    n = ts.n_types
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for members in groups.values():
        for pa, pi_pa in pi0.items():
            by_x: dict[object, list[int]] = {x: [] for x in domain_x}
            for t in members:
                x = _treatment_value(ts, t, treatment, parents, pa)
                by_x[x].append(t)
            active = [x for x in domain_x if by_x[x]]
            if not active:
                continue
            # For each x: (π0(x)-c) Σq ≤ q(x) ≤ (π0(x)+c) Σq
            for x in domain_x:
                lo = pi_pa.get(x, 0.0) - c
                hi = pi_pa.get(x, 0.0) + c
                # q(x) - hi * Σq <= 0
                r_hi = np.zeros(n)
                for t in members:
                    r_hi[t] -= hi
                for t in by_x[x]:
                    r_hi[t] += 1.0
                if np.any(np.abs(r_hi) > _EPS):
                    rows.append(r_hi)
                    rhs.append(0.0)
                # lo * Σq - q(x) <= 0
                r_lo = np.zeros(n)
                for t in members:
                    r_lo[t] += lo
                for t in by_x[x]:
                    r_lo[t] -= 1.0
                if np.any(np.abs(r_lo) > _EPS):
                    rows.append(r_lo)
                    rhs.append(0.0)
    return rows, rhs


def build_pi0_for_treatment(
    ts: TypeSpace,
    cells: Sequence[tuple],
    P: np.ndarray,
    treatment: str,
) -> dict[tuple, dict[object, float]]:
    parents = list(ts.diagram.parents(treatment))
    domain_x = list(ts.variables.domains[treatment])
    # ``cells`` / ``P`` are indexed by observed coordinates (may omit latents)
    obs_order = ts.variables.observed_names(ts.order)
    return observed_conditional_propensity(obs_order, cells, P, treatment, parents, domain_x)


def binary_marginal_e0(
    pi0: Mapping[tuple, Mapping[object, float]],
    *,
    treated_level: object = 1,
) -> float | None:
    """Marginal P(X=treated) when there is a single empty-parent stratum."""
    if list(pi0.keys()) == [()]:
        return float(pi0[()].get(treated_level, 0.0))
    # Mixture over parent strata is not stored here; caller may pass raw P.
    return None
