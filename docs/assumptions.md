# Assumptions

The core idea of **partialid**: pick assumptions that appear in the causal
inference literature, **compose** them with `&`, and compute numeric bounds.

```python
from partialid import Assumptions, Query, Variables, bound

V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

a = (
    Assumptions.order("X", "Y")
    & Assumptions.mtr("Y", "X")
    & Assumptions.tan_msm("X", 2.0)
)
r = bound(V, P, Q, a)
print(r.summary(Q))
```

`&` is a meet: the feasible set is the **intersection** of the constraints.
Adding assumptions can only shrink (or leave unchanged) the identified
interval.

## Catalog

| Factory | Literature idea |
|---|---|
| `order(...)` / `manski(...)` | Temporal / causal order; Manski maximal confounding |
| `diagram(...)` / `from_diagram` | Explicit DAG + latents (exclusions, confounding) |
| `exclusion(child, parent)` | Missing directed edge |
| `mtr(outcome, treatment)` | Monotone treatment response |
| `mts(treatment, outcome)` | Monotone treatment selection (Manski–Pepper) |
| `no_defiers(treatment, instrument)` | Compliance monotonicity |
| `iv(Z, X, Y, monotone=...)` | Classic IV diagram (+ optional no-defiers) |
| `tan_msm(treatment, Λ)` | Tan marginal sensitivity model |
| `rosenbaum_gamma(treatment, Γ)` | Anchored Rosenbaum Γ (via Tan MSM) |
| `c_dependence(treatment, c)` | Masten–Poirier \(c\)-dependence |
| `margin_budget(κ)` | Soft independence / deviation from a reference margin |
| `f_sensitivity(treatment, ρ, f=...)` | \(f\)-divergence sensitivity (CVXPY) |
| `support_bound(k)` | At most \(k\) positive response types (MIP) |
| `independence(*groups)` | Hard independence / c-split between groups |
| `empty()` | Empty assumption set (no structural atoms) |

Recipes (`manski`, `sensitivity_msm`, `iv`) are thin wrappers over these
atoms. Prefer atoms + `&` when a recipe does not match your setting.

`independence` is structural: v1 does **not** solve the sharp product
factorization. Formulate always forces a merged-latent **outer** LP
(`tight=False`) — see [Solvers](advanced/solvers.md). Unanchored
`tan_msm(..., anchored=False)` and unanchored Rosenbaum raise at the factory.

## Structure vs sensitivity

Two complementary families:

1. **Structural / qualitative** — order, diagram, exclusions, MTR/MTS,
   instruments, independence.
2. **Sensitivity / quantitative budgets** — Tan MSM, Rosenbaum Γ,
   \(c\)-dependence, margin budgets, \(f\)-sensitivity.

You can mix both in one `&` expression. Sweeping a budget with
[`bound_path`](sensitivity.md) is optional — useful when you want a curve,
not required for ordinary point bounds.

## Inspecting a set

```python
print(a.describe())           # human-readable atoms
print(len(a))
print([type(atom).__name__ for atom in a])
# a.of_type(SomeAssumptionClass) filters by dataclass type when needed
```

Full signatures: [API · Assumptions](api/assumptions.md).
