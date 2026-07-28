# Sensitivity analysis

When one of your assumptions is a **quantitative budget** (Tan MSM Λ,
Rosenbaum Γ, \(c\)-dependence, margin κ, …), you can either fix the budget and
call `bound`, or sweep it with `bound_path` / `bound_grid` to see how the
identified interval moves.

This page is about that optional curve workflow. For composing structural
assumptions (order, MTR, IV, …) see [Assumptions](assumptions.md).

## Tan MSM (Λ)

Anchored Tan marginal sensitivity model: relative-propensity likelihood ratios
bounded by \(\Lambda \ge 1\). \(\Lambda=1\) is no unobserved confounding on the
propensity; larger \(\Lambda\) widens bounds.

```python
from partialid import Assumptions, Query, Variables, bound, bound_path

V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

r = bound(V, P, Q, Assumptions.sensitivity_msm("X", "Y", Lambda=3.0))
print(r.summary(Q))

path = bound_path(
    V, P, Q, Assumptions.manski("X", "Y"),
    param="Lambda", theta_min=1.0, theta_max=5.0,
)
for lam in (1, 2, 3, 5):
    lo, hi, w = path.at(lam)
    print(f"Λ={lam}: [{lo:.4f}, {hi:.4f}]  width={w:.4f}")
```

On binary \(X,Y\), many MSM paths use a closed-form margin (no LP). Check
`path.closed_form`.

### Breakdown

Where does a null enter the interval?

```python
print(path.breakdown(null=0.5))  # budget where 0.5 ∈ [ℓ,u], when defined
```

## Rosenbaum Γ

Anchored Rosenbaum \(\Gamma\) maps to Tan MSM with \(\Lambda=\sqrt{\Gamma}\):

```python
a = Assumptions.manski("X", "Y") & Assumptions.rosenbaum_gamma("X", Gamma=4.0)
# anchored=True required (unanchored Γ is not LP)
```

Use `bound_path(..., param="Gamma", ...)`.

## c-dependence

Masten–Poirier \(\ell_\infty\) budget on the propensity:

```python
a = Assumptions.manski("X", "Y") & Assumptions.c_dependence("X", c=0.1)
path = bound_path(V, P, Q, Assumptions.manski("X", "Y"), param="c", theta_min=0.0, theta_max=0.5)
```

## Margin budget (soft independence)

Soft independence on an identifying margin \(|S - S_0| \le \kappa\):

```python
a = Assumptions.manski("X", "Y") & Assumptions.margin_budget(kappa=0.05)
```

See [Soft independence recipe](recipes/margin_budget.md).

## Dense grids

```python
from partialid import bound_grid
import numpy as np

for lam, r in bound_grid(
    V, P, Q, Assumptions.manski("X", "Y"),
    param="Lambda", values=np.linspace(1, 5, 9),
):
    print(lam, r.interval)
```

Install `partialid[fast]` (highspy) for persistent HiGHS updates across the grid.

## Next

- [MSM sensitivity recipe](recipes/msm_path.md)
- [Graphs & IV](graphs.md)
