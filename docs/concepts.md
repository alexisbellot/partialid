# Concepts

## The identification problem

You observe a discrete joint law \(P\) on endogenous variables. A causal query
(e.g. \(P(Y=1\mid \mathrm{do}(X=1))\)) is typically **not point-identified**
under confounding. **Partial identification** returns the sharp interval of
values compatible with \(P\) and your **assumptions**.

## Building blocks

### Variables

```python
from partialid import Variables

V = Variables.from_binary("X", "Y")
# or
V = Variables.from_mapping({"X": (0, 1), "Y": (0, 1, 2)})
```

Declaration order is **not** a causal order — pass order/diagram via assumptions.

### Observational data

Pass a probability table keyed by tuples in the causal (or declaration) order,
or build one with helpers:

```python
from partialid import from_counts, from_records

data = from_counts(("X", "Y"), {(0, 0): 30, (0, 1): 20, (1, 0): 15, (1, 1): 35})
# data.variables, data.P
```

Masses are normalized to sum to 1 by default. If you pass `normalize=False`,
the table must already be a simplex (helpful error otherwise).

### Queries

```python
from partialid import Query

Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
Q = Query.ate(outcome="Y", treatment="X")
Q = Query.pns(treatment="X", outcome="Y")
```

Use `Q.describe()` in logs / `BoundResult.summary(Q)`.

### Assumptions

The main API is **composition**. Factories map onto common literature
assumptions; `&` intersects their constraints:

```python
from partialid import Assumptions

a = Assumptions.order("X", "Y") & Assumptions.mtr("Y", "X") & Assumptions.tan_msm("X", 2.0)
a = Assumptions.sensitivity_msm("X", "Y", Lambda=3.0)  # recipe = manski & tan_msm
a = Assumptions.iv("Z", "X", "Y", monotone=True)       # diagram (+ no-defiers)
```

See the [Assumptions](assumptions.md) catalog for the full list (MTR/MTS, IV,
Tan MSM, Rosenbaum Γ, \(c\)-dependence, margin budgets, …).

## Tight vs outer

| Signal | Meaning |
|---|---|
| `tight=True` | Sharp bounds under the stated discrete program |
| `tight=False` | Valid **outer** approximation (may be loose) |
| `UserWarning` | Same issue, human-readable |

Typical cause of `tight=False`: multi c-component / merged-latent fallback.
See [Stability](stability.md).

## Method argument

Prefer `method="auto"`. See [Auto routing](auto_routing.md). You do not need
to choose `dual` / `colgen` / `fiber` for ordinary work.
