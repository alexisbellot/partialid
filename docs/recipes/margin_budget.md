# Recipe: soft independence (margin budget)

**Job:** what if a hard identifying assumption is only approximately true?
Use a margin budget \(|S-S_0|\le\kappa\).

```python
from partialid import Assumptions, Query, Variables, bound, bound_path

V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

# Soften a point-identified reference on the query margin
a = Assumptions.manski("X", "Y") & Assumptions.margin_budget(kappa=0.05)
r = bound(V, P, Q, a)
print(r.summary(Q))

path = bound_path(
    V, P, Q, Assumptions.manski("X", "Y"),
    param="kappa", theta_min=0.0, theta_max=0.2,
)
print(path.summary(at=0.05))
```

!!! note
    `reference="point"` (default) anchors \(S_0\) to a no-confounding /
    g-formula pole when available. See `Assumptions.margin_budget` in the API.

Compare to hard MSM budgets in [Sensitivity analysis](../sensitivity.md).
