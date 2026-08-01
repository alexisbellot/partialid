# Recipe: measurement error (optional)

**Job:** true outcome is latent; you observe a misclassified measurement.

Declare latent endogenous variables with `observed=...`. They still have
response types and may appear in queries, but do **not** index \(P\).

```python
from partialid import Assumptions, CausalDiagram, Query, Variables, bound

# Latent Y; observe Y2. X → Y → Y2, confounding Y↔Y2; MTR of Y2 in Y.
V = Variables.from_binary("X", "Y", "Y2", observed=["X", "Y2"])
g = CausalDiagram.from_edges(
    ["X", "Y", "Y2"],
    directed=[("X", "Y"), ("Y", "Y2")],
    bidirected=[("Y", "Y2")],
)
# P is on observed columns (X, Y2) only
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q = Query.ate(outcome="Y", treatment="X", y=1, t1=1, t0=0)
a = Assumptions.from_diagram(g) & Assumptions.mtr("Y2", "Y")

r = bound(V, P, Q, a)
print(r.summary(Q))
print("tight=", r.tight)
```

!!! warning
    Measurement-error graphs are easy to misspecify. Prefer published
    identification assumptions for the measurement channel when you have them.
