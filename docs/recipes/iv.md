# Recipe: IV + monotonicity

**Job:** Balke–Pearl-style interventional bounds with a binary instrument.

```python
from partialid import Assumptions, bound
from partialid.problems import binary_iv

# Observational law sampled from a valid IV SCM (Z→X→Y, X↔Y)
p = binary_iv(seed=2)

# Classic IV diagram (default: no no-defiers)
r = bound(p.variables, p.data, p.query, Assumptions.iv("Z", "X", "Y"))
print(r.summary(p.query))
print("tight=", r.tight, "method=", r.method)

# Opt in to compliance monotonicity only when justified *and* feasible for P
r_m = bound(
    p.variables, p.data, p.query,
    Assumptions.iv("Z", "X", "Y", monotone=True),
)
print("monotone success:", r_m.success)
```

!!! tip
    Instrumental inequalities must hold in \(P\). Prefer laws from a valid SCM
    (as above). `monotone=True` adds no-defiers — if that conflicts with the
    data, the LP is infeasible (`success=False`).
