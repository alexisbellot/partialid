# Recipe: MSM sensitivity curve

**Job:** how wide are \(P(Y=1\mid\mathrm{do}(X=1))\) bounds as Tan MSM Λ goes from 1 → 5?

```python
from partialid import Assumptions, Query, Variables, bound_path

V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

path = bound_path(
    V, P, Q, Assumptions.manski("X", "Y"),
    param="Lambda", theta_min=1.0, theta_max=5.0,
)
print(path.summary(at=3.0))
for lam in (1.0, 2.0, 3.0, 5.0):
    lo, hi, w = path.at(lam)
    print(f"Λ={lam:g}: [{lo:.4f}, {hi:.4f}]  width={w:.4f}")
```

At \(\Lambda=1\) the width is 0 (point-identified under no unobserved confounding
on this margin). At \(\Lambda=3\) the width is \(1/3\) for this classic table.

Point estimate at a fixed budget:

```python
from partialid import bound

r = bound(V, P, Q, Assumptions.sensitivity_msm("X", "Y", Lambda=3.0))
print(r.summary(Q))
```

Also see the [tutorial notebooks](../tutorials.md)
(especially `03_sensitivity_paths.ipynb`).
