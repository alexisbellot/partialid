# Recipe: Manski bounds from a 2×2 table

**Job:** worst-case interventional bounds under order-only maximal confounding.

```python
from partialid import Assumptions, Query, from_counts, bound

data = from_counts(
    ("X", "Y"),
    {(0, 0): 30, (0, 1): 20, (1, 0): 15, (1, 1): 35},
)
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

r = bound(data.variables, data.P, Q, Assumptions.manski("X", "Y"))
print(r.summary(Q))
print(r.to_dict()["interval"])
```

Expected shape: a wide interval (Manski). Adding `Assumptions.mtr("Y", "X")`
typically shrinks it.

```python
r_mtr = bound(
    data.variables, data.P, Q,
    Assumptions.manski("X", "Y") & Assumptions.mtr("Y", "X"),
)
print(r_mtr.width, "≤", r.width)
```
