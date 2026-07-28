# Quickstart

Install and bound an effect under composed literature assumptions.

## Install

```bash
pip install partialid
```

## Minimal example

```python
from partialid import Assumptions, Query, Variables, bound

V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

# Compose atoms from the literature with &
a = Assumptions.order("X", "Y") & Assumptions.mtr("Y", "X")
r = bound(V, P, Q, a)
print(r.summary(Q))

# Or use a recipe for a common bundle
r = bound(V, P, Q, Assumptions.manski("X", "Y"))
r = bound(V, P, Q, Assumptions.sensitivity_msm("X", "Y", Lambda=3.0))
print(round(r.width, 3))  # 0.333
```

Leave `method="auto"` (the default). Advanced solvers are documented under
[Auto routing](auto_routing.md).

## From a table or CSV

```python
from partialid import from_counts

data = from_counts(
    ("X", "Y"),
    {(0, 0): 30, (0, 1): 20, (1, 0): 15, (1, 1): 35},
)
r = bound(data.variables, data.P, Q, Assumptions.manski("X", "Y"))
```

With pandas (`pip install 'partialid[data]'`):

```python
import pandas as pd
from partialid import from_dataframe

data = from_dataframe(pd.read_csv("table.csv"), columns=["X", "Y"])
```

Runnable scripts: `examples/quickstart.py`, `examples/from_csv.py`.

## Next

- [Assumptions](assumptions.md) — catalog of literature factories + `&`
- [Concepts](concepts.md) — data, queries, tightness
- [Sensitivity analysis](sensitivity.md) — optional budget sweeps
- [Recipes](recipes/manski.md) — copy-paste jobs
