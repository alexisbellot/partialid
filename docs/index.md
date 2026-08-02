# partialid

> Bound causal effects from discrete data by composing explicit assumptions
> from the literature — without requiring a fully identified DAG.

**partialid** is a Python library for **numeric** partial identification in
discrete structural causal models. The main idea is **assumption composition**:
combine orders, diagrams, monotonicity, instruments, and sensitivity models
with `&`, pass observational data, and get an interval under those constraints.
Sensitivity *paths* (`bound_path`) are available when you want to sweep a
budget — they are optional, not required.

!!! tip "Start here"
    New users: [Quickstart](quickstart.md) → [Assumptions](assumptions.md) →
    [Recipes](recipes/manski.md). Leave `method="auto"`.

## Install

```bash
pip install partialid
```

Requires Python ≥ 3.10. Dependencies: NumPy, SciPy, pandas, HiGHS (`highspy`),
and CVXPY.

## 30-second example

```python
from partialid import Assumptions, Query, Variables, bound

V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

a = Assumptions.order("X", "Y") & Assumptions.mtr("Y", "X") & Assumptions.tan_msm("X", 2.0)
r = bound(V, P, Q, a)
print(r.summary(Q))
```

## What you get

| Capability | Entry point |
|---|---|
| Assumption composition | `Assumptions.*` factories + `&` |
| Point bounds | `bound(...)` |
| Sensitivity curves | `bound_path(...)`, `bound_grid(...)` |
| Data ingress | `from_counts`, `from_records`, `from_dataframe` |
| Recipes | `Assumptions.manski`, `sensitivity_msm`, `iv` |
| Diagrams & program class | `CausalDiagram`, `classify_program` |
| Identification (explicit DAG) | `is_identified`, `identify_query` |
| Solver knobs | `prefer_gpp`, `prefer_poly` on `bound` |
| Audit / CI (experimental) | `audit_assumptions`, `audit_bound`, `bound_ci` |

Always check `BoundResult.tight`. Degree ≥3 and some multi-component graphs use
a merged-latent **outer** LP by default (`tight=False`); opt into nested
multilinear with `prefer_poly=True`. See [Concepts](concepts.md),
[Graphs](graphs.md), and [Stability](stability.md).

## Status

Alpha (`0.1.0a4`). Public API may evolve on 0.x; see the
[stability contract](stability.md).

## Further reading

- [Assumptions](assumptions.md) — literature catalog and composition
- [Graphs](graphs.md) — diagrams, `classify_program`, instruments
- [API reference](api/index.md)
- [Auto routing](auto_routing.md) — what `method="auto"` chooses
- [Solvers (advanced)](advanced/solvers.md) — method scope and routing sketch
