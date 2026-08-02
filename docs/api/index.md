# API reference

Public symbols are those in `partialid.__all__`. Prefer:

```python
from partialid import Variables, Query, Assumptions, bound, bound_path, classify_program
```

Deep imports of internal modules are unsupported.

| Page | Contents |
|---|---|
| [Variables & data](data.md) | `Variables`, `ObservationalData`, `from_*` |
| [Assumptions](assumptions.md) | `Assumptions`, `AssumptionSet`, recipes |
| [Queries](query.md) | `Query` and AST helpers |
| [Bound](bound.md) | `bound`, `bound_path`, `bound_grid`, results (`prefer_poly` / `prefer_gpp`) |

Also exported: `classify_program`, `ProgramDegree`, `ProgramKind` — see
[Graphs → program degree](../graphs.md#program-degree-which-solver).

Conceptual catalog (literature → factory): [Assumptions guide](../assumptions.md).

Stability tiers: [Stability contract](../stability.md).
