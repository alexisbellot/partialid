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

Also exported (see [Stability](../stability.md) for tiers):

- `CausalDiagram`, `classify_program`, `ProgramDegree`, `ProgramKind`
- Identification: `is_identified`, `identify_query`, `IDResult`
- Query AST: `PO`, `Lit`, `Atom`, `LinComb`, `Cond`
- `audit_assumptions`, `audit_bound`, `bound_ci`, `try_closed_form`
- Literature `Problem` / `all_literature_problems`

Program-degree routing: [Graphs](../graphs.md#program-degree-which-solver).
Conceptual catalog: [Assumptions guide](../assumptions.md).
