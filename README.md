# partialid

[![CI](https://github.com/alexisbellot/partialid/actions/workflows/ci.yml/badge.svg)](https://github.com/alexisbellot/partialid/actions/workflows/ci.yml)

> Bound causal effects from discrete data by composing explicit assumptions
> from the literature — without requiring a fully identified DAG.

**partialid** computes **numeric** partial-identification bounds for discrete
structural causal models. The main idea is **assumption composition**: combine
orders, diagrams, monotonicity, instruments, and sensitivity models with `&`,
pass observational data, and get a sharp interval under those constraints.

**Status:** `0.1.0a4` (alpha). API may change; see
[stability contract](docs/stability.md).

**Docs:** [https://alexisbellot.github.io/partialid/](https://alexisbellot.github.io/partialid/)
(GitHub Pages; available once the repo is public / Pages is enabled).

## Install

```bash
pip install partialid
```

Requires Python ≥ 3.10. Dependencies: NumPy, SciPy, pandas, HiGHS (`highspy`),
and CVXPY.

```bash
pip install -e ".[dev]"   # contributors
pytest
```

## Quick start — compose assumptions

```python
from partialid import Assumptions, Query, Variables, bound

V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

# Literature atoms compose with &
a = Assumptions.order("X", "Y") & Assumptions.mtr("Y", "X") & Assumptions.tan_msm("X", 2.0)
r = bound(V, P, Q, a)
print(r.summary(Q))
```

Recipes cover common bundles (`method="auto"` by default):

```python
r = bound(V, P, Q, Assumptions.manski("X", "Y"))
r = bound(V, P, Q, Assumptions.sensitivity_msm("X", "Y", Lambda=3.0))
r = bound(V, P, Q, Assumptions.iv("Z", "X", "Y", monotone=True))
```

Optional: how wide do MSM bounds get as Λ varies?

```python
from partialid import bound_path

path = bound_path(
    V,
    P,
    Q,
    Assumptions.manski("X", "Y"),
    param="Lambda",
    theta_min=1.0,
    theta_max=5.0,
)
print(path.at(3.0))  # (lower, upper, width)
```

From a contingency table or CSV rows:

```python
from partialid import from_counts, from_records

data = from_counts(("X", "Y"), {(0, 0): 30, (0, 1): 20, (1, 0): 15, (1, 1): 35})
# or: data = from_records(("X", "Y"), rows)
# or: from_dataframe(df, columns=["X", "Y"])

r = bound(data.variables, data.P, Q, Assumptions.manski("X", "Y"))
```

See `examples/quickstart.py`, `examples/from_csv.py`, and the showcase
notebooks under [`notebooks/`](notebooks/) (see [`notebooks/README.md`](notebooks/README.md)).

## Assumptions (catalog)

| Factory / recipe | Meaning |
|---|---|
| `order` / `manski` | Causal order; Manski maximal confounding |
| `diagram` / `iv` | Explicit DAG (+ classic IV recipe) |
| `mtr` / `mts` / `no_defiers` | Response / selection / compliance monotonicity |
| `tan_msm` / `rosenbaum_gamma` / `c_dependence` | Sensitivity budgets |
| `margin_budget` / `f_sensitivity` | Soft independence / \(f\)-divergence budgets |
| `exclusion` / `independence` / `support_bound` | Structure and sparsity |

Compose any of these with `&`. Details: [docs/assumptions.md](docs/assumptions.md).

## Diagrams & program degree

With an explicit `CausalDiagram`, `bound` runs **Shpitser–Pearl ID** first
(point interval when identified), then routes by **program degree** via
`classify_program` (LP / bilinear GPP / outer for deg≥3). Check `r.tight`.
Deg≥3 defaults to an outer LP; opt in with `bound(..., prefer_poly=True)`.
Details: [docs/graphs.md](docs/graphs.md), [auto routing](docs/auto_routing.md).

```python
from partialid import CausalDiagram, classify_program, is_identified

g = CausalDiagram.from_edges(
    ["X", "M", "Y"],
    directed=[("X", "M"), ("M", "Y")],
    bidirected=[("X", "M"), ("M", "Y")],
)
print(classify_program(g, treatments=["X"], outcome="Y").kind)
# bilinear/latent
print(is_identified(g, ["X"], ["Y"]))  # False (pair chain)
```

## When bounds may be loose

If the solver falls back to a merged-latent **outer** approximation
(deg≥3 by default, size-gate failure, hard independence, …),
`BoundResult.tight` is `False` and a `UserWarning` is emitted. Check `r.tight`
in pipelines. Details: [stability](docs/stability.md),
[solvers (advanced)](docs/advanced/solvers.md).

Advanced solver flags (`dual`, `colgen`, `prefer_gpp`, `prefer_poly`, …) are
optional; leave `method="auto"`.

## Docs

Full site (MkDocs): [partialid documentation](https://alexisbellot.github.io/partialid/)

| Link | Contents |
|---|---|
| [Quickstart](docs/quickstart.md) | Install + first bounds |
| [Assumptions](docs/assumptions.md) | Composition + literature catalog |
| [Graphs](docs/graphs.md) | Diagrams, `classify_program`, IV |
| [Auto routing](docs/auto_routing.md) | What `method="auto"` chooses |
| [Sensitivity](docs/sensitivity.md) | MSM / paths when you want a curve |
| [Recipes](docs/recipes/manski.md) | Copy-paste jobs |
| [API](docs/api/index.md) | Public symbols |
| [Solvers](docs/advanced/solvers.md) | Advanced methods / routing |
| [examples/](examples/) | Runnable scripts |
| [notebooks/](notebooks/) | Numbered tutorial notebooks (start at `01_first_bounds`) |

Build locally: `pip install -e ".[docs]" && mkdocs serve`

## License

BSD-3-Clause. See [LICENSE](LICENSE).
