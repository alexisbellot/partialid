# partialid

[![CI](https://github.com/alexisbellot/partialid/actions/workflows/ci.yml/badge.svg)](https://github.com/alexisbellot/partialid/actions/workflows/ci.yml)

> From discrete data and a list of assumptions you are willing to make,
> compute the tightest interval that must contain a causal effect.

**partialid** answers questions like: *“If we intervene and set treatment to 1,
what fraction of people would have a good outcome?”* — when that effect is
**not** uniquely determined by the observed table alone.

Instead of forcing a single answer, the package returns a **bound**: a lower
and upper value consistent with the data and with every assumption you compose
(causal order, “treatment never hurts,” limited hidden bias, an instrument, …).
Assumptions are named pieces from the literature; you combine them with `&`.

**Status:** `0.1.0a5` (alpha). The API may change; see the
[stability contract](docs/stability.md).

**Docs:** [https://alexisbellot.github.io/partialid/](https://alexisbellot.github.io/partialid/)
(GitHub Pages; available once the repo is public / Pages is enabled).

## Install

Requires Python ≥ 3.10. Dependencies: NumPy, SciPy, pandas, HiGHS (`highspy`),
and CVXPY.

Until the package is on PyPI, install from the repository:

```bash
pip install "git+https://github.com/alexisbellot/partialid.git"
# or, from a clone:
pip install -e .
```

```bash
pip install -e ".[dev]"         # lint, tests, docs
pip install -e ".[notebooks]"   # Jupyter for tutorials under notebooks/
pytest
```

When published: `pip install partialid`.

## Quick start

Binary treatment `X` and outcome `Y`, with an observed joint distribution.
We ask for the probability that `Y = 1` if we set `X = 1` for everyone.

```python
from partialid import Assumptions, Query, Variables, bound

V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

# Compose assumptions: causal order, "treatment never hurts the outcome",
# and a bound on how much hidden confounding can distort treatment odds.
a = (
    Assumptions.order("X", "Y")
    & Assumptions.mtr("Y", "X")          # monotone treatment response
    & Assumptions.tan_msm("X", 2.0)      # odds of treatment at most 2× off
)
r = bound(V, P, Q, a)
print(r.summary(Q))
# → an interval [lower, upper], plus whether the bound is sharp (r.tight)
```

Common ready-made bundles:

```python
# Worst-case confounding given only the order X → Y (classic Manski bounds)
r = bound(V, P, Q, Assumptions.manski("X", "Y"))

# Manski plus a sensitivity budget on hidden bias (parameter Λ ≥ 1)
r = bound(V, P, Q, Assumptions.sensitivity_msm("X", "Y", Lambda=3.0))

# Instrument Z that affects Y only through treatment X
r = bound(V, P, Q, Assumptions.iv("Z", "X", "Y", monotone=True))
```

Optional: see how the interval widens as you allow more hidden bias:

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
print(path.at(3.0))  # (lower, upper, width) at Λ = 3
```

From a contingency table or row data:

```python
from partialid import from_counts, from_records

data = from_counts(("X", "Y"), {(0, 0): 30, (0, 1): 20, (1, 0): 15, (1, 1): 35})
# or: data = from_records(("X", "Y"), rows)
# or: from_dataframe(df, columns=["X", "Y"])

r = bound(data.variables, data.P, Q, Assumptions.manski("X", "Y"))
```

Longer walkthroughs: [`notebooks/`](notebooks/)
(start at [`01_first_bounds`](notebooks/01_first_bounds.ipynb)).

## What you can assume (catalog)

Each factory is one literature-style assumption. Combine any of them with `&`.

| Name | Plain meaning |
|---|---|
| `order` / `manski` | Variables follow a causal order; Manski = that order with worst-case hidden confounding |
| `diagram` / `iv` | An explicit causal graph; `iv` is the classic instrument setup (Z → X → Y, hidden link X ↔ Y) |
| `mtr` | Monotone treatment response: raising treatment never lowers the potential outcome |
| `mts` | Monotone treatment selection: people who take treatment tend to have better outcomes anyway |
| `no_defiers` | Under an instrument: no one does the opposite of encouragement |
| `tan_msm` | Limit how far hidden factors can push treatment odds (Tan marginal sensitivity model) |
| `rosenbaum_gamma` | Same idea in Rosenbaum’s Γ scale (anchored form) |
| `c_dependence` | Cap the absolute shift in treatment propensity due to confounding |
| `margin_budget` | Soften “no confounding” with a budget on how much margins may disagree |
| `f_sensitivity` | Soften independence with an \(f\)-divergence budget (KL / χ² / total variation) |
| `exclusion` | A variable does not affect another (remove an arrow) |
| `independence` | Groups of variables do not share a common cause (v1 returns a conservative outer bound) |
| `support_bound` | At most \(k\) latent response types have positive mass |

Details and citations: [docs/assumptions.md](docs/assumptions.md).

## Causal graphs

If you pass an explicit graph (`CausalDiagram`), the package first checks whether
the effect is **point-identified** (a single number, as in the usual do-calculus /
identification algorithms). If not, it computes **partial**-identification
bounds under that graph. Always check `r.tight`: when it is `False`, the
interval is valid but may be wider than the sharpest possible answer.

```python
from partialid import CausalDiagram, is_identified

g = CausalDiagram.from_edges(
    ["X", "M", "Y"],
    directed=[("X", "M"), ("M", "Y")],
    bidirected=[("X", "M"), ("M", "Y")],  # hidden common causes on those pairs
)
print(is_identified(g, ["X"], ["Y"]))  # False — need bounds, not a point
```

How the solver is chosen automatically (linear program vs nested methods vs a
conservative outer approximation) is documented under
[graphs](docs/graphs.md) and [auto routing](docs/auto_routing.md). For ordinary
use, leave `method="auto"`.

## When to trust the interval

- `r.tight is True` — bounds are sharp under the assumptions you stated.
- `r.tight is False` — still a reported interval, but either a **conservative
  outer** approximation (may be wider than necessary) or, for some hard
  bilinear programs, an **uncertified** numeric attainable (may be slightly
  too narrow). Inspect `r.notes` / warnings; see [stability](docs/stability.md).

## Docs

Full site: [partialid documentation](https://alexisbellot.github.io/partialid/)

| Link | Contents |
|---|---|
| [Quickstart](docs/quickstart.md) | Install + first bounds |
| [Assumptions](docs/assumptions.md) | Composition + catalog |
| [Graphs](docs/graphs.md) | Diagrams and instruments |
| [Sensitivity](docs/sensitivity.md) | Sweeping a bias parameter |
| [Recipes](docs/recipes/manski.md) | Copy-paste workflows |
| [API](docs/api/index.md) | Public symbols |
| [notebooks/](notebooks/) | Tutorial notebooks |

Build locally: `pip install -e ".[docs]" && mkdocs serve`

## License

BSD-3-Clause. See [LICENSE](LICENSE).
