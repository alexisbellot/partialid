#!/usr/bin/env python3
"""One-shot builder for the numbered tutorial notebooks. Not imported by CI."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

META = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}


def md(source: str) -> dict:
    text = source.strip("\n") + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(source: str) -> dict:
    text = source.strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_nb(name: str, cells: list[dict]) -> None:
    path = NB / name
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": META,
        "cells": cells,
    }
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", path.relative_to(ROOT), f"({len(cells)} cells)")


# Shared path helper used in every notebook that loads CSV.
PATH_HELPER = '''
from pathlib import Path

def tutorial_data(name: str) -> Path:
    """Resolve notebooks/data/<name> whether __file__ is set (CI) or not."""
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidates = [
        here / "data" / name,
        here / "notebooks" / "data" / name,
        Path.cwd() / "notebooks" / "data" / name,
        Path.cwd() / "data" / name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"could not find notebooks/data/{name}; tried {candidates}")
'''.strip()


def nb01() -> None:
    write_nb(
        "01_first_bounds.ipynb",
        [
            md(
                """
# 01 — First bounds: from a table to a causal interval

**partialid** computes *numeric partial-identification bounds* for causal
queries in discrete data. You do not need a fully identified DAG. The main
idea is **assumption composition**: pick assumptions that appear in the causal
inference literature, combine them with `&`, and call `bound`.

This notebook is a complete first analysis. You will:

1. Inspect a small observational extract from a job-training study.
2. State a causal question in words and as a `Query`.
3. Compute Manski (worst-case confounding) bounds.
4. Read a `BoundResult` (`width`, `tight`, `method`, …).
5. Tighten the interval by adding monotone-response assumptions.

You do not need any other package docs to follow along.
"""
            ),
            md(
                """
## The study

We evaluate an observational **job-training program**. For each person we see:

- `treated` — whether they participated in the program (0/1)
- `employed` — whether they are employed at follow-up (0/1)

People select into training for reasons we do not fully observe (motivation,
local labor markets, …), so a simple comparison of employment rates between
treated and untreated units is **not** the causal effect of the program.

**Scientific question:** if *everyone* had been treated, what fraction would be
employed? That is the interventional probability
\\(P(\\mathrm{employed}=1 \\mid \\mathrm{do}(\\mathrm{treated}=1))\\).
"""
            ),
            md("## Load and inspect the data"),
            code(
                PATH_HELPER
                + """

import pandas as pd

df = pd.read_csv(tutorial_data("job_training.csv"))
print(f"N = {len(df)} individuals")
print(df.head(8))
print()
print("treatment rate:", round(df["treated"].mean(), 3))
print("overall employment rate:", round(df["employed"].mean(), 3))
print()
print("Employment rate by treatment:")
print(df.groupby("treated")["employed"].mean().rename("P(employed=1 | treated)"))
print()
ct = pd.crosstab(df["treated"], df["employed"], margins=True)
print("Counts:")
print(ct)
print()
print("Row percentages (within treated):")
print(pd.crosstab(df["treated"], df["employed"], normalize="index").round(3))
"""
            ),
            md(
                """
Among the treated, employment is high — but that **associational** success rate
mixes the program's effect with selection. Partial identification asks: given
only the observed table and stated assumptions, which values of the
*interventional* employment rate remain possible?
"""
            ),
            md("## Turn the table into partialid objects"),
            code(
                """
from partialid import Assumptions, Query, bound, from_dataframe

data = from_dataframe(df, columns=["treated", "employed"])
print("variables:", data.variables.names)
print("domains:", data.variables.domains)
print("n_obs:", data.n_obs)
print("first probability atoms:")
for k, v in list(sorted(data.P.items()))[:4]:
    print(f"  P{k} = {v:.4f}")
"""
            ),
            md(
                """
`from_dataframe` builds an `ObservationalData` object: discrete `Variables`
plus an empirical joint law `P` (normalized cell probabilities). You will pass
`data.variables` and `data.P` to `bound`.
"""
            ),
            md("## State the causal question"),
            code(
                """
Q = Query.interventional(outcome="employed", do={"treated": 1}, value=1)
print(Q.describe())

# Associational benchmark: employment among those who were actually treated
assoc = df.loc[df["treated"] == 1, "employed"].mean()
print(f"associational P(employed=1 | treated=1) = {assoc:.4f}")
print("This is *not* the same object as Q under confounding.")
"""
            ),
            md(
                """
## Manski bounds (order-only / maximal confounding)

The Manski (natural) bounds keep only a causal **order**
`treated → employed` and allow arbitrary unobserved confounding. They are the
honest baseline when you are unwilling to restrict how selection works.
"""
            ),
            code(
                """
a_manski = Assumptions.manski("treated", "employed")
r = bound(data.variables, data.P, Q, a_manski)
print(r.summary(Q))
"""
            ),
            md(
                """
### How to read a `BoundResult`

| Field | Meaning |
|---|---|
| `lower`, `upper` | Endpoints of the identified interval |
| `width` | `upper - lower` (0 means point identification under these assumptions) |
| `success` | Whether the optimization succeeded / stayed feasible |
| `method` | Solver path chosen by `method="auto"` |
| `tight` | `True` if the interval is sharp for the stated assumptions; `False` means a valid **outer** approximation |

Always check `tight` before treating the interval as sharp. Leave
`method="auto"` unless you have a reason to override it.
"""
            ),
            code(
                """
print("interval:", r.interval)
print("width:", round(r.width, 4))
print("success:", r.success, "| method:", r.method, "| tight:", r.tight)
print("JSON-friendly:", r.to_dict()["interval"])
"""
            ),
            md(
                """
## Compose stronger assumptions with `&`

Adding assumptions **intersects** constraint sets: the feasible set can only
shrink, so the interval can only get narrower (or stay the same).

Here **MTR** (monotone treatment response) says employment is weakly increasing
in treatment for every latent type — “the program does not hurt anyone’s
employment prospects.” **MTS** (monotone treatment selection) restricts how
selection correlates with potential outcomes.
"""
            ),
            code(
                """
base = Assumptions.order("treated", "employed")
for label, a in [
    ("order only (Manski)", base),
    ("+ MTR", base & Assumptions.mtr("employed", "treated")),
    ("+ MTS", base & Assumptions.mts("treated", "employed")),
    ("+ MTR + MTS", base & Assumptions.mtr("employed", "treated") & Assumptions.mts("treated", "employed")),
]:
    rr = bound(data.variables, data.P, Q, a)
    print(f"{label:22s}  [{rr.lower:.4f}, {rr.upper:.4f}]  width={rr.width:.4f}")

print("\\nAssumption list for MTR + MTS:")
for line in (base & Assumptions.mtr("employed", "treated") & Assumptions.mts("treated", "employed")).describe():
    print(" ", line)
"""
            ),
            md(
                """
Under MTR the interval typically shrinks a lot relative to Manski: ruling out
“treatment hurts employment” removes many adversarial response types.
"""
            ),
            md(
                """
## Same table, different estimand: the ATE

Stakeholders often want the **average treatment effect**
\\(P(Y_1=1) - P(Y_0=1)\\) rather than the level under universal treatment.
"""
            ),
            code(
                """
Q_ate = Query.ate(outcome="employed", treatment="treated")
print(Q_ate.describe())
r_ate = bound(
    data.variables, data.P, Q_ate,
    Assumptions.manski("treated", "employed") & Assumptions.mtr("employed", "treated"),
)
print(r_ate.summary(Q_ate))
"""
            ),
            md(
                """
## You can now…

- Load discrete observational data and inspect it before bounding.
- Build a `Query` for an interventional probability or an ATE.
- Compute Manski bounds and tighten them by composing atoms with `&`.
- Read `width`, `tight`, `method`, and `success` on a `BoundResult`.

**Next:** [02_assumption_composition.ipynb](02_assumption_composition.ipynb) —
structural faces vs sensitivity budgets, and when to use recipes.
"""
            ),
        ],
    )


def nb02() -> None:
    write_nb(
        "02_assumption_composition.ipynb",
        [
            md(
                """
# 02 — Assumption composition: scientific stances as constraints

Partial identification is only as credible as the assumptions you state.
**partialid** treats those assumptions as composable objects: each literature
idea is an atom (or a thin recipe), and `&` takes their intersection.

This notebook reuses the job-training extract from notebook 01 and compares
intervals under different **scientific stances** — without introducing graphs
or sensitivity *paths* yet.
"""
            ),
            md("## Reload the study (self-contained setup)"),
            code(
                PATH_HELPER
                + """

import pandas as pd
from partialid import Assumptions, Query, bound, from_dataframe

df = pd.read_csv(tutorial_data("job_training.csv"))
data = from_dataframe(df, columns=["treated", "employed"])
Q = Query.interventional(outcome="employed", do={"treated": 1}, value=1)
print(f"N={len(df)}  |  {Q.describe()}")
print(df.groupby("treated")["employed"].mean().rename("employment rate"))
"""
            ),
            md(
                """
## Two families of assumptions

1. **Structural / qualitative** — order, diagrams, exclusions, MTR/MTS,
   instruments, hard independence. These carve faces of the response-type set.
2. **Sensitivity / quantitative budgets** — Tan MSM \\(\\Lambda\\), Rosenbaum
   \\(\\Gamma\\), \\(c\\)-dependence, margin budgets, \\(f\\)-sensitivity. These
   dial *how much* confounding you allow.

`&` is a meet: adding assumptions can only shrink (or leave unchanged) the
identified interval.
"""
            ),
            md("## Structural faces side-by-side"),
            code(
                """
base = Assumptions.order("treated", "employed")
faces = {
    "Manski / order only": Assumptions.manski("treated", "employed"),
    "MTR (program never hurts)": base & Assumptions.mtr("employed", "treated"),
    "MTS (selection monotone)": base & Assumptions.mts("treated", "employed"),
    "MTR + MTS": base & Assumptions.mtr("employed", "treated") & Assumptions.mts("treated", "employed"),
}
print(f"{'stance':28s}  {'interval':18s}  width   tight")
for name, a in faces.items():
    r = bound(data.variables, data.P, Q, a)
    print(f"{name:28s}  [{r.lower:.4f}, {r.upper:.4f}]  {r.width:.4f}  {r.tight}")
"""
            ),
            md(
                """
In the job-training story:

- **Manski** refuses almost all structure beyond temporal order — wide bounds.
- **MTR** says the program is not harmful for any latent type.
- **MTS** restricts how latent employability drives selection into training.
- Combining both is a stronger joint claim; check that you believe *each* piece.
"""
            ),
            md(
                """
## Fixed sensitivity budgets

Quantitative budgets answer: “if confounding is limited in *this* way, how
tight is the interval?” Below we compare common budgets at **one** defended
value each, always on top of the Manski order.
"""
            ),
            code(
                """
budgets = {
    "Tan MSM Λ=2": base & Assumptions.tan_msm("treated", 2.0),
    "Rosenbaum Γ=4": base & Assumptions.rosenbaum_gamma("treated", Gamma=4.0),
    "c-dependence c=0.1": base & Assumptions.c_dependence("treated", 0.1),
    "margin κ=0.05": base & Assumptions.margin_budget(0.05),
}
print(f"{'budget':22s}  {'interval / status':32s}  method")
for name, a in budgets.items():
    r = bound(data.variables, data.P, Q, a)
    if r.success:
        status = f"[{r.lower:.4f}, {r.upper:.4f}] w={r.width:.4f}"
    else:
        status = f"infeasible ({r.message})"
    print(f"{name:22s}  {status:32s}  {r.method}")
"""
            ),
            md(
                """
Reading the knobs (intuition, not proofs):

- **Tan MSM \\(\\Lambda\\)** — odds of treatment may differ by at most \\(\\Lambda\\)
  across latent strata (\\(\\Lambda=1\\) is no unobserved confounding on this margin).
- **Rosenbaum \\(\\Gamma\\)** — related odds-ratio sensitivity (anchored form).
- **\\(c\\)-dependence** — absolute deviation of propensity from a baseline.
- **Margin \\(\\kappa\\)** — soft independence / free-mass deviation from a
  reference factorization.
"""
            ),
            md("## Recipes are thin wrappers"),
            code(
                """
recipe = Assumptions.sensitivity_msm("treated", "employed", 3.0)
atoms = Assumptions.manski("treated", "employed") & Assumptions.tan_msm("treated", 3.0)
r1 = bound(data.variables, data.P, Q, recipe)
r2 = bound(data.variables, data.P, Q, atoms)
print("recipe describe:")
for line in recipe.describe():
    print(" ", line)
print(f"widths equal? {round(r1.width, 6)} == {round(r2.width, 6)}")
"""
            ),
            md(
                """
Prefer **atoms + `&`** when a recipe’s defaults do not match your setting
(e.g. you already have MTR and only want to add MSM).
"""
            ),
            md(
                """
## Advanced atoms (short)

Two less common tools: a hard **support** cap on the number of positive
response types (MIP), and **\\(f\\)-sensitivity** (divergence ball; uses CVXPY).
Reach for these when a referee asks for sparsity or a specific divergence
sensitivity model — not as day-one defaults.
"""
            ),
            code(
                """
for name, a in [
    ("support ≤ 4 types", base & Assumptions.support_bound(4)),
    ("f-sensitivity TV ρ=0.05", base & Assumptions.f_sensitivity("treated", 0.05, f="tv")),
]:
    r = bound(data.variables, data.P, Q, a)
    status = f"[{r.lower:.4f}, {r.upper:.4f}] w={r.width:.4f}" if r.success else f"fail: {r.message}"
    print(f"{name:28s}  {status}  method={r.method}")
"""
            ),
            md(
                """
## When assumptions conflict with the data

If the observational law is incompatible with your assumption set, `bound`
returns `success=False`. Always read `message` — do not interpret an empty or
failed solve as “effect is zero.”
"""
            ),
            code(
                """
# Extreme support cap: often infeasible for rich binary tables
r_bad = bound(data.variables, data.P, Q, base & Assumptions.support_bound(1))
print("success:", r_bad.success)
print("message:", r_bad.message)
"""
            ),
            md(
                """
## Decision guide

1. Start with **Manski** to see the worst-case width.
2. Add **structural faces** you actually believe (MTR, MTS, exclusions, IV…).
3. Add a **sensitivity budget** you can defend to a skeptic.
4. Report the interval **and** `a.describe()` — the assumptions are part of the result.

**Next:** [03_sensitivity_paths.ipynb](03_sensitivity_paths.ipynb) — vary the
budget continuously with `bound_path` and find breakdown points.
"""
            ),
        ],
    )


def nb03() -> None:
    write_nb(
        "03_sensitivity_paths.ipynb",
        [
            md(
                """
# 03 — Sensitivity paths: when the budget is the object

A single bound at \\(\\Lambda=3\\) is useful. Stakeholders often ask a harder
question: **how does the answer change as we relax the confounding limit?**

This notebook stays with the job-training extract and treats the sensitivity
parameter as the analysis object: paths, breakdown points, and short grids.
"""
            ),
            md("## Setup"),
            code(
                PATH_HELPER
                + """

import pandas as pd
from partialid import Assumptions, Query, bound, bound_grid, bound_path, from_dataframe

df = pd.read_csv(tutorial_data("job_training.csv"))
data = from_dataframe(df, columns=["treated", "employed"])
Q = Query.interventional(outcome="employed", do={"treated": 1}, value=1)
base = Assumptions.manski("treated", "employed")
print(Q.describe())
"""
            ),
            md("## Fixed budget reminder"),
            code(
                """
r = bound(data.variables, data.P, Q, Assumptions.sensitivity_msm("treated", "employed", 3.0))
print(r.summary(Q))
print(
    "At Λ=3, universal-treatment employment lies in "
    f"[{r.lower:.3f}, {r.upper:.3f}] under Manski + Tan MSM."
)
"""
            ),
            md(
                """
## Path in \\(\\Lambda\\)

`bound_path` sweeps a budget parameter. For non-default treatment names, pass
`treatment=` explicitly (it defaults to `"X"`).
"""
            ),
            code(
                """
path = bound_path(
    data.variables, data.P, Q, base,
    param="Lambda",
    treatment="treated",
    theta_min=1.0,
    theta_max=5.0,
    n_grid=9,
)
print(path.summary(at=3.0))
print()
print(f"{'Λ':>6}  {'lower':>8}  {'upper':>8}  {'width':>8}")
for lam in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
    lo, hi, w = path.at(lam)
    print(f"{lam:6g}  {lo:8.4f}  {hi:8.4f}  {w:8.4f}")
"""
            ),
            md(
                """
At \\(\\Lambda=1\\) the MSM forces no unobserved confounding on this margin
(width \\(\\to 0\\) here). As \\(\\Lambda\\) grows, the interval opens toward the
Manski width — the sensitivity curve makes that tradeoff visible.
"""
            ),
            md(
                """
## Breakdown: when does a threshold enter the interval?

`path.breakdown(null)` returns the smallest budget at which `null` lies inside
the bound interval (or `None` if it never does in the searched range).

Here we ask when the interval for universal-treatment employment expands enough
to include 0.50 — a round policy threshold an analyst might care about.
"""
            ),
            code(
                """
assoc = float(df.loc[df["treated"] == 1, "employed"].mean())
print(f"associational employment among treated = {assoc:.4f}")

for null in (0.50, 0.55, assoc):
    b = path.breakdown(null)
    print(f"null={null:.4f} enters at Λ ≈ {b}")
"""
            ),
            md(
                """
## Stricter path under MTR

If you also believe monotone treatment response, the whole sensitivity curve
shifts. Compose MTR into the **base** assumptions passed to `bound_path`.
"""
            ),
            code(
                """
base_mtr = base & Assumptions.mtr("employed", "treated")
path_mtr = bound_path(
    data.variables, data.P, Q, base_mtr,
    param="Lambda",
    treatment="treated",
    theta_min=1.0,
    theta_max=5.0,
    n_grid=9,
)
print(f"{'Λ':>6}  {'Manski+MSM width':>16}  {'+MTR width':>12}")
for lam in (1.0, 2.0, 3.0, 5.0):
    _, _, w0 = path.at(lam)
    _, _, w1 = path_mtr.at(lam)
    print(f"{lam:6g}  {w0:16.4f}  {w1:12.4f}")
"""
            ),
            md(
                """
## Other budget parameters

The same machinery works for Rosenbaum \\(\\Gamma\\) paths and for discrete grids
in \\(c\\) or \\(\\kappa\\) via `bound_grid`.
"""
            ),
            code(
                """
path_g = bound_path(
    data.variables, data.P, Q, base,
    param="Gamma",
    treatment="treated",
    theta_min=1.0,
    theta_max=6.0,
    n_grid=7,
)
print("Gamma path:")
for g in (1.0, 2.0, 4.0, 6.0):
    lo, hi, w = path_g.at(g)
    print(f"  Γ={g:g}: [{lo:.4f}, {hi:.4f}]  width={w:.4f}")

print("\\nc-dependence grid:")
grid = bound_grid(
    data.variables, data.P, Q, base,
    param="c",
    values=[0.0, 0.05, 0.1, 0.2],
    treatment="treated",
)
for c, rr in grid:
    print(f"  c={c:g}: [{rr.lower:.4f}, {rr.upper:.4f}]  width={rr.width:.4f}")
"""
            ),
            md(
                """
## How to report a sensitivity analysis

Example sentence you can adapt:

> Under order-only Manski assumptions and Tan’s marginal sensitivity model,
> the employment rate under universal treatment lies in
> \\([\\ell(\\Lambda), u(\\Lambda)]\\). At \\(\\Lambda=3\\) the interval is
> \\([\\ldots]\\). The conclusion that the rate exceeds 0.50 is robust for
> \\(\\Lambda\\) up to approximately \\(\\Lambda^\\star\\) (breakdown). Adding MTR
> further narrows the path.

Always state the query, the assumption list (`a.describe()`), and whether
`tight` is true.

**Next:** [04_instruments_and_graphs.ipynb](04_instruments_and_graphs.ipynb) —
encouragement designs, IV recipes, and explicit diagrams.
"""
            ),
        ],
    )


if __name__ == "__main__":
    nb01()
    nb02()
    nb03()
    print("partial build ok (01-03); continue in part 2")
