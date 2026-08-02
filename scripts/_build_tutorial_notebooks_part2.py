#!/usr/bin/env python3
"""Build tutorial notebooks 04–07. Companion to _build_tutorial_notebooks.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "nb1", ROOT / "scripts" / "_build_tutorial_notebooks.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

md, code, write_nb, PATH_HELPER = mod.md, mod.code, mod.write_nb, mod.PATH_HELPER


def nb04() -> None:
    write_nb(
        "04_instruments_and_graphs.ipynb",
        [
            md(
                """
# 04 — Instruments and graphs: design knowledge as assumptions

Sometimes you know more than “treatment precedes outcome.” An **encouragement
design** randomizes an *offer* of a program; people choose whether to enroll;
you care about the effect of *enrollment* on employment.

This notebook loads a synthetic encouragement trial, encodes the design with
`Assumptions.iv` and with an explicit `CausalDiagram`, contrasts that with
point identification under an observed confounder, and briefly treats
measurement error on the outcome.
"""
            ),
            md(
                """
## The encouragement design

Columns:

- `offer` — randomized offer of training (instrument)
- `enrolled` — whether the person took up training (treatment)
- `employed` — employment at follow-up (outcome)

**Assumptions of the classic IV diagram:** the offer affects employment only
through enrollment (exclusion), and unobserved confounding may still link
enrollment and employment.
"""
            ),
            code(
                PATH_HELPER
                + """

import pandas as pd
from partialid import (
    Assumptions, CausalDiagram, Query, Variables, bound, from_dataframe,
    identify_query, is_identified,
)  # is_identified(g, treatments, outcomes); identify_query(V, P, Q, a)

df = pd.read_csv(tutorial_data("encouragement_trial.csv"))
print(f"N = {len(df)}")
print(df.head(8))
print()
print("offer rate:", round(df["offer"].mean(), 3))
print("enrollment rate:", round(df["enrolled"].mean(), 3))
print()
print("First stage — enrollment by offer:")
print(df.groupby("offer")["enrolled"].mean().rename("P(enrolled=1 | offer)"))
print()
print("Employment by offer and enrollment:")
print(df.groupby(["offer", "enrolled"])["employed"].mean().unstack("enrolled").round(3))
"""
            ),
            md("## Causal question: effect of enrollment, not offer"),
            code(
                """
data = from_dataframe(df, columns=["offer", "enrolled", "employed"])
Q = Query.interventional(outcome="employed", do={"enrolled": 1}, value=1)
print(Q.describe())
print(
    "associational among enrolled:",
    round(df.loc[df["enrolled"] == 1, "employed"].mean(), 4),
)
"""
            ),
            md(
                """
## IV recipe

`Assumptions.iv(instrument, treatment, outcome)` builds the classic diagram
without asking you to write edges by hand.
"""
            ),
            code(
                """
a_iv = Assumptions.iv("offer", "enrolled", "employed")
r_iv = bound(data.variables, data.P, Q, a_iv)
print(r_iv.summary(Q))
print("assumption list:")
for line in a_iv.describe():
    print(" ", line)
"""
            ),
            md(
                """
## Compliance monotonicity (no-defiers)

`monotone=True` adds **no-defiers**: nobody enrolls when not offered but refuses
when offered. That is a substantive claim. If it conflicts with the observed
law, the LP is **infeasible** (`success=False`) — that is informative, not a
crash.
"""
            ),
            code(
                """
a_mono = Assumptions.iv("offer", "enrolled", "employed", monotone=True)
r_mono = bound(data.variables, data.P, Q, a_mono)
print("success:", r_mono.success)
if r_mono.success:
    print(r_mono.summary(Q))
else:
    print("infeasible under no-defiers for this sample.")
    print("message:", r_mono.message)
"""
            ),
            md(
                """
## Same assumptions from an explicit diagram

Recipes are convenience. Equivalent structure via `CausalDiagram`:
"""
            ),
            code(
                """
g = CausalDiagram.from_edges(
    nodes=["offer", "enrolled", "employed"],
    directed=[("offer", "enrolled"), ("enrolled", "employed")],
    bidirected=[("enrolled", "employed")],
)
a_diag = Assumptions.from_diagram(g)
r_diag = bound(data.variables, data.P, Q, a_diag)
print(r_diag.summary(Q))
print(
    "recipe width matches diagram?",
    abs(r_iv.width - r_diag.width) < 1e-8,
)
"""
            ),
            md(
                """
## Contrast: observed confounder → point identification

If the confounder is **observed**, classical identification can collapse the
interval to a point. Here a small Markovian back-door example (synthetic joint)
shows `method="id"` and width 0.

`is_identified` / `identify_query` are **experimental** APIs — useful for
checks, but prefer `bound` + reading `method` / `tight` in applications.
"""
            ),
            code(
                """
import numpy as np

V_bd = Variables.from_binary("risk", "enrolled", "employed")
g_bd = CausalDiagram.from_edges(
    ["risk", "enrolled", "employed"],
    directed=[("risk", "enrolled"), ("risk", "employed"), ("enrolled", "employed")],
    bidirected=[],
)
rng = np.random.default_rng(0)
keys = V_bd.cells()
w = np.exp(rng.normal(size=len(keys)))
w = w / w.sum()
P_bd = {k: float(v) for k, v in zip(keys, w)}
Q_bd = Query.interventional(outcome="employed", do={"enrolled": 1}, value=1)
a_bd = Assumptions.from_diagram(g_bd)

print(
    "is_identified (experimental):",
    is_identified(g_bd, ["enrolled"], ["employed"]),
)
id_res = identify_query(V_bd, P_bd, Q_bd, a_bd)
print("identify_query:", id_res.identified, "value=", id_res.value)

r_bd = bound(V_bd, P_bd, Q_bd, a_bd)
print(r_bd.summary(Q_bd))
"""
            ),
            md(
                """
## Ignoring the instrument widens the bounds

If we throw away `offer` and treat `(enrolled, employed)` as a plain
observational pair under Manski, we lose design information.
"""
            ),
            code(
                """
df_xy = df[["enrolled", "employed"]]
data_xy = from_dataframe(df_xy, columns=["enrolled", "employed"])
r_manski = bound(
    data_xy.variables, data_xy.P, Q,
    Assumptions.manski("enrolled", "employed"),
)
print(f"IV width:     {r_iv.width:.4f}")
print(f"Manski width: {r_manski.width:.4f}")
print("Design knowledge (the instrument) shrinks the identified set.")
"""
            ),
            md(
                """
## IV + sensitivity budget

You can compose diagram structure with a quantitative confounding limit on
enrollment. A **tight** MSM budget can be incompatible with the observed IV
law (the LP becomes infeasible) — the same `success=False` pattern as
no-defiers. Here Λ=3 is feasible for this sample and shrinks the IV interval;
try Λ=2 yourself to see a refutation.
"""
            ),
            code(
                """
for lam in (2.0, 3.0):
    a_mix = a_iv & Assumptions.tan_msm("enrolled", lam)
    r_mix = bound(data.variables, data.P, Q, a_mix)
    if r_mix.success:
        print(f"Λ={lam:g}: [{r_mix.lower:.4f}, {r_mix.upper:.4f}]  width={r_mix.width:.4f}")
        print(f"  IV only width {r_iv.width:.4f} → IV+MSM width {r_mix.width:.4f}")
    else:
        print(f"Λ={lam:g}: infeasible (MSM too tight for this IV law)")
        print(f"  message: {r_mix.message}")
"""
            ),
            md(
                """
## Measurement error (short case)

Sometimes the **true** outcome is latent and you observe a misclassified proxy.
Declare latent endogenous variables with `observed=...`: they still have
response types and may appear in queries, but they do **not** index `P`.
"""
            ),
            code(
                """
# Latent true employment; observe employed_obs. enrolled → employed → employed_obs,
# with confounding between true and measured employment.
V_me = Variables.from_binary(
    "enrolled", "employed", "employed_obs",
    observed=["enrolled", "employed_obs"],
)
g_me = CausalDiagram.from_edges(
    ["enrolled", "employed", "employed_obs"],
    directed=[("enrolled", "employed"), ("employed", "employed_obs")],
    bidirected=[("employed", "employed_obs")],
)
# Observational law on (enrolled, employed_obs) only — reuse Study A margins as a stand-in
P_me = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q_me = Query.interventional(outcome="employed", do={"enrolled": 1}, value=1)
a_me = Assumptions.from_diagram(g_me) & Assumptions.mtr("employed_obs", "employed")

r_me = bound(V_me, P_me, Q_me, a_me)
print(r_me.summary(Q_me))
print(f"tight={r_me.tight}  method={r_me.method}")

r_me0 = bound(V_me, P_me, Q_me, Assumptions.from_diagram(g_me))
print(f"diagram only width={r_me0.width:.4f}; + MTR on measurement channel width={r_me.width:.4f}")
"""
            ),
            md(
                """
## Always read `tight`

`tight=True` means the interval is sharp for the stated assumptions.
`tight=False` means a valid **outer** approximation (common for some
multi-component graphs). Warnings on the result explain when that happens.
Deeper routing (`classify_program`, `prefer_poly`) is covered in notebook 07.
"""
            ),
            code(
                """
print("IV tight:", r_iv.tight, "warnings:", r_iv.warnings)
print("back-door tight:", r_bd.tight, "method:", r_bd.method)
"""
            ),
            md(
                """
## You can now…

- Encode an encouragement / IV design with `Assumptions.iv` or `CausalDiagram`.
- Handle infeasible monotonicity cleanly via `success` / `message`.
- See when an observed confounder point-identifies vs when IV only partially identifies.
- Compose IV structure with sensitivity budgets.
- Declare latent outcomes for simple measurement-error graphs.

**Next:** [05_queries_and_results.ipynb](05_queries_and_results.ipynb) — more
estimands, data ingress, and reporting. Optional deep dive:
[07_advanced_identification.ipynb](07_advanced_identification.ipynb).
"""
            ),
        ],
    )


def nb05() -> None:
    write_nb(
        "05_queries_and_results.ipynb",
        [
            md(
                """
# 05 — Queries and results: estimands, data ingress, reporting

The same observational table supports many scientific questions. This notebook
returns to the job-training extract and shows how to:

- ask interventional, ATE, and necessity/sufficiency queries;
- build a custom linear combination (experimental AST);
- ingest data from counts, records, or a DataFrame;
- use experimental closed-form, audit, and CI helpers;
- report a bound responsibly.
"""
            ),
            md("## Setup"),
            code(
                PATH_HELPER
                + """

import pandas as pd
from partialid import (
    Assumptions, Query, audit_bound, bound, bound_ci, from_counts,
    from_dataframe, from_records, try_closed_form,
)
from partialid.query_ast import Atom, LinComb, Lit, PO

df = pd.read_csv(tutorial_data("job_training.csv"))
data = from_dataframe(df, columns=["treated", "employed"])
a = Assumptions.order("treated", "employed") & Assumptions.mtr("employed", "treated")
print(f"N={len(df)}")
"""
            ),
            md(
                """
## Built-in query faces

| Query | Plain-language ask |
|---|---|
| interventional | Employment rate if everyone were treated |
| ATE | Average gain from treating vs not |
| PNS | Probability treatment is necessary *and* sufficient for the outcome |
| PN / PS | Probabilities of necessity / sufficiency |
"""
            ),
            code(
                """
queries = {
    "interventional": Query.interventional(outcome="employed", do={"treated": 1}, value=1),
    "ATE": Query.ate(outcome="employed", treatment="treated"),
    "PNS": Query.pns(treatment="treated", outcome="employed"),
    "PN": Query.pn(treatment="treated", outcome="employed"),
    "PS": Query.ps(treatment="treated", outcome="employed"),
}
for name, Q in queries.items():
    r = bound(data.variables, data.P, Q, a)
    print(f"{name:14s}  {Q.describe()}")
    print(f"{'':14s}  [{r.lower:.4f}, {r.upper:.4f}]  w={r.width:.4f}  method={r.method}\\n")
"""
            ),
            md(
                """
## Custom linear combination (experimental)

The query AST (`PO`, `Lit`, `Atom`, `LinComb`) is **experimental**. Use named
factories when they suffice.
"""
            ),
            code(
                """
# 2 P(employed_1=1) - P(employed_0=1)
expr = LinComb((
    (2.0, Atom(Lit(PO("employed", {"treated": 1}), 1))),
    (-1.0, Atom(Lit(PO("employed", {"treated": 0}), 1))),
))
Q_custom = Query.from_expr(expr, outcome="employed")
r_custom = bound(data.variables, data.P, Q_custom, a)
print(Q_custom.describe())
print(r_custom.summary(Q_custom))
"""
            ),
            md(
                """
## Conditional interventional

`condition=` restricts attention to a pretreatment slice of \(P\) before
bounding (see package docs for details).
"""
            ),
            code(
                """
# Illustrative: condition on treated==1 is *not* a pretreatment covariate —
# use a real covariate when you have one (notebook 06). Here we show the API
# shape on a trivial self-contained slice by rebuilding a tiny table with a
# binary covariate.
df_c = df.copy()
df_c["urban"] = (df_c.index % 2).astype(int)  # fake covariate for API demo only
data_c = from_dataframe(df_c, columns=["urban", "treated", "employed"])
Q_cond = Query.interventional(
    outcome="employed", do={"treated": 1}, value=1, condition={"urban": 1},
)
r_cond = bound(
    data_c.variables, data_c.P, Q_cond,
    Assumptions.order("urban", "treated", "employed"),
)
print(Q_cond.describe())
print(r_cond.summary(Q_cond))
"""
            ),
            md("## Data ingress cookbook"),
            code(
                """
# 1) Aggregated counts
counts = {(0, 0): 30, (0, 1): 20, (1, 0): 15, (1, 1): 35}
d1 = from_counts(("treated", "employed"), counts)
# 2) Individual records
recs = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 1)]
d2 = from_records(("treated", "employed"), recs)
# 3) DataFrame / CSV (already used above)
d3 = from_dataframe(df, columns=["treated", "employed"])

Q = Query.interventional(outcome="employed", do={"treated": 1}, value=1)
for label, d in [("counts", d1), ("records", d2), ("dataframe", d3)]:
    rr = bound(d.variables, d.P, Q, Assumptions.manski("treated", "employed"))
    print(f"{label:10s}  width={rr.width:.4f}  n_obs={d.n_obs}")
"""
            ),
            md(
                """
## Closed form (experimental)

Some Manski / MTR settings admit analytic Moves. `try_closed_form` returns
them when available.
"""
            ),
            code(
                """
Q = Query.interventional(outcome="employed", do={"treated": 1}, value=1)
cf = try_closed_form(data.variables, data.P, Q, Assumptions.manski("treated", "employed"))
print(cf)
"""
            ),
            md(
                """
## Audit and confidence intervals (experimental)

`audit_bound` sanity-checks a result. `bound_ci` builds an Imbens–Manski-style
CI for the identified set (bootstrap; keep `n_boot` modest in notebooks).
"""
            ),
            code(
                """
Q = Query.interventional(outcome="employed", do={"treated": 1}, value=1)
r = bound(data.variables, data.P, Q, Assumptions.sensitivity_msm("treated", "employed", 3.0))
report = audit_bound(r, query=Q, assumptions=Assumptions.sensitivity_msm("treated", "employed", 3.0))
print(report)

ci = bound_ci(
    data.variables, data.P, Q,
    Assumptions.manski("treated", "employed"),
    n=data.n_obs,
    n_boot=40,
    seed=0,
)
print(ci)
"""
            ),
            md(
                """
## Reporting checklist

When you publish or hand off a bound, include:

1. **Sample** — N and how `P` was formed (counts / survey weights / …).
2. **Query** — `Q.describe()` in words.
3. **Assumptions** — full `a.describe()` list (not only a recipe name).
4. **Interval** — \\([\\ell, u]\\), width, and `tight`.
5. **Sensitivity** — a path or breakdown if a budget is material (notebook 03).
6. **Failures** — if `success=False`, report infeasibility rather than inventing a point.

## Core path complete

Optional next steps:

- [06_panel_and_scale.ipynb](06_panel_and_scale.ipynb) — realistic 8-variable panel and scale.
- [07_advanced_identification.ipynb](07_advanced_identification.ipynb) — routing, degree, `prefer_poly`.
"""
            ),
        ],
    )


def nb06() -> None:
    write_nb(
        "06_panel_and_scale.ipynb",
        [
            md(
                """
# 06 — Wider data and scale: multi-variable panels

Real extracts rarely have only two columns. This notebook loads an **8-variable**
synthetic job-training panel, explores it like microdata, and shows that
order-only interventional queries stay tractable because partialid **prunes
inert covariates** from the program. We then look at settings where scale
*does* bite (higher treatment cardinality, multiple causes).

Honesty check: under order-only Manski, the solver is **not** optimizing a full
8-way response-type LP — structure reduces the active program to
`{treated, employed}`. That reduction *is* the product capability you want when
files get wide.
"""
            ),
            md("## A wider extract"),
            code(
                PATH_HELPER
                + """

import time
import pandas as pd
from partialid import (
    Assumptions, Query, Variables, bound, bound_path, from_counts, from_dataframe,
)

cols = ["site", "age_bin", "educ", "prior_emp", "urban", "spouse", "treated", "employed"]
df = pd.read_csv(tutorial_data("job_training_panel.csv"))
print(f"N = {len(df)}  columns = {list(df.columns)}")
print(df.head(6))
print()
print("missing values:", int(df.isna().sum().sum()))
print(df[cols].mean().round(3).rename("mean"))
"""
            ),
            md("## Look at employment patterns"),
            code(
                """
print("Employment by treatment:")
print(df.groupby("treated")["employed"].mean())
print()
print("Employment by site × treatment:")
print(df.groupby(["site", "treated"])["employed"].mean().unstack("treated").round(3))
print()
print("Employment by educ:")
print(df.groupby("educ")["employed"].mean().round(3))
"""
            ),
            md(
                """
## Question on the full 8-column table

Same scientific question as notebook 01, but `from_dataframe` keeps every
column. Causal order: pretreatment covariates, then treatment, then outcome.
"""
            ),
            code(
                """
data = from_dataframe(df, columns=cols)
Q = Query.interventional(outcome="employed", do={"treated": 1}, value=1)
# Pretreatment → treatment → outcome (explicit order; clearer than manski's treatment-first covariates)
a = Assumptions.order(*cols)
print(Q.describe())
print("variables:", data.variables.names)
"""
            ),
            md(
                """
## Manski on the wide table — watch the prune notes
"""
            ),
            code(
                """
t0 = time.perf_counter()
r = bound(data.variables, data.P, Q, a)
t1 = time.perf_counter()
print(r.summary(Q))
print(f"elapsed: {t1 - t0:.3f}s")
print("notes (structure / prune):")
for n in r.notes:
    if "inert" in n or "prune" in n or "program:" in n or "c-degree" in n:
        print(" ", n)
"""
            ),
            md(
                """
The note `inert prune dropped [...]` means covariates were removed from the
*program* as irrelevant for this order-only interventional query. The dataset
can stay wide; the LP stays small.
"""
            ),
            md("## Equivalence on the collapsed margin"),
            code(
                """
ct = df.groupby(["treated", "employed"]).size()
counts = {k: int(v) for k, v in ct.items()}
data_xy = from_counts(("treated", "employed"), counts)
r_xy = bound(
    data_xy.variables, data_xy.P, Q,
    Assumptions.manski("treated", "employed"),
)
print(f"wide table: [{r.lower:.6f}, {r.upper:.6f}]")
print(f"2×2 margin: [{r_xy.lower:.6f}, {r_xy.upper:.6f}]")
print("match?", abs(r.lower - r_xy.lower) < 1e-8 and abs(r.upper - r_xy.upper) < 1e-8)
"""
            ),
            md("## Sensitivity path on the wide table"),
            code(
                """
path = bound_path(
    data.variables, data.P, Q, a,
    param="Lambda",
    treatment="treated",
    theta_min=1.0,
    theta_max=4.0,
    n_grid=7,
)
print(path.summary(at=2.0))
for lam in (1.0, 2.0, 3.0, 4.0):
    lo, hi, w = path.at(lam)
    print(f"Λ={lam:g}: [{lo:.4f}, {hi:.4f}]  width={w:.4f}")
"""
            ),
            md(
                """
## Stratum-specific analysis

A large file is not one monolithic question. Slice the cohort when the science
is conditional (e.g. one study site).
"""
            ),
            code(
                """
df_site = df[df["site"] == 1]
data_site = from_dataframe(df_site, columns=cols)
r_site = bound(data_site.variables, data_site.P, Q, a)
print(f"site==1  N={len(df_site)}")
print(r_site.summary(Q))

# Or use Query.condition on a pretreatment covariate
Q_educ = Query.interventional(
    outcome="employed", do={"treated": 1}, value=1, condition={"educ": 1},
)
r_educ = bound(data.variables, data.P, Q_educ, a)
print(Q_educ.describe())
print(r_educ.summary(Q_educ))
"""
            ),
            md(
                """
## When scale bites: ternary dose

Raising treatment cardinality grows the response-type space. Here a small
ternary exposure table (shown explicitly) still runs quickly, but timings and
widths illustrate the cost of richer treatments.
"""
            ),
            code(
                """
# Synthetic ternary dose × binary outcome (visible table, not a black-box helper)
import numpy as np
dose_cols = ["dose", "employed"]
# probabilities over dose∈{0,1,2}, employed∈{0,1}
P_dose = {
    (0, 0): 0.18, (0, 1): 0.12,
    (1, 0): 0.10, (1, 1): 0.20,
    (2, 0): 0.15, (2, 1): 0.25,
}
rows = []
rng = np.random.default_rng(3)
keys = list(P_dose)
p = np.array([P_dose[k] for k in keys]); p = p / p.sum()
idx = rng.choice(len(keys), size=2000, p=p)
df_dose = pd.DataFrame([keys[i] for i in idx], columns=dose_cols)
print(pd.crosstab(df_dose["dose"], df_dose["employed"], normalize="index").round(3))

data_dose = from_dataframe(df_dose, columns=dose_cols)
Q_dose = Query.interventional(outcome="employed", do={"dose": 1}, value=1)
t0 = time.perf_counter()
r_dose = bound(
    data_dose.variables, data_dose.P, Q_dose,
    Assumptions.order("dose", "employed") & Assumptions.tan_msm("dose", 2.0),
)
print(r_dose.summary(Q_dose), f"  elapsed={time.perf_counter()-t0:.3f}s")
"""
            ),
            md(
                """
## When scale bites: multi-cause

Two treatments and one outcome under order-only: intervening on one cause
keeps a larger active set than a simple binary XY problem.
"""
            ),
            code(
                """
V_mc = Variables.from_binary("T0", "T1", "employed")
# Small joint with full support
P_mc = {
    (0, 0, 0): 0.10, (0, 0, 1): 0.15,
    (0, 1, 0): 0.08, (0, 1, 1): 0.12,
    (1, 0, 0): 0.09, (1, 0, 1): 0.14,
    (1, 1, 0): 0.11, (1, 1, 1): 0.21,
}
s = sum(P_mc.values())
P_mc = {k: v / s for k, v in P_mc.items()}
Q_mc = Query.interventional(outcome="employed", do={"T0": 1}, value=1)
t0 = time.perf_counter()
r_mc = bound(V_mc, P_mc, Q_mc, Assumptions.order("T0", "T1", "employed"))
print(r_mc.summary(Q_mc), f"  elapsed={time.perf_counter()-t0:.3f}s")
print("notes:")
for n in r_mc.notes:
    if "inert" in n or "program:" in n or "c-degree" in n:
        print(" ", n)
"""
            ),
            md(
                """
## Practical guidance

1. **Wide binary panels** with order-only interventional queries are fine —
   inert pruning keeps the LP on the active margin.
2. **Higher cardinality** and **multiple active causes** grow cost; start binary
   when you can.
3. Prefer **design structure** (IV, exclusions, observed confounders) over
   brute-force joints.
4. Always check **`tight`**.

**Next (optional):** [07_advanced_identification.ipynb](07_advanced_identification.ipynb)
for program degree, outer approximations, and `prefer_poly`.
"""
            ),
        ],
    )


def nb07() -> None:
    write_nb(
        "07_advanced_identification.ipynb",
        [
            md(
                """
# 07 — Advanced identification and solver routing (optional)

**Audience:** users debugging hard graph problems or reading `tight=False`
warnings. Several APIs here are **experimental** (`classify_program`,
`is_identified`, `identify_query`) and may evolve on the 0.x train.

Prefer `method="auto"` in applications. Use this notebook to understand *why*
a result looks the way it does.
"""
            ),
            code(
                """
import numpy as np
from partialid import (
    Assumptions, CausalDiagram, Query, Variables, bound, classify_program,
    identify_query, is_identified,
)

def softmax_P(V, seed=0):
    rng = np.random.default_rng(seed)
    keys = V.cells()
    logits = rng.normal(size=len(keys))
    w = np.exp(logits - logits.max())
    w = w / w.sum()
    return {k: float(v) for k, v in zip(keys, w)}
"""
            ),
            md(
                """
## Point ID vs partial ID

Back-door (observed confounder) and front-door can be point-identified.
Order-only Manski on the same variables is not.
"""
            ),
            code(
                """
V = Variables.from_binary("Z", "X", "Y")
g_bd = CausalDiagram.from_edges(
    ["Z", "X", "Y"],
    directed=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
    bidirected=[],
)
P = softmax_P(V, seed=0)
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
a_bd = Assumptions.from_diagram(g_bd)
print("back-door identified?", is_identified(g_bd, ["X"], ["Y"]))
print(identify_query(V, P, Q, a_bd))
r_bd = bound(V, P, Q, a_bd)
print(r_bd.summary(Q))

r_order = bound(V, P, Q, Assumptions.order("Z", "X", "Y"))
print(f"order-only width {r_order.width:.4f} >> ID width {r_bd.width:.4f}")
"""
            ),
            code(
                """
V_fd = Variables.from_binary("X", "M", "Y")
g_fd = CausalDiagram.from_edges(
    ["X", "M", "Y"],
    directed=[("X", "M"), ("M", "Y")],
    bidirected=[("X", "Y")],
)
P_fd = softmax_P(V_fd, seed=1)
Q_fd = Query.interventional(outcome="Y", do={"X": 1}, value=1)
r_fd = bound(V_fd, P_fd, Q_fd, Assumptions.from_diagram(g_fd))
print(r_fd.summary(Q_fd))
print("classify:", classify_program(g_fd, query=Q_fd, variables=V_fd))
"""
            ),
            md(
                """
## `classify_program` (experimental)

Returns a `ProgramDegree` with `kind` ∈ {point, linear, bilinear/…, polynomial}
and a numeric `degree`. This is the signal `bound(..., method="auto")` uses for
routing.
"""
            ),
            code(
                """
for label, g in [
    ("back-door", g_bd),
    ("front-door", g_fd),
    ("bow (X↔Y, X→Y)", CausalDiagram.from_edges(
        ["X", "Y"], directed=[("X", "Y")], bidirected=[("X", "Y")],
    )),
]:
    Vloc = Variables.from_binary(*g.nodes)
    Qi = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    prog = classify_program(g, query=Qi, variables=Vloc)
    print(f"{label:20s}  kind={prog.kind.value:18s}  degree={prog.degree}")
"""
            ),
            md(
                """
## Outer default for degree ≥ 3

For polynomial programs, `bound` defaults to a merged-latent **outer** LP
(`tight=False`). Opt into nested multilinear search with `prefer_poly=True`
(more expensive; still check `tight`).
"""
            ),
            code(
                """
V_t = Variables.from_binary("X", "Y", "Z")
g_t = CausalDiagram.from_edges(
    ["X", "Y", "Z"],
    directed=[("X", "Y"), ("Y", "Z"), ("X", "Z")],
    bidirected=[("X", "Y"), ("Y", "Z"), ("X", "Z")],
)
P_t = {k: 1.0 / 8 for k in V_t.cells()}
Q_t = Query.interventional(outcome="Y", do={"X": 1}, value=1)
prog_t = classify_program(g_t, query=Q_t, variables=V_t)
print("classify:", prog_t.kind, "degree=", prog_t.degree)

r_outer = bound(V_t, P_t, Q_t, Assumptions.from_diagram(g_t))
print("default:", r_outer.summary(Q_t))
print("tight=", r_outer.tight, "warnings=", r_outer.warnings)

r_poly = bound(
    V_t, P_t, Q_t, Assumptions.from_diagram(g_t), prefer_poly=True,
)
print("prefer_poly:", r_poly.summary(Q_t))
print("tight=", r_poly.tight, "method=", r_poly.method)
"""
            ),
            md(
                """
## Exclusions and latent hyperedges

Missing directed edges are first-class via `Assumptions.exclusion`. Latent
confounding is encoded as bidirected / hyperedges on `CausalDiagram`.
"""
            ),
            code(
                """
# Order X→M→Y with confounding X↔Y, plus an explicit exclusion already in the diagram.
# Compare to adding an exclusion atom on a denser baseline graph.
V = Variables.from_binary("X", "M", "Y")
P = softmax_P(V, seed=2)
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

g_dense = CausalDiagram.from_edges(
    ["X", "M", "Y"],
    directed=[("X", "M"), ("M", "Y"), ("X", "Y")],
    bidirected=[("X", "Y")],
)
# Drop the direct X→Y edge via exclusion
a_ex = Assumptions.from_diagram(g_dense) & Assumptions.exclusion("Y", "X")
r_ex = bound(V, P, Q, a_ex)
print(r_ex.summary(Q))
print("tight=", r_ex.tight)
"""
            ),
            md(
                """
## Trust contract

1. Prefer **`method="auto"`**; treat explicit solver names as escape hatches.
2. Always read **`tight`** and **`warnings`**.
3. Deg ≥ 3 outer approximations are valid but conservative unless you opt into
   `prefer_poly=True`.
4. Experimental classifiers and ID helpers help explain routing — they are not
   required for ordinary `bound` workflows.

You have finished the optional advanced track. Return to the core curriculum
([01](01_first_bounds.ipynb)–[05](05_queries_and_results.ipynb)) for applied
analyses, or continue with the 8-variable panel in
[06_panel_and_scale.ipynb](06_panel_and_scale.ipynb).
"""
            ),
        ],
    )


if __name__ == "__main__":
    nb04()
    nb05()
    nb06()
    nb07()
    print("built 04-07")
