"""Regenerate every number quoted in Section 5 of latex/main.tex.

Run from the repository root:  PYTHONPATH=src python scripts/paper_numbers.py
Timings are the median of five calls after one warm-up call.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pandas as pd

from partialid import (
    Assumptions,
    Query,
    Variables,
    bound,
    bound_ci,
    bound_path,
    classify_program,
    from_dataframe,
)
from partialid.problems import binary_iv, ternary_exposure, two_instruments

ROOT = Path(__file__).resolve().parents[1]


def timed(*args, **kw):
    r = bound(*args, **kw)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        bound(*args, **kw)
        ts.append(time.perf_counter() - t0)
    return r, 1e3 * statistics.median(ts)


def show(label, r, ms=None):
    if not r.success:
        print(f"  {label:34s} infeasible            method={r.method}")
        return
    t = f"{ms:8.2f} ms" if ms is not None else ""
    print(f"  {label:34s} [{r.lower:.3f}, {r.upper:.3f}] w={r.width:.3f} "
          f"tight={r.tight!s:5s} method={r.method:28s} {t}")


# ---------------------------------------------------------------------------
print("5.1 Job-training table, assumption catalogue")
V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Qy = Query.interventional(outcome="Y", do={"X": 1}, value=1)
Qate = Query.ate(outcome="Y", treatment="X", y=1, t1=1, t0=0)
base = Assumptions.order("X", "Y")
mtr = Assumptions.mtr("Y", "X")
mts = Assumptions.mts("X", "Y")
catalogue = {
    "order": base,
    "MTR": base & mtr,
    "MTS": base & mts,
    "MTR & MTS": base & mtr & mts,
    "Tan L=2": base & Assumptions.tan_msm("X", 2.0),
    "Tan L=3": base & Assumptions.tan_msm("X", 3.0),
    "MTR & Tan L=2": base & mtr & Assumptions.tan_msm("X", 2.0),
    "Rosenbaum G=9": base & Assumptions.rosenbaum_gamma("X", 9.0),
    "c-dep c=0.1": base & Assumptions.c_dependence("X", 0.1),
    "margin k=0.05": base & Assumptions.margin_budget(0.05),
    "f-sens KL rho=0.1": base & Assumptions.f_sensitivity("X", 0.1, f="kl"),
    "f-sens TV rho=0.1": base & Assumptions.f_sensitivity("X", 0.1, f="tv"),
    "support k=4": base & Assumptions.support_bound(4),
    "support k=3": base & Assumptions.support_bound(3),
}
for name, a in catalogue.items():
    r, ms = timed(V, P, Qy, a)
    show(f"P(Y1=1) {name}", r, ms)
    try:
        show(f"ATE     {name}", bound(V, P, Qate, a))
    except (ValueError, NotImplementedError) as e:
        print(f"  ATE     {name:26s} not supported: {e}")

print(" other queries")
for qn, Q in {
    "PNS": Query.pns(treatment="X", outcome="Y"),
    "PN": Query.pn(treatment="X", outcome="Y"),
    "PS": Query.ps(treatment="X", outcome="Y"),
}.items():
    for an, a in {"order": base, "MTR": base & mtr}.items():
        show(f"{qn} {an}", bound(V, P, Q, a))

print(" Tan path")
path = bound_path(V, P, Qy, base, param="Lambda", theta_min=1, theta_max=7, n_grid=7)
for lam in range(1, 8):
    lo, hi, w = path.at(float(lam))
    print(f"  Lambda={lam}: [{lo:.3f}, {hi:.3f}] w={w:.3f}")

# ---------------------------------------------------------------------------
print("5.2 Designs")
p = binary_iv(seed=2)
iv = Assumptions.iv("Z", "X", "Y")
r, ms = timed(p.variables, p.data, p.query, iv)
show("IV", r, ms)
d = classify_program(iv.diagram(), query=p.query)
print("   classify:", d.kind.value, d.degree, d.wl_wr_cut)
show("IV order-only", bound(p.variables, p.data, p.query, Assumptions.order("Z", "X", "Y")))
show("IV monotone", bound(p.variables, p.data, p.query, Assumptions.iv("Z", "X", "Y", monotone=True)))
for lam in (1.0, 1.5, 2.0, 3.0, 10.0):
    r, ms = timed(p.variables, p.data, p.query, iv & Assumptions.tan_msm("X", lam))
    show(f"IV & Tan L={lam}", r, ms)
ivzx = Assumptions.diagram(["Z", "X", "Y"], [("Z", "X"), ("X", "Y")], [("Z", "X"), ("X", "Y")])
r, ms = timed(p.variables, p.data, p.query, ivzx)
show("IV with Z<->X", r, ms)
print("   classify:", classify_program(ivzx.diagram(), query=p.query).kind.value)

p2 = two_instruments()
a2 = Assumptions.diagram(["Z1", "Z2", "X", "Y"], [("Z1", "X"), ("Z2", "X"), ("X", "Y")], [("X", "Y")])
r, ms = timed(p2.variables, p2.data, Qy, a2)
show("two instruments", r, ms)

Vfd = Variables.from_binary("X", "Z", "Y")
px1, pz1, py1 = 0.5, {0: 0.4, 1: 0.7}, {0: 0.3, 1: 0.8}
Pfd = {}
for x in (0, 1):
    for z in (0, 1):
        for y in (0, 1):
            Pfd[x, z, y] = ((px1 if x else 1 - px1) * (pz1[x] if z else 1 - pz1[x])
                            * (py1[z] if y else 1 - py1[z]))
fd = Assumptions.diagram(["X", "Z", "Y"], [("X", "Z"), ("Z", "Y")], [("X", "Y")])
r, ms = timed(Vfd, Pfd, Qy, fd)
show("front door", r, ms)
show("front door table, order-only", bound(Vfd, Pfd, Qy, Assumptions.order("X", "Z", "Y")))

Vme = Variables.from_binary("X", "Y", "Y2", observed=["X", "Y2"])
me = Assumptions.diagram(["X", "Y", "Y2"], [("X", "Y"), ("Y", "Y2")], [("Y", "Y2")]) & Assumptions.mtr("Y2", "Y")
r, ms = timed(Vme, P, Qate, me)
show("measurement error ATE", r, ms)

pt = ternary_exposure()
r, ms = timed(pt.variables, pt.data, Qy, base)
show("ternary exposure", r, ms)
print(f"   1 - P(X=1) = {pt.expected_width:.3f}")

cols = ["site", "age_bin", "educ", "prior_emp", "urban", "spouse", "treated", "employed"]
df = pd.read_csv(ROOT / "notebooks" / "data" / "job_training_panel.csv")
Qp = Query.interventional(outcome="employed", do={"treated": 1}, value=1)
data = from_dataframe(df, columns=cols)
r, ms = timed(data.variables, data.P, Qp, Assumptions.order(*cols))
show(f"panel pooled (n={len(df)})", r, ms)
s0 = df[df["site"] == 0]
ds = from_dataframe(s0, columns=cols)
show(f"panel site 0 (n={len(s0)})", bound(ds.variables, ds.P, Qp, Assumptions.order(*cols)))
show("panel site 0 & MTR", bound(ds.variables, ds.P, Qp,
                                 Assumptions.order(*cols) & Assumptions.mtr("employed", "treated")))
Qu = Query.interventional(outcome="employed", do={"treated": 1}, value=1, condition={"urban": 1})
show("panel | urban=1", bound(data.variables, data.P, Qu, Assumptions.order(*cols)))
ci = bound_ci(ds.variables, ds.P, Qp, Assumptions.order(*cols) & Assumptions.mtr("employed", "treated"),
              n=len(s0), n_boot=200, seed=0)
print(f"  panel site 0 & MTR, 95% CI ({ci.method}): [{ci.lower:.3f}, {ci.upper:.3f}]")

# ---------------------------------------------------------------------------
print("5.3 Mediated table (eq:re3)")
V3 = Variables.from_binary("X", "W", "Y")
Pxw = {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.15, (1, 1): 0.35}
pyw = {0: 0.4, 1: 0.6}
P3 = {(x, w, y): Pxw[x, w] * (pyw[w] if y else 1 - pyw[w])
      for x in (0, 1) for w in (0, 1) for y in (0, 1)}
Qpns = Query.pns(treatment="X", outcome="Y")
graphs = {
    "mediated X<->W": [("X", "W")],
    "pair chain": [("X", "W"), ("W", "Y")],
    "one oval": [("X", "W", "Y")],
    "triangle": [("X", "W"), ("W", "Y"), ("X", "Y")],
}
for gname, bi in graphs.items():
    a = Assumptions.diagram(["X", "W", "Y"], [("X", "W"), ("W", "Y")], bi)
    for qn, Q in {"P(Y1=1)": Qy, "PNS": Qpns}.items():
        prog = classify_program(a.diagram(), query=Q, variables=V3)
        r, ms = timed(V3, P3, Q, a, prefer_gpp=False)
        show(f"{gname} {qn} [{prog.kind.value}, deg {prog.degree}]", r, ms)

Vm = Variables.from_binary("X", "Y")
markov = Assumptions.diagram(["X", "Y"], [("X", "Y")], [])
show("Markovian bow P(Y1=1)", bound(Vm, P, Qy, markov))
show("independence X,Y", bound(V, P, Qy, base & Assumptions.independence(["X"], ["Y"])))
