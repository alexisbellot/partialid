"""Universe roadmap examples: PN/PS, MTS, conditional do, paths, audit, CI."""

from partialid import (
    Assumptions,
    Query,
    Variables,
    audit_bound,
    bound,
    bound_ci,
    bound_path,
)

V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
a = Assumptions.order("X", "Y")
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

print("=== P1: queries & faces ===")
print("Manski:", bound(V, P, Q, a))
print("MTS:   ", bound(V, P, Q, a & Assumptions.mts("X", "Y")))
print("PNS:   ", bound(V, P, Query.pns(), a))
print("PN:    ", bound(V, P, Query.pn(), a))
print("PS:    ", bound(V, P, Query.ps(), a))

print("\n=== Conditional interventional ===")
V3 = Variables.from_binary("W", "X", "Y")
P3 = {c: 0.125 for c in V3.cells()}
Qc = Query.interventional(outcome="Y", do={"X": 1}, value=1, condition={"W": 0})
print("P(Y=1|do(X=1),W=0):", bound(V3, P3, Qc, Assumptions.order("W", "X", "Y")))

print("\n=== P2: budget paths ===")
for param in ("Lambda", "c", "kappa"):
    path = bound_path(V, P, Q, a, param=param, n_grid=7)
    print(f"  {param}: n={len(path.values)}, width[{path.width[0]:.3f}→{path.width[-1]:.3f}]")

print("\n=== P3: audit + CI ===")
r = bound(V, P, Q, a & Assumptions.mtr("Y", "X") & Assumptions.tan_msm("X", 2.0))
print(
    audit_bound(
        r, query=Q, assumptions=a & Assumptions.mtr("Y", "X") & Assumptions.tan_msm("X", 2.0)
    )
)
print(bound_ci(V, P, Q, a, n_boot=50, n=500, seed=0))
