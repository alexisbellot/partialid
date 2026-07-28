"""Practitioner quickstart: compose assumptions, then optional sensitivity path."""

from partialid import Assumptions, Query, Variables, bound, bound_path

V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

print("Compose order + MTR + Tan MSM:")
a = Assumptions.order("X", "Y") & Assumptions.mtr("Y", "X") & Assumptions.tan_msm("X", 2.0)
print(bound(V, P, Q, a).summary(Q))

print("\nManski recipe:")
print(bound(V, P, Q, Assumptions.manski("X", "Y")).summary(Q))

print("\nMSM recipe Λ=3:")
r = bound(V, P, Q, Assumptions.sensitivity_msm("X", "Y", 3.0))
print(r.summary(Q))
print("width", round(r.width, 3))

print("\nOptional: how wide are bounds as Λ goes from 1 → 5?")
path = bound_path(
    V,
    P,
    Q,
    Assumptions.manski("X", "Y"),
    param="Lambda",
    theta_min=1.0,
    theta_max=5.0,
)
print(path.summary(at=3.0))
for lam in (1.0, 2.0, 3.0, 5.0):
    lo, hi, w = path.at(lam)
    print(f"  Λ={lam:g}: [{lo:.4f}, {hi:.4f}]  width={w:.4f}")
