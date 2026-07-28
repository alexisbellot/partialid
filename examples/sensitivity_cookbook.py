"""Cookbook: sensitivity / soft-independence / sparsity knobs on RE1.

Run: PYTHONPATH=src python examples/sensitivity_cookbook.py
"""

from __future__ import annotations

from partialid import Assumptions, Query, Variables, bound, bound_grid
from partialid.convex_solve import cvxpy_available

V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
base = Assumptions.order("X", "Y")

print("=== Manski (order-only) ===")
r = bound(V, P, Q, base)
print(f"  interval={r.interval} width={r.width:.4f} method={r.method}")

print("\n=== Tan MSM grid (persistent HiGHS if highspy installed) ===")
for lam, rr in bound_grid(
    V, P, Q, base, param="Lambda", values=[1.0, 1.5, 3.0, 7.0], warm_start=True
):
    print(f"  Λ={lam:g}: [{rr.lower:.4f}, {rr.upper:.4f}] w={rr.width:.4f}")

print("\n=== Soft independence via margin_budget (THEORY §8.8.3) ===")
q_fd = 139 / 240
for kappa in (0.0, 0.1, 11 / 48, 13 / 48, 1.0):
    rr = bound(V, P, Q, base & Assumptions.margin_budget(kappa, s0=q_fd), method="lp")
    print(f"  κ={kappa:.4f}: [{rr.lower:.4f}, {rr.upper:.4f}] w={rr.width:.4f}")

print("\n=== support_bound (Cor 7.7: step at |X|=4) ===")
for k in (3, 4, 8):
    rr = bound(V, P, Q, base & Assumptions.support_bound(k))
    if rr.success:
        print(f"  k={k}: [{rr.lower:.4f}, {rr.upper:.4f}] w={rr.width:.4f}")
    else:
        print(f"  k={k}: infeasible ({rr.message})")

print("\n=== f-sensitivity (needs partialid[convex]) ===")
if not cvxpy_available():
    print("  skipped — install with: pip install 'partialid[convex]'")
else:
    for rho in (0.0, 0.05, 0.5, 5.0):
        rr = bound(V, P, Q, base & Assumptions.f_sensitivity("X", rho, f="tv"))
        print(f"  TV ρ={rho:g}: [{rr.lower:.4f}, {rr.upper:.4f}] w={rr.width:.4f}")
