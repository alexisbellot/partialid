"""Rounded D-descendant table: certified full-model bounds versus the package."""

import itertools

from partialid import Assumptions, Query, Variables, bound
from s3_trim_strict import certified, law_from_scm, manski

P = law_from_scm(1)
P = {k: round(v, 3) for k, v in P.items()}
s = sum(P.values())
P = {k: v / s for k, v in P.items()}
print("P(x,y,d):", {k: round(v, 4) for k, v in P.items()}, "sum", round(sum(P.values()), 6))
print("trimmed (Manski): [%.4f, %.4f]" % manski(P))
li, lo, _ = certified(P, "min")
ui, uo, _ = certified(P, "max")
print(f"full model: lower in [{lo:.4f}, {li:.4f}], upper in [{ui:.4f}, {uo:.4f}]")

V = Variables.from_binary("X", "Y", "D")
a = Assumptions.diagram(["X", "Y", "D"], [("X", "Y"), ("Y", "D")], [("X", "Y"), ("X", "D")])
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
r = bound(V, {k: P[k] for k in itertools.product((0, 1), repeat=3)}, Q, a)
print(f"package: [{r.lower:.4f}, {r.upper:.4f}] tight={r.tight} method={r.method}")
