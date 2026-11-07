"""Package output on the mediated graph X -> W -> Y, X <-> W, with table (eq:re2).

The model implies Y _||_ X | W (Y = f_Y(W, U_Y), U_Y independent of (X, W)),
so we first test that constraint on the table.
"""

from partialid import Assumptions, Query, Variables, bound

RE2 = {
    (0, 0, 0): 0.20, (0, 0, 1): 0.05, (0, 1, 0): 0.10, (0, 1, 1): 0.15,
    (1, 0, 0): 0.05, (1, 0, 1): 0.10, (1, 1, 0): 0.10, (1, 1, 1): 0.25,
}

for w in (0, 1):
    for x in (0, 1):
        pxw = RE2[(x, w, 0)] + RE2[(x, w, 1)]
        print(f"P(Y=1 | X={x}, W={w}) = {RE2[(x, w, 1)] / pxw:.3f}")

V = Variables.from_binary("X", "W", "Y")
a = Assumptions.diagram(["X", "W", "Y"], [("X", "W"), ("W", "Y")], [("X", "W")])
for name, Q in {
    "P(Y_1=1)": Query.interventional(outcome="Y", do={"X": 1}, value=1),
    "PNS": Query.pns(treatment="X", outcome="Y"),
}.items():
    r = bound(V, RE2, Q, a)
    print(f"{name:9s} success={r.success} [{r.lower:.3f}, {r.upper:.3f}] "
          f"tight={r.tight} method={r.method}")
