"""What the package currently returns on the pair chain and IV with Z<->X (eq:re2)."""

from partialid import Assumptions, Query, Variables, bound

RE2 = {
    (0, 0, 0): 0.20, (0, 0, 1): 0.05, (0, 1, 0): 0.10, (0, 1, 1): 0.15,
    (1, 0, 0): 0.05, (1, 0, 1): 0.10, (1, 1, 0): 0.10, (1, 1, 1): 0.25,
}
cases = {
    "pair chain": (("X", "W", "Y"), [("X", "W"), ("W", "Y")], [("X", "W"), ("W", "Y")], "X"),
    "IV Z<->X": (("Z", "X", "Y"), [("Z", "X"), ("X", "Y")], [("Z", "X"), ("X", "Y")], "X"),
}
for name, (nodes, de, be, t) in cases.items():
    V = Variables.from_binary(*nodes)
    a = Assumptions.diagram(list(nodes), de, be)
    for qn, Q in {
        "P(Y_1=1)": Query.interventional(outcome="Y", do={t: 1}, value=1),
        "PNS": Query.pns(treatment=t, outcome="Y"),
    }.items():
        r = bound(V, RE2, Q, a, prefer_gpp=False)
        print(f"{name:10s} {qn:9s} [{r.lower:.4f}, {r.upper:.4f}] tight={r.tight} method={r.method}")
