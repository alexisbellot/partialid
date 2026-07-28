# Graphs & instruments

Assumptions can encode an explicit causal diagram — one atom among many you
can compose with `&`. See the [Assumptions](assumptions.md) catalog.

## Diagrams

```python
from partialid import Assumptions, CausalDiagram

g = CausalDiagram.from_edges(
    nodes=["Z", "X", "Y"],
    directed=[("Z", "X"), ("X", "Y")],
    bidirected=[("X", "Y")],
)
a = Assumptions.from_diagram(g)
# or Assumptions.diagram(nodes, directed=..., bidirected=...)
```

With an **explicit diagram**, `bound` first tries identification (g-formula /
back-door / front-door). When identified, you get a point interval
(`method` starts with `id/`).

Order-only problems (`Assumptions.order` / `manski`) skip ID and go straight to
bounds under maximal confounding.

## Instrumental variables

Classic binary IV recipe:

```python
from partialid import Assumptions, Query, Variables, bound

# Z → X → Y with latent confounding X ↔ Y
a = Assumptions.iv("Z", "X", "Y")  # add monotone=True for no-defiers when justified

V = Variables.from_binary("Z", "X", "Y")
# P must be a valid law on (Z,X,Y); IV inequalities should hold for tight BP bounds
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
r = bound(V, P, Q, a)
print(r.summary(Q), r.tight)
```

Add `monotone=True` for no-defiers when that assumption is justified **and**
compatible with \(P\).

See [IV recipe](recipes/iv.md) for a complete runnable example.

## Tightness on diagrams

Multi c-component problems may fall back to a merged-latent **outer** LP.
Then `BoundResult.tight` is `False` and a warning is emitted. Always check
`r.tight` in automated pipelines.

## Next

- [IV recipe](recipes/iv.md)
- [Auto routing](auto_routing.md)
- [Solvers (advanced)](advanced/solvers.md)
