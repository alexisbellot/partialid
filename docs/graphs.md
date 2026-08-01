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

`bidirected` entries are **latent hyperedges**: each tuple is one independent
exogenous variable. There is no separate `maximal_confounding` flag — write the
latent structure explicitly.

| Writing | Meaning | Program (for typical `P(Y\|do(X))`) |
|---|---|---|
| `[("X", "Y")]` | one latent on `{X,Y}` | linear (degree 1) |
| `[("X", "M"), ("M", "Y")]` | two latents | bilinear (degree 2) |
| `[("X", "M", "Y")]` | one latent on all three | linear (degree 1) |
| `[tuple(nodes)]` / `from_order` | one latent on everyone (Manski) | linear |
| `[]` | no latent confounding | point ID / Markovian |

With an **explicit diagram**, `bound` first tries identification (g-formula /
back-door / front-door). When identified, you get a point interval
(`method` starts with `id/`).

`Assumptions.order("Z", "X", "Y")` is Manski: complete directed order plus
`bidirected=[("Z","X","Y")]`.

## Program degree (which solver?)

Degree is derived from the GPP, not guessed: if free confounded districts have
c-degrees \(d_C\) (# hyperedges each), then

\[
\text{degree} = \sum_C d_C
\]

(each hyperedge is one free simplex factor). Then:

| Degree | Kind | Solver |
|---|---|---|
| 0 | point | ID / g-formula |
| 1 | linear | fiber / LP / instrument |
| 2 | bilinear/latent or bilinear/factor | `gpp/latent` or `gpp` |
| ≥3 | polynomial | outer LP (`tight=False`) for now |

```python
from partialid import CausalDiagram, Query, classify_program

g = CausalDiagram.from_edges(
    ["X", "M", "Y"],
    directed=[("X", "M"), ("M", "Y")],
    bidirected=[("X", "M"), ("M", "Y")],
)
print(classify_program(g, treatments=["X"], outcome="Y"))
# ProgramKind.BILINEAR_LATENT, degree=2

# Any AST query (not only named factories): pass query=
from partialid.query_ast import Atom, Lit, PO

g2 = CausalDiagram.from_edges(
    ["X", "M", "Y"],
    directed=[("X", "M"), ("M", "Y")],
    bidirected=[("X", "M")],
)
print(classify_program(g2, query=Query.interventional(outcome="Y", do={"X": 1}, value=1)).kind)
# linear
print(classify_program(g2, query=Query.pns()).kind)
# bilinear/factor

# e.g. P(Y_z=1, W_x=0 | R=1)
Q = Query.conditional(
    Atom(Lit(PO("Y", {"Z": 1}), 1), Lit(PO("W", {"X": 0}), 0)),
    Atom(Lit("R", 1)),
)
classify_program(diagram, query=Q)  # degree from multi-regime + evidence
```

Worked examples live in the docstring of
[`program_degree.py`](../src/partialid/program_degree.py).

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

Multi c-component / degree ≥3 problems may fall back to a merged-latent
**outer** LP (`bidirected=[tuple(nodes)]`). Then `BoundResult.tight` is `False`
and a warning is emitted. Always check `r.tight` in automated pipelines.

## Next

- [IV recipe](recipes/iv.md)
- [Auto routing](auto_routing.md)
- [Solvers (advanced)](advanced/solvers.md)
