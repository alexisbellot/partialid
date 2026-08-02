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

`classify_program` implements THEORY §6 under hypothesis (H): ancestral trim,
footprint free blocks (C1), directed-cut C2 (WL–WR), then
\(\mathrm{degree}=\sum\lambda\) over surviving free districts.

| Degree | Kind | Default solver |
|---|---|---|
| 0 | point | ancestral TypeSpace / ID path in `bound` |
| 1 | linear | fiber / LP / instrument (Prop. 8.4) |
| 2 | bilinear/latent or bilinear/factor | `gpp/latent` or `gpp` |
| ≥3 | polynomial | **outer LP** (`tight=False`) |

For deg≥3, opt into the nested multilinear solver with
`bound(..., prefer_poly=True)` / `formulate(..., prefer_poly=True)`
(`method` → `gpp/poly` when the size gate passes; factor \(K=3\) only today).
Latent \(\lambda\ge 3\) still falls back to outer even with `prefer_poly=True`.

**`classify` is not full ID.** Empty free set → `POINT`, but identified
front-door / back-door graphs may still classify as `LINEAR` until the separate
ID pass in `bound` returns a point interval (`method` starts with `id/`).
Use `bound` (or `try_identify`) for identification, not `ProgramKind` alone.

```python
from partialid import CausalDiagram, Query, classify_program

g = CausalDiagram.from_edges(
    ["X", "M", "Y"],
    directed=[("X", "M"), ("M", "Y")],
    bidirected=[("X", "M"), ("M", "Y")],
)
prog = classify_program(g, treatments=["X"], outcome="Y")
# ProgramKind.BILINEAR_LATENT, degree=2, in_hypothesis_H=True

# Any AST query (not only named factories): pass query=
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
from partialid.query_ast import Atom, Lit, PO

Q = Query.conditional(
    Atom(Lit(PO("Y", {"Z": 1}), 1), Lit(PO("W", {"X": 0}), 0)),
    Atom(Lit("R", 1)),
)
classify_program(diagram, query=Q)  # degree from multi-regime + evidence
```

`ProgramDegree` also records `wl_wr_cut`, `in_hypothesis_H`, and
`pre_cut_free_districts`. Worked examples live in the docstring of
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

Degree ≥3 defaults to a merged-latent **outer** LP
(`bidirected=[tuple(nodes)]`, `tight=False`). Bilinear size-gate failures and
hard independence splits can also outer. Always check `r.tight` in automated
pipelines.

## Next

- [IV recipe](recipes/iv.md)
- [Auto routing](auto_routing.md)
- [Solvers (advanced)](advanced/solvers.md)
