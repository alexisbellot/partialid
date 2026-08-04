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

With an **explicit diagram**, `bound` runs **Shpitser–Pearl ID** for
interventional queries (and ATE as a difference of two ID’d margins). When
identified, you get a point interval (`method` is `id` or `id/ate`). Classical
g-formula / back-door / front-door are special cases of ID, not separate engines.

Check identification without solving an LP:

```python
from partialid import CausalDiagram, is_identified, identify_query

g = CausalDiagram.from_edges(
    ["X", "Z", "Y"],
    directed=[("X", "Z"), ("Z", "Y")],
    bidirected=[("X", "Y")],
)
assert is_identified(g, ["X"], ["Y"])  # classic front-door
```

`Assumptions.order("Z", "X", "Y")` is Manski: complete directed order plus
`bidirected=[("Z","X","Y")]` — ID is skipped (never point-identified for `do`).

## Program degree (which solver?)

`classify_program` implements **program-degree theory**
(`THEORY_PROGRAM_DEGREE` §6) under hypothesis (H): ancestral trim,
footprint free blocks (C1), directed-cut C2 (WL–WR), then **ID** (empty free
set → `POINT`), else \(\mathrm{degree}=\sum\lambda\) over surviving free
districts.

| Degree | Kind | Default solver |
|---|---|---|
| 0 | point | ID in `bound` (`method` `id` / `id/ate`), or ancestral TypeSpace |
| 1 | linear | fiber / LP / instrument via classify C2 (WL–WR cut) |
| 2 | bilinear/latent or bilinear/factor | `gpp/latent` or `gpp` (`prefer_gpp=True`) |
| ≥3 | polynomial | **outer LP** (`tight=False`) |

Knobs on `bound` / `formulate`:

- `prefer_gpp=True` (default) — bilinear → nested GPP; set `False` for outer.
- `prefer_poly=False` (default) — deg≥3 → outer; set `True` for size-gated
  factor-\(K=3\) nested multilinear (`method` → `gpp/poly`).
  Latent \(\lambda\ge 3\) still falls back to outer even with `prefer_poly=True`.

For interventional / ATE queries, `classify_program` and `bound` agree on ID:
front-door / napkin → `POINT` / `method="id"`. Counterfactuals (PNS, …) are
not handled by interventional ID.

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
from partialid import Atom, Lit, PO

Q = Query.conditional(
    Atom(Lit(PO("Y", {"Z": 1}), 1), Lit(PO("W", {"X": 0}), 0)),
    Atom(Lit("R", 1)),
)
classify_program(diagram, query=Q)  # degree from multi-regime + evidence
```

`ProgramDegree` also records `wl_wr_cut`, `in_hypothesis_H`, and
`pre_cut_free_districts`. Worked latent expansions live in the module docstring of
[`program_degree.py`](https://github.com/alexisbellot/partialid/blob/main/src/partialid/program_degree.py)
(and in [`classify_program`](api/index.md) in the API reference).

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

Degree ≥3 defaults to a merged-latent **outer** LP (`tight=False`). That
fallback is built by `_merged_latent_diagram` (one latent over the surviving
ancestral nodes) — **not** the same object as intentional Manski order-only
setup, which uses `bidirected=[tuple(nodes)]` / `Assumptions.order` from the
start and remains a sharp degree‑1 LP.

Bilinear size-gate failures and hard `independence` also force outer. Uncertified
GPP attainables set `tight=False` with a **possibly-inner** warning (not a
certified outer). Always check `r.tight` (and `r.notes`) in automated pipelines.

## Next

- [IV recipe](recipes/iv.md)
- [Auto routing](auto_routing.md)
- [Solvers (advanced)](advanced/solvers.md)
