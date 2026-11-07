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

### When is the canonical normal form a single linear program?

The checklist below is the theorem. The model, the proofs, the examples, and what is known once a component has three or more hyperedges are written out in `WHEN_THE_BOUNDS_ARE_AN_LP.md`, at the repository root.

Suppose you are given a causal diagram \(G\) (a DAG with latent hyperedges) and a linear counterfactual query \(Q\). Before looking at any dataset, you simplify the optimization problem over the canonical response-type laws using only equivalences that come directly from the graph and the query:

1. **Delete redundant hyperedges:** Any hyperedge contained inside another is merged into the larger one.
2. **Factor by \(c\)-components:** Because distinct \(c\)-components share no latent variables, the joint response-type law factors as a product \(q_1 \otimes \dots \otimes q_m\), and Tian's \(c\)-factor decomposition splits the observational distribution \(P\) into one factor \(Q_C\) per \(c\)-component.
3. **Eliminate determined pieces from the objective:**
   - **Pinning:** Any unconfounded variable \(V\) evaluated in at most one world per event of \(Q\) has its response law replaced in \(Q\) by the observed conditional \(P(V \mid \mathrm{Pa}_V)\).
   - **Dropping:** Any \(c\)-component that \(Q\) never reads, or reads only through root variables (whose values are directly observed margins of \(P\)), is removed from the objective.

What remains is the **normal form** of \((G, Q)\): an optimization of a multilinear objective \(f(q_1, \dots, q_K)\) over the remaining \(c\)-components \(C_1, \dots, C_K\), together with the requirement that every component's \(c\)-factor is **proper** (depends only on that component and its parents).

**The theorem answers the following question:**

> *Fix \(G\) and \(Q\) (in the class where every \(c\)-component has at most two hyperedges). When is this normal form a **single linear program** in \((q_1, \dots, q_K)\) that, for **every** positive observational table \(P\), either (a) is infeasible when no canonical SCM on \(G\) can generate \(P\), or (b) has minimum and maximum equal to the exact sharp bounds on \(Q\)?*

This holds **if and only if** three conditions—read purely off \(G\) and \(Q\)—are met:

- **Saturation (on the dropped components):** Every \(c\)-component dropped from the objective can realize *every* proper \(c\)-factor that a table \(P\) can present to it. Otherwise, there is a table \(P\) that the dropped component cannot generate (so the true model is empty/falsified), yet the normal form LP in the remaining variables \((q_1, \dots, q_K)\) never sees the dropped component and falsely returns finite bounds.
  - *Graphically:* A 1-hyperedge component is saturated \(\iff\) for every \(Y \in C\) and every outside parent \(Z\) of \(C\), either \(Z \in \mathrm{Pa}_Y\) or \(Y\) has a directed path to \(Z\) (**reachability**). A 2-hyperedge component is saturated \(\iff\) **reachability** holds, every variable private to the first hyperedge and every variable private to the second are linked by a direct edge (**cross-edge test**), and no variable in the intersection of the two hyperedges has a **blind** private variable on both sides (**no two-sided blindness**: a private variable \(w\) is blind to an intersection variable \(\rho\) when some outside parent \(Z\) is a parent of \(w\), is not a parent of \(\rho\), and \(\rho\) is not a parent of \(w\)).
- **Covering (on the remaining components):** For every remaining component \(C_j\), the set of realizable response laws matching \(P\) is a polytope (cut out by linear equalities and inequalities) for every table \(P\).
  - *Graphically:* For \(\le 2\) hyperedges, this holds \(\iff\) at least one hyperedge of \(C_j\) contains all non-root variables of \(C_j\) (variables that have a parent in \(G\), inside or outside \(C_j\)), so the private variables of the other hyperedge are all global roots of \(G\), turning the latent-independence constraint \(R_{S_1} \perp R_{S_2}\) into the linear constraint \(\sum_{r_{12}} q(r_1, r_{12}, r_2) = P(S_1 = r_1) q(r_2)\).
- **Linearity (on the objective over remaining components):** The multilinear objective \(f(q_1, \dots, q_K)\) is actually linear on the product of the feasible polytopes for every table \(P\) (equivalently, the pairwise four-term difference vanishes on the feasible sets; e.g., each event reads at most one remaining component, or any cross-component factor is fixed by that component's linear \(c\)-factor constraints).

**Why the exact graphical characterization stops at \(\lambda(C) \le 2\):**

- **One bipartition at \(\lambda \le 2\):** With at most two hyperedges \(H\) and \(K\) (latents \(U \perp V\)), the entire realizability constraint on the joint response law \(q_C(r_L, r_I, r_R)\) is a **single unconditional independence** \(R_L \perp R_R\), with \(R_I\) completely free given \((R_L, R_R)\). Both **Covering** and **Saturation** therefore reduce to two-sided graphical tests across the single cut \((L, R)\).
- **Higher-order couplings at \(\lambda \ge 3\):** Once a component has three or more hyperedges, realizability is governed by overlapping marginal and conditional couplings (as in the triangle network \(\{A,B\}, \{A,C\}, \{B,C\}\)). For remaining components (**Covering**), whether every fiber is a polytope at \(\lambda \ge 3\) depends on recursive latent-conditioning reductions intertwined with which specific roots are parents of which specific non-roots. For dropped components (**Saturation**), variables inside the component that miss one latent can act as *internal switches* on the remaining latents (for example, on the 4-node chain \(A \to B \to C \to D\) with hyperedges \(\{A,C\}, \{A,D\}, \{B,D\}\)), imposing quantitative functional-equation constraints even when the graph is a total chain and every non-adjacent pair shares a hyperedge.

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
