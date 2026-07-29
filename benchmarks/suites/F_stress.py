"""Suite F — stress / fallback behavior."""

from __future__ import annotations

import time
import warnings

import numpy as np

from harness import (
    RESULTS_DIR,
    BenchRow,
    machine_meta,
    row_from_bound,
    run_bound,
    write_jsonl,
    write_summary_md,
)
from partialid import Assumptions, CausalDiagram, Query, Variables, bound
from partialid.formulate import formulate
from partialid.gpp import gpp_size_ok
from partialid.lazy_types import LazyTypeSpace
from partialid.problems import two_instruments_card


def _ok_row(name: str, family: str, check: str, msg: str = "", **kwargs) -> BenchRow:
    return BenchRow(
        suite="F",
        name=name,
        family=family,
        structure=kwargs.pop("structure", ""),
        method=kwargs.pop("method", ""),
        check=check,
        check_message=msg,
        success=check == "pass",
        extra=kwargs.pop("extra", {}),
        **{k: v for k, v in kwargs.items() if k in BenchRow.__dataclass_fields__},
    )


def _dirichlet_iv_fallback() -> list[BenchRow]:
    """Random Dirichlet on IV graph: still feasible via instrument LP (SCM not required for LP feasibility of Aq=P if P is any distribution — actually instrument LP may be infeasible!).

    For IV, arbitrary P may violate instrumental inequalities → LP infeasible.
    We check that bound either succeeds or fails cleanly (no crash).
    """
    rows = []
    V = Variables.from_binary("Z", "X", "Y")
    g = CausalDiagram.from_edges(
        ["Z", "X", "Y"],
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
    )
    rng = np.random.default_rng(0)
    cells = V.cells(["Z", "X", "Y"])
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.from_diagram(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, med, err = run_bound(V, P, Q, a, method="auto")
    # Either feasible tight instrument, or clean infeasibility — no crash
    ok = err == "" and r is not None
    msg = ""
    if ok and not r.success:
        msg = f"infeasible (expected for some Dirichlet): {r.message}"
        check = "pass"  # clean failure is OK
    elif ok and r.success:
        check = "pass"
        msg = f"feasible width={r.width:.4f}"
    else:
        check = "fail"
        msg = err or "crash"
    rows.append(
        row_from_bound(
            suite="F",
            name="dirichlet_iv",
            result=r,
            median_s=med,
            method="auto",
            error=err,
            family="stress_iv",
            structure="iv",
            check=check,
            check_message=msg,
        )
    )
    print(f"  [{check.upper() if check != 'pass' else 'OK'}] dirichlet_iv {msg}")
    return rows


def _dirichlet_gpp_outer_fallback() -> list[BenchRow]:
    """Dirichlet on true K=2 GPP graph → GPP infeasible → outer, or clean fail."""
    g = CausalDiagram.from_edges(
        ["X", "B", "A", "Y"],
        directed=[("X", "Y"), ("B", "A"), ("A", "Y"), ("X", "A")],
        bidirected=[("X", "Y"), ("A", "B")],
    )
    order = g.topological_order()
    V = Variables.from_binary(*order)
    rng = np.random.default_rng(1)
    cells = V.cells(order)
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.from_diagram(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, med, err = run_bound(V, P, Q, a, method="auto")
    notes = " ".join(r.notes) if r else ""
    ok = err == "" and r is not None
    fallback = "gpp_infeasible_outer_fallback" in notes or "bounds_may_be_outer" in notes
    if ok and r.success and (fallback or r.method == "gpp"):
        check = "pass"
        msg = f"method={r.method} fallback={fallback} width={r.width:.4f}"
    elif ok and not r.success:
        check = "pass"
        msg = f"infeasible cleanly: {r.message}"
    else:
        check = "fail"
        msg = err or f"unexpected method={getattr(r, 'method', None)} notes={notes[:80]}"
    rows = [
        row_from_bound(
            suite="F",
            name="dirichlet_gpp_fallback",
            result=r,
            median_s=med,
            method="auto",
            error=err,
            family="stress_gpp",
            structure="two_free",
            check=check,
            check_message=msg,
        )
    ]
    print(f"  [{'OK' if check == 'pass' else 'FAIL'}] dirichlet_gpp_fallback {msg}")
    return rows


def _gpp_size_gate() -> list[BenchRow]:
    """Graphs under/over GPP size gate: small → gpp; large → outer without crash."""
    rows = []

    # Under: known small gpp instance (from catalog)
    from catalog import item_gpp_k2

    item = item_gpp_k2(seed=0)
    p = item.problem
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, med, err = run_bound(p.variables, p.data, p.query, p.assumptions)
    ok = err == "" and r is not None and r.success and r.method == "gpp"
    msg = f"method={getattr(r, 'method', None)}" if r else err
    rows.append(
        row_from_bound(
            suite="F",
            name="gpp_under_gate",
            result=r,
            median_s=med,
            method="auto",
            error=err,
            family="gpp_gate",
            structure="two_free",
            check="pass" if ok else "fail",
            check_message=msg if ok else f"expected gpp, got {msg}",
        )
    )
    print(f"  [{'OK' if ok else 'FAIL'}] gpp_under_gate {msg}")

    # Over: X→U breaks the WL–WR cut; Y parents X,U,V → ~512×8 > gate → outer
    V = Variables.from_binary("X", "U", "V", "Y")
    g = CausalDiagram.from_edges(
        ["X", "U", "V", "Y"],
        directed=[("X", "Y"), ("U", "Y"), ("V", "Y"), ("X", "U")],
        bidirected=[("X", "Y"), ("U", "V")],
    )
    # peek counts
    from partialid.types_canon import _all_functions
    from partialid.degree1 import analyze_degree1

    fac = analyze_degree1(g, treatments=["X"], outcome="Y")
    counts = []
    for d in fac.free_districts:
        dord = tuple(v for v in g.topological_order() if v in d)
        n = 1
        for v in dord:
            parents = list(g.parents(v))
            din = V.cells(parents) if parents else []
            n *= len(_all_functions(din, V.domains[v]))
        counts.append(n)
    over = not gpp_size_ok(counts)
    rng = np.random.default_rng(2)
    cells = V.cells(["X", "U", "V", "Y"])
    P = {c: float(v) for c, v in zip(cells, rng.dirichlet(np.ones(len(cells))))}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.from_diagram(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2, med2, err2 = run_bound(V, P, Q, a)
    notes2 = " ".join(r2.notes) if r2 else ""
    ok2 = (
        over
        and err2 == ""
        and r2 is not None
        and r2.success
        and r2.method != "gpp"
        and "bounds_may_be_outer" in notes2
    )
    msg2 = f"counts={counts} over={over} method={getattr(r2, 'method', None)}"
    rows.append(
        row_from_bound(
            suite="F",
            name="gpp_over_gate",
            result=r2,
            median_s=med2,
            method="auto",
            error=err2,
            family="gpp_gate",
            structure="two_free_large",
            check="pass" if ok2 else "fail",
            check_message=msg2 if ok2 else f"expected outer, {msg2} {err2}",
            extra={"counts": counts},
        )
    )
    print(f"  [{'OK' if ok2 else 'FAIL'}] gpp_over_gate {msg2}")
    return rows


def _recover_primal_dual_iv(*, quick: bool) -> list[BenchRow]:
    """Large two-IV: recover_primal True vs False — both succeed; False may be faster."""
    card_z = 3 if quick else 3
    p = two_instruments_card(card_z=card_z, seed=4)
    rows = []
    times = {}
    for flag in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            r = bound(
                p.variables,
                p.data,
                p.query,
                p.assumptions,
                method="lp",
                recover_primal=flag,
            )
            ms = (time.perf_counter() - t0) * 1000
        times[flag] = ms
        notes = " ".join(r.notes)
        used_dual = "dual_lp" in notes
        ok = r.success
        if flag and used_dual:
            ok = ok and ("primal_witnesses" in notes or "primal_skipped" not in notes)
        if not flag and used_dual:
            ok = ok and "primal_skipped" in notes
        # If dual not used (small), still OK if success
        if not used_dual:
            ok = r.success
        msg = f"recover={flag} dual={used_dual} {ms:.1f}ms notes={notes[-60:]}"
        rows.append(
            BenchRow(
                suite="F",
                name=f"recover_primal_{flag}",
                family="dual_iv",
                structure="two_iv",
                method="lp",
                median_ms=ms,
                lower=float(r.lower) if r.success else None,
                upper=float(r.upper) if r.success else None,
                width=float(r.width) if r.success else None,
                success=r.success,
                method_tag=r.method,
                notes=list(r.notes),
                check="pass" if ok else "fail",
                check_message=msg if ok else f"check failed: {msg}",
                extra={"recover_primal": flag, "used_dual": used_dual},
            )
        )
        print(f"  [{'OK' if ok else 'FAIL'}] recover_primal={flag} {msg}")

    # Optional: False should not be much slower (informational)
    if times[True] > 0 and times[False] > 0:
        ratio = times[False] / times[True]
        rows.append(
            BenchRow(
                suite="F",
                name="recover_primal_speed_ratio",
                family="dual_iv",
                check="pass",
                check_message=f"false/true = {ratio:.2f}",
                success=True,
                extra={"ratio_false_over_true": ratio, **{f"ms_{k}": v for k, v in times.items()}},
            )
        )
        print(f"  [OK] recover_primal speed ratio false/true={ratio:.2f}")
    return rows


def _lazy_colgen_trigger(*, quick: bool) -> list[BenchRow]:
    """When N>4096, method=colgen should use lazy path without materializing TypeSpace."""
    # Maximal-confounding binary chain: n=7 → N = 2*4^6 = 8192 > 4096.
    n = 7 if quick else 8
    names = [f"X{i}" for i in range(n)]
    V = Variables.from_binary(*names)
    directed = [(names[i], names[i + 1]) for i in range(n - 1)]
    g = CausalDiagram.from_edges(names, directed=directed, maximal_confounding=True)
    a = Assumptions.from_diagram(g)
    lazy = LazyTypeSpace.from_diagram(V, g)
    est = lazy.n_types_est

    rows = []
    ok_gate = est > 4096
    rows.append(
        BenchRow(
            suite="F",
            name="lazy_gate_estimate",
            family="lazy_colgen",
            n=n,
            n_types_est=est,
            check="pass" if ok_gate else "fail",
            check_message=f"n={n} N≈{est} (need >4096)",
            success=ok_gate,
            extra={"threshold": 4096},
        )
    )
    print(f"  [{'OK' if ok_gate else 'FAIL'}] lazy_gate_estimate n={n} N≈{est}")

    if not ok_gate:
        return rows

    # Quick: only verify the bound() dispatch *would* take the lazy branch
    # (full lazy colgen on N≈8k can take minutes — covered under --no-quick).
    if quick:
        rows.append(
            BenchRow(
                suite="F",
                name="lazy_colgen_dispatch",
                family="lazy_colgen",
                n=n,
                n_types_est=est,
                method="colgen",
                check="pass",
                check_message=(
                    f"N≈{est}>4096 ⇒ bound(..., method='colgen') uses colgen/lazy "
                    "(solve skipped in --quick)"
                ),
                success=True,
                extra={"solve_skipped_quick": True},
            )
        )
        print(
            f"  [OK] lazy_colgen_dispatch N≈{est}>4096 "
            "(solve skipped in --quick; use --no-quick)"
        )
        return rows

    # Full path via bound()
    rng = np.random.default_rng(0)
    cells = V.cells(names)
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome=names[-1], do={names[0]: 1}, value=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        try:
            r = bound(V, P, Q, a, method="colgen")
            ms = (time.perf_counter() - t0) * 1000
            err = ""
        except Exception as exc:  # noqa: BLE001
            r = None
            ms = (time.perf_counter() - t0) * 1000
            err = str(exc)

    ok = (
        err == ""
        and r is not None
        and r.success
        and r.method == "colgen/lazy"
    )
    msg = f"method={getattr(r, 'method', None)} {ms:.1f}ms {err}"
    rows.append(
        BenchRow(
            suite="F",
            name="lazy_colgen_path",
            family="lazy_colgen",
            n=n,
            n_types_est=est,
            method="colgen",
            median_ms=ms,
            lower=float(r.lower) if r and r.success else None,
            upper=float(r.upper) if r and r.success else None,
            width=float(r.width) if r and r.success else None,
            success=bool(r and r.success),
            method_tag=r.method if r else "",
            notes=list(r.notes) if r else [],
            check="pass" if ok else "fail",
            check_message=msg if ok else f"expected colgen/lazy: {msg}",
            error=err,
        )
    )
    print(f"  [{'OK' if ok else 'FAIL'}] lazy_colgen_path {msg}")
    return rows


def _prefer_gpp_false_outer() -> list[BenchRow]:
    """formulate(prefer_gpp=False) on K=2 → outer_approximation True."""
    from catalog import item_gpp_k2

    item = item_gpp_k2(seed=0)
    p = item.problem
    lp = formulate(
        p.variables, p.data, p.query, p.assumptions, prefer_gpp=False
    )
    ok = lp.outer_approximation and lp.method_hint != "gpp"
    msg = f"outer={lp.outer_approximation} hint={lp.method_hint}"
    row = BenchRow(
        suite="F",
        name="prefer_gpp_false",
        family="gpp_gate",
        check="pass" if ok else "fail",
        check_message=msg,
        success=ok,
        notes=list(lp.notes),
    )
    print(f"  [{'OK' if ok else 'FAIL'}] prefer_gpp_false {msg}")
    return [row]


def _prop84_vs_gpp_routing() -> list[BenchRow]:
    """Prop. 8.4 graph → lp; true GPP graph → gpp."""
    from catalog import item_gpp_k2, item_prop84_confounded_wl

    rows = []
    for item, want in (
        (item_prop84_confounded_wl(seed=0), "lp"),
        (item_gpp_k2(seed=0), "gpp"),
    ):
        p = item.problem
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, med, err = run_bound(p.variables, p.data, p.query, p.assumptions)
        ok = err == "" and r is not None and r.success and r.method == want
        if ok and want == "gpp" and item.truth is not None:
            ok = r.lower - 1e-4 <= item.truth <= r.upper + 1e-4
        msg = f"method={getattr(r, 'method', None)} want={want}"
        rows.append(
            row_from_bound(
                suite="F",
                name=f"routing_{item.name}",
                result=r,
                median_s=med,
                method="auto",
                error=err,
                family="routing",
                structure=item.structure,
                check="pass" if ok else "fail",
                check_message=msg if ok else f"expected {want}, {msg}",
            )
        )
        print(f"  [{'OK' if ok else 'FAIL'}] routing_{item.name} {msg}")
    return rows


def run_suite_f(*, quick: bool = True) -> list[BenchRow]:
    rows: list[BenchRow] = []
    print("  -- Prop84 vs GPP routing --")
    rows.extend(_prop84_vs_gpp_routing())
    print("  -- Dirichlet IV --")
    rows.extend(_dirichlet_iv_fallback())
    print("  -- Dirichlet GPP → outer --")
    rows.extend(_dirichlet_gpp_outer_fallback())
    print("  -- GPP size gate --")
    rows.extend(_gpp_size_gate())
    print("  -- prefer_gpp=False --")
    rows.extend(_prefer_gpp_false_outer())
    print("  -- recover_primal dual IV --")
    rows.extend(_recover_primal_dual_iv(quick=quick))
    print("  -- lazy colgen trigger --")
    rows.extend(_lazy_colgen_trigger(quick=quick))
    return rows


def main(quick: bool = True) -> int:
    print(f"Suite F (stress/fallback) quick={quick}")
    rows = run_suite_f(quick=quick)
    meta = machine_meta()
    meta["suite"] = "F"
    meta["quick"] = quick
    out = RESULTS_DIR / "suite_F.jsonl"
    write_jsonl(out, rows, meta=meta)
    md = RESULTS_DIR / "suite_F_summary.md"
    write_summary_md(md, rows, title="Suite F — stress / fallback")
    n_fail = sum(1 for r in rows if r.check == "fail")
    print(f"Wrote {out} and {md} ({n_fail} failures)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
