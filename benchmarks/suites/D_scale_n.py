"""Suite D — scale number of variables (cardinality fixed at 2 unless noted)."""

from __future__ import annotations

import warnings

import numpy as np

from harness import (
    RESULTS_DIR,
    BenchRow,
    Frontier,
    machine_meta,
    row_from_bound,
    run_bound,
    scale_stop,
    write_jsonl,
    write_summary_md,
)
from partialid import Assumptions, CausalDiagram, Query, Variables
from partialid.problems import (
    chain_order_only,
    binary_iv,
    shridharan_multi_treatment,
)


def _schedules(*, quick: bool) -> dict:
    if quick:
        return {
            "timeout_s": 5.0,
            "order_chain_ns": [2, 3, 4, 6, 8],
            "markov_ns": [2, 3, 4, 6, 8],
            "bow_inert_ns": [2, 3, 4, 6],
            "multi_Ts": [1, 2, 3],
            "n_instruments": [1, 2],
            "methods_order": ("auto", "fiber", "lp"),
            "methods_multi": ("auto", "fiber"),
            "methods_iv": ("auto", "lp"),
            "include_naive": False,
            "include_colgen": False,
        }
    return {
        "timeout_s": 30.0,
        "order_chain_ns": list(range(2, 13)),
        "markov_ns": list(range(2, 13)),
        "bow_inert_ns": list(range(2, 11)),
        "multi_Ts": list(range(1, 7)),
        "n_instruments": [1, 2, 3, 4],
        "methods_order": ("auto", "fiber", "lp", "colgen"),
        "methods_multi": ("auto", "fiber", "lp"),
        "methods_iv": ("auto", "lp"),
        "include_naive": True,
        "include_colgen": True,
    }


def _markov_chain(n: int, *, seed: int = 0):
    names = [f"X{i}" for i in range(n)]
    V = Variables.from_binary(*names)
    directed = [(names[i], names[i + 1]) for i in range(n - 1)]
    g = CausalDiagram.from_edges(names, directed=directed)
    rng = np.random.default_rng(seed)
    cells = V.cells(names)
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome=names[-1], do={names[0]: 1}, value=1)
    return V, P, Q, Assumptions.from_diagram(g)


def _bow_with_inert(n: int, *, seed: int = 1):
    """X→Y with X↔Y plus (n-2) isolated inert nodes (dropped by ancestral)."""
    if n < 2:
        raise ValueError(n)
    names = ["X", "Y"] + [f"Z{i}" for i in range(n - 2)]
    V = Variables.from_binary(*names)
    g = CausalDiagram.from_edges(
        names, directed=[("X", "Y")], bidirected=[("X", "Y")]
    )
    rng = np.random.default_rng(seed)
    cells = V.cells(names)
    # RE1 on XY margin, uniform on Z
    base = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    mid = names[2:]
    P = {c: 0.0 for c in cells}
    if not mid:
        P = {(x, y): p for (x, y), p in base.items()}
    else:
        mcells = V.cells(mid)
        share = 1.0 / len(mcells)
        for (x, y), p in base.items():
            for mc in mcells:
                P[(x, y, *mc)] = p * share
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    return V, P, Q, Assumptions.from_diagram(g)


def _iv_n_instruments(n_inst: int, *, seed: int = 2):
    if n_inst == 1:
        return binary_iv(seed=seed)
    z_names = [f"Z{i}" for i in range(n_inst)]
    names = z_names + ["X", "Y"]
    V = Variables.from_binary(*names)
    directed = [(z, "X") for z in z_names] + [("X", "Y")]
    g = CausalDiagram.from_edges(
        names, directed=directed, bidirected=[("X", "Y")]
    )
    from partialid.problems import _iv_data_from_scm

    data = _iv_data_from_scm(V, g, tuple(z_names), ("X", "Y"), seed=seed)
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    from partialid.problems import Problem

    return Problem(
        name=f"iv_ninst{n_inst}",
        source="scaled instruments",
        variables=V,
        data=data,
        query=Q,
        assumptions=Assumptions.from_diagram(g),
    )


def _run_scale_family(
    *,
    suite: str,
    family: str,
    structure: str,
    sizes: list[int],
    methods: tuple[str, ...],
    make_problem,
    size_key: str,  # "n"
    timeout_s: float,
    frontier: Frontier,
    include_naive: bool = False,
    expected_tight: bool | None = None,
    expect_method_prefix: str | None = None,
) -> list[BenchRow]:
    rows: list[BenchRow] = []
    stopped: set[str] = set()
    if include_naive:
        methods = tuple(methods) + ("naive",)

    for size in sizes:
        try:
            obj = make_problem(size)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                BenchRow(
                    suite=suite,
                    name=f"{family}_n{size}__build",
                    family=family,
                    structure=structure,
                    n=size,
                    check="fail",
                    check_message=str(exc),
                    error=str(exc),
                )
            )
            break

        if hasattr(obj, "variables"):
            V, P, Q, a = obj.variables, obj.data, obj.query, obj.assumptions
        else:
            V, P, Q, a = obj

        for method in methods:
            if method in stopped:
                rows.append(
                    BenchRow(
                        suite=suite,
                        name=f"{family}_n{size}__{method}",
                        family=family,
                        structure=structure,
                        n=size,
                        method=method,
                        check="skip",
                        check_message="stopped at smaller scale",
                    )
                )
                continue

            optimize = method != "naive"
            meth = "lp" if method == "naive" else method
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result, med, err = run_bound(
                    V, P, Q, a, method=meth, optimize=optimize, repeats=1
                )

            stop, reason = scale_stop(
                success=bool(result and result.success),
                median_s=med if med == med else float("inf"),
                timeout_s=timeout_s,
                error=err,
            )
            check = "pass"
            msg = ""
            if err or not (result and result.success):
                check = "fail" if not stop else "fail"
                msg = err or (result.message if result else "fail")
            elif stop:
                check = "fail"
                msg = reason
            else:
                if expected_tight is True and result and not result.tight:
                    # ID / degree1 may still be tight; warn only for order families
                    if family in ("order_chain", "bow_inert", "multi_treatment"):
                        check = "fail"
                        msg = "expected tight"
                if (
                    expect_method_prefix
                    and method == "auto"
                    and result
                    and not result.method.startswith(expect_method_prefix)
                ):
                    check = "fail"
                    msg = f"tag {result.method} ≠ prefix {expect_method_prefix}"

            row = row_from_bound(
                suite=suite,
                name=f"{family}_n{size}__{method}",
                result=result,
                median_s=med,
                method=method,
                error=err,
                family=family,
                structure=structure,
                check=check,
                check_message=msg,
                extra={"timeout_s": timeout_s},
            )
            row.n = size
            rows.append(row)
            frontier.update(
                family,
                method,
                n=size,
                median_ms=row.median_ms,
                success=(check == "pass"),
            )
            status = {"pass": "OK", "fail": "FAIL", "skip": "SKIP"}[check]
            ms = f"{row.median_ms:.1f}" if row.median_ms is not None else "—"
            tag = result.method if result else "—"
            print(
                f"  [{status}] {family} n={size} {method:8s} "
                f"tag={tag:12s} {ms:>8s}ms {msg}"
            )
            if stop:
                stopped.add(method)
    return rows


def run_suite_d(*, quick: bool = True) -> list[BenchRow]:
    cfg = _schedules(quick=quick)
    frontier = Frontier()
    rows: list[BenchRow] = []
    timeout = cfg["timeout_s"]

    print("  -- order-only chains --")
    rows.extend(
        _run_scale_family(
            suite="D",
            family="order_chain",
            structure="order_chain",
            sizes=cfg["order_chain_ns"],
            methods=cfg["methods_order"],
            make_problem=lambda n: chain_order_only(n, 2, seed=7),
            size_key="n",
            timeout_s=timeout,
            frontier=frontier,
            include_naive=cfg["include_naive"],
            expected_tight=True,
            expect_method_prefix="fiber",
        )
    )

    print("  -- Markov chains (ID) --")
    rows.extend(
        _run_scale_family(
            suite="D",
            family="markov_chain",
            structure="markov",
            sizes=cfg["markov_ns"],
            methods=("auto",),
            make_problem=lambda n: _markov_chain(n),
            size_key="n",
            timeout_s=timeout,
            frontier=frontier,
            expect_method_prefix="id/",
        )
    )

    print("  -- bow + inert nodes --")
    rows.extend(
        _run_scale_family(
            suite="D",
            family="bow_inert",
            structure="bow_inert",
            sizes=cfg["bow_inert_ns"],
            methods=("auto", "fiber", "lp"),
            make_problem=lambda n: _bow_with_inert(n),
            size_key="n",
            timeout_s=timeout,
            frontier=frontier,
            expected_tight=True,
            expect_method_prefix="fiber",
        )
    )

    print("  -- multi-treatment --")
    rows.extend(
        _run_scale_family(
            suite="D",
            family="multi_treatment",
            structure="multi_cause",
            sizes=cfg["multi_Ts"],
            methods=cfg["methods_multi"],
            make_problem=lambda t: shridharan_multi_treatment(t, 2, seed=6),
            size_key="n",
            timeout_s=timeout,
            frontier=frontier,
            expected_tight=True,
        )
    )

    print("  -- IV #instruments --")
    rows.extend(
        _run_scale_family(
            suite="D",
            family="iv_instruments",
            structure="iv",
            sizes=cfg["n_instruments"],
            methods=cfg["methods_iv"],
            make_problem=lambda k: _iv_n_instruments(k),
            size_key="n",
            timeout_s=timeout,
            frontier=frontier,
            expected_tight=True,
            expect_method_prefix="lp",
        )
    )

    rows.extend(frontier.as_rows("D"))
    print("  -- frontiers --")
    for r in frontier.as_rows("D"):
        print(f"  frontier {r.family}/{r.method}: n={r.n} ({r.median_ms:.1f}ms)")
    return rows


def main(quick: bool = True) -> int:
    print(f"Suite D (scale n) quick={quick}")
    rows = run_suite_d(quick=quick)
    meta = machine_meta()
    meta["suite"] = "D"
    meta["quick"] = quick
    out = RESULTS_DIR / "suite_D.jsonl"
    write_jsonl(out, rows, meta=meta)
    md = RESULTS_DIR / "suite_D_summary.md"
    write_summary_md(md, rows, title="Suite D — scale n")
    # Scaling suites: failures that are intentional timeouts still count as
    # recorded; only unexpected build errors fail the exit code hard.
    # Treat check=fail from timeout as non-fatal for exit if at least one
    # success exists per family — report count only.
    n_fail = sum(1 for r in rows if r.check == "fail" and "timeout" not in (r.check_message or "") and "stopped" not in (r.check_message or ""))
    # For CI quick: fail only if a small-n case fails
    hard = [
        r
        for r in rows
        if r.check == "fail"
        and r.n is not None
        and r.n <= (4 if quick else 3)
        and "frontier" not in r.name
    ]
    print(f"Wrote {out} and {md} ({len(hard)} hard failures at small n)")
    return 0 if not hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
