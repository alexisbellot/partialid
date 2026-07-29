"""Suite G — naive vs optimized head-to-head (+ method Pareto on small set)."""

from __future__ import annotations

import time
import warnings

from harness import (
    RESULTS_DIR,
    BenchRow,
    machine_meta,
    write_jsonl,
    write_summary_md,
)
from partialid import bound
from partialid.formulate import formulate
from partialid.problems import (
    chain_order_only,
    binary_iv,
    ternary_cardinalities,
    ternary_exposure,
    two_instruments,
    shridharan_multi_treatment,
)

MAX_NAIVE_TYPES = 5000


def _estimate_joint_types(p) -> int:
    from partialid.assumptions import as_assumption_set
    from partialid.graph import CausalDiagram

    a = as_assumption_set(p.assumptions)
    g = a.diagram()
    if g is None:
        g = CausalDiagram.from_order(p.variables.names, maximal_confounding=True)
    n = 1
    for v in g.topological_order():
        parents = list(g.parents(v))
        if not parents:
            n *= p.variables.size(v)
        else:
            in_size = 1
            for pa in parents:
                in_size *= p.variables.size(pa)
            n *= p.variables.size(v) ** in_size
    return int(n)


def _time_bound(p, *, optimize: bool, method: str = "auto") -> dict:
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if not optimize:
                est = _estimate_joint_types(p)
                if est > MAX_NAIVE_TYPES:
                    return {
                        "success": False,
                        "skipped": True,
                        "n_types": est,
                        "error": f"skipped: ~{est} types",
                        "median_ms": None,
                    }
            lp = formulate(
                p.variables, p.data, p.query, p.assumptions, optimize=optimize
            )
            # GPP stub has tiny type_space; use notes instead
            n_types = int(lp.type_space.n_types)
            if lp.method_hint == "gpp":
                n_types = -1
            r = bound(
                p.variables,
                p.data,
                p.query,
                p.assumptions,
                method=method if optimize else "lp",
                optimize=optimize,
            )
            ms = (time.perf_counter() - t0) * 1000
            return {
                "success": bool(r.success),
                "skipped": False,
                "n_types": n_types,
                "median_ms": ms,
                "width": float(r.width) if r.success else None,
                "lower": float(r.lower) if r.success else None,
                "upper": float(r.upper) if r.success else None,
                "method_tag": r.method,
                "tight": bool(r.tight),
                "notes": list(r.notes),
                "error": "" if r.success else r.message,
                "outer": "bounds_may_be_outer" in r.notes,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "skipped": False,
                "n_types": None,
                "median_ms": (time.perf_counter() - t0) * 1000,
                "error": str(exc),
            }


def _problems(*, quick: bool):
    probs = [
        ternary_exposure(),
        ternary_cardinalities(4, 2),
        chain_order_only(3, 2),
        chain_order_only(4, 2),
        binary_iv(seed=2),
        two_instruments(seed=3),
        shridharan_multi_treatment(2, 2),
    ]
    if not quick:
        probs.extend(
            [
                ternary_cardinalities(5, 2, seed=10),
                chain_order_only(3, 3),
                shridharan_multi_treatment(3, 2),
            ]
        )
    return probs


def run_suite_g(*, quick: bool = True) -> list[BenchRow]:
    rows: list[BenchRow] = []
    print("  -- naive vs optimized --")
    print(
        f"  {'problem':36s} {'n_types':>8s} {'opt_t':>8s} {'naive_ms':>10s} "
        f"{'opt_ms':>10s} {'speedup':>8s} {'w_ok':>6s}"
    )
    for p in _problems(quick=quick):
        naive = _time_bound(p, optimize=False)
        opt = _time_bound(p, optimize=True, method="auto")

        speedup = None
        width_ok = None
        check = "pass"
        msg = ""

        if naive.get("skipped"):
            # Optimized must succeed; naive skip is OK (enabling win)
            if not opt.get("success"):
                check = "fail"
                msg = f"opt failed while naive skipped: {opt.get('error')}"
            else:
                msg = f"naive skipped (~{naive.get('n_types')} types); opt ok"
            speedup = float("inf") if opt.get("success") else None
            width_ok = None  # N/A
        elif naive.get("success") and opt.get("success"):
            speedup = naive["median_ms"] / max(opt["median_ms"], 1e-9)
            # Width: instrument cases may disagree (naive outer looser)
            if abs(naive["width"] - opt["width"]) < 1e-4:
                width_ok = True
            elif naive.get("outer") or (naive["width"] + 1e-8 >= opt["width"]):
                # naive outer wider or equal — acceptable
                width_ok = True
                msg = "widths differ (naive outer/valid)"
            else:
                width_ok = False
                check = "fail"
                msg = (
                    f"opt wider than naive? opt={opt['width']} naive={naive['width']}"
                )
            if speedup is not None and speedup < 0.5 and check == "pass":
                # optimized much slower — soft note, still pass
                msg = (msg + "; ").lstrip("; ") + f"slow opt speedup={speedup:.2f}x"
        else:
            check = "fail"
            msg = f"naive={naive.get('error')} opt={opt.get('error')}"

        row = BenchRow(
            suite="G",
            name=f"naive_vs_opt__{p.name}",
            family="naive_compare",
            structure=p.name,
            method="auto",
            n_types_est=naive.get("n_types"),
            n_types_used=opt.get("n_types"),
            median_ms=opt.get("median_ms"),
            width=opt.get("width"),
            lower=opt.get("lower"),
            upper=opt.get("upper"),
            success=bool(opt.get("success")),
            method_tag=opt.get("method_tag", ""),
            tight=opt.get("tight"),
            notes=list(opt.get("notes") or []),
            check=check,
            check_message=msg,
            extra={
                "naive": naive,
                "optimized": opt,
                "speedup": speedup,
                "width_agree": width_ok,
                "source": p.source,
            },
        )
        rows.append(row)
        nt = str(naive.get("n_types"))
        ot = str(opt.get("n_types"))
        nms = f"{naive['median_ms']:.1f}" if naive.get("median_ms") else "skip"
        oms = f"{opt['median_ms']:.1f}" if opt.get("median_ms") else "—"
        sp = f"{speedup:.1f}x" if speedup not in (None, float("inf")) else (
            "∞" if speedup == float("inf") else "—"
        )
        print(
            f"  [{'OK' if check == 'pass' else 'FAIL'}] {p.name:36s} "
            f"{nt:>8s} {ot:>8s} {nms:>10s} {oms:>10s} {sp:>8s} "
            f"{str(width_ok):>6s} {msg}"
        )

    # Method Pareto on RE1-sized / small IV: which method is fastest among successful
    print("  -- method Pareto (small instances) --")
    from catalog import RE1_P
    from partialid import Assumptions, Query, Variables

    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    methods = ["fiber", "lp", "colgen", "hyperarc", "c_canonical"]
    best = None
    for m in methods:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            r = bound(V, RE1_P, Q, a, method=m)
            ms = (time.perf_counter() - t0) * 1000
        ok = r.success and abs(r.width - 0.5) < 1e-6
        rows.append(
            BenchRow(
                suite="G",
                name=f"pareto_re1__{m}",
                family="pareto",
                structure="order_xy",
                method=m,
                median_ms=ms,
                width=float(r.width) if r.success else None,
                success=r.success,
                method_tag=r.method,
                check="pass" if ok else "fail",
                check_message="" if ok else r.message,
            )
        )
        if ok and (best is None or ms < best[1]):
            best = (m, ms)
        print(f"  [{'OK' if ok else 'FAIL'}] pareto RE1 {m:12s} {ms:.2f}ms")
    if best:
        rows.append(
            BenchRow(
                suite="G",
                name="pareto_re1_winner",
                family="pareto",
                method=best[0],
                median_ms=best[1],
                check="pass",
                check_message=f"winner {best[0]} at {best[1]:.2f}ms",
                success=True,
                extra={"winner": best[0], "ms": best[1]},
            )
        )
        print(f"  [OK] RE1 winner: {best[0]} ({best[1]:.2f}ms)")

    return rows


def main(quick: bool = True) -> int:
    print(f"Suite G (naive vs opt) quick={quick}")
    rows = run_suite_g(quick=quick)
    meta = machine_meta()
    meta["suite"] = "G"
    meta["quick"] = quick
    out = RESULTS_DIR / "suite_G.jsonl"
    write_jsonl(out, rows, meta=meta)
    md = RESULTS_DIR / "suite_G_summary.md"
    write_summary_md(md, rows, title="Suite G — naive vs optimized")
    n_fail = sum(1 for r in rows if r.check == "fail")
    print(f"Wrote {out} and {md} ({n_fail} failures)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
