"""Suite E — scale cardinality (topology fixed)."""

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
    sachs_ternary_cardinalities,
    sachs_two_instruments_card,
)


def _schedules(*, quick: bool) -> dict:
    if quick:
        return {
            "timeout_s": 8.0,
            "xy_cards": [2, 3, 4],
            "iv_z_cards": [2, 3, 4],
            "two_iv_z_cards": [2, 3],
            "chain_cards": [2, 3],
            "methods_xy": ("auto", "fiber", "lp"),
            "methods_iv": ("auto", "lp"),
            "methods_chain": ("auto", "fiber"),
            "include_naive": False,
            "include_c_canonical": True,
        }
    return {
        "timeout_s": 45.0,
        "xy_cards": [2, 3, 4, 5],
        "iv_z_cards": [2, 3, 4, 5, 6],
        "two_iv_z_cards": [2, 3, 4],
        "chain_cards": [2, 3, 4],
        "methods_xy": ("auto", "fiber", "lp", "colgen"),
        "methods_iv": ("auto", "lp"),
        "methods_chain": ("auto", "fiber", "lp"),
        "include_naive": True,
        "include_c_canonical": True,
    }


def _confounded_xy(card: int, *, seed: int = 0):
    """Order-only / bow-equivalent: X,Y with |Ω|=card, Manski width = 1-P(X=do)."""
    from partialid.problems import _softmax_table

    V = Variables.from_mapping({"X": tuple(range(card)), "Y": tuple(range(card))})
    keys = V.cells(["X", "Y"])
    raw = _softmax_table(keys, seed=seed)
    data = {k: max(v, 1e-4) for k, v in raw.items()}
    s = sum(data.values())
    data = {k: v / s for k, v in data.items()}
    do_x = 1 if card > 1 else 0
    val_y = 1 if card > 1 else 0
    px = sum(v for (x, y), v in data.items() if x == do_x)
    Q = Query.interventional(outcome="Y", do={"X": do_x}, value=val_y)
    from partialid.problems import Problem

    return Problem(
        name=f"xy_card{card}",
        source="confounded XY card sweep",
        variables=V,
        data=data,
        query=Q,
        assumptions=Assumptions.order("X", "Y"),
        expected_width=1.0 - px,
    )


def _iv_z_card(card_z: int, *, seed: int = 2):
    """Single instrument Z with |Z|=card_z, binary X,Y."""
    V = Variables.from_mapping(
        {"Z": tuple(range(card_z)), "X": (0, 1), "Y": (0, 1)}
    )
    g = CausalDiagram.from_edges(
        ["Z", "X", "Y"],
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
    )
    from partialid.problems import Problem, _iv_data_from_scm

    data = _iv_data_from_scm(V, g, ("Z",), ("X", "Y"), seed=seed)
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    return Problem(
        name=f"iv_Z{card_z}",
        source="IV Z-cardinality",
        variables=V,
        data=data,
        query=Q,
        assumptions=Assumptions.from_diagram(g),
    )


def _run_card_family(
    *,
    suite: str,
    family: str,
    structure: str,
    cards: list[int],
    methods: tuple[str, ...],
    make_problem,
    timeout_s: float,
    frontier: Frontier,
    include_naive: bool = False,
    check_width: bool = False,
) -> list[BenchRow]:
    rows: list[BenchRow] = []
    stopped: set[str] = set()
    meths = list(methods)
    if include_naive:
        meths.append("naive")

    for card in cards:
        try:
            prob = make_problem(card)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                BenchRow(
                    suite=suite,
                    name=f"{family}_c{card}__build",
                    family=family,
                    structure=structure,
                    card=card,
                    check="fail",
                    check_message=str(exc),
                    error=str(exc),
                )
            )
            break

        V, P, Q, a = prob.variables, prob.data, prob.query, prob.assumptions
        for method in meths:
            if method in stopped:
                rows.append(
                    BenchRow(
                        suite=suite,
                        name=f"{family}_c{card}__{method}",
                        family=family,
                        structure=structure,
                        card=card,
                        method=method,
                        check="skip",
                        check_message="stopped at smaller card",
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
                check = "fail"
                msg = err or (result.message if result else "fail")
            elif stop:
                check = "fail"
                msg = reason
            elif (
                check_width
                and method in ("auto", "fiber", "lp")
                and prob.expected_width is not None
                and result is not None
            ):
                if abs(result.width - prob.expected_width) > 1e-4:
                    check = "fail"
                    msg = (
                        f"width {result.width} != expected {prob.expected_width}"
                    )

            row = row_from_bound(
                suite=suite,
                name=f"{family}_c{card}__{method}",
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
            row.card = card
            rows.append(row)
            frontier.update(
                family,
                method,
                card=card,
                median_ms=row.median_ms,
                success=(check == "pass"),
            )
            status = {"pass": "OK", "fail": "FAIL", "skip": "SKIP"}[check]
            ms = f"{row.median_ms:.1f}" if row.median_ms is not None else "—"
            tag = result.method if result else "—"
            print(
                f"  [{status}] {family} card={card} {method:10s} "
                f"tag={tag:12s} {ms:>8s}ms {msg}"
            )
            if stop:
                stopped.add(method)
    return rows


def run_suite_e(*, quick: bool = True) -> list[BenchRow]:
    cfg = _schedules(quick=quick)
    frontier = Frontier()
    rows: list[BenchRow] = []
    timeout = cfg["timeout_s"]

    print("  -- confounded XY card --")
    rows.extend(
        _run_card_family(
            suite="E",
            family="xy_card",
            structure="order_xy",
            cards=cfg["xy_cards"],
            methods=cfg["methods_xy"],
            make_problem=lambda c: _confounded_xy(c),
            timeout_s=timeout,
            frontier=frontier,
            include_naive=cfg["include_naive"],
            check_width=True,
        )
    )

    if cfg["include_c_canonical"]:
        print("  -- XY c_canonical --")
        rows.extend(
            _run_card_family(
                suite="E",
                family="xy_c_canonical",
                structure="order_xy",
                cards=cfg["xy_cards"],
                methods=("c_canonical",),
                make_problem=lambda c: _confounded_xy(c, seed=1),
                timeout_s=timeout,
                frontier=frontier,
                check_width=False,
            )
        )

    print("  -- IV |Z|=card --")
    rows.extend(
        _run_card_family(
            suite="E",
            family="iv_z_card",
            structure="iv",
            cards=cfg["iv_z_cards"],
            methods=cfg["methods_iv"],
            make_problem=lambda c: _iv_z_card(c),
            timeout_s=timeout,
            frontier=frontier,
        )
    )

    print("  -- two IV |Z|=card --")
    rows.extend(
        _run_card_family(
            suite="E",
            family="two_iv_z_card",
            structure="two_iv",
            cards=cfg["two_iv_z_cards"],
            methods=cfg["methods_iv"],
            make_problem=lambda c: sachs_two_instruments_card(card_z=c, seed=4),
            timeout_s=timeout,
            frontier=frontier,
        )
    )

    print("  -- chain n=3 card --")
    rows.extend(
        _run_card_family(
            suite="E",
            family="chain3_card",
            structure="order_chain",
            cards=cfg["chain_cards"],
            methods=cfg["methods_chain"],
            make_problem=lambda c: chain_order_only(3, c, seed=7),
            timeout_s=timeout,
            frontier=frontier,
            check_width=True,
        )
    )

    # Optional literature ternary exposure cards in full mode
    if not quick:
        print("  -- Sachs ternary card_x --")
        rows.extend(
            _run_card_family(
                suite="E",
                family="sachs_ternary_x",
                structure="exposure",
                cards=[3, 4, 5],
                methods=("auto", "fiber", "lp"),
                make_problem=lambda c: sachs_ternary_cardinalities(c, 2, seed=1),
                timeout_s=timeout,
                frontier=frontier,
            )
        )

    rows.extend(frontier.as_rows("E"))
    print("  -- frontiers --")
    for r in frontier.as_rows("E"):
        print(
            f"  frontier {r.family}/{r.method}: card={r.card} ({r.median_ms:.1f}ms)"
        )
    return rows


def main(quick: bool = True) -> int:
    print(f"Suite E (scale card) quick={quick}")
    rows = run_suite_e(quick=quick)
    meta = machine_meta()
    meta["suite"] = "E"
    meta["quick"] = quick
    out = RESULTS_DIR / "suite_E.jsonl"
    write_jsonl(out, rows, meta=meta)
    md = RESULTS_DIR / "suite_E_summary.md"
    write_summary_md(md, rows, title="Suite E — scale cardinality")
    hard = [
        r
        for r in rows
        if r.check == "fail"
        and r.card is not None
        and r.card <= 2
        and "frontier" not in r.name
        and "timeout" not in (r.check_message or "")
    ]
    print(f"Wrote {out} and {md} ({len(hard)} hard failures at card≤2)")
    return 0 if not hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
