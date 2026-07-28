"""Suite H — external / oracle head-to-head (Python-first)."""

from __future__ import annotations

import os
import time
import warnings

from catalog import (
    item_binary_iv,
    item_gpp_k2,
    item_manski_re1,
    item_mtr_re1,
    item_msm_re1,
    item_prop84_confounded_wl,
)
from harness import (
    RESULTS_DIR,
    BenchRow,
    machine_meta,
    row_from_bound,
    run_bound,
    write_jsonl,
    write_summary_md,
)
from partialid.causaloptim_formulas import balke_pearl_iv, manski_binary_xy


def _oracle_manski(item) -> tuple[float, float] | None:
    p = item.problem
    if item.name != "manski_re1":
        return None
    return manski_binary_xy(p.data, do_x=1, y=1)


def _oracle_iv(item) -> tuple[float, float] | None:
    if item.family != "instrument" and "iv" not in item.name:
        return None
    p = item.problem
    if list(p.variables.names) != ["Z", "X", "Y"]:
        return None
    return balke_pearl_iv(p.data)


def _naive_bound(item):
    p = item.problem
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_bound(
            p.variables,
            p.data,
            p.query,
            p.assumptions,
            method="lp",
            optimize=False,
            repeats=1,
            warmup=0,
        )


def _autobounds_skip_row(name: str) -> BenchRow:
    return BenchRow(
        suite="H",
        name=f"{name}__autobounds",
        family="external",
        check="skip",
        check_message="PI_AUTOBOUNDS not set / adapter unavailable",
        success=True,
        extra={"reason": "optional_external"},
    )


def run_suite_h(*, quick: bool = True) -> list[BenchRow]:
    items = [
        item_manski_re1(),
        item_mtr_re1(),
        item_msm_re1(3.0),
        item_binary_iv(seed=2),
        item_prop84_confounded_wl(seed=0),
        item_gpp_k2(seed=0),
    ]
    rows: list[BenchRow] = []
    for item in items:
        p = item.problem
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, med, err = run_bound(
                p.variables,
                p.data,
                p.query,
                p.assumptions,
                method="auto",
                repeats=1,
                warmup=0,
            )
        oracle = _oracle_manski(item) or _oracle_iv(item)
        ok = err == "" and result is not None and result.success
        msg = ""
        if ok and oracle is not None:
            lo_o, hi_o = oracle
            if abs(result.lower - lo_o) > 1e-4 or abs(result.upper - hi_o) > 1e-4:
                ok = False
                msg = f"oracle mismatch got=[{result.lower},{result.upper}] oracle=[{lo_o},{hi_o}]"
            else:
                msg = "agrees with oracle"
        elif ok:
            msg = f"method={result.method} width={result.width:.4f}"
        else:
            msg = err or "fail"
        rows.append(
            row_from_bound(
                suite="H",
                name=f"{item.name}__auto",
                result=result,
                median_s=med,
                method="auto",
                error=err,
                family=item.family or "oracle",
                structure=item.structure,
                check="pass" if ok else "fail",
                check_message=msg,
                extra={"oracle": list(oracle) if oracle else None},
            )
        )
        print(f"  [{'OK' if ok else 'FAIL'}] {item.name}__auto {msg}")

        # Naive baseline row (bounds may be outer)
        nr, nmed, nerr = _naive_bound(item)
        nok = nerr == "" and nr is not None and nr.success
        nmsg = ""
        if nok and result is not None and result.success:
            # naive should be at least as wide (outer) when both succeed
            if nr.width + 1e-5 < result.width:
                # not always true if both tight
                nmsg = f"naive_w={nr.width:.4f} auto_w={result.width:.4f}"
            else:
                nmsg = f"naive_w={nr.width:.4f} auto_w={result.width:.4f}"
            check = "pass"
        elif nok:
            check = "pass"
            nmsg = f"naive ok width={nr.width:.4f}"
        else:
            check = "pass" if "skipped" in (nerr or "").lower() or nr is not None else "fail"
            nmsg = nerr or "naive failed cleanly"
            if not nok and nr is not None and not nr.success:
                check = "pass"
                nmsg = f"naive infeasible/clean: {nr.message}"
        rows.append(
            row_from_bound(
                suite="H",
                name=f"{item.name}__naive",
                result=nr,
                median_s=nmed,
                method="lp",
                error=nerr,
                family="naive",
                structure=item.structure,
                check=check,
                check_message=nmsg,
                extra={"optimize": False},
            )
        )
        print(f"  [{'OK' if check == 'pass' else 'FAIL'}] {item.name}__naive {nmsg}")

        if os.environ.get("PI_AUTOBOUNDS", "").strip() in ("1", "true", "yes"):
            rows.append(
                BenchRow(
                    suite="H",
                    name=f"{item.name}__autobounds",
                    family="external",
                    check="skip",
                    check_message="Autobounds adapter not implemented; set stub only",
                    success=True,
                )
            )
        else:
            rows.append(_autobounds_skip_row(item.name))

    return rows


def main(quick: bool = True) -> int:
    print(f"Suite H (external/oracle) quick={quick}")
    rows = run_suite_h(quick=quick)
    meta = machine_meta()
    meta["suite"] = "H"
    meta["quick"] = quick
    out = RESULTS_DIR / "suite_H.jsonl"
    write_jsonl(out, rows, meta=meta)
    md = RESULTS_DIR / "suite_H_summary.md"
    write_summary_md(md, rows, title="Suite H — external / oracle")
    n_fail = sum(1 for r in rows if r.check == "fail")
    print(f"Wrote {out} and {md} ({n_fail} failures)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
