"""Suite A — correctness anchors for promised functionality."""

from __future__ import annotations

import warnings

from catalog import CatalogItem, suite_a_items
from harness import (
    RESULTS_DIR,
    assert_close,
    assert_interval_contains,
    machine_meta,
    row_from_bound,
    run_bound,
    write_jsonl,
    write_summary_md,
    BenchRow,
)


def _check_item(item: CatalogItem, result, error: str) -> tuple[str, str]:
    """Return (check, message)."""
    if error:
        return "fail", error
    if result is None or not result.success:
        return "fail", getattr(result, "message", "no result")

    msgs: list[str] = []
    ok = True
    atol = item.atol

    if item.expected_lower is not None:
        good, msg = assert_close(
            result.lower, item.expected_lower, atol=atol, label="lower"
        )
        ok &= good
        if msg:
            msgs.append(msg)
    if item.expected_upper is not None:
        good, msg = assert_close(
            result.upper, item.expected_upper, atol=atol, label="upper"
        )
        ok &= good
        if msg:
            msgs.append(msg)
    if item.expected_width is not None:
        good, msg = assert_close(
            result.width, item.expected_width, atol=atol, label="width"
        )
        ok &= good
        if msg:
            msgs.append(msg)
    if item.expected_method_prefix is not None:
        tag = result.method or ""
        if not tag.startswith(item.expected_method_prefix):
            ok = False
            msgs.append(
                f"method_tag {tag!r} does not start with "
                f"{item.expected_method_prefix!r}"
            )
    if item.expect_tight is not None and bool(result.tight) != item.expect_tight:
        ok = False
        msgs.append(f"tight={result.tight}, expected {item.expect_tight}")
    if item.truth is not None:
        good, msg = assert_interval_contains(
            result.lower, result.upper, item.truth, atol=atol
        )
        ok &= good
        if msg:
            msgs.append(msg)
    # IV / instrument: notes should mention instrument when not ID
    if item.family == "instrument" and not result.method.startswith("id/"):
        blob = " ".join(result.notes)
        if "instrument LP" not in blob and "instrument" not in blob.lower():
            # still ok if tight Balke–Pearl via lp
            if not result.tight:
                ok = False
                msgs.append("instrument instance not tight / missing instrument note")

    # GPP vs outer: if gpp, width should be finite and contain truth (above)
    if item.family == "gpp" and result.method == "gpp":
        from partialid.formulate import formulate
        from partialid.backends import solve_bounds_lp

        p = item.problem
        lp_outer = formulate(
            p.variables, p.data, p.query, p.assumptions, prefer_gpp=False
        )
        lo_r, hi_r = solve_bounds_lp(lp_outer)
        if lo_r.success and hi_r.success:
            if result.lower + 1e-4 < lo_r.value or result.upper - 1e-4 > hi_r.value:
                ok = False
                msgs.append(
                    f"GPP [{result.lower},{result.upper}] not inside outer "
                    f"[{lo_r.value},{hi_r.value}]"
                )

    return ("pass" if ok else "fail"), "; ".join(msgs)


def run_suite_a(
    *,
    quick: bool = True,
    repeats: int = 1,
) -> list[BenchRow]:
    items = suite_a_items(quick=quick)
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
                method=item.method,
                repeats=repeats,
                warmup=0,
            )
        check, msg = _check_item(item, result, err)
        assum = ", ".join(p.assumptions.describe()) if hasattr(p.assumptions, "describe") else ""
        rows.append(
            row_from_bound(
                suite="A",
                name=item.name,
                result=result,
                median_s=med,
                method=item.method,
                error=err,
                family=item.family,
                structure=item.structure,
                assumptions=assum,
                seed=(item.extra or {}).get("seed") if item.extra else None,
                check=check,
                check_message=msg,
                extra=dict(item.extra or {}),
            )
        )
        status = "OK" if check == "pass" else "FAIL"
        tag = result.method if result is not None else "—"
        w = f"{result.width:.4f}" if result is not None and result.success else "—"
        print(f"  [{status}] {item.name:28s} tag={tag:16s} width={w:8s} {msg}")
    return rows


def main(quick: bool = True) -> int:
    print(f"Suite A (correctness) quick={quick}")
    rows = run_suite_a(quick=quick)
    meta = machine_meta()
    meta["suite"] = "A"
    meta["quick"] = quick
    out = RESULTS_DIR / "suite_A.jsonl"
    write_jsonl(out, rows, meta=meta)
    md = RESULTS_DIR / "suite_A_summary.md"
    write_summary_md(md, rows, title="Suite A — correctness")
    n_fail = sum(1 for r in rows if r.check == "fail")
    print(f"Wrote {out} and {md} ({n_fail} failures)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
