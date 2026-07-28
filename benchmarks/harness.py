"""Shared benchmark harness: timing, timeouts, JSONL schema."""

from __future__ import annotations

import json
import platform
import statistics
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from partialid import BoundResult, bound

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = 1


@dataclass
class BenchRow:
    """Uniform result row for all suites."""

    suite: str
    name: str
    family: str = ""
    n: int | None = None
    card: int | None = None
    structure: str = ""
    assumptions: str = ""
    method: str = "auto"
    n_types_est: int | None = None
    n_types_used: int | None = None
    formulate_ms: float | None = None
    solve_ms: float | None = None
    total_ms: float | None = None
    median_ms: float | None = None
    lower: float | None = None
    upper: float | None = None
    width: float | None = None
    tight: bool | None = None
    success: bool = False
    fallback: str = ""
    method_tag: str = ""
    notes: list[str] = field(default_factory=list)
    seed: int | None = None
    check: str = ""  # "pass" | "fail" | "skip"
    check_message: str = ""
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # drop empty extra
        if not d["extra"]:
            del d["extra"]
        return d


def machine_meta() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import highspy  # noqa: F401

        meta["highspy"] = True
    except ImportError:
        meta["highspy"] = False
    try:
        import cvxpy  # noqa: F401

        meta["cvxpy"] = True
    except ImportError:
        meta["cvxpy"] = False
    return meta


def median_time(
    fn: Callable[[], Any],
    *,
    repeats: int = 3,
    warmup: int = 1,
) -> tuple[float, Any]:
    """Return (median_seconds, last_return_value)."""
    out = None
    for _ in range(warmup):
        out = fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def run_bound(
    variables,
    data: Mapping,
    query,
    assumptions,
    *,
    method: str = "auto",
    optimize: bool = True,
    recover_primal: bool = True,
    repeats: int = 1,
    warmup: int = 0,
) -> tuple[BoundResult | None, float, str]:
    """Time ``bound``; return (result_or_None, median_s, error)."""

    def _call():
        return bound(
            variables,
            data,
            query,
            assumptions,
            method=method,  # type: ignore[arg-type]
            optimize=optimize,
            recover_primal=recover_primal,
        )

    try:
        med, result = median_time(_call, repeats=repeats, warmup=warmup)
        return result, med, ""
    except Exception as exc:  # noqa: BLE001
        return None, float("nan"), f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}"


@dataclass
class Frontier:
    """Track largest successful scale per (family, method)."""

    records: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    def update(
        self,
        family: str,
        method: str,
        *,
        n: int | None = None,
        card: int | None = None,
        median_ms: float | None = None,
        success: bool,
    ) -> None:
        if not success:
            return
        key = (family, method)
        prev = self.records.get(key, {})
        better = False
        if n is not None and n >= prev.get("n", -1):
            better = True
        if card is not None and card >= prev.get("card", -1):
            better = True
        if better or key not in self.records:
            self.records[key] = {
                "family": family,
                "method": method,
                "n": n,
                "card": card,
                "median_ms": median_ms,
            }

    def as_rows(self, suite: str) -> list[BenchRow]:
        out = []
        for (fam, meth), rec in sorted(self.records.items()):
            out.append(
                BenchRow(
                    suite=suite,
                    name=f"frontier__{fam}__{meth}",
                    family=fam,
                    method=meth,
                    n=rec.get("n"),
                    card=rec.get("card"),
                    median_ms=rec.get("median_ms"),
                    success=True,
                    check="pass",
                    extra={"frontier": True},
                )
            )
        return out


def scale_stop(
    *,
    success: bool,
    median_s: float,
    timeout_s: float,
    error: str,
) -> tuple[bool, str]:
    """Return (should_stop_scaling, reason)."""
    if error:
        return True, f"error: {error.splitlines()[0][:120]}"
    if not success:
        return True, "unsuccessful solve"
    if median_s == median_s and median_s > timeout_s:
        return True, f"timeout ({median_s:.2f}s > {timeout_s}s)"
    return False, ""


def row_from_bound(
    *,
    suite: str,
    name: str,
    result: BoundResult | None,
    median_s: float,
    method: str = "auto",
    error: str = "",
    family: str = "",
    structure: str = "",
    assumptions: str = "",
    seed: int | None = None,
    check: str = "",
    check_message: str = "",
    extra: dict | None = None,
) -> BenchRow:
    if result is None:
        return BenchRow(
            suite=suite,
            name=name,
            family=family,
            structure=structure,
            assumptions=assumptions,
            method=method,
            median_ms=median_s * 1000 if median_s == median_s else None,
            success=False,
            check=check or "fail",
            check_message=check_message or error,
            error=error,
            seed=seed,
            extra=extra or {},
        )
    notes = list(result.notes)
    fallback = ""
    if "gpp_infeasible_outer_fallback" in notes:
        fallback = "gpp_outer"
    elif "bounds_may_be_outer" in notes:
        fallback = "outer"
    return BenchRow(
        suite=suite,
        name=name,
        family=family,
        structure=structure,
        assumptions=assumptions,
        method=method,
        formulate_ms=(result.formulate_time or 0.0) * 1000,
        solve_ms=(result.solve_time or 0.0) * 1000,
        total_ms=((result.formulate_time or 0.0) + (result.solve_time or 0.0)) * 1000,
        median_ms=median_s * 1000,
        lower=float(result.lower) if result.success else None,
        upper=float(result.upper) if result.success else None,
        width=float(result.width) if result.success else None,
        tight=bool(result.tight),
        success=bool(result.success),
        fallback=fallback,
        method_tag=result.method,
        notes=notes,
        seed=seed,
        check=check,
        check_message=check_message,
        error=error,
        extra=extra or {},
    )


def write_jsonl(path: Path, rows: Sequence[BenchRow], *, meta: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        header = meta if meta is not None else machine_meta()
        f.write(json.dumps({"_meta": header}) + "\n")
        for row in rows:
            f.write(json.dumps(row.to_dict()) + "\n")
    return path


def write_summary_md(path: Path, rows: Sequence[BenchRow], *, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"Rows: {len(rows)}", ""]
    n_pass = sum(1 for r in rows if r.check == "pass")
    n_fail = sum(1 for r in rows if r.check == "fail")
    n_skip = sum(1 for r in rows if r.check == "skip")
    lines.append(f"- pass: **{n_pass}**")
    lines.append(f"- fail: **{n_fail}**")
    lines.append(f"- skip: **{n_skip}**")
    lines.append("")
    lines.append("| name | method | tag | width | ms | check |")
    lines.append("|---|---|---|---:|---:|---|")
    for r in rows:
        w = f"{r.width:.4f}" if r.width is not None else "—"
        ms = f"{r.median_ms:.2f}" if r.median_ms is not None else "—"
        lines.append(
            f"| {r.name} | {r.method} | {r.method_tag or '—'} | {w} | {ms} | "
            f"{r.check or '—'} |"
        )
        if r.check == "fail" and r.check_message:
            lines.append(f"| | | | | | _{r.check_message}_ |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def assert_close(
    got: float | None,
    expected: float,
    *,
    atol: float = 1e-6,
    label: str = "value",
) -> tuple[bool, str]:
    if got is None or got != got:
        return False, f"{label}: got {got}"
    if abs(got - expected) > atol:
        return False, f"{label}: got {got}, expected {expected} (atol={atol})"
    return True, ""


def assert_interval_contains(
    lo: float | None,
    hi: float | None,
    truth: float,
    *,
    atol: float = 1e-5,
) -> tuple[bool, str]:
    if lo is None or hi is None:
        return False, f"missing interval [{lo},{hi}]"
    if lo - atol > truth or hi + atol < truth:
        return False, f"truth {truth} not in [{lo},{hi}]"
    return True, ""
