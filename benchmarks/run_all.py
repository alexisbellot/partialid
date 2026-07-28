#!/usr/bin/env python3
"""Unified benchmark entrypoint.

Examples
--------
  PYTHONPATH=src python benchmarks/run_all.py --suite A --quick
  PYTHONPATH=src python benchmarks/run_all.py --suite F G --quick
  PYTHONPATH=src python benchmarks/run_all.py --suite J --quick
  PYTHONPATH=src python benchmarks/run_all.py --suite all --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="partialid benchmark suites")
    p.add_argument(
        "--suite",
        nargs="+",
        default=["A"],
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "all"],
        help="Suites to run (default: A). J=Track A4 method Pareto.",
    )
    p.add_argument(
        "--quick",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Smaller grids suitable for CI (default: true)",
    )
    p.add_argument(
        "--lite",
        action="store_true",
        help="Laptop-safe: for Suite I, Ex F only (no heavy |H| enumeration)",
    )
    args = p.parse_args(argv)

    suites = list(args.suite)
    if "all" in suites:
        suites = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    runners = {
        "A": "suites.A_correctness",
        "B": "suites.B_structure_matrix",
        "C": "suites.C_sensitivity",
        "D": "suites.D_scale_n",
        "E": "suites.E_scale_card",
        "F": "suites.F_stress",
        "G": "suites.G_naive_compare",
        "H": "suites.H_external",
        "I": "suites.I_shridharan",
        "J": "suites.J_method_pareto",
    }

    exit_code = 0
    for s in suites:
        mod = __import__(runners[s], fromlist=["main"])
        if s == "I":
            exit_code = max(exit_code, mod.main(quick=args.quick, lite=args.lite or args.quick))
        else:
            exit_code = max(exit_code, mod.main(quick=args.quick))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
