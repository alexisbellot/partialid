"""CSV / table → MSM bounds in a few lines.

Uses the bundled counts file by default. With pandas::

    data = from_dataframe(pd.read_csv(path), columns=["X", "Y"])
"""

from __future__ import annotations

from pathlib import Path

from partialid import Assumptions, Query, bound, bound_path, from_counts, from_records

HERE = Path(__file__).resolve().parent
CSV = HERE / "data" / "xy_counts.csv"


def _load_without_pandas(path: Path):
    lines = path.read_text().strip().splitlines()
    header = lines[0].split(",")
    rows = [tuple(int(x) for x in line.split(",")) for line in lines[1:]]
    return from_records(header, rows)


def main() -> None:
    try:
        import pandas as pd

        from partialid import from_dataframe

        data = from_dataframe(pd.read_csv(CSV), columns=["X", "Y"])
        print("loaded via from_dataframe", data.notes)
    except ImportError:
        data = _load_without_pandas(CSV)
        print("loaded via from_records (pandas not installed)")

    # Equivalent aggregated table (RE1 proportions × 10)
    _ = from_counts(
        ("X", "Y"),
        {(0, 0): 3, (0, 1): 2, (1, 0): 1.5, (1, 1): 3.5},
    )

    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r = bound(data.variables, data.P, Q, Assumptions.sensitivity_msm("X", "Y", 3.0))
    print(r.summary(Q))

    path = bound_path(
        data.variables,
        data.P,
        Q,
        Assumptions.manski("X", "Y"),
        param="Lambda",
        theta_min=1.0,
        theta_max=5.0,
    )
    print(path.summary(at=3.0))


if __name__ == "__main__":
    main()
