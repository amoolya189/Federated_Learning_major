from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGETS = {"label", "type"}
LEAKAGE_COLUMNS = {
    "ts", "timestamp", "time",
    "src_ip", "dst_ip", "srcip", "dstip",
}
CLIENT_COLUMNS = ("src_ip", "srcip", "source_ip", "sourceip")


def normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def class_distribution(data: pd.DataFrame, column: str) -> pd.DataFrame:
    result = (
        data[column]
        .fillna("<missing>")
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
    )
    result.insert(0, "target", column)
    result["percentage"] = (result["count"] / len(data) * 100).round(4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TON_IoT network CSV.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Dataset not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input, low_memory=False)

    normalized = {column: normalize(column) for column in data.columns}
    reverse = {value: key for key, value in normalized.items()}
    normalized_columns = set(normalized.values())

    targets = sorted(TARGETS.intersection(normalized_columns))
    leakage = sorted(LEAKAGE_COLUMNS.intersection(normalized_columns))

    summary = {
        "input_file": str(args.input),
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "column_names": list(data.columns),
        "targets_found": targets,
        "candidate_leakage_columns": leakage,
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_values": {
            column: int(count)
            for column, count in data.isna().sum().items()
            if count > 0
        },
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
    }

    (args.output_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    distributions = [
        class_distribution(data, reverse[target])
        for target in targets
    ]
    if distributions:
        pd.concat(distributions, ignore_index=True).to_csv(
            args.output_dir / "class_distribution.csv",
            index=False,
        )

    client_column = next(
        (column for column in CLIENT_COLUMNS if column in normalized_columns),
        None,
    )
    if client_column:
        original_name = reverse[client_column]
        clients = (
            data[original_name]
            .fillna("<missing>")
            .value_counts(dropna=False)
            .rename_axis("candidate_client")
            .reset_index(name="flows")
        )
        clients["percentage"] = (clients["flows"] / len(data) * 100).round(4)
        clients.to_csv(args.output_dir / "client_candidates.csv", index=False)

    print(f"Audited {len(data):,} rows and {data.shape[1]} columns.")
    print(f"Targets found: {', '.join(targets) if targets else 'none'}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()