from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


NUMERIC_FEATURES = [
    "src_port",
    "dst_port",
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "http_request_body_len",
    "http_response_body_len",
    "http_status_code",
]

CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "conn_state",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected",
    "ssl_version",
    "ssl_cipher",
    "ssl_resumed",
    "ssl_established",
    "http_trans_depth",
    "http_method",
    "http_version",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "weird_name",
    "weird_notice",
]

TARGET_COLUMNS = ["label", "type"]


def distribution(data: pd.DataFrame) -> dict[str, int]:
    return {
        str(name): int(count)
        for name, count in data["type"].value_counts().sort_index().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create duplicate-free TON_IoT train, validation, and test data."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Dataset not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, low_memory=False)
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGET_COLUMNS
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise SystemExit(f"Required columns missing: {missing_columns}")

    original_rows = len(data)
    data = data.drop_duplicates().copy()
    duplicates_removed = original_rows - len(data)

    data["type"] = data["type"].astype(str).str.strip().str.lower()
    data["label"] = data["label"].astype(int)

    selected_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGET_COLUMNS
    data = data[selected_columns]

    train_and_validation, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["type"],
    )

    federated_pool, server_validation = train_test_split(
        train_and_validation,
        test_size=0.125,
        random_state=42,
        stratify=train_and_validation["type"],
    )

    federated_pool.to_csv(args.output_dir / "federated_pool.csv", index=False)
    server_validation.to_csv(args.output_dir / "server_validation.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)

    report = {
        "original_rows": original_rows,
        "duplicates_removed": duplicates_removed,
        "remaining_rows": len(data),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_identity_features": [
            "src_ip",
            "dst_ip",
            "dns_query",
            "ssl_subject",
            "ssl_issuer",
            "http_uri",
            "http_user_agent",
            "weird_addl",
        ],
        "splits": {
            "federated_pool_rows": len(federated_pool),
            "server_validation_rows": len(server_validation),
            "test_rows": len(test),
            "federated_pool_class_distribution": distribution(federated_pool),
            "server_validation_class_distribution": distribution(server_validation),
            "test_class_distribution": distribution(test),
        },
    }

    (args.output_dir / "preparation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Original rows: {original_rows:,}")
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Remaining rows: {len(data):,}")
    print(f"Federated-training pool: {len(federated_pool):,}")
    print(f"Server validation: {len(server_validation):,}")
    print(f"Final test set: {len(test):,}")
    print(f"Saved processed files to: {args.output_dir}")


if __name__ == "__main__":
    main()