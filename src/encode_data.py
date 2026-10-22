from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "src_port", "dst_port", "duration", "src_bytes", "dst_bytes",
    "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts",
    "dst_ip_bytes", "dns_qclass", "dns_qtype", "dns_rcode",
    "http_request_body_len", "http_response_body_len", "http_status_code",
]

CATEGORICAL_FEATURES = [
    "proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA",
    "dns_rejected", "ssl_version", "ssl_cipher", "ssl_resumed",
    "ssl_established", "http_trans_depth", "http_method", "http_version",
    "http_orig_mime_types", "http_resp_mime_types", "weird_name",
    "weird_notice",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def save_encoded(
    data: pd.DataFrame,
    preprocessor: ColumnTransformer,
    label_map: dict[str, int],
    output_path: Path,
) -> None:
    x = preprocessor.transform(data[FEATURES]).astype(np.float32)
    y_binary = data["label"].to_numpy(dtype=np.int64)
    y_multiclass = data["type"].map(label_map).to_numpy(dtype=np.int64)

    np.savez_compressed(
        output_path,
        x=x,
        y_binary=y_binary,
        y_multiclass=y_multiclass,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--clients-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoded_clients_dir = args.output_dir / "clients"
    encoded_clients_dir.mkdir()

    pool = pd.read_csv(args.pool, low_memory=False)
    validation = pd.read_csv(args.validation, low_memory=False)
    test = pd.read_csv(args.test, low_memory=False)

    numeric_pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20,
                sparse_output=False,
            ),
            CATEGORICAL_FEATURES,
        ),
    ])

    preprocessor.fit(pool[FEATURES])

    class_names = sorted(pool["type"].unique())
    label_map = {name: index for index, name in enumerate(class_names)}

    save_encoded(
        validation,
        preprocessor,
        label_map,
        args.output_dir / "server_validation.npz",
    )
    save_encoded(
        test,
        preprocessor,
        label_map,
        args.output_dir / "test.npz",
    )

    client_files = sorted(args.clients_dir.glob("client_*.csv"))
    for client_file in client_files:
        client_data = pd.read_csv(client_file, low_memory=False)
        save_encoded(
            client_data,
            preprocessor,
            label_map,
            encoded_clients_dir / f"{client_file.stem}.npz",
        )

    joblib.dump(preprocessor, args.output_dir / "preprocessor.joblib")

    report = {
        "input_feature_count": len(FEATURES),
        "encoded_feature_count": int(
            len(preprocessor.get_feature_names_out())
        ),
        "multiclass_label_map": label_map,
        "client_files_encoded": len(client_files),
        "feature_names": preprocessor.get_feature_names_out().tolist(),
    }

    (args.output_dir / "encoding_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Encoded input features: {len(FEATURES)}")
    print(f"Encoded model features: {report['encoded_feature_count']}")
    print(f"Encoded clients: {len(client_files)}")
    print(f"Saved encoded data to: {args.output_dir}")


if __name__ == "__main__":
    main()