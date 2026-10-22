from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def allocate_counts(total: int, proportions: np.ndarray) -> np.ndarray:
    raw = proportions * total
    counts = np.floor(raw).astype(int)

    remaining = total - counts.sum()
    if remaining > 0:
        largest_fractional_parts = np.argsort(raw - counts)[-remaining:]
        counts[largest_fractional_parts] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create non-IID federated clients from TON_IoT training data."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Dirichlet concentration. Lower values create stronger non-IID clients.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {args.output_dir}. "
            "Choose a new directory instead."
        )

    if args.clients < 2:
        raise SystemExit("Use at least two clients.")

    if args.alpha <= 0:
        raise SystemExit("Alpha must be greater than zero.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, low_memory=False)
    if "type" not in data.columns or "label" not in data.columns:
        raise SystemExit("Input must contain both 'label' and 'type' columns.")

    rng = np.random.default_rng(args.seed)
    client_indices = [[] for _ in range(args.clients)]

    for attack_type in sorted(data["type"].unique()):
        class_indices = data.index[data["type"] == attack_type].to_numpy(copy=True)
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(np.full(args.clients, args.alpha))
        counts = allocate_counts(len(class_indices), proportions)

        start = 0
        for client_id, count in enumerate(counts):
            end = start + count
            client_indices[client_id].extend(class_indices[start:end])
            start = end

    manifest_clients = []

    for client_id, indices in enumerate(client_indices):
        client_data = data.loc[indices].sample(
            frac=1,
            random_state=args.seed + client_id,
        )

        filename = f"client_{client_id:02d}.csv"
        client_data.to_csv(args.output_dir / filename, index=False)

        type_counts = {
            str(name): int(count)
            for name, count in client_data["type"].value_counts().sort_index().items()
        }

        manifest_clients.append(
            {
                "client_id": client_id,
                "file": filename,
                "rows": len(client_data),
                "attack_type_distribution": type_counts,
                "attack_rows": int((client_data["label"] == 1).sum()),
                "normal_rows": int((client_data["label"] == 0).sum()),
            }
        )

    manifest = {
        "source_file": str(args.input),
        "number_of_clients": args.clients,
        "partition_method": "label-wise Dirichlet non-IID split",
        "dirichlet_alpha": args.alpha,
        "random_seed": args.seed,
        "clients": manifest_clients,
    }

    (args.output_dir / "client_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Created {args.clients} non-IID federated clients.")
    for client in manifest_clients:
        print(
            f"Client {client['client_id']:02d}: "
            f"{client['rows']:,} rows | "
            f"{len(client['attack_type_distribution'])} attack classes"
        )
    print(f"Manifest saved to: {args.output_dir / 'client_manifest.json'}")


if __name__ == "__main__":
    main()