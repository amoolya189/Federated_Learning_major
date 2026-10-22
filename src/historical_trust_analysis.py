
import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_historical_trust(combined_data, decay=0.7):
    """
    Calculate historical trust recursively.

    T_i^r = decay * T_i^(r-1)
            + (1 - decay) * C_i^r

    Initial trust is 0.5 for every client.
    """

    if not 0.0 <= decay <= 1.0:
        raise ValueError("Decay must be between 0 and 1.")

    rounds_data = combined_data["combined_signals"]

    client_ids = sorted(
        {
            str(client_id)
            for clients in rounds_data.values()
            for client_id in clients
        },
        key=int
    )

    trust = {client_id: 0.5 for client_id in client_ids}
    history = {}

    for round_id in sorted(rounds_data, key=int):
        round_results = {}

        for client_id in client_ids:
            if client_id not in rounds_data[round_id]:
                continue

            row = rounds_data[round_id][client_id]
            evidence = float(row["combined_evidence"])

            previous_trust = trust[client_id]

            current_trust = (
                decay * previous_trust
                + (1.0 - decay) * evidence
            )

            # Evidence and initial trust are in [0, 1],
            # so trust should remain in that interval.
            current_trust = float(
                np.clip(current_trust, 0.0, 1.0)
            )

            round_results[client_id] = {
                "client_id": int(client_id),
                "malicious": bool(row["malicious"]),
                "previous_trust": float(previous_trust),
                "combined_evidence": evidence,
                "historical_trust": current_trust
            }

            trust[client_id] = current_trust

        history[round_id] = round_results

    return history


def summarize_trust(history):
    observations = [
        row
        for clients in history.values()
        for row in clients.values()
    ]

    benign = [
        row["historical_trust"]
        for row in observations
        if not row["malicious"]
    ]

    malicious = [
        row["historical_trust"]
        for row in observations
        if row["malicious"]
    ]

    def mean(values):
        return float(np.mean(values)) if values else None

    summary = {
        "observation_count": len(observations),
        "benign_observations": len(benign),
        "malicious_observations": len(malicious),
        "benign_mean_historical_trust": mean(benign),
        "malicious_mean_historical_trust": mean(malicious)
    }

    if benign and malicious:
        summary["benign_minus_malicious_trust_gap"] = (
            mean(benign) - mean(malicious)
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze historical trust from combined FL signals."
    )

    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to combined_trust_signals.json"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for historical trust output"
    )

    parser.add_argument(
        "--decay",
        type=float,
        default=0.7,
        help="Historical trust retention factor (default: 0.7)"
    )

    args = parser.parse_args()

    if not 0.0 <= args.decay <= 1.0:
        parser.error("--decay must be between 0 and 1.")

    combined_data = load_json(args.input_json)

    history = calculate_historical_trust(
        combined_data,
        decay=args.decay
    )

    summary = summarize_trust(history)

    output = {
        "experiment": "Historical trust analysis",
        "formula": "T_r = decay * T_(r-1) + (1-decay) * C_r",
        "decay": args.decay,
        "initial_trust": 0.5,
        "rounds": len(history),
        "trust_history": history,
        "summary": summary
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "historical_trust_analysis.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print()
    print("=" * 65)
    print("HISTORICAL TRUST ANALYSIS")
    print("=" * 65)
    print(f"Decay factor: {args.decay}")
    print(f"Rounds processed: {len(history)}")
    print(f"Benign observations: {summary['benign_observations']}")
    print(f"Malicious observations: {summary['malicious_observations']}")
    print(
        "Mean benign trust: "
        f"{summary['benign_mean_historical_trust']}"
    )
    print(
        "Mean malicious trust: "
        f"{summary['malicious_mean_historical_trust']}"
    )

    if "benign_minus_malicious_trust_gap" in summary:
        print(
            "Benign - malicious trust gap: "
            f"{summary['benign_minus_malicious_trust_gap']:.4f}"
        )

    print(f"Saved to: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
