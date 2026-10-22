
import argparse
import json
from pathlib import Path

import numpy as np


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(
            f"Vector shape mismatch: {a.shape} versus {b.shape}"
        )

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def min_max_normalize(values):
    """Normalize values to the range [0, 1]."""
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    minimum = np.min(values)
    maximum = np.max(values)

    if np.isclose(minimum, maximum):
        return np.full_like(values, 0.5)

    return (values - minimum) / (maximum - minimum)


def normalize_round_id(round_id):
    """
    Convert round identifiers such as 'round_02', '02',
    and '2' into the same integer round number.
    """
    value = str(round_id).strip().lower()

    if value.startswith("round_"):
        value = value[len("round_"):]

    return int(value)


def calculate_shap_consistency(shap_data, reference):
    """
    Calculate explanation consistency for each client and round.

    E_i^r = cosine_similarity(
        client's round-wise SHAP signature,
        median SHAP reference
    )
    """
    results = {}

    for client_id, client_info in shap_data["clients"].items():
        client_key = str(client_id)

        results[client_key] = {
            "client_id": int(client_info.get("client_id", client_id)),
            "malicious": bool(client_info.get("malicious", False)),
            "rounds": {}
        }

        for round_id, signature in client_info["round_signatures"].items():
            round_number = normalize_round_id(round_id)

            score = cosine_similarity(signature, reference)

            # Store the round as a numeric string, e.g. "2".
            results[client_key]["rounds"][str(round_number)] = score

    return results


def convert_client_signal_to_rounds(client_data, signal_name):
    """
    Convert client-oriented JSON into a round-oriented structure.

    Expected similarity format:
        clients -> client -> similarities -> round -> score

    Expected usefulness format:
        clients -> client -> usefulness -> round -> score
    """
    if "clients" not in client_data:
        raise ValueError(
            f"The {signal_name} JSON file does not contain 'clients'."
        )

    round_data = {}

    for client in client_data["clients"]:
        client_id = str(client["client_id"])

        if signal_name == "similarity":
            signal_values = client.get("similarities", {})
        elif signal_name == "usefulness":
            signal_values = client.get("usefulness", {})
        else:
            raise ValueError(f"Unknown signal: {signal_name}")

        for round_id, score in signal_values.items():
            round_number = normalize_round_id(round_id)
            round_key = str(round_number)

            round_data.setdefault(round_key, {})
            round_data[round_key][client_id] = float(score)

    return round_data


def build_combined_signals(
    similarity_data,
    usefulness_data,
    shap_consistency,
    alpha=0.5,
    beta=0.2,
    gamma=0.3
):
    """
    Combine three signals:

        S = update similarity
        U = update usefulness
        E = SHAP explanation consistency

    C_i^r = alpha * S_normalized
          + beta  * U_normalized
          + gamma * E_normalized

    Each signal is normalized independently within each round.
    """

    similarity_by_round = convert_client_signal_to_rounds(
        similarity_data, "similarity"
    )

    usefulness_by_round = convert_client_signal_to_rounds(
        usefulness_data, "usefulness"
    )

    shap_by_round = {}

    for client_id, client_info in shap_consistency.items():
        for round_id, score in client_info["rounds"].items():
            round_key = str(normalize_round_id(round_id))

            shap_by_round.setdefault(round_key, {})
            shap_by_round[round_key][str(client_id)] = float(score)

    # Process only rounds available in all three signal sources.
    common_rounds = (
        set(similarity_by_round)
        & set(usefulness_by_round)
        & set(shap_by_round)
    )

    combined = {}

    for round_key in sorted(common_rounds, key=int):
        similarity_values = similarity_by_round[round_key]
        usefulness_values = usefulness_by_round[round_key]
        shap_values = shap_by_round[round_key]

        # Include only clients with all three signals.
        client_ids = sorted(
            set(similarity_values)
            & set(usefulness_values)
            & set(shap_values),
            key=int
        )

        if not client_ids:
            continue

        raw_s = [
            similarity_values[client_id]
            for client_id in client_ids
        ]

        raw_u = [
            usefulness_values[client_id]
            for client_id in client_ids
        ]

        raw_e = [
            shap_values[client_id]
            for client_id in client_ids
        ]

        norm_s = min_max_normalize(raw_s)
        norm_u = min_max_normalize(raw_u)
        norm_e = min_max_normalize(raw_e)

        round_result = {}

        for index, client_id in enumerate(client_ids):
            combined_score = (
                alpha * norm_s[index]
                + beta * norm_u[index]
                + gamma * norm_e[index]
            )

            round_result[client_id] = {
                "client_id": int(client_id),
                "malicious": bool(
                    shap_consistency[client_id]["malicious"]
                ),
                "raw_update_similarity": float(raw_s[index]),
                "raw_update_usefulness": float(raw_u[index]),
                "raw_shap_consistency": float(raw_e[index]),
                "normalized_update_similarity": float(norm_s[index]),
                "normalized_update_usefulness": float(norm_u[index]),
                "normalized_shap_consistency": float(norm_e[index]),
                "combined_evidence": float(combined_score)
            }

        combined[round_key] = round_result

    return combined


def summarize(combined):
    """Summarize signal values for benign and malicious clients."""
    all_rows = []

    for round_id, clients in combined.items():
        for client_id, row in clients.items():
            all_rows.append({
                "round": int(round_id),
                "client_id": int(client_id),
                "malicious": row["malicious"],
                "S": row["raw_update_similarity"],
                "U": row["raw_update_usefulness"],
                "E": row["raw_shap_consistency"],
                "C": row["combined_evidence"]
            })

    if not all_rows:
        return {
            "all_round_client_observations": 0,
            "benign": {"count": 0},
            "malicious": {"count": 0},
            "warning": "No common rounds or clients had all three signals."
        }

    benign = [
        row for row in all_rows
        if not row["malicious"]
    ]

    malicious = [
        row for row in all_rows
        if row["malicious"]
    ]

    def mean(rows, key):
        if not rows:
            return None

        return float(np.mean([row[key] for row in rows]))

    summary = {
        "all_round_client_observations": len(all_rows),
        "benign": {
            "count": len(benign),
            "mean_similarity": mean(benign, "S"),
            "mean_usefulness": mean(benign, "U"),
            "mean_shap_consistency": mean(benign, "E"),
            "mean_combined_evidence": mean(benign, "C")
        },
        "malicious": {
            "count": len(malicious),
            "mean_similarity": mean(malicious, "S"),
            "mean_usefulness": mean(malicious, "U"),
            "mean_shap_consistency": mean(malicious, "E"),
            "mean_combined_evidence": mean(malicious, "C")
        }
    }

    if benign and malicious:
        summary["separation"] = {
            "similarity_gap_benign_minus_malicious": (
                mean(benign, "S") - mean(malicious, "S")
            ),
            "usefulness_gap_benign_minus_malicious": (
                mean(benign, "U") - mean(malicious, "U")
            ),
            "shap_gap_benign_minus_malicious": (
                mean(benign, "E") - mean(malicious, "E")
            ),
            "combined_gap_benign_minus_malicious": (
                mean(benign, "C") - mean(malicious, "C")
            )
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Combine federated learning trust signals."
    )

    parser.add_argument(
        "--similarity-json",
        required=True,
        help="Path to update_similarity_analysis.json"
    )

    parser.add_argument(
        "--usefulness-json",
        required=True,
        help="Path to update_usefulness_analysis.json"
    )

    parser.add_argument(
        "--shap-json",
        required=True,
        help="Path to shap_trajectory_analysis.json"
    )

    parser.add_argument(
        "--reference-json",
        required=True,
        help="Path to shap_reference_profile.json"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the output JSON file"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for update similarity"
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.2,
        help="Weight for update usefulness"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.3,
        help="Weight for SHAP consistency"
    )

    args = parser.parse_args()

    # The weights must be valid proportions.
    weights = [args.alpha, args.beta, args.gamma]

    if any(weight < 0 for weight in weights):
        parser.error("Signal weights cannot be negative.")

    if not np.isclose(sum(weights), 1.0):
        parser.error("Signal weights must sum to 1.0.")

    # Load the four input files.
    similarity_data = load_json(args.similarity_json)
    usefulness_data = load_json(args.usefulness_json)
    shap_data = load_json(args.shap_json)
    reference_data = load_json(args.reference_json)

    reference = reference_data["median_reference"]

    if len(reference) != reference_data["feature_count"]:
        raise ValueError(
            "SHAP reference length does not match feature_count."
        )

    if len(reference) == 0:
        raise ValueError("The SHAP reference vector is empty.")

    # Calculate SHAP consistency for each client and round.
    shap_consistency = calculate_shap_consistency(
        shap_data,
        reference
    )

    # Combine all available signals.
    combined = build_combined_signals(
        similarity_data,
        usefulness_data,
        shap_consistency,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )

    summary = summarize(combined)

    # Prepare the output directory.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "Combined trust-signal analysis",
        "weights": {
            "alpha_update_similarity": args.alpha,
            "beta_update_usefulness": args.beta,
            "gamma_shap_consistency": args.gamma
        },
        "shap_reference_method": reference_data.get(
            "method", "unspecified"
        ),
        "rounds": len(combined),
        "combined_signals": combined,
        "summary": summary
    }

    output_path = output_dir / "combined_trust_signals.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    # Print a readable summary.
    print()
    print("=" * 70)
    print("COMBINED TRUST-SIGNAL ANALYSIS")
    print("=" * 70)

    print(
        f"Weights: alpha={args.alpha}, "
        f"beta={args.beta}, gamma={args.gamma}"
    )
    print(f"Rounds successfully combined: {len(combined)}")

    benign = summary.get("benign", {})
    malicious = summary.get("malicious", {})

    print()
    print("BENIGN CLIENT OBSERVATIONS")
    print(f"  Count                 : {benign.get('count')}")
    print(f"  Mean update similarity: {benign.get('mean_similarity')}")
    print(f"  Mean usefulness       : {benign.get('mean_usefulness')}")
    print(f"  Mean SHAP consistency : {benign.get('mean_shap_consistency')}")
    print(f"  Mean combined evidence: {benign.get('mean_combined_evidence')}")

    print()
    print("MALICIOUS CLIENT OBSERVATIONS")
    print(f"  Count                 : {malicious.get('count')}")
    print(f"  Mean update similarity: {malicious.get('mean_similarity')}")
    print(f"  Mean usefulness       : {malicious.get('mean_usefulness')}")
    print(f"  Mean SHAP consistency : {malicious.get('mean_shap_consistency')}")
    print(f"  Mean combined evidence: {malicious.get('mean_combined_evidence')}")

    if "separation" in summary:
        separation = summary["separation"]

        print()
        print("BENIGN - MALICIOUS GAP")
        print(
            f"  Similarity : "
            f"{separation['similarity_gap_benign_minus_malicious']:.4f}"
        )
        print(
            f"  Usefulness : "
            f"{separation['usefulness_gap_benign_minus_malicious']:.4f}"
        )
        print(
            f"  SHAP       : "
            f"{separation['shap_gap_benign_minus_malicious']:.4f}"
        )
        print(
            f"  Combined   : "
            f"{separation['combined_gap_benign_minus_malicious']:.4f}"
        )

    if not combined:
        print()
        print("WARNING: No matching rounds were found across the input files.")
        print("Check the round IDs and client IDs in the JSON files.")

    print()
    print(f"Saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
