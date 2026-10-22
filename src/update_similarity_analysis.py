from pathlib import Path
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F


def parse_client_ids(text):
    if not text:
        return set()
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def load_model(path, input_size, output_size):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.20),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.20),
        torch.nn.Linear(64, output_size),
    )

    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model


def model_vector(model):
    parts = []

    for param in model.parameters():
        parts.append(param.detach().cpu().reshape(-1))

    return torch.cat(parts)


def weighted_average_vectors(vectors, weights):
    result = torch.zeros_like(vectors[0])

    total_weight = sum(weights)

    for vector, weight in zip(vectors, weights):
        result += vector * (weight / total_weight)

    return result


def cosine_similarity(a, b):
    return float(
        F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
            dim=1
        ).item()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Experiment directory containing local_models"
    )

    parser.add_argument(
        "--encoded-dir",
        default="data/encoded_final"
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10
    )

    parser.add_argument(
        "--malicious-client-ids",
        default=""
    )

    args = parser.parse_args()

    experiment_dir = Path(args.model_dir)
    encoded_dir = Path(args.encoded_dir)
    output_dir = Path(args.output_dir)

    local_model_root = experiment_dir / "local_models"

    output_dir.mkdir(parents=True, exist_ok=True)

    malicious_clients = parse_client_ids(
        args.malicious_client_ids
    )

    # ---------------------------------------------------------
    # Discover clients
    # ---------------------------------------------------------

    client_files = sorted(
        (encoded_dir / "clients").glob("client_*.npz")
    )

    if not client_files:
        raise FileNotFoundError(
            f"No client files found in {encoded_dir / 'clients'}"
        )

    client_ids = [
        int(path.stem.split("_")[1])
        for path in client_files
    ]

    sample_counts = {}

    for path in client_files:
        data = np.load(path)
        sample_counts[
            int(path.stem.split("_")[1])
        ] = len(data["y_multiclass"])

    print("=" * 70)
    print("UPDATE SIMILARITY ANALYSIS")
    print("=" * 70)

    print(f"Experiment directory : {experiment_dir}")
    print(f"Clients              : {len(client_ids)}")
    print(f"Rounds               : {args.rounds}")
    print(
        f"Malicious clients    : "
        f"{sorted(malicious_clients) if malicious_clients else '[]'}"
    )

    # ---------------------------------------------------------
    # Load one model to determine parameter size
    # ---------------------------------------------------------

    first_model_path = (
        local_model_root
        / "round_01"
        / f"client_{client_ids[0]:02d}.pt"
    )

    if not first_model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {first_model_path}"
        )

    # We don't need to know input/output dimensions because
    # the saved state dictionaries already contain parameters.
    first_state = torch.load(
        first_model_path,
        map_location="cpu",
        weights_only=True
    )

    # ---------------------------------------------------------
    # Load all local model vectors
    # ---------------------------------------------------------

    round_models = {}

    for round_id in range(1, args.rounds + 1):

        round_dir = (
            local_model_root
            / f"round_{round_id:02d}"
        )

        if not round_dir.exists():
            raise FileNotFoundError(
                f"Missing round directory: {round_dir}"
            )

        round_models[round_id] = {}

        for client_id in client_ids:

            model_path = (
                round_dir
                / f"client_{client_id:02d}.pt"
            )

            state = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True
            )

            vector = torch.cat([
                tensor.detach().cpu().reshape(-1)
                for tensor in state.values()
            ])

            round_models[round_id][client_id] = vector

    # ---------------------------------------------------------
    # Reconstruct global models
    #
    # The original FedAvg experiment uses sample-count
    # weighted averaging.
    # ---------------------------------------------------------

    global_models = {}

    for round_id in range(1, args.rounds + 1):

        vectors = [
            round_models[round_id][client_id]
            for client_id in client_ids
        ]

        weights = [
            sample_counts[client_id]
            for client_id in client_ids
        ]

        global_models[round_id] = weighted_average_vectors(
            vectors,
            weights
        )

    # ---------------------------------------------------------
    # Calculate update similarity
    #
    # For round r:
    #
    #   update_i = local_model_i(r)
    #               - global_model(r-1)
    #
    # Then compare each client's update against the
    # mean update of all OTHER clients.
    #
    # This is leave-one-out similarity, so a client does
    # not define its own reference.
    # ---------------------------------------------------------

    results = []

    print()
    print("Consecutive-round update similarity")
    print(
        "Client       R2       R3       R4       R5       "
        "R6       R7       R8       R9       R10"
    )

    for client_id in client_ids:

        similarities = []

        for round_id in range(2, args.rounds + 1):

            previous_global = global_models[round_id - 1]

            client_update = (
                round_models[round_id][client_id]
                - previous_global
            )

            other_updates = []

            for other_id in client_ids:

                if other_id == client_id:
                    continue

                other_update = (
                    round_models[round_id][other_id]
                    - previous_global
                )

                other_updates.append(other_update)

            reference_update = torch.stack(
                other_updates
            ).mean(dim=0)

            similarity = cosine_similarity(
                client_update,
                reference_update
            )

            similarities.append(similarity)

        label = (
            "M"
            if client_id in malicious_clients
            else "L"
        )

        values = " ".join(
            f"{x:.4f}"
            for x in similarities
        )

        print(
            f"{client_id:02d} [{label}]     {values}"
        )

        results.append({
            "client_id": client_id,
            "label": (
                "malicious"
                if client_id in malicious_clients
                else "legitimate"
            ),
            "similarities": {
                f"round_{r:02d}": similarities[r - 2]
                for r in range(2, args.rounds + 1)
            },
            "mean_similarity": float(
                np.mean(similarities)
            ),
            "min_similarity": float(
                np.min(similarities)
            ),
            "max_similarity": float(
                np.max(similarities)
            ),
            "std_similarity": float(
                np.std(similarities)
            ),
        })

    # ---------------------------------------------------------
    # Group statistics
    # ---------------------------------------------------------

    legitimate = [
        x for x in results
        if x["label"] == "legitimate"
    ]

    malicious = [
        x for x in results
        if x["label"] == "malicious"
    ]

    def group_mean(items):
        if not items:
            return None

        return float(
            np.mean([
                x["mean_similarity"]
                for x in items
            ])
        )

    summary = {
        "experiment_directory": str(experiment_dir),
        "rounds": args.rounds,
        "clients": len(client_ids),
        "malicious_client_ids": sorted(
            malicious_clients
        ),
        "legitimate_mean_similarity": group_mean(
            legitimate
        ),
        "malicious_mean_similarity": group_mean(
            malicious
        ),
        "clients": results,
    }

    output_file = (
        output_dir
        / "update_similarity_analysis.json"
    )

    with open(output_file, "w") as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print("Group summary")

    if legitimate:
        print(
            "Legitimate mean: "
            f"{group_mean(legitimate):.4f}"
        )

    if malicious:
        print(
            "Malicious mean:  "
            f"{group_mean(malicious):.4f}"
        )
    else:
        print("Malicious mean:  N/A")

    print()
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()