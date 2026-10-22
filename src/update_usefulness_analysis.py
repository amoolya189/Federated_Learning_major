from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# EXACT SAME MODEL ARCHITECTURE AS federated_poisoned.py
# ============================================================

class IDSModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, output_size),
        )

    def forward(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        return self.network(features)


# ============================================================
# CLIENT ID PARSER
# ============================================================

def parse_client_ids(value: str) -> set[int]:

    if not value.strip():
        return set()

    try:
        return {
            int(client_id.strip())
            for client_id in value.split(",")
            if client_id.strip()
        }

    except ValueError as error:
        raise SystemExit(
            "Malicious client IDs must be integers, "
            "for example: 1,6"
        ) from error


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    path: Path
) -> tuple[np.ndarray, np.ndarray]:

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    data = np.load(path)

    x = data["x"].astype(
        np.float32
    )

    y = data["y_multiclass"].astype(
        np.int64
    )

    return x, y


# ============================================================
# LOAD STATE DICTIONARY
# ============================================================

def load_state(
    path: Path
) -> dict[str, torch.Tensor]:

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )


# ============================================================
# LOAD STATE INTO IDS MODEL
# ============================================================

def set_model_state(
    model: IDSModel,
    state: dict[str, torch.Tensor]
) -> None:

    model.load_state_dict(state)
    model.eval()


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate(
    model: IDSModel,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int
) -> dict[str, float]:

    model.eval()

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y)
        ),
        batch_size=batch_size,
        shuffle=False
    )

    predictions = []
    labels = []

    with torch.no_grad():

        for features, targets in loader:

            logits = model(
                features
            )

            batch_predictions = torch.argmax(
                logits,
                dim=1
            )

            predictions.extend(
                batch_predictions.cpu().numpy()
            )

            labels.extend(
                targets.numpy()
            )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1)
    }


# ============================================================
# SAMPLE-COUNT-WEIGHTED FEDAVG
#
# This matches federated_poisoned.py
# ============================================================

def weighted_average_states(
    states: list[dict[str, torch.Tensor]],
    sample_counts: list[int]
) -> dict[str, torch.Tensor]:

    total_samples = sum(
        sample_counts
    )

    aggregated = {}

    for key in states[0]:

        weighted_value = sum(
            state[key].float()
            * (
                count / total_samples
            )
            for state, count in zip(
                states,
                sample_counts
            )
        )

        aggregated[key] = weighted_value

    return aggregated


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze local-model update usefulness "
            "for the TON_IoT FedAvg experiments."
        )
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--encoded-dir",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512
    )

    parser.add_argument(
        "--malicious-client-ids",
        default=""
    )

    args = parser.parse_args()

    experiment_dir = args.model_dir
    encoded_dir = args.encoded_dir
    output_dir = args.output_dir

    local_model_root = (
        experiment_dir / "local_models"
    )

    malicious_clients = parse_client_ids(
        args.malicious_client_ids
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LOAD SERVER VALIDATION DATA
    # ========================================================

    validation_path = (
        encoded_dir /
        "server_validation.npz"
    )

    validation_x, validation_y = (
        load_dataset(validation_path)
    )

    # ========================================================
    # FIND CLIENT FILES
    # ========================================================

    client_files = sorted(
        (
            encoded_dir / "clients"
        ).glob("client_*.npz")
    )

    if not client_files:
        raise FileNotFoundError(
            "No client files found in "
            f"{encoded_dir / 'clients'}"
        )

    client_ids = [
        int(
            path.stem.split("_")[1]
        )
        for path in client_files
    ]

    # ========================================================
    # SAMPLE COUNTS
    # ========================================================

    sample_counts = {}

    for client_id in client_ids:

        client_path = (
            encoded_dir
            / "clients"
            / f"client_{client_id:02d}.npz"
        )

        client_x, client_y = (
            load_dataset(client_path)
        )

        sample_counts[client_id] = len(
            client_y
        )

    # ========================================================
    # FIND FIRST SAVED MODEL
    # ========================================================

    first_model_path = (
        local_model_root
        / "round_01"
        / f"client_{client_ids[0]:02d}.pt"
    )

    first_state = load_state(
        first_model_path
    )

    # ========================================================
    # DETERMINE MODEL DIMENSIONS
    #
    # These keys are from the ACTUAL IDSModel:
    #
    # network.0.weight
    # network.6.weight
    # ========================================================

    input_size = (
        first_state[
            "network.0.weight"
        ].shape[1]
    )

    output_size = (
        first_state[
            "network.6.weight"
        ].shape[0]
    )

    model = IDSModel(
        input_size=input_size,
        output_size=output_size
    )

    # ========================================================
    # PRINT EXPERIMENT INFORMATION
    # ========================================================

    print("=" * 70)
    print("UPDATE USEFULNESS ANALYSIS")
    print("=" * 70)

    print(
        f"Experiment directory : "
        f"{experiment_dir}"
    )

    print(
        f"Validation samples    : "
        f"{len(validation_y)}"
    )

    print(
        f"Clients               : "
        f"{len(client_ids)}"
    )

    print(
        f"Rounds                : "
        f"{args.rounds}"
    )

    print(
        f"Input features        : "
        f"{input_size}"
    )

    print(
        f"Output classes        : "
        f"{output_size}"
    )

    if malicious_clients:

        print(
            "Malicious clients     : "
            f"{sorted(malicious_clients)}"
        )

    else:

        print(
            "Malicious clients     : []"
        )

    # ========================================================
    # LOAD ALL LOCAL MODEL STATES
    # ========================================================

    round_states = {}

    for round_number in range(
        1,
        args.rounds + 1
    ):

        round_dir = (
            local_model_root
            / f"round_{round_number:02d}"
        )

        if not round_dir.exists():
            raise FileNotFoundError(
                f"Missing round directory: "
                f"{round_dir}"
            )

        round_states[
            round_number
        ] = {}

        for client_id in client_ids:

            model_path = (
                round_dir
                / f"client_{client_id:02d}.pt"
            )

            round_states[
                round_number
            ][client_id] = load_state(
                model_path
            )

    # ========================================================
    # RECONSTRUCT GLOBAL MODEL AFTER EACH ROUND
    #
    # This exactly follows the FedAvg aggregation used
    # in federated_poisoned.py.
    # ========================================================

    global_states = {}

    for round_number in range(
        1,
        args.rounds + 1
    ):

        states = [
            round_states[
                round_number
            ][client_id]
            for client_id in client_ids
        ]

        counts = [
            sample_counts[client_id]
            for client_id in client_ids
        ]

        global_states[
            round_number
        ] = weighted_average_states(
            states,
            counts
        )

    # ========================================================
    # UPDATE USEFULNESS
    #
    # For round r:
    #
    # previous global model
    #          ↓
    #       client trains
    #          ↓
    # client's local model
    #
    # usefulness =
    #
    # local model macro-F1
    # -
    # previous global macro-F1
    #
    # Positive = local model performs better
    # Negative = local model performs worse
    # ========================================================

    results = []

    print()
    print(
        "Client update usefulness "
        "(validation macro-F1 change)"
    )

    print(
        "Client       R2       R3       R4       R5       "
        "R6       R7       R8       R9       R10"
    )

    # ========================================================
    # ROUND-BY-ROUND ANALYSIS
    # ========================================================

    for client_id in client_ids:

        usefulness_values = []

        for round_number in range(
            2,
            args.rounds + 1
        ):

            # ------------------------------------------------
            # Previous global model
            # ------------------------------------------------

            previous_global_state = (
                global_states[
                    round_number - 1
                ]
            )

            set_model_state(
                model,
                previous_global_state
            )

            previous_global_metrics = evaluate(
                model,
                validation_x,
                validation_y,
                args.batch_size
            )

            previous_global_f1 = (
                previous_global_metrics[
                    "macro_f1"
                ]
            )

            # ------------------------------------------------
            # Current client's local model
            # ------------------------------------------------

            client_state = (
                round_states[
                    round_number
                ][client_id]
            )

            set_model_state(
                model,
                client_state
            )

            client_metrics = evaluate(
                model,
                validation_x,
                validation_y,
                args.batch_size
            )

            client_f1 = (
                client_metrics[
                    "macro_f1"
                ]
            )

            # ------------------------------------------------
            # USEFULNESS
            # ------------------------------------------------

            usefulness = (
                client_f1
                - previous_global_f1
            )

            usefulness_values.append(
                usefulness
            )

        label = (
            "M"
            if client_id in malicious_clients
            else "L"
        )

        values = " ".join(
            f"{value:+.4f}"
            for value in usefulness_values
        )

        print(
            f"{client_id:02d} [{label}]     "
            f"{values}"
        )

        results.append(
            {
                "client_id": client_id,

                "label": (
                    "malicious"
                    if client_id
                    in malicious_clients
                    else "legitimate"
                ),

                "usefulness": {
                    f"round_{round_number:02d}":
                        usefulness_values[
                            round_number - 2
                        ]
                    for round_number in range(
                        2,
                        args.rounds + 1
                    )
                },

                "mean_usefulness": float(
                    np.mean(
                        usefulness_values
                    )
                ),

                "min_usefulness": float(
                    np.min(
                        usefulness_values
                    )
                ),

                "max_usefulness": float(
                    np.max(
                        usefulness_values
                    )
                ),

                "positive_round_fraction":
                    float(
                        np.mean(
                            np.array(
                                usefulness_values
                            ) > 0
                        )
                    )
            }
        )

    # ========================================================
    # GROUP STATISTICS
    # ========================================================

    legitimate = [
        result
        for result in results
        if result["label"] == "legitimate"
    ]

    malicious = [
        result
        for result in results
        if result["label"] == "malicious"
    ]

    def group_mean(
        items: list[dict]
    ):

        if not items:
            return None

        return float(
            np.mean(
                [
                    item["mean_usefulness"]
                    for item in items
                ]
            )
        )

    legitimate_mean = (
        group_mean(legitimate)
    )

    malicious_mean = (
        group_mean(malicious)
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    summary = {
        "experiment_directory":
            str(experiment_dir),

        "rounds":
            args.rounds,

        "validation_samples":
            len(validation_y),

        "input_features":
            input_size,

        "output_classes":
            output_size,

        "malicious_client_ids":
            sorted(malicious_clients),

        "legitimate_mean_usefulness":
            legitimate_mean,

        "malicious_mean_usefulness":
            malicious_mean,

        "clients":
            results
    }

    output_file = (
        output_dir
        / "update_usefulness_analysis.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("Group summary")

    if legitimate_mean is not None:

        print(
            "Legitimate mean: "
            f"{legitimate_mean:+.4f}"
        )

    else:

        print(
            "Legitimate mean: N/A"
        )

    if malicious_mean is not None:

        print(
            "Malicious mean:  "
            f"{malicious_mean:+.4f}"
        )

    else:

        print(
            "Malicious mean:  N/A"
        )

    print()
    print(
        f"Saved: {output_file}"
    )


if __name__ == "__main__":
    main()