from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import shap
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn


# ============================================================
# Model
# ============================================================

class IDSModel(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Round-wise SHAP trajectory analysis"
)

parser.add_argument(
    "--model-dir",
    type=Path,
    required=True,
    help="Experiment directory containing local_models/",
)

parser.add_argument(
    "--encoded-dir",
    type=Path,
    default=Path("data/encoded_final"),
    help="Encoded dataset directory",
)

parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Directory where analysis results will be saved",
)

parser.add_argument(
    "--rounds",
    type=int,
    default=10,
    help="Number of FL rounds",
)

parser.add_argument(
    "--background-samples",
    type=int,
    default=50,
    help="Number of SHAP background samples",
)

parser.add_argument(
    "--explanation-samples",
    type=int,
    default=20,
    help="Number of samples to explain per round",
)

parser.add_argument(
    "--malicious-client-ids",
    type=str,
    default="",
    help='Comma-separated malicious client IDs, e.g. "1,6"',
)

args = parser.parse_args()


# ============================================================
# Paths and settings
# ============================================================

ENCODED_DIR = args.encoded_dir
EXPERIMENT_DIR = args.model_dir
LOCAL_MODEL_ROOT = EXPERIMENT_DIR / "local_models"
OUTPUT_DIR = args.output_dir

ROUNDS = args.rounds
BACKGROUND_SAMPLES = args.background_samples
EXPLANATION_SAMPLES = args.explanation_samples


def parse_client_ids(value: str) -> set[int]:
    value = value.strip()

    if not value:
        return set()

    return {
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    }


MALICIOUS_CLIENTS = parse_client_ids(
    args.malicious_client_ids
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR / "shap_trajectory_analysis.json"
)


# ============================================================
# Load encoding information
# ============================================================

with open(
    ENCODED_DIR / "encoding_report.json",
    "r",
    encoding="utf-8",
) as f:
    encoding_data = json.load(f)


input_size = len(
    encoding_data["feature_names"]
)

output_size = len(
    encoding_data["multiclass_label_map"]
)


# ============================================================
# Load clients
# ============================================================

client_files = sorted(
    (ENCODED_DIR / "clients").glob("client_*.npz")
)

if not client_files:
    raise FileNotFoundError(
        "No client files found."
    )


clients = {}

for path in client_files:
    client_id = int(
        path.stem.split("_")[1]
    )

    data = np.load(path)

    clients[client_id] = (
        data["x"].astype(np.float32)
    )


# ============================================================
# SHAP signature calculation
# ============================================================

def calculate_shap_signature(
    model: nn.Module,
    x: np.ndarray,
) -> np.ndarray:

    model.eval()

    background_count = min(
        BACKGROUND_SAMPLES,
        len(x),
    )

    explanation_count = min(
        EXPLANATION_SAMPLES,
        len(x),
    )

    background = torch.from_numpy(
        x[:background_count]
    ).float()

    explain_x = torch.from_numpy(
        x[:explanation_count]
    ).float()

    explainer = shap.GradientExplainer(
        model,
        background,
    )

    shap_values = explainer.shap_values(
        explain_x
    )

    # Handle SHAP output formats.
    if isinstance(shap_values, list):

        shap_array = np.stack(
            [
                np.asarray(values)
                for values in shap_values
            ],
            axis=-1,
        )

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 2:
            shap_array = (
                shap_array[:, :, np.newaxis]
            )

    # Predictions determine which class
    # explanation is selected for each sample.
    with torch.no_grad():

        logits = model(
            explain_x
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        ).numpy()

    selected_values = []

    for sample_index, predicted_class in enumerate(
        predictions
    ):

        selected_values.append(
            shap_array[
                sample_index,
                :,
                predicted_class,
            ]
        )

    selected_values = np.asarray(
        selected_values
    )

    signature = np.mean(
        np.abs(selected_values),
        axis=0,
    )

    return signature.astype(
        np.float32
    )


# ============================================================
# Calculate SHAP trajectories
# ============================================================

all_results = {}

print()
print("=" * 90)
print("ROUND-WISE SHAP TRAJECTORY ANALYSIS")
print("=" * 90)

print(
    f"Experiment directory: {EXPERIMENT_DIR}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print(
    f"Rounds: {ROUNDS}"
)

print(
    f"Clients: {len(clients)}"
)

print(
    f"Background samples: {BACKGROUND_SAMPLES}"
)

print(
    f"Explanation samples: {EXPLANATION_SAMPLES}"
)

print(
    f"Malicious clients: "
    f"{sorted(MALICIOUS_CLIENTS)}"
)


for client_id in sorted(clients):

    print()
    print(
        f"Processing Client {client_id:02d}"
        + (
            " [MALICIOUS]"
            if client_id in MALICIOUS_CLIENTS
            else " [LEGITIMATE]"
        )
    )

    client_x = clients[client_id]

    round_signatures = {}

    # --------------------------------------------------------
    # Calculate SHAP signature for every round.
    # --------------------------------------------------------

    for round_number in range(
        1,
        ROUNDS + 1,
    ):

        model_path = (
            LOCAL_MODEL_ROOT
            / f"round_{round_number:02d}"
            / f"client_{client_id:02d}.pt"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model: {model_path}"
            )

        model = IDSModel(
            input_size,
            output_size,
        )

        model.load_state_dict(
            torch.load(
                model_path,
                map_location="cpu",
            )
        )

        model.eval()

        signature = calculate_shap_signature(
            model,
            client_x,
        )

        round_signatures[
            str(round_number)
        ] = signature

        print(
            f"  Round {round_number:02d} "
            f"SHAP calculated"
        )

    # --------------------------------------------------------
    # Calculate consecutive-round consistency.
    # --------------------------------------------------------

    transition_scores = {}

    for round_number in range(
        2,
        ROUNDS + 1,
    ):

        previous_signature = (
            round_signatures[
                str(round_number - 1)
            ]
        )

        current_signature = (
            round_signatures[
                str(round_number)
            ]
        )

        similarity = cosine_similarity(
            previous_signature.reshape(
                1,
                -1,
            ),
            current_signature.reshape(
                1,
                -1,
            ),
        )[0, 0]

        transition_scores[
            str(round_number)
        ] = float(similarity)

    # --------------------------------------------------------
    # Save client results.
    # --------------------------------------------------------

    all_results[
        str(client_id)
    ] = {
        "client_id": client_id,
        "malicious": (
            client_id in MALICIOUS_CLIENTS
        ),
        "sample_count": len(client_x),
        "round_signatures": {
            round_number: signature.tolist()
            for round_number, signature
            in round_signatures.items()
        },
        "consecutive_round_similarity": (
            transition_scores
        ),
    }


# ============================================================
# Print transition summary
# ============================================================

print()
print("=" * 90)
print(
    "CONSECUTIVE-ROUND SHAP SIMILARITY"
)
print("=" * 90)

print(
    "Client       R2       R3       R4       "
    "R5       R6       R7       R8       R9       R10"
)

print("-" * 90)

for client_id in sorted(clients):

    scores = all_results[
        str(client_id)
    ]["consecutive_round_similarity"]

    values = [
        scores[str(round_number)]
        for round_number in range(
            2,
            ROUNDS + 1,
        )
    ]

    label = (
        "M"
        if client_id in MALICIOUS_CLIENTS
        else "L"
    )

    formatted = " ".join(
        f"{value:.4f}"
        for value in values
    )

    print(
        f"{client_id:02d} [{label}]     "
        f"{formatted}"
    )


# ============================================================
# Per-client summary statistics
# ============================================================

print()
print("=" * 90)
print(
    "CLIENT SHAP STABILITY SUMMARY"
)
print("=" * 90)

for client_id in sorted(clients):

    scores = np.array(
        list(
            all_results[
                str(client_id)
            ]["consecutive_round_similarity"].values()
        )
    )

    label = (
        "MALICIOUS"
        if client_id in MALICIOUS_CLIENTS
        else "LEGITIMATE"
    )

    print(
        f"Client {client_id:02d} | "
        f"{label:10s} | "
        f"Mean: {scores.mean():.4f} | "
        f"Min: {scores.min():.4f} | "
        f"Max: {scores.max():.4f} | "
        f"Std: {scores.std():.4f}"
    )


# ============================================================
# Group-level summary
# ============================================================

legitimate_means = []
malicious_means = []

for client_id in sorted(clients):

    scores = np.array(
        list(
            all_results[
                str(client_id)
            ]["consecutive_round_similarity"].values()
        )
    )

    client_mean = scores.mean()

    if client_id in MALICIOUS_CLIENTS:
        malicious_means.append(
            client_mean
        )
    else:
        legitimate_means.append(
            client_mean
        )


print()
print("=" * 90)
print("GROUP SUMMARY")
print("=" * 90)

if legitimate_means:
    print(
        f"Legitimate clients mean "
        f"trajectory similarity: "
        f"{np.mean(legitimate_means):.4f}"
    )

if malicious_means:
    print(
        f"Malicious clients mean "
        f"trajectory similarity: "
        f"{np.mean(malicious_means):.4f}"
    )
else:
    print(
        "Malicious clients mean "
        "trajectory similarity: N/A "
        "(no malicious clients)"
    )


# ============================================================
# Save results
# ============================================================

output_data = {
    "experiment": (
        "Round-wise SHAP trajectory analysis"
    ),
    "model_directory": str(
        EXPERIMENT_DIR
    ),
    "output_directory": str(
        OUTPUT_DIR
    ),
    "rounds": ROUNDS,
    "background_samples": BACKGROUND_SAMPLES,
    "explanation_samples": EXPLANATION_SAMPLES,
    "malicious_clients": sorted(
        MALICIOUS_CLIENTS
    ),
    "feature_count": input_size,
    "clients": all_results,
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output_data,
        f,
        indent=2,
    )


print()
print("=" * 90)
print("SHAP TRAJECTORY ANALYSIS COMPLETED")
print("=" * 90)

print(
    f"Saved: {OUTPUT_FILE.resolve()}"
)