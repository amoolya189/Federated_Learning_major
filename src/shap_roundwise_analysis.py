from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shap
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn


# ============================================================
# Model definition
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
# Paths
# ============================================================

ENCODED_DIR = Path("data/encoded_final")

POISONED_DIR = Path(
    "outputs/poisoned_fedavg_roundwise"
)

LOCAL_MODEL_DIR = (
    POISONED_DIR / "local_models" / "round_10"
)

REFERENCE_FILE = Path(
    "outputs/shap_signatures/shap_reference_profile.json"
)

ENCODING_REPORT = (
    ENCODED_DIR / "encoding_report.json"
)

OUTPUT_FILE = (
    POISONED_DIR / "round_10_shap_analysis.json"
)


# ============================================================
# Experiment settings
# ============================================================

BACKGROUND_SAMPLES = 50
EXPLANATION_SAMPLES = 20

MALICIOUS_CLIENTS = {1, 6}


# ============================================================
# Load reference profile
# ============================================================

with open(
    REFERENCE_FILE,
    "r",
    encoding="utf-8",
) as f:
    reference_data = json.load(f)

median_reference = np.array(
    reference_data["median_reference"],
    dtype=np.float32,
)


# ============================================================
# Load feature names
# ============================================================

with open(
    ENCODING_REPORT,
    "r",
    encoding="utf-8",
) as f:
    encoding_data = json.load(f)

feature_names = encoding_data["feature_names"]


# ============================================================
# Load client files
# ============================================================

client_files = sorted(
    (ENCODED_DIR / "clients").glob("client_*.npz")
)

if not client_files:
    raise FileNotFoundError(
        "No encoded client files found."
    )


print(
    f"Clients found: {len(client_files)}"
)

print(
    f"Round being analyzed: 10"
)

print(
    f"Background samples per client: "
    f"{BACKGROUND_SAMPLES}"
)

print(
    f"Explanation samples per client: "
    f"{EXPLANATION_SAMPLES}"
)


# ============================================================
# Determine model dimensions
# ============================================================

first_client = np.load(
    client_files[0]
)

input_size = first_client["x"].shape[1]

output_size = (
    len(
        encoding_data["multiclass_label_map"]
    )
)

print(
    f"Input features: {input_size}"
)

print(
    f"Output classes: {output_size}"
)


# ============================================================
# SHAP helper
# ============================================================

def calculate_client_shap_signature(
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

    # --------------------------------------------------------
    # GradientExplainer
    # --------------------------------------------------------

    explainer = shap.GradientExplainer(
        model,
        background,
    )

    shap_values = explainer.shap_values(
        explain_x
    )

    # --------------------------------------------------------
    # Convert SHAP output to numpy
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list,
    ):
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

        # Some SHAP versions return:
        # samples x features x classes
        #
        # Other versions can return:
        # samples x features
        #
        # We handle both.

        if shap_array.ndim == 2:
            shap_array = (
                shap_array[:, :, np.newaxis]
            )

    # --------------------------------------------------------
    # Obtain predictions
    # --------------------------------------------------------

    with torch.no_grad():
        logits = model(explain_x)

        predictions = torch.argmax(
            logits,
            dim=1,
        ).numpy()

    # --------------------------------------------------------
    # Build per-sample predicted-class SHAP values
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Mean absolute SHAP importance
    # --------------------------------------------------------

    signature = np.mean(
        np.abs(selected_values),
        axis=0,
    )

    return signature.astype(
        np.float32
    )


# ============================================================
# Analyze clients
# ============================================================

results = []

print()
print("=" * 90)
print("ROUND 10 SHAP CONSISTENCY ANALYSIS")
print("=" * 90)

for client_file in client_files:

    client_id = int(
        client_file.stem.split("_")[1]
    )

    model_file = (
        LOCAL_MODEL_DIR
        / f"client_{client_id:02d}.pt"
    )

    if not model_file.exists():
        print(
            f"Skipping client {client_id:02d}: "
            f"model not found."
        )
        continue

    data = np.load(
        client_file
    )

    x = data["x"].astype(
        np.float32
    )

    # --------------------------------------------------------
    # Load local model
    # --------------------------------------------------------

    model = IDSModel(
        input_size,
        output_size,
    )

    model.load_state_dict(
        torch.load(
            model_file,
            map_location="cpu",
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Calculate SHAP signature
    # --------------------------------------------------------

    signature = (
        calculate_client_shap_signature(
            model,
            x,
        )
    )

    # --------------------------------------------------------
    # Compare with median reference
    # --------------------------------------------------------

    similarity = cosine_similarity(
        signature.reshape(1, -1),
        median_reference.reshape(1, -1),
    )[0, 0]

    is_malicious = (
        client_id in MALICIOUS_CLIENTS
    )

    result = {
        "client_id": client_id,
        "malicious": is_malicious,
        "sample_count": len(x),
        "shap_reference_cosine_similarity": float(
            similarity
        ),
        "shap_signature": signature.tolist(),
    }

    results.append(result)

    label = (
        "MALICIOUS"
        if is_malicious
        else "legitimate"
    )

    print(
        f"Client {client_id:02d} | "
        f"{label:10s} | "
        f"SHAP similarity: {similarity:.4f}"
    )


# ============================================================
# Sort results by client ID
# ============================================================

results.sort(
    key=lambda item: item["client_id"]
)


# ============================================================
# Save complete analysis
# ============================================================

output_data = {
    "experiment": (
        "Round-wise SHAP analysis "
        "of poisoned FedAvg"
    ),
    "round": 10,
    "malicious_clients": sorted(
        MALICIOUS_CLIENTS
    ),
    "background_samples": BACKGROUND_SAMPLES,
    "explanation_samples": EXPLANATION_SAMPLES,
    "reference_method": (
        "feature-wise median SHAP reference"
    ),
    "feature_count": input_size,
    "clients": results,
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


# ============================================================
# Summary
# ============================================================

legitimate_scores = [
    item["shap_reference_cosine_similarity"]
    for item in results
    if not item["malicious"]
]

malicious_scores = [
    item["shap_reference_cosine_similarity"]
    for item in results
    if item["malicious"]
]


print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)

if legitimate_scores:
    print(
        "Legitimate clients:"
    )

    print(
        f"  Mean similarity: "
        f"{np.mean(legitimate_scores):.4f}"
    )

    print(
        f"  Min similarity:  "
        f"{np.min(legitimate_scores):.4f}"
    )

    print(
        f"  Max similarity:  "
        f"{np.max(legitimate_scores):.4f}"
    )

if malicious_scores:
    print(
        "Malicious clients:"
    )

    print(
        f"  Mean similarity: "
        f"{np.mean(malicious_scores):.4f}"
    )

    print(
        f"  Min similarity:  "
        f"{np.min(malicious_scores):.4f}"
    )

    print(
        f"  Max similarity:  "
        f"{np.max(malicious_scores):.4f}"
    )

print()
print(
    f"Saved: {OUTPUT_FILE.resolve()}"
)
