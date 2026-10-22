from pathlib import Path
import json

import numpy as np
import shap
import torch

from federated_poisoned import IDSModel


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CLIENT_DIR = (
    PROJECT_DIR
    / "data"
    / "encoded_final"
    / "clients"
)

MODEL_FILE = (
    PROJECT_DIR
    / "outputs"
    / "poisoned_fedavg"
    / "global_model.pt"
)

REPORT_FILE = (
    PROJECT_DIR
    / "data"
    / "encoded_final"
    / "encoding_report.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "shap_signatures"
)


# ============================================================
# SETTINGS
# ============================================================

BACKGROUND_SIZE = 50
EXPLAIN_SIZE = 20
TOP_K = 10


# ============================================================
# FIND CLIENT FILES
# ============================================================

client_files = sorted(
    CLIENT_DIR.glob("client_*.npz")
)

if not client_files:
    raise FileNotFoundError(
        f"No client files found in: {CLIENT_DIR}"
    )

print("Clients found:", len(client_files))

for client_file in client_files:
    print(" -", client_file.name)


# ============================================================
# LOAD FIRST CLIENT TO GET INPUT SIZE
# ============================================================

first_data = np.load(client_files[0])

x_first = first_data["x"].astype(np.float32)

input_size = x_first.shape[1]

print("\nInput features:", input_size)


# ============================================================
# DETERMINE NUMBER OF OUTPUT CLASSES
# ============================================================

output_size = 0

for client_file in client_files:

    client_data = np.load(client_file)

    client_y = client_data["y_multiclass"]

    output_size = max(
        output_size,
        int(np.max(client_y)) + 1
    )


print("Output classes:", output_size)


# ============================================================
# LOAD GLOBAL MODEL
# ============================================================

model = IDSModel(
    input_size,
    output_size
)

state_dict = torch.load(
    MODEL_FILE,
    map_location="cpu"
)

model.load_state_dict(state_dict)

model.eval()

print("Global model loaded")


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

with open(
    REPORT_FILE,
    "r",
    encoding="utf-8"
) as file:

    report = json.load(file)


feature_names = report["feature_names"]


if len(feature_names) != input_size:

    raise ValueError(
        f"Feature-name count ({len(feature_names)}) "
        f"does not match model input size ({input_size})."
    )


print("Feature names loaded:", len(feature_names))


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PROCESS EACH CLIENT
# ============================================================

all_signatures = {}


for client_file in client_files:

    # --------------------------------------------------------
    # CLIENT ID FROM ACTUAL FILE NAME
    # --------------------------------------------------------

    client_id = client_file.stem.replace(
        "client_",
        ""
    )

    print("\n" + "=" * 80)
    print(f"Processing Client {client_id}")
    print("=" * 80)


    # --------------------------------------------------------
    # LOAD CLIENT DATA
    # --------------------------------------------------------

    data = np.load(client_file)

    x = data["x"].astype(np.float32)
    y = data["y_multiclass"].astype(np.int64)

    print("Samples:", len(x))
    print("Features:", x.shape[1])


    if x.shape[1] != input_size:

        raise ValueError(
            f"Client {client_id} has {x.shape[1]} features, "
            f"expected {input_size}."
        )


    # --------------------------------------------------------
    # SELECT BACKGROUND
    # --------------------------------------------------------

    background_size = min(
        BACKGROUND_SIZE,
        len(x)
    )

    background = torch.from_numpy(
        x[:background_size]
    )

    print(
        "SHAP background samples:",
        background_size
    )


    # --------------------------------------------------------
    # CREATE SHAP EXPLAINER
    # --------------------------------------------------------

    explainer = shap.GradientExplainer(
        model,
        background
    )


    # --------------------------------------------------------
    # SELECT SAMPLES TO EXPLAIN
    # --------------------------------------------------------

    explain_size = min(
        EXPLAIN_SIZE,
        len(x)
    )

    samples = torch.from_numpy(
        x[:explain_size]
    )


    # --------------------------------------------------------
    # MODEL PREDICTIONS
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(samples)

        predictions = torch.argmax(
            logits,
            dim=1
        ).numpy()


    print(
        "Samples explained:",
        explain_size
    )

    print(
        "Predicted classes:",
        predictions.tolist()
    )


    # --------------------------------------------------------
    # GENERATE SHAP VALUES
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        samples
    )

    shap_array = np.asarray(
        shap_values
    )

    print(
        "Raw SHAP shape:",
        shap_array.shape
    )


    # --------------------------------------------------------
    # CHECK SHAP SHAPE
    # --------------------------------------------------------

    if shap_array.ndim != 3:

        raise ValueError(
            "Expected 3-dimensional SHAP output "
            "(samples, features, classes), "
            f"but received {shap_array.shape}"
        )


    sample_count, feature_count, class_count = (
        shap_array.shape
    )


    if feature_count != input_size:

        raise ValueError(
            f"SHAP feature count ({feature_count}) "
            f"does not match model input size ({input_size})."
        )


    # --------------------------------------------------------
    # CLIENT-LEVEL SHAP IMPORTANCE
    # --------------------------------------------------------

    client_importance = np.zeros(
        feature_count,
        dtype=np.float64
    )


    for sample_index in range(sample_count):

        predicted_class = int(
            predictions[sample_index]
        )

        sample_shap = shap_array[
            sample_index,
            :,
            predicted_class
        ]

        client_importance += np.abs(
            sample_shap
        )


    client_importance /= sample_count


    # --------------------------------------------------------
    # TOP-K FEATURES
    # --------------------------------------------------------

    top_indices = np.argsort(
        client_importance
    )[::-1][:TOP_K]


    print("\nClient-level SHAP signature")
    print("-" * 80)

    print("Client:", client_id)
    print("Samples used:", sample_count)
    print("Features:", feature_count)
    print("Classes:", class_count)

    print()


    top_features = []


    for rank, feature_index in enumerate(
        top_indices,
        start=1
    ):

        feature_name = feature_names[
            feature_index
        ]

        importance = float(
            client_importance[
                feature_index
            ]
        )

        print(
            f"{rank:2d}. "
            f"{feature_name:<55} "
            f"importance={importance:.6f}"
        )


        top_features.append(
            {
                "rank": rank,
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "importance": importance
            }
        )


    # --------------------------------------------------------
    # SAVE FULL 90-FEATURE SIGNATURE
    # --------------------------------------------------------

    signature = {
        "client_id": client_id,
        "samples_available": int(len(x)),
        "samples_explained": int(sample_count),
        "feature_count": int(feature_count),
        "class_count": int(class_count),
        "top_k": TOP_K,
        "top_features": top_features,
        "feature_names": feature_names,
        "importance": client_importance.tolist()
    }


    output_file = (
        OUTPUT_DIR
        / f"client_{client_id}_shap.json"
    )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            signature,
            file,
            indent=2
        )


    # Store for combined file
    all_signatures[
        client_id
    ] = signature


    print(
        "\nSaved:",
        output_file
    )


# ============================================================
# SAVE ALL CLIENT SIGNATURES TO ONE FILE
# ============================================================

combined_file = (
    OUTPUT_DIR
    / "all_client_shap_signatures.json"
)


with open(
    combined_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_signatures,
        file,
        indent=2
    )


print("\n" + "=" * 80)
print("ALL CLIENT SHAP EVALUATION COMPLETED")
print("=" * 80)

print(
    "Clients processed:",
    len(all_signatures)
)

print(
    "Output directory:",
    OUTPUT_DIR
)

print(
    "Combined file:",
    combined_file
)