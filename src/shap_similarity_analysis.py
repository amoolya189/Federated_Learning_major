from pathlib import Path
import json

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SIGNATURE_DIR = (
    PROJECT_DIR
    / "outputs"
    / "shap_signatures"
)

OUTPUT_FILE = (
    SIGNATURE_DIR
    / "shap_similarity_matrix.json"
)


# ============================================================
# LOAD CLIENT SIGNATURES
# ============================================================

signature_files = sorted(
    SIGNATURE_DIR.glob("client_*_shap.json")
)

if not signature_files:
    raise FileNotFoundError(
        f"No SHAP signature files found in: {SIGNATURE_DIR}"
    )


signatures = {}
client_ids = []


for signature_file in signature_files:

    with open(
        signature_file,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    client_id = data["client_id"]

    importance = np.asarray(
        data["importance"],
        dtype=np.float64
    )

    signatures[client_id] = importance
    client_ids.append(client_id)


print("SHAP signatures loaded:", len(signatures))
print("Clients:", client_ids)


# ============================================================
# CHECK SIGNATURES
# ============================================================

feature_counts = {
    client_id: len(signature)
    for client_id, signature in signatures.items()
}

unique_feature_counts = set(
    feature_counts.values()
)

if len(unique_feature_counts) != 1:

    raise ValueError(
        "Clients do not have the same number of SHAP features: "
        f"{feature_counts}"
    )


feature_count = next(
    iter(unique_feature_counts)
)

print(
    "Features per signature:",
    feature_count
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        / (norm_a * norm_b)
    )


# ============================================================
# PAIRWISE SIMILARITY MATRIX
# ============================================================

similarity_matrix = {}


for client_a in client_ids:

    similarity_matrix[client_a] = {}

    for client_b in client_ids:

        similarity = cosine_similarity(
            signatures[client_a],
            signatures[client_b]
        )

        similarity_matrix[client_a][
            client_b
        ] = similarity


# ============================================================
# PRINT MATRIX
# ============================================================

print("\n" + "=" * 90)
print("PAIRWISE SHAP COSINE SIMILARITY")
print("=" * 90)

header = (
    "Client".ljust(10)
    + "".join(
        client_id.rjust(10)
        for client_id in client_ids
    )
)

print(header)
print("-" * len(header))


for client_a in client_ids:

    row = client_a.ljust(10)

    for client_b in client_ids:

        value = similarity_matrix[
            client_a
        ][
            client_b
        ]

        row += f"{value:10.4f}"

    print(row)


# ============================================================
# OFF-DIAGONAL STATISTICS
# ============================================================

pairwise_values = []


for i, client_a in enumerate(client_ids):

    for j, client_b in enumerate(client_ids):

        if i < j:

            value = similarity_matrix[
                client_a
            ][
                client_b
            ]

            pairwise_values.append(value)


pairwise_values = np.asarray(
    pairwise_values,
    dtype=np.float64
)


print("\n" + "=" * 90)
print("PAIRWISE SIMILARITY STATISTICS")
print("=" * 90)

print(
    "Number of client pairs:",
    len(pairwise_values)
)

print(
    f"Minimum similarity: {pairwise_values.min():.4f}"
)

print(
    f"Maximum similarity: {pairwise_values.max():.4f}"
)

print(
    f"Mean similarity:    {pairwise_values.mean():.4f}"
)

print(
    f"Median similarity:  {np.median(pairwise_values):.4f}"
)

print(
    f"Std deviation:      {pairwise_values.std():.4f}"
)


# ============================================================
# MOST SIMILAR / LEAST SIMILAR PAIRS
# ============================================================

pairs = []


for i, client_a in enumerate(client_ids):

    for j, client_b in enumerate(client_ids):

        if i < j:

            similarity = similarity_matrix[
                client_a
            ][
                client_b
            ]

            pairs.append(
                (
                    client_a,
                    client_b,
                    similarity
                )
            )


pairs_sorted = sorted(
    pairs,
    key=lambda item: item[2]
)


print("\nLeast similar client pairs:")

for client_a, client_b, similarity in pairs_sorted[:5]:

    print(
        f"  Client {client_a} ↔ Client {client_b}"
        f" : {similarity:.4f}"
    )


print("\nMost similar client pairs:")

for client_a, client_b, similarity in reversed(
    pairs_sorted[-5:]
):

    print(
        f"  Client {client_a} ↔ Client {client_b}"
        f" : {similarity:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_data = {
    "clients": client_ids,
    "feature_count": feature_count,
    "similarity_metric": "cosine_similarity",
    "similarity_matrix": similarity_matrix,
    "pairwise_statistics": {
        "number_of_pairs": int(len(pairwise_values)),
        "minimum": float(pairwise_values.min()),
        "maximum": float(pairwise_values.max()),
        "mean": float(pairwise_values.mean()),
        "median": float(np.median(pairwise_values)),
        "std": float(pairwise_values.std())
    }
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output_data,
        file,
        indent=2
    )


print("\n" + "=" * 90)
print("SHAP SIMILARITY ANALYSIS COMPLETED")
print("=" * 90)

print(
    "Saved:",
    OUTPUT_FILE
)