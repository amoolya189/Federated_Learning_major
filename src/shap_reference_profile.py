from pathlib import Path
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Paths
# ============================================================

SIGNATURE_DIR = Path("outputs/shap_signatures")
OUTPUT_FILE = SIGNATURE_DIR / "shap_reference_profile.json"


# ============================================================
# Load client SHAP signatures
# ============================================================

signature_files = sorted(
    SIGNATURE_DIR.glob("client_*_shap.json")
)

if not signature_files:
    raise FileNotFoundError(
        f"No client SHAP signature files found in {SIGNATURE_DIR}"
    )

signatures = {}
feature_names = None

for file_path in signature_files:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    client_id = data["client_id"]
    importance = np.array(data["importance"], dtype=float)

    signatures[client_id] = importance

    if feature_names is None:
        feature_names = data.get("feature_names")


print(f"SHAP signatures loaded: {len(signatures)}")
print(f"Clients: {list(signatures.keys())}")
print(f"Features per signature: {len(next(iter(signatures.values())))}")


# ============================================================
# Build feature-wise median reference
# ============================================================

client_ids = list(signatures.keys())

signature_matrix = np.vstack(
    [signatures[client_id] for client_id in client_ids]
)

median_reference = np.median(
    signature_matrix,
    axis=0
)


# ============================================================
# Calculate similarity to reference
# ============================================================

reference_vector = median_reference.reshape(1, -1)

similarities = {}

for client_id in client_ids:
    client_vector = signatures[client_id].reshape(1, -1)

    similarity = cosine_similarity(
        client_vector,
        reference_vector
    )[0, 0]

    similarities[client_id] = float(similarity)


# ============================================================
# Display results
# ============================================================

print()
print("=" * 80)
print("CLIENT SHAP CONSISTENCY WITH MEDIAN REFERENCE")
print("=" * 80)

for client_id in client_ids:
    print(
        f"Client {client_id} : "
        f"{similarities[client_id]:.4f}"
    )


# ============================================================
# Summary statistics
# ============================================================

similarity_values = np.array(
    list(similarities.values())
)

print()
print("=" * 80)
print("REFERENCE SIMILARITY STATISTICS")
print("=" * 80)

print(f"Minimum similarity : {similarity_values.min():.4f}")
print(f"Maximum similarity : {similarity_values.max():.4f}")
print(f"Mean similarity    : {similarity_values.mean():.4f}")
print(f"Median similarity  : {np.median(similarity_values):.4f}")
print(f"Std deviation      : {similarity_values.std():.4f}")


# ============================================================
# Save reference profile
# ============================================================

output_data = {
    "method": "feature-wise median SHAP reference",
    "client_count": len(client_ids),
    "feature_count": len(median_reference),
    "client_ids": client_ids,
    "feature_names": feature_names,
    "median_reference": median_reference.tolist(),
    "client_reference_similarity": similarities,
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        output_data,
        f,
        indent=2
    )


print()
print("=" * 80)
print("SHAP REFERENCE PROFILE COMPLETED")
print("=" * 80)
print(f"Saved: {OUTPUT_FILE.resolve()}")