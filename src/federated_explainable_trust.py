from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import shap
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RAW_FEATURES = [
    "src_port", "dst_port", "duration", "src_bytes", "dst_bytes",
    "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts",
    "dst_ip_bytes", "dns_qclass", "dns_qtype", "dns_rcode",
    "http_request_body_len", "http_response_body_len", "http_status_code",
    "proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA",
    "dns_rejected", "ssl_version", "ssl_cipher", "ssl_resumed",
    "ssl_established", "http_trans_depth", "http_method", "http_version",
    "http_orig_mime_types", "http_resp_mime_types", "weird_name",
    "weird_notice",
]


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["x"].astype(np.float32), data["y_multiclass"].astype(np.int64)


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=False,
    )

    predictions = []
    labels = []

    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(device))
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(targets.numpy())

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def train_client(
    global_model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
) -> nn.Module:
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        for features, targets in loader:
            optimizer.zero_grad()
            logits = local_model(features.to(device))
            loss = loss_function(logits, targets.to(device))
            loss.backward()
            optimizer.step()

    return local_model.cpu()


def model_update_vector(
    local_model: nn.Module,
    global_model: nn.Module,
) -> np.ndarray:
    updates = []

    for local_parameter, global_parameter in zip(
        local_model.parameters(),
        global_model.parameters(),
    ):
        delta = (
            local_parameter.detach().cpu()
            - global_parameter.detach().cpu()
        )
        updates.append(delta.flatten().numpy())

    return np.concatenate(updates)


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if np.isclose(minimum, maximum):
        return np.full_like(values, 0.5, dtype=np.float64)

    return (values - minimum) / (maximum - minimum)


def cosine_score(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)

    if denominator == 0:
        return 0.0

    return float(np.clip(np.dot(first, second) / denominator, 0.0, 1.0))


def calculate_update_similarity(update_vectors: list[np.ndarray]) -> np.ndarray:
    updates = np.stack(update_vectors)
    total_update = np.sum(updates, axis=0)
    scores = []

    for index, update in enumerate(updates):
        peer_reference = (total_update - update) / (len(updates) - 1)
        scores.append(cosine_score(update, peer_reference))

    return np.asarray(scores, dtype=np.float64)


def make_feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    groups = {feature: [] for feature in RAW_FEATURES}

    for index, encoded_name in enumerate(feature_names):
        base_name = encoded_name.split("__", 1)[-1]

        matches = [
            feature
            for feature in RAW_FEATURES
            if base_name == feature or base_name.startswith(f"{feature}_")
        ]

        if matches:
            groups[max(matches, key=len)].append(index)

    return groups


def reduce_shap_values(
    shap_values: object,
    encoded_feature_count: int,
) -> np.ndarray:
    if isinstance(shap_values, list):
        values = np.stack([np.asarray(item) for item in shap_values], axis=0)
        return np.mean(np.abs(values), axis=(0, 1))

    values = np.asarray(shap_values)

    if values.ndim == 2:
        return np.mean(np.abs(values), axis=0)

    feature_axes = [
        axis
        for axis, size in enumerate(values.shape)
        if size == encoded_feature_count
    ]

    if not feature_axes:
        raise ValueError(f"Could not identify SHAP feature axis: {values.shape}")

    feature_axis = feature_axes[0]
    average_axes = tuple(
        axis for axis in range(values.ndim) if axis != feature_axis
    )

    return np.mean(np.abs(values), axis=average_axes)


def shap_signature(
    model: nn.Module,
    shared_background_x: np.ndarray,
    shared_anchor_x: np.ndarray,
    feature_groups: dict[str, list[int]],
) -> dict[str, float]:
    background = torch.from_numpy(shared_background_x)
    anchor = torch.from_numpy(shared_anchor_x)

    model.eval()
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(anchor)

    encoded_importance = reduce_shap_values(
        shap_values,
        shared_background_x.shape[1],
    )

    grouped_importance = {}
    for feature, indices in feature_groups.items():
        grouped_importance[feature] = float(np.sum(encoded_importance[indices]))

    total_importance = sum(grouped_importance.values())

    if total_importance > 0:
        grouped_importance = {
            feature: value / total_importance
            for feature, value in grouped_importance.items()
        }

    return grouped_importance


def signature_vector(signature: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [signature.get(feature, 0.0) for feature in RAW_FEATURES],
        dtype=np.float64,
    )


def calculate_explanation_consistency(
    signatures: list[dict[str, float]],
) -> np.ndarray:
    vectors = np.stack([signature_vector(item) for item in signatures])
    scores = []

    for index, vector in enumerate(vectors):
        peer_vectors = np.delete(vectors, index, axis=0)
        robust_reference = np.median(peer_vectors, axis=0)
        scores.append(cosine_score(vector, robust_reference))

    return np.asarray(scores, dtype=np.float64)


def top_features(
    signature: dict[str, float],
    count: int = 5,
) -> list[dict[str, float]]:
    ordered = sorted(
        signature.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:count]

    return [
        {
            "feature": feature,
            "importance": round(float(importance), 6),
        }
        for feature, importance in ordered
    ]


def aggregate_models(
    models: list[nn.Module],
    raw_weights: np.ndarray,
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    normalized_weights = raw_weights / np.sum(raw_weights)
    aggregated_state = {}

    for key in models[0].state_dict():
        aggregated_state[key] = sum(
            model.state_dict()[key].float() * float(weight)
            for model, weight in zip(models, normalized_weights)
        )

    return aggregated_state, normalized_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explainable trust-aware FL for TON_IoT IDS."
    )
    parser.add_argument("--encoded-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-size", type=int, default=16)
    parser.add_argument("--background-size", type=int, default=16)
    parser.add_argument("--trust-momentum", type=float, default=0.6)
    parser.add_argument(
        "--malicious-client-ids",
        default="",
        help="Comma-separated zero-based client IDs that perform label flipping.",
    )
    args = parser.parse_args()

    malicious_client_ids = {
        int(value.strip())
        for value in args.malicious_client_ids.split(",")
        if value.strip()
    }

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")

    set_seed(args.seed)
    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client_files = sorted((args.encoded_dir / "clients").glob("client_*.npz"))
    clients = [load_dataset(path) for path in client_files]

    if any(
        client_id < 0 or client_id >= len(clients)
        for client_id in malicious_client_ids
    ):
        raise SystemExit("A malicious client ID is outside the valid range.")

    validation_x, validation_y = load_dataset(
        args.encoded_dir / "server_validation.npz"
    )
    test_x, test_y = load_dataset(args.encoded_dir / "test.npz")

    encoding_report = json.loads(
        (args.encoded_dir / "encoding_report.json").read_text(encoding="utf-8")
    )
    feature_groups = make_feature_groups(encoding_report["feature_names"])

    anchor_rng = np.random.default_rng(args.seed)
    anchor_indices = anchor_rng.choice(
        len(validation_x),
        size=min(args.anchor_size, len(validation_x)),
        replace=False,
    )
    shared_anchor_x = validation_x[anchor_indices]
    background_indices = anchor_rng.choice(
    len(validation_x),
    size=min(args.background_size, len(validation_x)),
    replace=False,
    )
    shared_background_x = validation_x[background_indices]

    input_size = clients[0][0].shape[1]
    output_size = int(max(np.max(y) for _, y in clients)) + 1
    global_model = IDSModel(input_size, output_size).to(device)

    trust_scores = np.ones(len(clients), dtype=np.float64)
    history = []

    print(f"Malicious clients: {sorted(malicious_client_ids)}")
    print("Attack: multiclass label flipping")
    print("Trust: update similarity + usefulness + SHAP consistency")

    for round_number in range(1, args.rounds + 1):
        previous_global_metrics = evaluate(
            global_model,
            validation_x,
            validation_y,
            device,
            args.batch_size,
        )

        local_models = []
        sample_counts = []
        local_f1_scores = []
        update_vectors = []
        explanation_signatures = []

        for client_id, (client_x, client_y) in enumerate(clients):
            local_y = client_y.copy()

            if client_id in malicious_client_ids:
                local_y = (local_y + 1) % output_size

            local_model = train_client(
                global_model,
                client_x,
                local_y,
                device,
                args.batch_size,
                args.local_epochs,
                args.learning_rate,
            )

            local_metrics = evaluate(
                local_model,
                validation_x,
                validation_y,
                device,
                args.batch_size,
            )

            signature = shap_signature(
                local_model,
                shared_background_x,
                shared_anchor_x,
                feature_groups,
            )

            local_models.append(local_model)
            sample_counts.append(len(client_y))
            local_f1_scores.append(local_metrics["macro_f1"])
            update_vectors.append(model_update_vector(local_model, global_model))
            explanation_signatures.append(signature)

        similarity_scores = calculate_update_similarity(update_vectors)
        usefulness = (
            np.asarray(local_f1_scores)
            - previous_global_metrics["macro_f1"]
        )
        usefulness_scores = min_max_normalize(usefulness)
        explanation_scores = calculate_explanation_consistency(
            explanation_signatures
        )

        evidence_scores = (
            0.45 * similarity_scores
            + 0.20 * usefulness_scores
            + 0.35 * explanation_scores
        )

        updated_trust = (
            args.trust_momentum * trust_scores
            + (1.0 - args.trust_momentum) * evidence_scores
        )
        updated_trust = np.clip(updated_trust, 0.05, 1.0)

        sample_weights = np.asarray(sample_counts, dtype=np.float64)

        if round_number == 1:
            aggregation_name = "FedAvg warm-up"
            aggregation_weights = sample_weights
        else:
            aggregation_name = "explanation-consistent trust-weighted FedAvg"
            aggregation_weights = sample_weights * updated_trust

        aggregated_state, normalized_weights = aggregate_models(
            local_models,
            aggregation_weights,
        )
        global_model.load_state_dict(aggregated_state)
        trust_scores = updated_trust

        validation_metrics = evaluate(
            global_model,
            validation_x,
            validation_y,
            device,
            args.batch_size,
        )

        client_evidence = []
        for client_id in range(len(clients)):
            client_evidence.append(
                {
                    "client_id": client_id,
                    "malicious_simulation": client_id in malicious_client_ids,
                    "sample_count": int(sample_counts[client_id]),
                    "update_similarity": round(float(similarity_scores[client_id]), 6),
                    "usefulness_score": round(float(usefulness_scores[client_id]), 6),
                    "explanation_consistency": round(
                        float(explanation_scores[client_id]),
                        6,
                    ),
                    "trust_score": round(float(trust_scores[client_id]), 6),
                    "aggregation_weight": round(
                        float(normalized_weights[client_id]),
                        6,
                    ),
                    "top_shap_features": top_features(
                        explanation_signatures[client_id]
                    ),
                }
            )

        history.append(
            {
                "round": round_number,
                "aggregation": aggregation_name,
                "validation": validation_metrics,
                "client_evidence": client_evidence,
            }
        )

        print(
            f"Round {round_number}/{args.rounds} | "
            f"{aggregation_name} | "
            f"validation macro-F1: {validation_metrics['macro_f1']:.4f}"
        )
        print(
            "Trust scores: "
            + ", ".join(f"{value:.3f}" for value in trust_scores)
        )

    test_metrics = evaluate(
        global_model,
        test_x,
        test_y,
        device,
        args.batch_size,
    )

    results = {
        "method": "Explanation-consistent trust-weighted FedAvg",
        "dataset": "TON_IoT network dataset",
        "attack": "multiclass label flipping",
        "malicious_clients": sorted(malicious_client_ids),
        "trust_signals": {
            "update_similarity_weight": 0.45,
            "validation_usefulness_weight": 0.20,
            "shap_explanation_consistency_weight": 0.35,
            "historical_trust_momentum": args.trust_momentum,
            "shared_anchor_size": args.anchor_size,
            "background_size": args.background_size,
        },
        "rounds": args.rounds,
        "history": history,
        "final_test_metrics": test_metrics,
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    torch.save(
        global_model.state_dict(),
        args.output_dir / "global_model.pt",
    )

    print(
        f"Final test accuracy: {test_metrics['accuracy']:.4f} | "
        f"Final test macro-F1: {test_metrics['macro_f1']:.4f}"
    )
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()