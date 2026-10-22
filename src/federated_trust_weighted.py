from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
    parts = []

    for local_parameter, global_parameter in zip(
        local_model.parameters(),
        global_model.parameters(),
    ):
        update = (
            local_parameter.detach().cpu()
            - global_parameter.detach().cpu()
        )
        parts.append(update.flatten().numpy())

    return np.concatenate(parts)


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if np.isclose(minimum, maximum):
        return np.full_like(values, 0.5, dtype=np.float64)

    return (values - minimum) / (maximum - minimum)


def calculate_similarity_scores(update_vectors: list[np.ndarray]) -> np.ndarray:
    updates = np.stack(update_vectors)
    total_update = np.sum(updates, axis=0)
    scores = []

    for index, update in enumerate(updates):
        reference = (total_update - update) / (len(updates) - 1)

        denominator = np.linalg.norm(update) * np.linalg.norm(reference)
        cosine_similarity = 0.0 if denominator == 0 else (
            np.dot(update, reference) / denominator
        )

        scores.append((cosine_similarity + 1.0) / 2.0)

    return np.asarray(scores, dtype=np.float64)


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
        description="Trust-weighted FL for TON_IoT intrusion detection."
    )
    parser.add_argument("--encoded-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--malicious-client-ids",
        default="",
        help="Comma-separated zero-based client IDs that perform label flipping.",
    )
    parser.add_argument(
        "--trust-momentum",
        type=float,
        default=0.6,
        help="Historical trust weight between 0 and 1.",
    )
    args = parser.parse_args()

    malicious_client_ids = {
        int(value.strip())
        for value in args.malicious_client_ids.split(",")
        if value.strip()
    }

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")

    if not 0.0 <= args.trust_momentum <= 1.0:
        raise SystemExit("--trust-momentum must be between 0 and 1.")

    set_seed(args.seed)
    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client_files = sorted((args.encoded_dir / "clients").glob("client_*.npz"))
    if not client_files:
        raise SystemExit("No encoded client files found.")

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

    input_size = clients[0][0].shape[1]
    output_size = int(max(np.max(y) for _, y in clients)) + 1
    global_model = IDSModel(input_size, output_size).to(device)

    trust_scores = np.ones(len(clients), dtype=np.float64)
    history = []

    print(f"Malicious clients: {sorted(malicious_client_ids)}")
    print("Attack: multiclass label flipping")
    print("Trust signals: update similarity + validation usefulness")

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
        local_validation_f1 = []
        update_vectors = []

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

            local_models.append(local_model)
            sample_counts.append(len(client_y))
            local_validation_f1.append(local_metrics["macro_f1"])
            update_vectors.append(model_update_vector(local_model, global_model))

        similarity_scores = calculate_similarity_scores(update_vectors)

        usefulness = (
            np.asarray(local_validation_f1)
            - previous_global_metrics["macro_f1"]
        )
        usefulness_scores = min_max_normalize(usefulness)

        evidence_scores = (
            0.7 * similarity_scores
            + 0.3 * usefulness_scores
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
            aggregation_name = "trust-weighted FedAvg"
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
                    "validation_usefulness": round(float(usefulness[client_id]), 6),
                    "usefulness_score": round(float(usefulness_scores[client_id]), 6),
                    "trust_score": round(float(trust_scores[client_id]), 6),
                    "aggregation_weight": round(
                        float(normalized_weights[client_id]),
                        6,
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
        "method": "Trust-weighted FedAvg",
        "dataset": "TON_IoT network dataset",
        "attack": "multiclass label flipping",
        "malicious_clients": sorted(malicious_client_ids),
        "trust_signals": {
            "update_similarity_weight": 0.7,
            "validation_usefulness_weight": 0.3,
            "historical_trust_momentum": args.trust_momentum,
        },
        "rounds": args.rounds,
        "history": history,
        "final_test_metrics": test_metrics,
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    torch.save(global_model.state_dict(), args.output_dir / "global_model.pt")

    print(
        f"Final test accuracy: {test_metrics['accuracy']:.4f} | "
        f"Final test macro-F1: {test_metrics['macro_f1']:.4f}"
    )
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()