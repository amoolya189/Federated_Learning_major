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
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "macro_f1": round(
            float(f1_score(labels, predictions, average="macro", zero_division=0)),
            6,
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


def fedavg(models: list[nn.Module], sample_counts: list[int]) -> dict[str, torch.Tensor]:
    total_samples = sum(sample_counts)
    aggregated = {}

    for key in models[0].state_dict():
        weighted_value = sum(
            model.state_dict()[key].float() * (count / total_samples)
            for model, count in zip(models, sample_counts)
        )
        aggregated[key] = weighted_value

    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="FedAvg baseline for TON_IoT IDS.")
    parser.add_argument("--encoded-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")

    set_seed(args.seed)
    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client_files = sorted((args.encoded_dir / "clients").glob("client_*.npz"))
    if not client_files:
        raise SystemExit("No encoded client files found.")

    clients = [load_dataset(path) for path in client_files]
    validation_x, validation_y = load_dataset(args.encoded_dir / "server_validation.npz")
    test_x, test_y = load_dataset(args.encoded_dir / "test.npz")

    input_size = clients[0][0].shape[1]
    output_size = int(max(np.max(y) for _, y in clients)) + 1
    global_model = IDSModel(input_size, output_size).to(device)

    history = []

    for round_number in range(1, args.rounds + 1):
        local_models = []
        sample_counts = []

        for client_x, client_y in clients:
            local_model = train_client(
                global_model,
                client_x,
                client_y,
                device,
                args.batch_size,
                args.local_epochs,
                args.learning_rate,
            )
            local_models.append(local_model)
            sample_counts.append(len(client_y))

        global_model.load_state_dict(fedavg(local_models, sample_counts))
        validation_metrics = evaluate(
            global_model,
            validation_x,
            validation_y,
            device,
            args.batch_size,
        )

        history.append({
            "round": round_number,
            "validation": validation_metrics,
        })

        print(
            f"Round {round_number}/{args.rounds} | "
            f"validation accuracy: {validation_metrics['accuracy']:.4f} | "
            f"validation macro-F1: {validation_metrics['macro_f1']:.4f}"
        )

    test_metrics = evaluate(
        global_model,
        test_x,
        test_y,
        device,
        args.batch_size,
    )

    results = {
        "method": "FedAvg baseline",
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "input_features": input_size,
        "classes": output_size,
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