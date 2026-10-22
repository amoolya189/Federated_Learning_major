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
    x = data["x"].astype(np.float32)
    y = data["y_multiclass"].astype(np.int64)
    return x, y


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

    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(device))
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(targets.numpy())

    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "macro_f1": round(
            float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
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

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=learning_rate,
    )

    loss_function = nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        for features, targets in loader:
            optimizer.zero_grad()

            logits = local_model(features.to(device))
            loss = loss_function(
                logits,
                targets.to(device),
            )

            loss.backward()
            optimizer.step()

    return local_model.cpu()


def fedavg(
    models: list[nn.Module],
    sample_counts: list[int],
) -> dict[str, torch.Tensor]:
    total_samples = sum(sample_counts)
    aggregated: dict[str, torch.Tensor] = {}

    for key in models[0].state_dict():
        weighted_value = sum(
            model.state_dict()[key].float()
            * (count / total_samples)
            for model, count in zip(models, sample_counts)
        )

        aggregated[key] = weighted_value

    return aggregated


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
            "Malicious client IDs must be integers, for example: 1,6"
        ) from error


def save_local_model(
    model: nn.Module,
    output_dir: Path,
    round_number: int,
    client_id: int,
) -> None:
    round_dir = output_dir / "local_models" / f"round_{round_number:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    model_path = round_dir / f"client_{client_id:02d}.pt"

    torch.save(
        model.state_dict(),
        model_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Poisoned FedAvg baseline for TON_IoT intrusion detection "
            "with per-round local model logging."
        )
    )

    parser.add_argument(
        "--encoded-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--malicious-client-ids",
        default="",
        help=(
            "Comma-separated, zero-based client IDs "
            "that perform label flipping."
        ),
    )

    args = parser.parse_args()

    malicious_client_ids = parse_client_ids(
        args.malicious_client_ids
    )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {args.output_dir}"
        )

    set_seed(args.seed)

    device = torch.device("cpu")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    client_files = sorted(
        (args.encoded_dir / "clients").glob("client_*.npz")
    )

    if not client_files:
        raise SystemExit(
            "No encoded client files found."
        )

    clients = [
        load_dataset(path)
        for path in client_files
    ]

    if any(
        client_id < 0 or client_id >= len(clients)
        for client_id in malicious_client_ids
    ):
        raise SystemExit(
            f"Client IDs must be between 0 and {len(clients) - 1}."
        )

    validation_x, validation_y = load_dataset(
        args.encoded_dir / "server_validation.npz"
    )

    test_x, test_y = load_dataset(
        args.encoded_dir / "test.npz"
    )

    input_size = clients[0][0].shape[1]

    output_size = int(
        max(
            np.max(y)
            for _, y in clients
        )
    ) + 1

    global_model = IDSModel(
        input_size,
        output_size,
    ).to(device)

    history = []

    print(
        f"Malicious clients: "
        f"{sorted(malicious_client_ids)}"
    )

    print(
        "Attack: multiclass label flipping"
    )

    print(
        "Local model logging: enabled"
    )

    for round_number in range(
        1,
        args.rounds + 1,
    ):
        local_models = []
        sample_counts = []

        round_client_info = []

        for client_id, (
            client_x,
            client_y,
        ) in enumerate(clients):

            local_y = client_y.copy()

            is_malicious = (
                client_id in malicious_client_ids
            )

            if is_malicious:
                local_y = (
                    local_y + 1
                ) % output_size

            local_model = train_client(
                global_model=global_model,
                x=client_x,
                y=local_y,
                device=device,
                batch_size=args.batch_size,
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate,
            )

            # ------------------------------------------------
            # Save the trained local model for later SHAP
            # analysis.
            # ------------------------------------------------

            save_local_model(
                model=local_model,
                output_dir=args.output_dir,
                round_number=round_number,
                client_id=client_id,
            )

            local_models.append(
                local_model
            )

            sample_counts.append(
                len(client_y)
            )

            round_client_info.append(
                {
                    "client_id": client_id,
                    "malicious": is_malicious,
                    "sample_count": len(client_y),
                    "label_flipped": is_malicious,
                }
            )

        # ----------------------------------------------------
        # FedAvg aggregation remains unchanged.
        # ----------------------------------------------------

        global_model.load_state_dict(
            fedavg(
                local_models,
                sample_counts,
            )
        )

        validation_metrics = evaluate(
            global_model,
            validation_x,
            validation_y,
            device,
            args.batch_size,
        )

        history.append(
            {
                "round": round_number,
                "validation": validation_metrics,
                "clients": round_client_info,
            }
        )

        print(
            f"Round {round_number}/{args.rounds} | "
            f"validation accuracy: "
            f"{validation_metrics['accuracy']:.4f} | "
            f"validation macro-F1: "
            f"{validation_metrics['macro_f1']:.4f}"
        )

    test_metrics = evaluate(
        global_model,
        test_x,
        test_y,
        device,
        args.batch_size,
    )

    results = {
        "method": "Poisoned FedAvg baseline",
        "attack_type": "multiclass label flipping",
        "malicious_clients": sorted(
            malicious_client_ids
        ),
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "input_features": input_size,
        "classes": output_size,
        "local_models_saved": True,
        "history": history,
        "final_test_metrics": test_metrics,
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    torch.save(
        global_model.state_dict(),
        args.output_dir / "global_model.pt",
    )

    print(
        f"Final test accuracy: "
        f"{test_metrics['accuracy']:.4f} | "
        f"Final test macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Results saved to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
