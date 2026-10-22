"""Create reproducible, report-ready tables and figures from experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENTS = [
    ("Clean FedAvg", "fedavg_10rounds"),
    ("Poisoned FedAvg", "poisoned_fedavg"),
    ("Trust-weighted FL", "trust_weighted_v1"),
    ("Explanation-consistent trust FL", "explainable_trust_sharedref_final"),
]


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs = project_root / "outputs"
    report_dir = outputs / "professional_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    trajectories: dict[str, list[dict]] = {}
    explainable = None

    for label, folder in EXPERIMENTS:
        result = read_json(outputs / folder / "metrics.json")
        metrics = result["final_test_metrics"]
        histories = result["history"]

        summaries.append(
            {
                "experiment": label,
                "test_accuracy": round(metrics["accuracy"], 6),
                "test_macro_f1": round(metrics["macro_f1"], 6),
                "rounds": result["rounds"],
            }
        )
        trajectories[label] = histories

        if folder == "explainable_trust_sharedref_final":
            explainable = result

    poisoned_f1 = next(
        row["test_macro_f1"]
        for row in summaries
        if row["experiment"] == "Poisoned FedAvg"
    )
    for row in summaries:
        row["macro_f1_recovery_vs_poisoned"] = round(
            row["test_macro_f1"] - poisoned_f1,
            6,
        )

    write_csv(
        report_dir / "experiment_summary.csv",
        list(summaries[0]),
        summaries,
    )

    trajectory_rows = []
    for label, history in trajectories.items():
        for item in history:
            trajectory_rows.append(
                {
                    "experiment": label,
                    "round": item["round"],
                    "validation_accuracy": item["validation"]["accuracy"],
                    "validation_macro_f1": item["validation"]["macro_f1"],
                }
            )

    write_csv(
        report_dir / "roundwise_validation_metrics.csv",
        list(trajectory_rows[0]),
        trajectory_rows,
    )

    final_evidence = explainable["history"][-1]["client_evidence"]
    trust_rows = []
    shap_rows = []

    for item in final_evidence:
        trust_rows.append(
            {
                "client_id": item["client_id"],
                "malicious_simulation": item["malicious_simulation"],
                "sample_count": item["sample_count"],
                "update_similarity": item["update_similarity"],
                "usefulness_score": item["usefulness_score"],
                "explanation_consistency": item["explanation_consistency"],
                "trust_score": item["trust_score"],
                "aggregation_weight": item["aggregation_weight"],
            }
        )

        for rank, feature in enumerate(item["top_shap_features"], start=1):
            shap_rows.append(
                {
                    "client_id": item["client_id"],
                    "malicious_simulation": item["malicious_simulation"],
                    "rank": rank,
                    "feature": feature["feature"],
                    "importance": feature["importance"],
                }
            )

    write_csv(
        report_dir / "final_round_client_trust.csv",
        list(trust_rows[0]),
        trust_rows,
    )
    write_csv(
        report_dir / "final_round_top_shap_features.csv",
        list(shap_rows[0]),
        shap_rows,
    )

    plt.style.use("seaborn-v0_8-whitegrid")

    labels = [row["experiment"] for row in summaries]
    values = [row["test_macro_f1"] for row in summaries]
    colors = ["#4c78a8", "#e45756", "#72b7b2", "#54a24b"]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(labels, values, color=colors)
    axis.set_title("Final Test Macro-F1 Comparison")
    axis.set_ylabel("Macro-F1")
    axis.set_ylim(0, 0.75)
    axis.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(report_dir / "performance_comparison.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for label, history in trajectories.items():
        rounds = [item["round"] for item in history]
        macro_f1 = [item["validation"]["macro_f1"] for item in history]
        axis.plot(rounds, macro_f1, marker="o", label=label)
    axis.set_title("Validation Macro-F1 Across Federated Rounds")
    axis.set_xlabel("Federated round")
    axis.set_ylabel("Macro-F1")
    axis.set_xticks(range(1, 11))
    axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "roundwise_validation_macro_f1.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    client_labels = [f"Client {row['client_id']}" for row in trust_rows]
    trust_values = [row["trust_score"] for row in trust_rows]
    trust_colors = [
        "#e45756" if row["malicious_simulation"] else "#4c78a8"
        for row in trust_rows
    ]
    axis.bar(client_labels, trust_values, color=trust_colors)
    axis.set_title("Final-Round Client Trust Scores")
    axis.set_ylabel("Trust score")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(report_dir / "final_round_trust_scores.png", dpi=220)
    plt.close(figure)

    clean_f1 = summaries[0]["test_macro_f1"]
    explainable_f1 = summaries[-1]["test_macro_f1"]
    recovery = explainable_f1 - poisoned_f1

    summary_text = f"""# Experimental Results Summary

## Dataset and setup

- Dataset: TON_IoT network traffic dataset.
- Clients: 10 simulated non-IID federated clients.
- Task: 10-class intrusion classification.
- Attack: multiclass label flipping by clients 1 and 6.
- Main metric: macro-F1, because attack classes are imbalanced.

## Main result

The clean FedAvg reference achieved a test macro-F1 of **{clean_f1:.4f}**.
Label-flipping poisoning reduced FedAvg to **{poisoned_f1:.4f}**.
The explanation-consistent trust-aware method achieved **{explainable_f1:.4f}**,
recovering **{recovery:.4f}** macro-F1 over poisoned FedAvg.

## Honest interpretation

The proposed method improves robustness relative to poisoned FedAvg. In this
single controlled configuration, SHAP consistency produced performance comparable
to update-similarity/usefulness trust rather than a large independent gain. Its
main contribution is auditable trust: the server can inspect which features local
models rely on when determining whether their updates should influence aggregation.

## Generated artefacts

- `experiment_summary.csv`
- `roundwise_validation_metrics.csv`
- `final_round_client_trust.csv`
- `final_round_top_shap_features.csv`
- `performance_comparison.png`
- `roundwise_validation_macro_f1.png`
- `final_round_trust_scores.png`
"""
    (report_dir / "RESULTS_SUMMARY.md").write_text(summary_text, encoding="utf-8")

    print(f"Professional report artefacts written to: {report_dir}")


if __name__ == "__main__":
    main()
