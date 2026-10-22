
"""Interactive dashboard for TON_IoT federated IDS experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "outputs" / "professional_report"

st.set_page_config(
    page_title="Trust-Aware Explainable FL IDS",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Trust-Aware Explainable Federated IDS")
st.caption(
    "TON_IoT | 10 simulated non-IID clients | "
    "10-class intrusion detection | Simulated label-flipping attack"
)

st.markdown(
    """
    This dashboard presents experimental results for federated intrusion
    detection, comparing standard FedAvg, poisoned FedAvg, trust-weighted
    federated learning, and explanation-consistent trust-aware aggregation.
    """
)


# ---------------------------------------------------------
# Load report artefacts
# ---------------------------------------------------------

REQUIRED_FILES = [
    "experiment_summary.csv",
    "final_round_client_trust.csv",
    "final_round_top_shap_features.csv",
    "roundwise_validation_metrics.csv",
]

missing_files = [
    filename
    for filename in REQUIRED_FILES
    if not (REPORT_DIR / filename).is_file()
]

if missing_files:
    st.error(
        "Some report files are missing. Run "
        "`python src/generate_report.py` from the project directory."
    )
    st.write("Missing files:", missing_files)
    st.stop()


@st.cache_data
def load_report_data(report_dir: str):
    directory = Path(report_dir)

    summary_df = pd.read_csv(directory / "experiment_summary.csv")
    trust_df = pd.read_csv(directory / "final_round_client_trust.csv")
    shap_df = pd.read_csv(directory / "final_round_top_shap_features.csv")
    roundwise_df = pd.read_csv(directory / "roundwise_validation_metrics.csv")

    return summary_df, trust_df, shap_df, roundwise_df


try:
    summary, trust, shap_features, roundwise = load_report_data(
        str(REPORT_DIR)
    )
except (OSError, pd.errors.ParserError) as exc:
    st.error(f"Could not read the report files: {exc}")
    st.stop()


# ---------------------------------------------------------
# Validate expected experiment data
# ---------------------------------------------------------

if summary.empty:
    st.error("The experiment summary contains no data.")
    st.stop()

required_experiments = {
    "Clean FedAvg",
    "Poisoned FedAvg",
    "Trust-weighted FL",
    "Explanation-consistent trust FL",
}

if not required_experiments.issubset(set(summary["experiment"])):
    st.error(
        "The experiment summary does not contain all four expected "
        "experiments. Regenerate the report and check its source metrics."
    )
    st.stop()

if trust.empty:
    st.warning("No final-round client trust records were found.")

if shap_features.empty:
    st.warning("No SHAP feature records were found.")

if roundwise.empty:
    st.warning("No round-wise validation records were found.")


def experiment_score(experiment_name: str) -> float:
    """Return the test macro-F1 for a named experiment."""
    values = summary.loc[
        summary["experiment"] == experiment_name,
        "test_macro_f1",
    ]

    if values.empty:
        raise ValueError(f"Missing experiment: {experiment_name}")

    return float(values.iloc[0])


clean = experiment_score("Clean FedAvg")
poisoned = experiment_score("Poisoned FedAvg")
trust_weighted = experiment_score("Trust-weighted FL")
proposed = experiment_score("Explanation-consistent trust FL")

recovery = proposed - poisoned
clean_gap = clean - proposed
proposed_vs_trust = proposed - trust_weighted


# ---------------------------------------------------------
# Key performance indicators
# ---------------------------------------------------------

st.header("1. Experimental performance")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Clean FedAvg",
    f"{clean:.4f}",
    help="Test macro-F1 of the clean federated baseline.",
)

metric2.metric(
    "Poisoned FedAvg",
    f"{poisoned:.4f}",
    delta=f"{poisoned - clean:+.4f} vs clean",
    delta_color="inverse",
    help="Test macro-F1 under simulated label-flipping poisoning.",
)

metric3.metric(
    "Trust-weighted FL",
    f"{trust_weighted:.4f}",
    help="Test macro-F1 of the update-trust baseline.",
)

metric4.metric(
    "Proposed method",
    f"{proposed:.4f}",
    delta=f"{recovery:+.4f} vs poisoned",
    help=(
        "Test macro-F1 of explanation-consistent trust-aware FL. "
        "The delta compares it with poisoned FedAvg."
    ),
)

st.caption(
    "Macro-F1 is used as the primary metric because it evaluates "
    "class-wise performance and is useful when classes are imbalanced."
)


# ---------------------------------------------------------
# Performance comparison
# ---------------------------------------------------------

st.header("2. Test performance comparison")

comparison = summary[["experiment", "test_macro_f1"]].copy()
comparison["test_macro_f1"] = pd.to_numeric(
    comparison["test_macro_f1"],
    errors="coerce",
)
comparison = comparison.dropna(subset=["test_macro_f1"])

st.bar_chart(
    comparison.set_index("experiment"),
    y="test_macro_f1",
    y_label="Test macro-F1",
    x_label="Experiment",
    height=380,
)

with st.expander("View complete experiment summary"):
    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------
# Performance recovery interpretation
# ---------------------------------------------------------

st.header("3. Robustness and recovery")

recovery_col1, recovery_col2, recovery_col3 = st.columns(3)

recovery_col1.metric(
    "Recovery over poisoned FedAvg",
    f"{recovery:+.4f}",
)

recovery_col2.metric(
    "Gap from clean baseline",
    f"{clean_gap:.4f}",
)

recovery_col3.metric(
    "Proposed vs trust-weighted FL",
    f"{proposed_vs_trust:+.5f}",
)

st.markdown(
    f"""
    - The simulated poisoning attack reduced macro-F1 from **{clean:.4f}**
      to **{poisoned:.4f}**.
    - The proposed method recovered **{recovery:.4f} macro-F1**
      relative to poisoned FedAvg.
    - Its macro-F1 difference from trust-weighted FL was
      **{proposed_vs_trust:+.5f}**. This small difference does not establish
      a substantial independent performance benefit from explanation
      consistency.
    """
)

st.info(
    "These are results from a controlled simulation. They do not, by "
    "themselves, establish statistical significance or performance on "
    "real deployed IoT devices."
)


# ---------------------------------------------------------
# Round-wise validation performance
# ---------------------------------------------------------

st.header("4. Validation performance across communication rounds")

if not roundwise.empty:
    if {"round", "experiment", "validation_macro_f1"}.issubset(
        roundwise.columns
    ):
        trajectory = roundwise.pivot_table(
            index="round",
            columns="experiment",
            values="validation_macro_f1",
            aggfunc="mean",
        ).sort_index()

        st.line_chart(
            trajectory,
            x_label="Communication round",
            y_label="Validation macro-F1",
            height=420,
        )

        with st.expander("View round-wise validation data"):
            st.dataframe(
                roundwise.sort_values(["experiment", "round"]),
                width="stretch",
                hide_index=True,
            )
    else:
        st.warning(
            "The round-wise metrics file does not contain the expected "
            "columns: round, experiment, validation_macro_f1."
        )


# ---------------------------------------------------------
# Client trust and aggregation weights
# ---------------------------------------------------------

st.header("5. Final-round client trust analysis")

if not trust.empty:
    expected_trust_columns = {
        "client_id",
        "trust_score",
        "aggregation_weight",
    }

    if expected_trust_columns.issubset(trust.columns):
        trust_view = trust.copy()
        trust_view["client"] = (
            "Client " + trust_view["client_id"].astype(str)
        )

        if "malicious_simulation" in trust_view.columns:
            trust_view["simulation_status"] = trust_view[
                "malicious_simulation"
            ].map({True: "Simulated malicious", False: "Benign simulation"})

        st.caption(
            "Trust scores and aggregation weights are shown for the final "
            "communication round. Low trust is a risk signal, not proof "
            "of malicious behavior."
        )

        trust_chart_col, weight_chart_col = st.columns(2)

        with trust_chart_col:
            st.subheader("Client trust scores")
            trust_chart = trust_view.set_index("client")[["trust_score"]]
            st.bar_chart(
                trust_chart,
                y="trust_score",
                y_label="Trust score",
                height=360,
            )

        with weight_chart_col:
            st.subheader("Aggregation weights")
            weight_chart = trust_view.set_index("client")[
                ["aggregation_weight"]
            ]
            st.bar_chart(
                weight_chart,
                y="aggregation_weight",
                y_label="Aggregation weight",
                height=360,
            )

        st.subheader("Client evidence table")

        sort_by = st.selectbox(
            "Sort clients by",
            ["trust_score", "aggregation_weight", "client_id"],
        )

        st.dataframe(
            trust_view.sort_values(sort_by),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "The malicious-client labels in this experiment describe the "
            "simulation setup. They should not be interpreted as labels "
            "available to a real-world detection system."
        )
    else:
        st.warning(
            "The client trust file is missing one or more required columns."
        )


# ---------------------------------------------------------
# SHAP explanation audit
# ---------------------------------------------------------

st.header("6. SHAP explanation audit")

if not shap_features.empty:
    expected_shap_columns = {
        "client_id",
        "rank",
        "feature",
        "importance",
    }

    if expected_shap_columns.issubset(shap_features.columns):
        shap_view = shap_features.copy()
        shap_view["importance"] = pd.to_numeric(
            shap_view["importance"],
            errors="coerce",
        )
        shap_view = shap_view.dropna(subset=["importance"])

        clients = sorted(shap_view["client_id"].unique().tolist())

        if clients:
            selected_client = st.selectbox(
                "Select a client to inspect",
                clients,
                format_func=lambda client_id: f"Client {client_id}",
            )

            client_features = shap_view.loc[
                shap_view["client_id"] == selected_client
            ].sort_values("rank")

            st.caption(
                "Features are ranked by mean absolute SHAP attribution on "
                "the shared reference samples. Higher importance indicates "
                "greater attribution magnitude in this analysis; it does not "
                "by itself imply malicious behavior or causality."
            )

            st.bar_chart(
                client_features.set_index("feature")[["importance"]],
                y="importance",
                y_label="Mean absolute SHAP attribution",
                height=400,
            )

            st.dataframe(
                client_features,
                width="stretch",
                hide_index=True,
            )
        else:
            st.warning("No clients are available in the SHAP feature file.")
    else:
        st.warning(
            "The SHAP file is missing one or more required columns."
        )


# ---------------------------------------------------------
# Methodology and interpretation
# ---------------------------------------------------------

st.header("7. Methodology and limitations")

with st.expander("Experiment setup"):
    st.markdown(
        """
        - **Dataset:** TON_IoT network traffic.
        - **Federated setup:** 10 simulated non-IID clients.
        - **Task:** 10-class intrusion classification.
        - **Attack simulation:** Multiclass label flipping by clients 1 and 6.
        - **Evaluation metric:** Test macro-F1.
        - **Explainability:** SHAP feature-attribution consistency is used
          as part of the trust assessment.
        - **Deployment status:** Experiments are simulated locally; they
          are not evidence of validation on deployed IoT devices.
        """
    )

with st.expander("Interpretation guidance"):
    st.markdown(
        """
        1. Compare all methods using the same evaluation setup.
        2. Distinguish improvements over poisoned FedAvg from improvements
           over other trust-aware methods.
        3. Treat trust scores as evidence for investigation, not as definitive
           malicious-client classifications.
        4. Interpret SHAP importance as feature attribution, not proof of
           causality.
        5. Evaluate multiple random seeds, attack strengths, and client
           configurations before making broad robustness claims.
        """
    )

st.caption(
    "Research dashboard | Generated from the project's existing report "
    "artefacts. Regenerate the report after updating experiment results."
)
