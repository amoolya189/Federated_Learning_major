# Trust-Aware Explainable Federated IDS

This project implements a privacy-preserving intrusion-detection prototype for
the TON_IoT network dataset. Ten simulated, non-IID clients collaboratively train
a multiclass IDS without sharing raw client data.

## Research contribution

The proposed method uses three client-trust signals during federated aggregation:

1. **Update similarity**: whether a client update aligns with peer updates.
2. **Validation usefulness**: whether the local model improves a clean server-side validation set.
3. **SHAP explanation consistency**: whether the local model relies on comparable original feature groups when evaluated on shared reference samples.

The aggregation weight is proportional to:

`client sample count × historical trust score`

## Reproducible workflow

Activate the virtual environment first:

```powershell
.\env\Scripts\Activate.ps1
```

Run the full workflow in this order:

```powershell
python src\data_audit.py --input data\raw\train_test_network.csv --output-dir outputs\data_audit
python src\prepare_data.py --input data\raw\train_test_network.csv --output-dir data\processed
python src\create_clients.py --input data\processed\federated_pool.csv --output-dir data\clients --clients 10 --alpha 0.3
python src\encode_data.py --pool data\processed\federated_pool.csv --validation data\processed\server_validation.csv --test data\processed\test.csv --clients-dir data\clients --output-dir data\encoded_final
```

Existing experiment scripts:

```powershell
python src\federated_baseline.py --encoded-dir data\encoded_final --output-dir outputs\fedavg_10rounds --rounds 10
python src\federated_poisoned.py --encoded-dir data\encoded_final --output-dir outputs\poisoned_fedavg --rounds 10 --malicious-client-ids 1,6
python src\federated_trust_weighted.py --encoded-dir data\encoded_final --output-dir outputs\trust_weighted_v1 --rounds 10 --malicious-client-ids 1,6
python src\federated_explainable_trust.py --encoded-dir data\encoded_final --output-dir outputs\explainable_trust_sharedref_final --rounds 10 --malicious-client-ids 1,6
```

## Professional report and dashboard

Install the presentation dependencies once:

```powershell
pip install matplotlib streamlit
```

Generate tables and figures:

```powershell
python src\generate_report.py
```

Launch the dashboard:

```powershell
streamlit run app.py
```

## Results obtained in the current controlled experiment

| Method | Test macro-F1 |
|---|---:|
| Clean FedAvg | 0.6128 |
| Poisoned FedAvg | 0.5380 |
| Trust-weighted FL | 0.5758 |
| Explanation-consistent trust FL | 0.5761 |

The proposed method recovers 0.0381 macro-F1 compared with poisoned FedAvg.
The SHAP signal makes the client-trust decision auditable; in this configuration,
its numerical gain beyond the non-explainable trust baseline is negligible and
should be reported honestly.

## Limitations

- Clients are simulated from a public dataset rather than independent real organizations.
- The evaluation presently uses one controlled label-flipping configuration.
- The shared validation set is an experimental trust anchor; a production system
  would require a carefully governed reference dataset.
