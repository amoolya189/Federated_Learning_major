1. Create a virtual environment for python to activate(after python is installed)
   
python -m venv venv

3. After cloning the repository, and installing requirements.txt
   
pip install requirements.txt

5. Also following this format the data folder is placed and the data is downloaded as zipa nd extracteed from link<br> (https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset)
```text
majorproject/
├── data/
├── env/
├── outputs/
├── src/
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

6. Once the environment is set up, and installations are done(Please take the help of AI (Chats))<br>
    keep running the src files<br>
   python src\FILENAME.py --arguments

   (AI will help for No.of rounds for training and others)


   the following are some of them for reference
   ## Running the Project (Windows PowerShell)

Run all commands from the project root directory.

### Step 1: Activate the virtual environment

```powershell
cd "$env:LOCALAPPDATA\Projects\majorproject"
.\env\Scripts\Activate.ps1
```

### Step 2: Prepare the dataset

```powershell
python src\prepare_data.py --input data\raw\train_test_network.csv --output-dir data\processed
```

### Step 3: Create non-IID federated clients

```powershell
python src\create_clients.py --input data\processed\federated_pool.csv --output-dir data\clients --clients 10 --alpha 0.3 --seed 42
```

### Step 4: Check the encoding script arguments

```powershell
python src\encode_data.py --help
```

Run the encoder using its supported arguments to generate the `.npz` files required in `data\encoded_final`. The expected inputs for federated training are:

- `data\encoded_final\clients\client_*.npz`
- `data\encoded_final\server_validation.npz`
- `data\encoded_final\test.npz`

### Step 5: Run benign FedAvg

```powershell
python src\federated_poisoned.py --encoded-dir data\encoded_final --output-dir outputs\benign_fedavg_roundwise --rounds 10
```

### Step 6: Run poisoned FedAvg

```powershell
python src\federated_poisoned.py --encoded-dir data\encoded_final --output-dir outputs\poisoned_fedavg --rounds 10 --malicious-client-ids 1,6
```

### Step 7: Run explainable trust-aware federated learning

```powershell
python src\federated_explainable_trust.py --encoded-dir data\encoded_final --output-dir outputs\explainable_trust_smoke --rounds 3 --malicious-client-ids 1,6
```

### Step 8: Analyze SHAP explanations across rounds

```powershell
python src\shap_roundwise_trajectory.py --model-dir outputs\benign_fedavg_roundwise --encoded-dir data\encoded_final --output-dir outputs\benign_fedavg_roundwise
```

### Step 9: Inspect results

```powershell
Get-ChildItem outputs -Recurse
Get-Content outputs\benign_fedavg_roundwise\metrics.json
```

**Notes:**
- The data preparation and client-generation commands assume the input CSV is located at `data\raw\train_test_network.csv`.
- Run the encoding step before starting federated training.
- Use a fresh or empty output directory when rerunning scripts that refuse to overwrite existing outputs.
- Generated datasets, client partitions, trained models, and experiment results can be recreated; they generally do not need to be committed to GitHub.
- The commands above reflect the scripts and paths discussed in the project conversation. Verify script-specific arguments with `--help` if the local files have changed.



6. for viewing the dashboard<br>
   streamlit run app.py
