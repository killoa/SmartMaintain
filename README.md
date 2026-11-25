# SmartMaintain MLOps — Machine Failure Detection

Supervised binary classification on the AI4I 2020 Predictive Maintenance dataset (UCI, synthetic):
given product type and five operating measurements, predict `Machine failure` (0/1) and its probability.
Users: industrial maintenance teams.

**Limitation:** the data is synthetic and describes failure under the recorded conditions. It defines no prediction
horizon or remaining useful life, and the model is not validated for real industrial deployment.

## Dataset
Source: https://archive.ics.uci.edu/dataset/601/ai4i (file `ai4i2020.csv`, place it in `data/`).
Verified locally: 10,000 rows, 14 columns, 0 missing values, 0 duplicate rows, 339 failures (3.39%), 9,661 non-failures.
Inputs (only these six): `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
Excluded to prevent leakage: `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

## Install and run (Windows PowerShell; use `source .venv/bin/activate` on Linux/macOS)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python eda.py            # prints stats, saves figures in reports/
python train.py          # trains 2 models, logs 2 MLflow runs, saves models/ and reports/
mlflow ui --backend-store-uri sqlite:///mlflow.db     # open http://127.0.0.1:5000
```
`data/ai4i2020.csv` is not stored in Git; it is versioned with DVC (pointer file `data/ai4i2020.csv.dvc`).
DVC remote: DagsHub (`https://dagshub.com/killoa/SmartMaintain.dvc`). To fetch the data, set your own DagsHub
credentials locally (never commit them), then pull:
```
dvc remote modify --local storage auth basic
dvc remote modify --local storage user <dagshub-username>
dvc remote modify --local storage password <dagshub-token>
dvc pull
```
Fallback: download the CSV from the UCI link above into `data/`; `dvc status` should then report it up to date.

Optional threshold tuning (Random Forest, threshold chosen on out-of-fold train predictions): `python threshold.py`
-> `reports/threshold_tuning.csv`. At threshold 0.32 test recall rises 0.66 -> 0.82 while precision falls 0.71 -> 0.54.

## Results (held-out 20% test set, seed 42, stratified split; failure class = positive)
| Model | Precision | Recall | F1 | Avg. precision | Accuracy | CV F1 (5-fold, train only) |
|---|---|---|---|---|---|---|
| Logistic Regression (balanced, scaled) | 0.142 | 0.824 | 0.242 | 0.382 | 0.825 | 0.235 ± 0.010 |
| Random Forest (200 trees, balanced) | 0.714 | 0.662 | 0.687 | 0.762 | 0.980 | 0.668 ± 0.043 |

Random Forest was preferred on training-partition cross-validation; the test set was not used for selection.
Always predicting "no failure" would already give 96.61% accuracy, so accuracy is only context.
The test set holds 68 failures, so these figures are noisy. Evidence is in `reports/`.

## Status
Implemented: EDA, two baseline pipelines (Pipeline + ColumnTransformer), local MLflow tracking (SQLite), local Git, DVC data tracking with a DagsHub remote (data pushed).
Planned (not done): GitHub remote, FastAPI, Docker, GitHub Actions CI/CD, Evidently monitoring, optional Prefect.
Note: the model is logged to MLflow as a joblib artifact because MLflow's skops serializer rejected the Random Forest tree type.
