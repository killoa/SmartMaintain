"""Baseline training: Logistic Regression vs Random Forest, tracked with local MLflow."""
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, ConfusionMatrixDisplay, accuracy_score,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "Machine failure"
# Only these six inputs are allowed (no UDI, Product ID, TWF/HDF/PWF/OSF/RNF -> avoids target leakage)
CAT = ["Type"]
NUM = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
Path("models").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)

df = pd.read_csv("data/ai4i2020.csv")
X, y = df[CAT + NUM], df[TARGET]
# One stratified 80/20 split shared by both models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

def make_pipeline(model, scale):
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT), ("num", num, NUM)])
    return Pipeline([("prep", pre), ("model", model)])  # preprocessing is fit on training data only

models = {
    "logistic_regression": (make_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED), True),
                            {"class_weight": "balanced", "max_iter": 1000, "scaling": "StandardScaler"}),
    "random_forest": (make_pipeline(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1), False),
                      {"n_estimators": 200, "class_weight": "balanced", "scaling": "none"}),
}

mlflow.set_tracking_uri("sqlite:///mlflow.db")   # same URI must be used for `mlflow ui`
mlflow.set_experiment("SmartMaintain-baseline")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, (pipe, params) in models.items():
    # 5-fold CV on the TRAIN partition only (F1 of failure class); test set is not used for selection
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    m = {"precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred),
         "f1": f1_score(y_test, pred), "average_precision": average_precision_score(y_test, proba),
         "accuracy": accuracy_score(y_test, pred), "cv_f1_mean": cv_f1.mean(), "cv_f1_std": cv_f1.std()}
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No failure", "Failure"], ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {name}"); fig.tight_layout()
    cm_path = f"reports/confusion_matrix_{name}.png"; fig.savefig(cm_path, dpi=150); plt.close(fig)
    joblib.dump(pipe, f"models/{name}.joblib")
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_name", name); mlflow.log_param("seed", SEED)
        mlflow.log_param("test_size", 0.2); mlflow.log_params(params)
        mlflow.log_metrics(m); mlflow.log_artifact(cm_path)
        mlflow.log_artifact(f"models/{name}.joblib")
    rows.append({"model": name, **{k: round(v, 4) for k, v in m.items()}})
    print(name, {k: round(v, 4) for k, v in m.items()})

res = pd.DataFrame(rows)
res.to_csv("reports/model_comparison.csv", index=False)
print("\n", res.to_string(index=False))
