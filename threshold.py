"""Threshold tuning for the Random Forest, chosen on TRAIN only (out-of-fold), reported once on TEST."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

SEED = 42
CAT = ["Type"]
NUM = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
df = pd.read_csv("data/ai4i2020.csv")
X, y = df[CAT + NUM], df["Machine failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)  # same split as train.py

fitted = joblib.load("models/random_forest.joblib")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = cross_val_predict(clone(fitted), X_train, y_train, cv=cv, method="predict_proba")[:, 1]

# F2 weights recall twice as much as precision. Pick the threshold maximising it on out-of-fold train predictions.
grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
f2 = [fbeta_score(y_train, oof >= t, beta=2) for t in grid]
best = float(grid[int(np.argmax(f2))])
print(f"Chosen threshold (max OOF F2 on train): {best}")

proba = fitted.predict_proba(X_test)[:, 1]
rows = []
for name, t in [("default 0.50", 0.5), (f"tuned {best}", best)]:
    pred = proba >= t
    rows.append({"threshold": name, "precision": round(precision_score(y_test, pred), 3),
                 "recall": round(recall_score(y_test, pred), 3), "f1": round(fbeta_score(y_test, pred, beta=1), 3),
                 "f2": round(fbeta_score(y_test, pred, beta=2), 3), "false_alarms": int(((pred == 1) & (y_test == 0)).sum()),
                 "missed": int(((pred == 0) & (y_test == 1)).sum())})
res = pd.DataFrame(rows); print(res.to_string(index=False))
res.to_csv("reports/threshold_tuning.csv", index=False)
