"""Basic EDA for the AI4I 2020 dataset. Prints facts and saves two figures in reports/."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("data/ai4i2020.csv")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
TARGET = "Machine failure"

df = pd.read_csv(DATA)
print("Shape:", df.shape)
print("\nColumns and dtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())
print("\nDuplicate rows (all columns):", df.duplicated().sum())
print("Duplicate Product IDs:", df["Product ID"].duplicated().sum(), "| Duplicate UDI:", df["UDI"].duplicated().sum())
print("\nDescriptive statistics:\n", df.describe().T)

counts = df[TARGET].value_counts().sort_index()
print("\nTarget distribution:\n", counts, "\nFailure rate: %.2f%%" % (100 * counts[1] / len(df)))
print("Always-predict-no-failure accuracy: %.2f%%" % (100 * counts[0] / len(df)))
print("\nFailure rate by Type:\n", df.groupby("Type")[TARGET].agg(["count", "sum", "mean"]))

# Figure 1: class distribution
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["No failure (0)", "Failure (1)"], counts.values, color=["#4C78A8", "#E45756"])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v} ({100*v/len(df):.2f}%)", ha="center", va="bottom")
ax.set_ylabel("Observations"); ax.set_title("Target class distribution"); ax.set_ylim(0, counts.max() * 1.12)
fig.tight_layout(); fig.savefig(REPORTS / "class_distribution.png", dpi=150); plt.close(fig)

# Figure 2: torque and tool wear by target
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, col in zip(axes, ["Torque [Nm]", "Tool wear [min]"]):
    for cls, color, label in [(0, "#4C78A8", "No failure"), (1, "#E45756", "Failure")]:
        ax.hist(df.loc[df[TARGET] == cls, col], bins=40, alpha=0.6, density=True, color=color, label=label)
    ax.set_xlabel(col); ax.set_ylabel("Density")
axes[0].legend(); fig.suptitle("Feature distributions by target (normalised)")
fig.tight_layout(); fig.savefig(REPORTS / "feature_by_target.png", dpi=150); plt.close(fig)
print("\nFigures saved in reports/")
