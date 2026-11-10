import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "output"
df = pd.read_csv(OUT / "test_predictions.csv")

correct = df[df["label"] == df["pred"]]
wrong = df[df["label"] != df["pred"]]

plt.figure(figsize=(9, 4))
plt.hist(correct["prob_positive"], bins=30, alpha=0.5, label="correct", color="steelblue")
plt.hist(wrong["prob_positive"], bins=30, alpha=0.7, label="wrong", color="indianred")
plt.axvline(0.5, color="black", linestyle="--", label="decision threshold")
plt.xlabel("P(positive)")
plt.ylabel("Count")
plt.legend()
plt.title("Distribution of predicted probabilities")
plt.tight_layout()
plt.savefig(OUT / "prob_distribution.png", dpi=200)
plt.close()
print("saved output/prob_distribution.png")
