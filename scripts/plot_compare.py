"""Plot test accuracy per epoch: the from-scratch C++ network vs the same network in PyTorch.

Reads results/train.csv and results/pytorch.csv, saves results/cpp_vs_pytorch.png.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = Path(__file__).resolve().parent.parent / "results"


def load(name):
    with open(results / name) as f:
        rows = list(csv.DictReader(f))
    return [int(r["epoch"]) for r in rows], [float(r["test_acc"]) for r in rows]


fig, ax = plt.subplots(figsize=(7, 4.5))
for name, label, color in [
    ("train.csv", "C++ from scratch", "#2a6fdb"),
    ("pytorch.csv", "PyTorch (same network)", "#e8710a"),
]:
    epochs, acc = load(name)
    ax.plot(epochs, acc, marker="o", color=color, label=label)
    ax.annotate(f"{acc[-1]:.2f}%", (epochs[-1], acc[-1]), textcoords="offset points",
                xytext=(8, 0), va="center", color=color, fontweight="bold")

ax.set_xlabel("Epoch (one pass over 60,000 training images)")
ax.set_ylabel("Test accuracy (%) on 10,000 unseen images")
ax.set_title("MNIST: hand-written C++ network vs PyTorch")
ax.set_xticks(range(1, 6))
ax.set_xlim(0.8, 5.6)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()

out = results / "cpp_vs_pytorch.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
