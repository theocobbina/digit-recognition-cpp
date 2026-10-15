"""Plot GFLOPS for each matrix-multiply version in results/cpu.csv.

Saves results/matmul_gflops.png, the main chart for the README.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = Path(__file__).resolve().parent.parent / "results"
series = defaultdict(list)  # impl -> [(size, gflops)]

for name in ("cpu.csv",):
    path = results / name
    if not path.exists():
        print(f"Skipping {name} (not found)")
        continue
    with open(path) as f:
        for row in csv.DictReader(f):
            # Skip unfinished implementations: a wrong answer's speed means nothing.
            if float(row["max_err"]) >= 1e-2:
                continue
            series[row["impl"]].append((int(row["size"]), float(row["gflops"])))

if not series:
    raise SystemExit("No passing results to plot yet. Run the benchmarks first.")

fig, ax = plt.subplots(figsize=(8, 5))
for impl, points in series.items():
    points.sort()
    sizes, gflops = zip(*points)
    ax.plot(sizes, gflops, marker="o", label=impl)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Matrix size N (N x N)")
ax.set_ylabel("Throughput (GFLOPS, log scale)")
ax.set_title("Matrix multiply: naive vs cache-friendly loop order")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()

out = results / "matmul_gflops.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
