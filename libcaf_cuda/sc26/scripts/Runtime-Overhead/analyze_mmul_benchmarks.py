#!/usr/bin/env python3

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR.parent.parent / "Runtime-Overhead" / "results"
INPUT_FILE = DATA_DIR / "benchmark_results.txt"
OUTPUT_PLOT = DATA_DIR / "mmul_benchmark_plot.png"

# -----------------------------
# Load file
# -----------------------------
with open(INPUT_FILE, "r") as f:
    text = f.read()


# ----------------------------------------------------------
# Extract statistics table
# ----------------------------------------------------------
def extract_stats(text, implementation):

    pattern = (
        rf"Implementation:\s*{implementation}"
        rf".*?--- Statistics ---"
        rf".*?^-+\n"
        rf"(.*?)\n\n"
    )

    m = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    if not m:
        raise RuntimeError(f"Could not locate statistics for {implementation}")

    stats_text = m.group(1)

    result = {}

    for line in stats_text.splitlines():
        row = re.match(r"\s*(\d+)\s+([\d.]+)", line)
        if not row:
            continue

        n = int(row.group(1))

        # filter out run indices / noise
        if n < 1000:
            continue

        result[n] = float(row.group(2))

    return result


# ----------------------------------------------------------
# Parse benchmark data
# ----------------------------------------------------------
driver_means = extract_stats(text, "cuda_native")
actor_means = extract_stats(text, "command_runner")

latency_data = {}

for n, t in re.findall(
    r"\[LATENCY TEST\]\s+matrix_size=(\d+),\s*time=(\d+)\s*ms",
    text,
):
    latency_data.setdefault(int(n), []).append(float(t))

latency_means = {
    n: np.mean(v)
    for n, v in latency_data.items()
}


# ----------------------------------------------------------
# Align sizes
# ----------------------------------------------------------
sizes = sorted(
    set(driver_means.keys())
    & set(actor_means.keys())
    & set(latency_means.keys())
)

driver_vals = [driver_means[s] for s in sizes]
actor_vals = [actor_means[s] for s in sizes]
latency_vals = [latency_means[s] for s in sizes]


# ----------------------------------------------------------
# Debug print (optional but useful)
# ----------------------------------------------------------
print("sizes       =", sizes)
print("cuda        =", driver_vals)
print("runner      =", actor_vals)
print("facade      =", latency_vals)


# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(
    sizes,
    driver_vals,
    marker='o',
    markersize=8,
    linewidth=2.5,
    linestyle='-',
    label='CUDA Native'
)

plt.plot(
    sizes,
    actor_vals,
    marker='s',
    markersize=8,
    linewidth=2.5,
    linestyle='--',
    label='Command Runner'
)

plt.plot(
    sizes,
    latency_vals,
    marker='^',
    markersize=8,
    linewidth=2.5,
    linestyle=':',
    label='Actor Facade'
)

# Log scale helps since you span ~11 ms → ~1885 ms
plt.yscale("log")

plt.xlabel("Matrix Size (N)")
plt.ylabel("Mean Runtime (ms)")
plt.title("Runtime Overhead Comparison")

plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)

print(f"\nSaved graph to: {OUTPUT_PLOT}")

plt.show()