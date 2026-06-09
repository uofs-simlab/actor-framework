#!/usr/bin/env python3

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Path configuration
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR.parent.parent / "Runtime-Overhead" / "results"
INPUT_FILE = DATA_DIR / "benchmark_results.txt"

OUTPUT_PLOT = DATA_DIR / "mmul_benchmark_plot.png"

# -----------------------------
# Data containers
# -----------------------------
driver_means = {}
actor_means = {}
latency_data = {}

# -----------------------------
# Read benchmark file
# -----------------------------
with open(INPUT_FILE, "r") as f:
    text = f.read()

# --------------------------------------------------
# CUDA Native Statistics Section
# --------------------------------------------------
cuda_section = re.search(
    r"Implementation:\s*cuda_native(.*?)Implementation:\s*actor_facade",
    text,
    re.DOTALL,
)

if cuda_section:
    stats = re.findall(
        r"^\s*(\d+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+",
        cuda_section.group(1),
        re.MULTILINE,
    )

    for n, mean in stats:
        n = int(n)

        # Skip raw run-number rows
        if n < 1000:
            continue

        driver_means[n] = float(mean)

# --------------------------------------------------
# Command Runner Statistics Section
# --------------------------------------------------
runner_section = re.search(
    r"Implementation:\s*command_runner(.*?)={10,}",
    text,
    re.DOTALL,
)

if runner_section:
    stats = re.findall(
        r"^\s*(\d+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+",
        runner_section.group(1),
        re.MULTILINE,
    )

    for n, mean in stats:
        n = int(n)

        # Skip raw run-number rows
        if n < 1000:
            continue

        actor_means[n] = float(mean)

# --------------------------------------------------
# Actor Facade Latency Data
# --------------------------------------------------
latency_matches = re.findall(
    r"\[LATENCY TEST\]\s+matrix_size=(\d+),\s*time=(\d+)\s*ms",
    text,
)

for n, t in latency_matches:
    n = int(n)
    t = float(t)

    latency_data.setdefault(n, []).append(t)

latency_means = {
    n: np.mean(times)
    for n, times in latency_data.items()
}

# --------------------------------------------------
# Sizes
# --------------------------------------------------
sizes = sorted(
    set(driver_means.keys())
    & set(actor_means.keys())
    & set(latency_means.keys())
)

# --------------------------------------------------
# Extract ordered values
# --------------------------------------------------
driver_vals = [driver_means[s] for s in sizes]
actor_vals = [actor_means[s] for s in sizes]
latency_vals = [latency_means[s] for s in sizes]

# --------------------------------------------------
# Metrics
# --------------------------------------------------
abs_diff = [
    a - d
    for d, a in zip(driver_vals, actor_vals)
]

speedup = [
    d / a if a != 0 else float("inf")
    for d, a in zip(driver_vals, actor_vals)
]

percent_diff = [
    ((a - d) / d) * 100
    if d != 0 else 0
    for d, a in zip(driver_vals, actor_vals)
]

# --------------------------------------------------
# Print results
# --------------------------------------------------
print("===== MEAN RESULTS =====")
print(
    "N     | CUDA (ms) | Command-Runner (ms) | "
    "Actor-Facade (ms) | % Diff | Speedup"
)
print("-" * 100)

for s, d, a, l, pct, sp in zip(
    sizes,
    driver_vals,
    actor_vals,
    latency_vals,
    percent_diff,
    speedup,
):
    print(
        f"N={s:5d} | "
        f"{d:10.3f} | "
        f"{a:19.3f} | "
        f"{l:17.3f} | "
        f"{pct:8.2f}% | "
        f"{sp:8.3f}"
    )

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    sizes,
    driver_vals,
    marker="o",
    label="CUDA (Driver)"
)

plt.plot(
    sizes,
    actor_vals,
    marker="s",
    label="Command-Runner"
)

plt.plot(
    sizes,
    latency_vals,
    marker="^",
    label="Actor-Facade"
)

plt.xlabel("Matrix Size (N)")
plt.ylabel("Mean Execution Time (ms)")
plt.title("CUDA vs Command-Runner vs Actor-Facade")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)

print(f"\nSaved graph: {OUTPUT_PLOT}")

plt.show()