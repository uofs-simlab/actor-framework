#!/usr/bin/env python3

import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# -----------------------------
# Path configuration (MATCHING STYLE OF FIRST SCRIPT)
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR.parent.parent / "Runtime-Overhead" / "results"

driver_files = glob.glob(f"{DATA_DIR}/matrix_mul_driver_run*.txt")
actor_files = glob.glob(f"{DATA_DIR}/test_run*.txt")
latency_files = glob.glob(f"{DATA_DIR}/latency_bench_test_run*.txt")

OUTPUT_PLOT = DATA_DIR / "mmul_benchmark_plot.png"

# -----------------------------
# Regex
# -----------------------------
size_pattern = re.compile(r"N=(\d+)")
driver_total_pattern = re.compile(r"TOTAL:\s+([\d.]+)")
actor_total_pattern = re.compile(r"TOTAL end-to-end:\s+([\d.]+)")

latency_pattern = re.compile(
    r"\[LATENCY TEST\]\s+matrix_size=(\d+),\s*time=(\d+)\s*ms"
)

# -----------------------------
# Data containers
# -----------------------------
driver_data = defaultdict(list)
actor_data = defaultdict(list)
latency_data = defaultdict(list)

# -----------------------------
# Parsers
# -----------------------------
def parse_driver(file):
    with open(file) as f:
        current_size = None
        for line in f:
            size_match = size_pattern.search(line)
            if size_match:
                current_size = int(size_match.group(1))

            total_match = driver_total_pattern.search(line)
            if total_match and current_size:
                driver_data[current_size].append(float(total_match.group(1)))


def parse_actor(file):
    with open(file) as f:
        current_size = None
        for line in f:
            size_match = size_pattern.search(line)
            if size_match:
                current_size = int(size_match.group(1))

            total_match = actor_total_pattern.search(line)
            if total_match and current_size:
                actor_data[current_size].append(float(total_match.group(1)))


def parse_latency(file):
    with open(file) as f:
        for line in f:
            m = latency_pattern.search(line)
            if m:
                size = int(m.group(1))
                time = float(m.group(2))
                latency_data[size].append(time)

# -----------------------------
# Parse files
# -----------------------------
for f in driver_files:
    parse_driver(f)

for f in actor_files:
    parse_actor(f)

for f in latency_files:
    parse_latency(f)

# -----------------------------
# Aggregate
# -----------------------------
sizes = sorted(driver_data.keys())

driver_means = [np.mean(driver_data[s]) for s in sizes]
actor_means = [np.mean(actor_data[s]) for s in sizes]
latency_means = [np.mean(latency_data[s]) for s in sizes]

# -----------------------------
# Metrics
# -----------------------------
abs_diff = [a - d for d, a in zip(driver_means, actor_means)]
speedup = [d / a if a != 0 else float('inf') for d, a in zip(driver_means, actor_means)]
percent_diff = [((a - d) / d) * 100 if d != 0 else 0 for d, a in zip(driver_means, actor_means)]

# -----------------------------
# Print
# -----------------------------
print("===== MEAN RESULTS =====")
print("N     | CUDA (ms) | Actors (ms) | Latency Facade (ms) | % Actor Diff | Speedup")
print("-" * 100)

for s, d, a, l, pct, sp in zip(
    sizes, driver_means, actor_means, latency_means, percent_diff, speedup
):
    print(f"N={s:5d} | {d:10.3f} | {a:11.3f} | {l:18.3f} | {pct:11.2f}% | {sp:8.3f}")

# -----------------------------
# Plot
# -----------------------------
plt.figure()

plt.plot(sizes, driver_means, marker='o', label="CUDA (Driver)")
plt.plot(sizes, actor_means, marker='s', label="Command-Runner")
plt.plot(sizes, latency_means, marker='^', label="Actor-Facade")

plt.xlabel("Matrix Size (N)")
plt.ylabel("Mean Execution Time (ms)")
plt.title("CUDA vs Actors vs Actor-Facade")
plt.legend()
plt.grid(True)

plt.savefig(OUTPUT_PLOT, dpi=300)
plt.show()