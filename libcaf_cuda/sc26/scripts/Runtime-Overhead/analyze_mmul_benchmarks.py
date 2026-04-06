#!/usr/bin/env python3

import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Directory containing benchmark outputs
DATA_DIR = "/student/nqr159/data/mmul-actor-test/benchmark-results"

# Patterns
driver_files = glob.glob(f"{DATA_DIR}/matrix_mul_driver_run*.txt")
actor_files = glob.glob(f"{DATA_DIR}/test_run*.txt")

# Regex
size_pattern = re.compile(r"N=(\d+)")
driver_total_pattern = re.compile(r"TOTAL:\s+([\d.]+)")
actor_total_pattern = re.compile(r"TOTAL end-to-end:\s+([\d.]+)")

driver_data = defaultdict(list)
actor_data = defaultdict(list)


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


# Parse files
for f in driver_files:
    parse_driver(f)

for f in actor_files:
    parse_actor(f)


# Compute means
sizes = sorted(driver_data.keys())

driver_means = [np.mean(driver_data[s]) for s in sizes]
actor_means = [np.mean(actor_data[s]) for s in sizes]

# Compute differences
abs_diff = [a - d for d, a in zip(driver_means, actor_means)]
speedup = [d / a if a != 0 else float('inf') for d, a in zip(driver_means, actor_means)]
percent_diff = [((a - d) / d) * 100 if d != 0 else 0 for d, a in zip(driver_means, actor_means)]

print("===== MEAN RESULTS =====")
print("N     | CUDA (ms) | Actors (ms) | Diff (ms) | % Diff | Speedup (CUDA/Actors)")
print("-" * 80)

for s, d, a, diff, pct, sp in zip(sizes, driver_means, actor_means, abs_diff, percent_diff, speedup):
    print(f"N={s:5d} | {d:10.3f} | {a:11.3f} | {diff:9.3f} | {pct:7.2f}% | {sp:8.3f}")


# Plot
plt.figure()

plt.plot(sizes, driver_means, marker='o', label="CUDA (Driver)")
plt.plot(sizes, actor_means, marker='s', label="CUDA Actors")

plt.xlabel("Matrix Size (N)")
plt.ylabel("Mean Execution Time (ms)")
plt.title("CUDA vs CUDA Actor Matrix Multiplication Performance")
plt.legend()
plt.grid(True)

plt.savefig(f"{DATA_DIR}/mmul_benchmark_plot.png", dpi=300)

plt.show()
