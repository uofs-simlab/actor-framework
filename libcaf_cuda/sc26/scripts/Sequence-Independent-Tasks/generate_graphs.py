#!/usr/bin/env python3

import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

cuda_data = defaultdict(list)
caf_data = defaultdict(list)

# -----------------------------
# Parse CUDA baseline logs
# -----------------------------
for i in range(1, 11):
    filename = f"matrix_mul_driver_run{i}.txt"
    with open(filename) as f:
        for line in f:
            m = re.search(r"iterations = (\d+), total GPU time = ([0-9.]+)", line)
            if m:
                it = int(m.group(1))
                time = float(m.group(2))
                cuda_data[it].append(time)

# -----------------------------
# Parse CAF CUDA logs
# -----------------------------
for i in range(1, 11):
    filename = f"test_run{i}.txt"
    with open(filename) as f:
        for line in f:
            m = re.search(r"iterations = (\d+), time=([0-9.]+)", line)
            if m:
                it = int(m.group(1))
                time = float(m.group(2))
                caf_data[it].append(time)

iterations = sorted(cuda_data.keys())

cuda_mean = [np.mean(cuda_data[it]) for it in iterations]
caf_mean = [np.mean(caf_data[it]) for it in iterations]

# -----------------------------
# CLI Output
# -----------------------------
print("\nMean Performance Comparison\n")
print(f"{'Iterations':>10} {'CUDA(ms)':>12} {'CAF CUDA(ms)':>15} {'Diff(ms)':>12} {'Overhead %':>12}")

for i, it in enumerate(iterations):
    diff = caf_mean[i] - cuda_mean[i]
    pct = (diff / cuda_mean[i]) * 100
    print(f"{it:>10} {cuda_mean[i]:>12.2f} {caf_mean[i]:>15.2f} {diff:>12.2f} {pct:>11.2f}%")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8,6))
plt.plot(iterations, cuda_mean, marker='o', label="CUDA")
plt.plot(iterations, caf_mean, marker='o', label="CAF CUDA")

plt.xlabel("Iterations")
plt.ylabel("Time (ms)")
plt.title("Matrix Multiplication Performance")
plt.legend()
plt.grid(True)

plt.savefig("mmul_comparison.png")
plt.show()
