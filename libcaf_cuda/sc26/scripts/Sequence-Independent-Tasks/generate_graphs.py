#!/usr/bin/env python3

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# -----------------------------
# Path Configuration
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Check if benchmark_results.txt is in the same directory as the script
INPUT_FILE = SCRIPT_DIR / "benchmark_results.txt"
OUTPUT_PLOT = SCRIPT_DIR / "mmul_comparison.png"

# Fallback to the original parent-directory structure if not found locally
if not INPUT_FILE.exists():
    BASE_DIR = SCRIPT_DIR.parent.parent / "Sequence-Independent-Tasks" / "results"
    INPUT_FILE = BASE_DIR / "benchmark_results.txt"
    OUTPUT_PLOT = BASE_DIR / "mmul_comparison.png"

# -----------------------------
# Data containers
# -----------------------------
cuda_data = defaultdict(list)
actor_facade_data = defaultdict(list)
command_runner_data = defaultdict(list)

# -----------------------------
# Regex Patterns
# -----------------------------
cuda_pattern = re.compile(r"iterations\s*=\s*(\d+),\s*total GPU time\s*=\s*([0-9.]+)")
actor_pattern = re.compile(r"iterations\s*=\s*(\d+),\s*total_time\s*=\s*([0-9.]+)")
runner_pattern = re.compile(r"iterations\s*=\s*(\d+),\s*time\s*=\s*([0-9.]+)")

# -----------------------------
# Parse benchmark_results.txt
# -----------------------------
current_impl = None

with open(INPUT_FILE, "r") as f:
    for line in f:
        # Detect block switches
        if "========== main_cuda_native ==========" in line:
            current_impl = "cuda"
            continue
        elif "========== main_actor_facade ==========" in line:
            current_impl = "facade"
            continue
        elif "========== main_command_runner ==========" in line:
            current_impl = "runner"
            continue
            
        # Parse data depending on the active block section
        if current_impl == "cuda":
            m = cuda_pattern.search(line)
            if m:
                it = int(m.group(1))
                cuda_data[it].append(float(m.group(2)))
        elif current_impl == "facade":
            m = actor_pattern.search(line)
            if m:
                it = int(m.group(1))
                actor_facade_data[it].append(float(m.group(2)))
        elif current_impl == "runner":
            m = runner_pattern.search(line)
            if m:
                it = int(m.group(1))
                command_runner_data[it].append(float(m.group(2)))

# -----------------------------
# Aggregate and Compute Means
# -----------------------------
iterations = sorted(cuda_data.keys())

cuda_mean = [np.mean(cuda_data[i]) for i in iterations]
actor_mean = [np.mean(actor_facade_data[i]) for i in iterations]
runner_mean = [np.mean(command_runner_data[i]) for i in iterations]

# -----------------------------
# Print Performance Table
# -----------------------------
print("\nMean Performance Comparison\n")
print(f"{'Iterations':>10} {'CUDA(ms)':>12} {'Facade(ms)':>12} {'Runner(ms)':>12} {'Facade Ovhd %':>15}")

for i, it in enumerate(iterations):
    pct = ((actor_mean[i] - cuda_mean[i]) / cuda_mean[i]) * 100
    print(f"{it:>10} {cuda_mean[i]:>12.2f} {actor_mean[i]:>12.2f} {runner_mean[i]:>12.2f} {pct:>14.2f}%")

# -----------------------------
# Plot and Save Chart
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(iterations, cuda_mean, marker='o', label="CUDA Native")
ax.plot(iterations, actor_mean, marker='s', label="Actor Facade")
ax.plot(iterations, runner_mean, marker='^', label="Command Runner")

ax.set_xlabel("Iterations")
ax.set_ylabel("Time (ms)")
ax.set_title("Matrix Multiplication Performance")
ax.legend()
ax.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_PLOT)
# plt.show() # Uncomment if running in an interactive graphical interface