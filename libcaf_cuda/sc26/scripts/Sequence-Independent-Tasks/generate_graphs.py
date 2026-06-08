#!/usr/bin/env python3

import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# -----------------------------
# Path Configuration (same style as first script)
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

BASE_DIR = SCRIPT_DIR.parent.parent / "Sequence-Independent-Tasks" / "results"

driver_files = glob.glob(str(BASE_DIR / "matrix_mul_driver_run*.txt"))
actor_files = glob.glob(str(BASE_DIR / "test_run*.txt"))
runner_files = glob.glob(str(BASE_DIR / "throughput_mapping_bench_test_run*.txt"))

OUTPUT_PLOT = BASE_DIR / "mmul_comparison.png"

# -----------------------------
# Data containers
# -----------------------------
cuda_data = defaultdict(list)
actor_facade_data = defaultdict(list)
command_runner_data = defaultdict(list)

# -----------------------------
# Regex
# -----------------------------
cuda_pattern = re.compile(r"iterations\s*=\s*(\d+),\s*total GPU time\s*=\s*([0-9.]+)")
actor_pattern = re.compile(r"iterations\s*=\s*(\d+),\s*time\s*=\s*([0-9.]+)")
runner_pattern = re.compile(r"iterations\s*=\s*(\d+).*total_time\s*=\s*([0-9.]+)\s*ms")

# -----------------------------
# Parsers
# -----------------------------
def parse_cuda(file):
    with open(file) as f:
        for line in f:
            m = cuda_pattern.search(line)
            if m:
                it = int(m.group(1))
                cuda_data[it].append(float(m.group(2)))

def parse_actor(file):
    with open(file) as f:
        for line in f:
            m = actor_pattern.search(line)
            if m:
                it = int(m.group(1))
                actor_facade_data[it].append(float(m.group(2)))

def parse_runner(file):
    with open(file) as f:
        for line in f:
            m = runner_pattern.search(line)
            if m:
                it = int(m.group(1))
                command_runner_data[it].append(float(m.group(2)))

# -----------------------------
# Parse all files
# -----------------------------
for f in driver_files:
    parse_cuda(f)

for f in actor_files:
    parse_actor(f)

for f in runner_files:
    parse_runner(f)

# -----------------------------
# Aggregate
# -----------------------------
iterations = sorted(cuda_data.keys())

cuda_mean = [np.mean(cuda_data[i]) for i in iterations]
actor_mean = [np.mean(actor_facade_data[i]) for i in iterations]
runner_mean = [np.mean(command_runner_data[i]) for i in iterations]

# -----------------------------
# Print
# -----------------------------
print("\nMean Performance Comparison\n")
print(f"{'Iterations':>10} {'CUDA(ms)':>12} {'Facade(ms)':>12} {'Runner(ms)':>12} {'Facade Ovhd %':>15}")

for i, it in enumerate(iterations):
    pct = ((actor_mean[i] - cuda_mean[i]) / cuda_mean[i]) * 100
    print(f"{it:>10} {cuda_mean[i]:>12.2f} {actor_mean[i]:>12.2f} {runner_mean[i]:>12.2f} {pct:>14.2f}%")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8,6))

plt.plot(iterations, cuda_mean, marker='o', label="CUDA Native")
plt.plot(iterations, actor_mean, marker='s', label="Actor Facade")
plt.plot(iterations, runner_mean, marker='^', label="Command Runner")

plt.xlabel("Iterations")
plt.ylabel("Time (ms)")
plt.title("Matrix Multiplication Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_PLOT)
plt.show()