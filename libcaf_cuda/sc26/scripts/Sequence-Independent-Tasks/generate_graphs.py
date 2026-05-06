#!/usr/bin/env python3

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Path Configuration
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent.parent / "Sequence-Independent-Tasks" / "results"
RESULTS_FILE = RESULTS_DIR / "benchmark_results.txt"
OUTPUT_PLOT = RESULTS_DIR / "mmul_comparison.png"

# -----------------------------
# Parse Benchmark Results
# -----------------------------
def parse_results():
    if not RESULTS_FILE.exists():
        print(f"Error: Results file not found at {RESULTS_FILE}")
        return None

    with open(RESULTS_FILE, 'r') as f:
        content = f.read()

    # Extract the comparison table data
    # Row format: series milestone main_cuda_native main_actor_facade main_command_runner ...
    table_pattern = re.compile(
        r"CROSS-IMPLEMENTATION COMPARISON.*?series\s+milestone.*?\n\s*-+\n(.*?)(?=\n\n|\Z)",
        re.DOTALL
    )
    
    match = table_pattern.search(content)
    if not match:
        print("Error: Could not find comparison table in results.")
        return None

    data = {"it": [], "cuda": [], "facade": [], "runner": []}
    for line in match.group(1).strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            data["it"].append(int(parts[1]))      # milestone
            data["cuda"].append(float(parts[2]))    # main_cuda_native
            data["facade"].append(float(parts[3]))  # main_actor_facade
            data["runner"].append(float(parts[4]))  # main_command_runner
    return data

data = parse_results()
if not data:
    exit(1)

# -----------------------------
# CLI Output
# -----------------------------
print("\nMean Performance Comparison\n")
print(f"{'Iterations':>10} {'CUDA(ms)':>12} {'Facade(ms)':>12} {'Runner(ms)':>12} {'Facade Ovhd %':>15}")

for i in range(len(data["it"])):
    pct = ((data["facade"][i] - data["cuda"][i]) / data["cuda"][i]) * 100
    print(f"{data['it'][i]:>10} {data['cuda'][i]:>12.2f} {data['facade'][i]:>12.2f} {data['runner'][i]:>12.2f} {pct:>14.2f}%")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8,6))
plt.plot(data["it"], data["cuda"], marker='o', label="CUDA Native")
plt.plot(data["it"], data["facade"], marker='s', label="CAF Actor Facade")
plt.plot(data["it"], data["runner"], marker='^', label="CAF Command Runner")

plt.xlabel("Iterations (Milestone)")
plt.ylabel("Time (ms)")
plt.title("Matrix Multiplication Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_PLOT)
plt.show()
