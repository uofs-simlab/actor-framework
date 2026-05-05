#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

# Path Configuration
# This script is located in sc26/scripts/Runtime-Overhead/
# We expect results in sc26/Runtime-Overhead/results/
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = SCRIPT_DIR.parent.parent / "Runtime-Overhead" / "results" / "benchmark_results.txt"
OUTPUT_PLOT = SCRIPT_DIR.parent.parent / "Runtime-Overhead" / "results" / "runtime_overhead_comparison.png"

def parse_comparison_table():
    """Parses the 'CROSS-IMPLEMENTATION COMPARISON' table from the results file."""
    if not RESULTS_FILE.exists():
        print(f"Error: Results file not found at {RESULTS_FILE}")
        return None

    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Locate the comparison section. It uses a specific header format.
    # We capture the data rows between the dashes and the next double newline.
    table_pattern = re.compile(
        r"CROSS-IMPLEMENTATION COMPARISON.*?N\s+cuda_native\s+actor_facade\s+command_runner.*?\n-+\n(.*?)(?=\n\n|\Z)",
        re.DOTALL
    )
    
    match = table_pattern.search(content)
    if not match:
        print("Error: Could not find the comparison table in the benchmark results.")
        return None

    data = {
        "N": [],
        "cuda_native": [],
        "actor_facade": [],
        "command_runner": []
    }

    rows = match.group(1).strip().splitlines()
    for row in rows:
        parts = row.split()
        # Expected format: N, cuda_native_mean, actor_facade_mean, command_runner_mean, ...
        if len(parts) >= 4:
            try:
                data["N"].append(int(parts[0]))
                data["cuda_native"].append(float(parts[1]))
                data["actor_facade"].append(float(parts[2]))
                data["command_runner"].append(float(parts[3]))
            except ValueError:
                continue # Skip header/footer noise if any

    return data

def generate_plot(data):
    """Generates a PNG graph comparing mean execution times."""
    if not data or not data["N"]:
        return

    plt.figure(figsize=(11, 7))

    # Plotting each implementation
    plt.plot(data["N"], data["cuda_native"], marker='o', linestyle='-', label='CUDA Native (Baseline)')
    plt.plot(data["N"], data["actor_facade"], marker='s', linestyle='--', label='CAF Actor Facade')
    plt.plot(data["N"], data["command_runner"], marker='^', linestyle=':', label='CAF Command Runner')

    plt.title('Matrix Multiplication Performance by Size', fontsize=14, fontweight='bold')
    plt.xlabel('Matrix Size N', fontsize=12)
    plt.ylabel('Mean Execution Time (ms)', fontsize=12)
    
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    # Apply log scale for better visualization if the N range is large
    if max(data["N"]) / min(data["N"]) > 10:
        plt.xscale('log')
        plt.yscale('log')

    # Set X-axis ticks to exactly the N values tested to avoid abbreviation
    plt.xticks(data["N"], data["N"])

    # Force scalar formatting for Y axis to avoid scientific notation (e.g., 10^x)
    ax = plt.gca()
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Successfully generated comparison graph: {OUTPUT_PLOT}")

if __name__ == "__main__":
    results = parse_comparison_table()
    if results:
        generate_plot(results)
