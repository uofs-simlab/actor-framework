#!/usr/bin/env python3
"""
Runtime-Overhead Benchmark Harness
====================================
Runs cuda_native, actor_facade, and command_runner each NUM_RUNS times.
Each run captures the total time for every matrix size, writes all raw data
to an output file, then appends mean / min / max / stddev statistics and a
cross-implementation comparison table at the bottom.

Usage:
    cd sc26/
    python3 benchmarks/run_runtime_overhead.py [--runs N] [--output PATH]

Output:
    Runtime-Overhead/results/benchmark_results.txt  (default)
"""

import argparse
import os
import re
import subprocess
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).resolve().parent.parent / "Runtime-Overhead"

BINARIES = {
    "cuda_native":    BENCH_DIR / "cuda_native",
    "actor_facade":   BENCH_DIR / "actor_facade",
    "command_runner": BENCH_DIR / "command_runner",
}

# Patterns shared across both implementations.
# cuda_native prints:  N=1000 ... TOTAL: 42.37 ms
# actor/command print: N=1000 ... TOTAL end-to-end: 42.37 ms
RE_SIZE  = re.compile(r"N=(\d+)")
RE_TOTAL_NATIVE = re.compile(r"^TOTAL:\s+([\d.]+)", re.MULTILINE)
RE_TOTAL_ACTOR  = re.compile(r"^TOTAL end-to-end:\s+([\d.]+)", re.MULTILINE)

# Warmup size — excluded from parsed results
WARMUP_SIZE = 64

ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_binary(binary: Path, run_index: int, timeout: int = 600) -> str:
    """Run a single binary and return its stdout as a string."""
    print(f"    run {run_index} ...", flush=True)
    result = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        cwd=str(BENCH_DIR),
        env=ENV,
        timeout=timeout,
    )
    if result.returncode != 0:
        print(f"      [WARNING] exit code {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr[:400], file=sys.stderr)
    return result.stdout


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_output(name: str, stdout: str) -> dict[int, float]:
    """
    Returns {matrix_size: total_ms} for one run output.
    Tracks the most-recently seen N= header so each TOTAL is paired correctly.
    """
    results: dict[int, float] = {}
    current_size: int | None = None
    total_re = RE_TOTAL_NATIVE if name == "cuda_native" else RE_TOTAL_ACTOR

    for line in stdout.splitlines():
        m_size = RE_SIZE.search(line)
        if m_size:
            current_size = int(m_size.group(1))
        m_total = total_re.search(line)
        if m_total and current_size is not None:
            if current_size != WARMUP_SIZE:
                results[current_size] = float(m_total.group(1))
            current_size = None  # reset; next hit belongs to the next block

    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def header_line(char: str = "=", width: int = 72) -> str:
    return char * width


def section(title: str, width: int = 72) -> str:
    pad = (width - len(title) - 2) // 2
    return "=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2)


def stats_table(data: dict[int, list[float]]) -> str:
    """Produce a per-size statistics table from {size: [times]}."""
    sizes = sorted(data.keys())
    lines = []
    col = 12
    lines.append(
        f"{'N':>6}  {'Mean (ms)':>{col}}  {'Min (ms)':>{col}}  "
        f"{'Max (ms)':>{col}}  {'StdDev (ms)':>{col}}  {'Runs':>5}"
    )
    lines.append("-" * 66)
    for s in sizes:
        vals = data[s]
        if not vals:
            continue
        mean   = statistics.mean(vals)
        lo     = min(vals)
        hi     = max(vals)
        stddev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lines.append(
            f"{s:>6}  {mean:>{col}.3f}  {lo:>{col}.3f}  "
            f"{hi:>{col}.3f}  {stddev:>{col}.3f}  {len(vals):>5}"
        )
    return "\n".join(lines)


def comparison_table(all_data: dict[str, dict[int, list[float]]]) -> str:
    """
    Cross-implementation comparison using mean times.
    Overhead columns show actor overhead vs cuda_native baseline.
    """
    names  = list(all_data.keys())
    # Collect sizes present in ALL implementations
    size_sets = [set(all_data[n].keys()) for n in names]
    sizes = sorted(size_sets[0].intersection(*size_sets[1:]))
    if not sizes:
        return "(no common matrix sizes found across all implementations)"

    native_means: dict[int, float] = {}
    impl_means:   dict[str, dict[int, float]] = {n: {} for n in names}
    for s in sizes:
        for n in names:
            vals = all_data[n].get(s, [])
            if vals:
                impl_means[n][s] = statistics.mean(vals)
        if "cuda_native" in impl_means and s in impl_means["cuda_native"]:
            native_means[s] = impl_means["cuda_native"][s]

    lines = []
    col = 14
    header = f"{'N':>6}"
    for n in names:
        header += f"  {n[:col]:>{col}}"
    if "cuda_native" in names:
        others = [n for n in names if n != "cuda_native"]
        for n in others:
            header += f"  {'vs_native_%':>12}"
    lines.append(header)
    lines.append("-" * (8 + col * len(names) + 14 * max(0, len(names) - 1)))

    for s in sizes:
        row = f"{s:>6}"
        for n in names:
            val = impl_means[n].get(s, float("nan"))
            row += f"  {val:>{col}.3f}"
        if "cuda_native" in names:
            baseline = native_means.get(s)
            others = [n for n in names if n != "cuda_native"]
            for n in others:
                val = impl_means[n].get(s)
                if val is not None and baseline and baseline > 0:
                    pct = (val - baseline) / baseline * 100
                    sign = "+" if pct >= 0 else ""
                    row += f"  {sign}{pct:>11.2f}%"
                else:
                    row += f"  {'N/A':>12}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime-Overhead benchmark harness")
    parser.add_argument("--runs",   type=int,  default=10, help="Number of times to run each binary (default: 10)")
    parser.add_argument("--output", type=str,  default=str(BENCH_DIR / "results" / "benchmark_results.txt"),
                        help="Path to the output file")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_runs = args.runs

    # {impl_name: {matrix_size: [time_ms, ...]}}
    all_data: dict[str, dict[int, list[float]]] = {}
    # {impl_name: list of raw stdout strings (one per run)}
    raw_outputs: dict[str, list[str]] = {}

    for name, binary in BINARIES.items():
        print(f"\n[{name}]", flush=True)
        if not binary.exists():
            print(f"  SKIP — binary not found: {binary}", file=sys.stderr)
            continue

        all_data[name]   = defaultdict(list)
        raw_outputs[name] = []

        for i in range(1, num_runs + 1):
            stdout = run_binary(binary, i)
            raw_outputs[name].append(stdout)
            parsed = parse_output(name, stdout)
            for size, t in parsed.items():
                all_data[name][size].append(t)

    # ------------------------------------------------------------------
    # Write output file
    # ------------------------------------------------------------------
    lines: list[str] = []

    lines.append(header_line())
    lines.append(section("RUNTIME OVERHEAD BENCHMARK"))
    lines.append(header_line())
    lines.append(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Runs       : {num_runs}")
    lines.append(f"GPU        : CUDA_VISIBLE_DEVICES=0")
    lines.append(f"Directory  : {BENCH_DIR}")
    lines.append("")

    for name in BINARIES:
        if name not in all_data:
            continue

        lines.append(header_line("-"))
        lines.append(f"  Implementation: {name}")
        lines.append(header_line("-"))

        lines.append("\n--- Raw Run Data ---")
        sizes = sorted(all_data[name].keys())
        # Per-run table
        col = 14
        header = f"{'Run':>5}"
        for s in sizes:
            header += f"  {'N='+str(s):>{col}}"
        lines.append(header)
        lines.append("-" * (7 + col * len(sizes)))

        for i, run_vals in enumerate(zip(*[all_data[name][s] for s in sizes]), start=1):
            row = f"{i:>5}"
            for val in run_vals:
                row += f"  {val:>{col}.3f}"
            lines.append(row)

        lines.append("")

        # Also include any sizes that might not appear in all runs
        # (some runs may have failed mid-way)
        uneven = {s for s in sizes if len(all_data[name][s]) != num_runs}
        if uneven:
            lines.append(
                f"  [NOTE] The following sizes had fewer than {num_runs} successful "
                f"data points: {sorted(uneven)}"
            )
            lines.append("")

        lines.append("--- Statistics ---")
        lines.append(stats_table(all_data[name]))
        lines.append("")

    # Cross-implementation comparison (only if we have at least 2 implementations)
    if len(all_data) >= 2:
        lines.append(header_line())
        lines.append(section("CROSS-IMPLEMENTATION COMPARISON (mean times, ms)"))
        lines.append(header_line())
        lines.append(comparison_table(all_data))
        lines.append("")
        lines.append(
            "vs_native_%: positive = actor implementation is slower than cuda_native"
        )
        lines.append("")

    # Full raw stdout dumps
    lines.append(header_line())
    lines.append(section("FULL RAW OUTPUT (all runs)"))
    lines.append(header_line())
    for name in BINARIES:
        if name not in raw_outputs:
            continue
        lines.append(f"\n{'='*10} {name} {'='*10}")
        for i, stdout in enumerate(raw_outputs[name], start=1):
            lines.append(f"\n--- {name} | run {i} ---")
            lines.append(stdout.strip())

    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
