#!/usr/bin/env python3
"""
Sequence-Independent-Tasks Benchmark Harness
=============================================
Runs main_cuda_native, main_actor_facade, and main_command_runner each
NUM_RUNS times. Each run is expected to execute one timed 10 000-iteration
series, emit cumulative [MILESTONE] lines every 1 000 iterations, and print
one final [SERIES RESULT] line. The output file contains:

  1. All raw [MILESTONE] and [SERIES RESULT] lines from every run.
  2. Statistics across runs for the 10 000-iteration series at each
      1 000-iteration milestone (mean / min / max / stddev).
  3. Incremental time per 1 000-iteration step (derived from adjacent
     milestones) — this shows how long each individual 1 000-iteration
     "slice" took on average.
  4. A cross-implementation comparison table.

Usage:
    cd sc26/
    python3 benchmarks/run_sequence_independent.py [--runs N] [--output PATH]

Output:
    Sequence-Independent-Tasks/results/benchmark_results.txt  (default)
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

BENCH_DIR = Path(__file__).resolve().parent.parent / "Sequence-Independent-Tasks"

BINARIES = {
    "main_cuda_native":    BENCH_DIR / "main_cuda_native",
    "main_actor_facade":   BENCH_DIR / "main_actor_facade",
    "main_command_runner": BENCH_DIR / "main_command_runner",
}

# [MILESTONE] 3000 / 10000 iterations, elapsed = 1564.72 ms
RE_MILESTONE = re.compile(
    r"\[MILESTONE\]\s+(\d+)\s*/\s*(\d+)\s+iterations,\s+elapsed\s*=\s*([\d.]+)"
)
# [SERIES RESULT] Matrix 1000x1000, iterations = 10000, total ... time = 2601.3 ms
RE_SERIES = re.compile(
    r"\[SERIES RESULT\].*iterations\s*=\s*(\d+).*?=\s*([\d.]+)\s*ms"
)

ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
TARGET_ITERATIONS = 10000

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_binary(binary: Path, run_index: int, timeout: int = 3600) -> str:
    """Run a single binary and return its stdout."""
    print(f"    run {run_index} ...", end=" ", flush=True)
    result = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        cwd=str(BENCH_DIR),
        env=ENV,
        timeout=timeout,
    )
    if result.returncode != 0:
        print(f"exit={result.returncode}", flush=True)
        if result.stderr:
            print(result.stderr[:400], file=sys.stderr)
    else:
        print("ok", flush=True)
    return result.stdout


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Data structure produced by parse_output:
#   milestones[series_total][milestone_iter] = elapsed_ms
#   series[series_total] = total_ms   (from [SERIES RESULT] line; may be absent)
RunData = tuple[dict[int, dict[int, float]], dict[int, float]]


def parse_output(stdout: str) -> RunData:
    """
    Parse one run's stdout.
    Returns:
        milestones: {series_total: {milestone_iter: elapsed_ms}}
        series_totals: {series_total: elapsed_ms}  (from [SERIES RESULT] lines)
    The 'series_total' key is the total iterations for that series.
    For this benchmark, each run should normally only contain total=10000.
    """
    milestones: dict[int, dict[int, float]] = {}
    series_totals: dict[int, float] = {}

    # As we scan lines we need to track which series we are in.
    # A milestone line carries both completed and total, so we use
    # the 'total' field as the series key directly.
    for line in stdout.splitlines():
        m = RE_MILESTONE.search(line)
        if m:
            completed   = int(m.group(1))
            series_total = int(m.group(2))
            elapsed_ms   = float(m.group(3))
            milestones.setdefault(series_total, {})[completed] = elapsed_ms

        s = RE_SERIES.search(line)
        if s:
            total_iters = int(s.group(1))
            total_ms    = float(s.group(2))
            series_totals[total_iters] = total_ms

    return milestones, series_totals


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def stat_row(vals: list[float], width: int = 12) -> str:
    if not vals:
        return f"{'N/A':>{width}}  {'N/A':>{width}}  {'N/A':>{width}}  {'N/A':>{width}}"
    mean   = statistics.mean(vals)
    lo     = min(vals)
    hi     = max(vals)
    stddev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (f"{mean:>{width}.2f}  {lo:>{width}.2f}  "
            f"{hi:>{width}.2f}  {stddev:>{width}.2f}")


def header_line(char: str = "=", width: int = 80) -> str:
    return char * width


def section(title: str, width: int = 80) -> str:
    pad = (width - len(title) - 2) // 2
    return char_pad(char="=", n=pad) + f" {title} " + char_pad(char="=", n=width - pad - len(title) - 2)


def char_pad(char: str, n: int) -> str:
    return char * max(0, n)


# ---------------------------------------------------------------------------
# Per-implementation report builder
# ---------------------------------------------------------------------------

def build_impl_report(name: str, all_runs: list[RunData], num_runs: int) -> list[str]:
    """Return lines for one implementation's section of the output file."""
    lines: list[str] = []
    lines.append(header_line("-"))
    lines.append(f"  Implementation: {name}")
    lines.append(header_line("-"))

    # Collect all series counts seen
    all_series: set[int] = set()
    for milestones, series_totals in all_runs:
        all_series.update(milestones.keys())
        all_series.update(series_totals.keys())
    sorted_series = sorted(all_series)

    # ----------------------------------------------------------------
    # 1. Raw run data  (one mini-table per series)
    # ----------------------------------------------------------------
    lines.append("\n--- Raw Milestone Data (cumulative elapsed ms) ---")

    for series_total in sorted_series:
        # Collect all milestone checkpoints seen for this series
        all_checkpoints: set[int] = set()
        for milestones, _ in all_runs:
            if series_total in milestones:
                all_checkpoints.update(milestones[series_total].keys())
        checkpoints = sorted(all_checkpoints)

        if not checkpoints:
            continue

        col = 12
        lines.append(f"\n  Series: {series_total} iterations")
        header = f"    {'Run':>5}"
        for cp in checkpoints:
            header += f"  {'@'+str(cp):>{col}}"
        header += f"  {'SERIES_END':>{col}}"
        lines.append(header)
        lines.append("    " + "-" * (7 + col * (len(checkpoints) + 1)))

        for i, (milestones, series_totals) in enumerate(all_runs, start=1):
            row = f"    {i:>5}"
            for cp in checkpoints:
                val = milestones.get(series_total, {}).get(cp)
                row += f"  {val:>{col}.2f}" if val is not None else f"  {'N/A':>{col}}"
            # Series-end total from [SERIES RESULT] line (may differ slightly from
            # last milestone if the last milestone IS the series-end)
            end_val = series_totals.get(series_total)
            row += f"  {end_val:>{col}.2f}" if end_val is not None else f"  {'N/A':>{col}}"
            lines.append(row)

    # ----------------------------------------------------------------
    # 2. Statistics per series per milestone checkpoint
    # ----------------------------------------------------------------
    lines.append("\n--- Statistics per Milestone (cumulative elapsed ms) ---")
    lines.append(
        f"  {'Series':>8}  {'Milestone':>10}  "
        f"{'Mean':>12}  {'Min':>12}  {'Max':>12}  {'StdDev':>12}  {'n':>4}"
    )
    lines.append("  " + "-" * 72)

    for series_total in sorted_series:
        all_checkpoints = set()
        for milestones, _ in all_runs:
            if series_total in milestones:
                all_checkpoints.update(milestones[series_total].keys())
        checkpoints = sorted(all_checkpoints)

        for cp in checkpoints:
            vals = [
                milestones.get(series_total, {}).get(cp)
                for milestones, _ in all_runs
                if milestones.get(series_total, {}).get(cp) is not None
            ]
            n = len(vals)
            lines.append(
                f"  {series_total:>8}  {cp:>10}  {stat_row(vals)}  {n:>4}"
            )

    # ----------------------------------------------------------------
    # 3. Average time per 1000-iteration increment  (incremental, not cumulative)
    # ----------------------------------------------------------------
    lines.append("\n--- Average Incremental Time per 1000 Iterations ---")
    lines.append(
        "  (derived from consecutive milestone differences; shows per-slice throughput)"
    )
    lines.append(
        f"  {'Series':>8}  {'Increment':>14}  "
        f"{'Mean (ms)':>12}  {'Min':>10}  {'Max':>10}  {'StdDev':>10}"
    )
    lines.append("  " + "-" * 72)

    for series_total in sorted_series:
        all_checkpoints = set()
        for milestones, _ in all_runs:
            if series_total in milestones:
                all_checkpoints.update(milestones[series_total].keys())
        checkpoints = sorted(all_checkpoints)
        if not checkpoints:
            continue

        # Build increments: [checkpoint[0] - 0, cp[1] - cp[0], ...]
        prev_cps = [0] + checkpoints[:-1]
        for prev_cp, cp in zip(prev_cps, checkpoints):
            incremental_vals = []
            for milestones, _ in all_runs:
                m = milestones.get(series_total, {})
                curr = m.get(cp)
                prev = m.get(prev_cp) if prev_cp != 0 else 0.0
                if curr is not None and prev is not None:
                    incremental_vals.append(curr - prev)

            label = f"{prev_cp+1}-{cp}"
            lines.append(
                f"  {series_total:>8}  {label:>14}  {stat_row(incremental_vals, width=10)}"
            )

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Cross-implementation comparison
# ---------------------------------------------------------------------------

def build_comparison(all_impl_data: dict[str, list[RunData]]) -> list[str]:
    """Compare mean elapsed time at each common milestone across implementations."""
    lines: list[str] = []

    # Find common (series_total, checkpoint) pairs
    impl_means: dict[str, dict[tuple[int, int], float]] = {}

    for name, runs in all_impl_data.items():
        impl_means[name] = {}
        by_key: dict[tuple[int, int], list[float]] = defaultdict(list)
        for milestones, _ in runs:
            for series_total, checkpoints in milestones.items():
                for cp, elapsed in checkpoints.items():
                    by_key[(series_total, cp)].append(elapsed)
        for key, vals in by_key.items():
            impl_means[name][key] = statistics.mean(vals)

    # Intersect all keys
    if not impl_means:
        return ["(no data)"]

    all_keys = set.intersection(*[set(d.keys()) for d in impl_means.values()])
    if not all_keys:
        return ["(no common milestone keys across implementations)"]

    sorted_keys = sorted(all_keys)
    names = list(all_impl_data.keys())
    col = 16

    header = f"  {'series':>8}  {'milestone':>10}"
    for n in names:
        header += f"  {n[:col]:>{col}}"
    # Overhead vs first implementation
    if len(names) > 1:
        base = names[0]
        for n in names[1:]:
            header += f"  {'vs_'+base[:8]+' %':>14}"
    lines.append(header)
    lines.append("  " + "-" * (22 + col * len(names) + 16 * max(0, len(names) - 1)))

    for series_total, cp in sorted_keys:
        row = f"  {series_total:>8}  {cp:>10}"
        base_val: float | None = None
        for i, n in enumerate(names):
            val = impl_means[n].get((series_total, cp))
            row += f"  {val:>{col}.2f}" if val is not None else f"  {'N/A':>{col}}"
            if i == 0:
                base_val = val
        if len(names) > 1 and base_val and base_val > 0:
            for n in names[1:]:
                val = impl_means[n].get((series_total, cp))
                if val is not None:
                    pct = (val - base_val) / base_val * 100
                    sign = "+" if pct >= 0 else ""
                    row += f"  {sign}{pct:>13.2f}%"
                else:
                    row += f"  {'N/A':>14}"
        lines.append(row)

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequence-Independent-Tasks benchmark harness"
    )
    parser.add_argument("--runs",   type=int,  default=10,
                        help="Number of times to run each binary (default: 10)")
    parser.add_argument("--output", type=str,
                        default=str(BENCH_DIR / "results" / "benchmark_results.txt"),
                        help="Path to the output file")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_runs = args.runs

    # {impl_name: [RunData, ...]}
    all_impl_data: dict[str, list[RunData]] = {}
    raw_outputs:   dict[str, list[str]]     = {}

    for name, binary in BINARIES.items():
        print(f"\n[{name}]", flush=True)
        if not binary.exists():
            print(f"  SKIP — binary not found: {binary}", file=sys.stderr)
            continue

        all_impl_data[name] = []
        raw_outputs[name]   = []

        for i in range(1, num_runs + 1):
            stdout = run_binary(binary, i)
            raw_outputs[name].append(stdout)
            all_impl_data[name].append(parse_output(stdout))

    # ------------------------------------------------------------------
    # Build output file
    # ------------------------------------------------------------------
    lines: list[str] = []

    lines.append(header_line())
    lines.append(section("SEQUENCE-INDEPENDENT TASKS BENCHMARK"))
    lines.append(header_line())
    lines.append(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Runs       : {num_runs}")
    lines.append(f"GPU        : CUDA_VISIBLE_DEVICES=0")
    lines.append(f"Directory  : {BENCH_DIR}")
    lines.append(
        "Milestones : cumulative elapsed time (ms) since series start, "
        "printed every 1 000 iterations by the binary"
    )
    lines.append("")

    for name in BINARIES:
        if name not in all_impl_data:
            continue
        lines.extend(build_impl_report(name, all_impl_data[name], num_runs))

    if len(all_impl_data) >= 2:
        lines.append(header_line())
        lines.append(section("CROSS-IMPLEMENTATION COMPARISON (mean cumulative elapsed ms)"))
        lines.append(header_line())
        lines.append(
            "  Positive % = the implementation is slower than the baseline (first column)"
        )
        lines.append("")
        lines.extend(build_comparison(all_impl_data))
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
