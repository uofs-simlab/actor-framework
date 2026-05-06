#!/usr/bin/env python3
"""SC26 Fault-Tolerance Benchmark — Paper Plot Generator

Parses structured output from the Monte Carlo π fault-tolerance binary and
generates the 5 visualizations recommended by the SC26 review:

  gantt.pdf           Recovery timeline  (Gantt chart per worker lane)
  throughput.pdf      Throughput under stress with fault injection markers
  pi_convergence.pdf  π-estimate convergence with fault annotation
  recovery_cdf.pdf    CDF of per-fault recovery latency (multi-run)
  degradation.pdf     Throughput degradation bar chart (baseline vs N faults)

The binary must be compiled with the SC26-hardened main.cpp that emits
[BATCH_START], [BATCH_END], [WORKER_RESPAWN], [FAULT_DETECTED],
[FAULT_DISPATCH], [FAULT_RECOVERED], [PROGRESS], and [RESULT] lines.

Usage examples
--------------
# Single fault-injection run (generates plots 1–3 and whatever CDF data exists):
  python3 plot.py --log run.log

# Add a no-fault baseline for throughput / degradation comparison:
  python3 plot.py --log fault.log --baseline baseline.log

# Pass multiple fault-injection logs (for CDF with many data points):
  python3 plot.py --log-dir logs/fault/ --baseline baseline.log

# Run the binary directly (captures output automatically):
  python3 plot.py --run ./main
  python3 plot.py --run ./main --args "--num-faults 3 --num-workers 4 --num-batches 200"

# Full degradation chart: provide logs for each fault count explicitly:
  python3 plot.py --baseline b.log --fault1 f1.log --fault2 f2.log --fault3 f3.log

All plots are written to --out-dir (default: ./plots/).
Use --no-pdf to save as PNG instead of PDF.
"""

import argparse
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Plot styling
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed TrueType fonts in PDF (required by most venues)
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

COLORS = {
    "normal":     "#4C8BBF",  # steel blue  — normal computation
    "recovery":   "#E8781A",  # orange      — recovery batch
    "baseline":   "#555555",  # dark grey   — baseline reference
    "fault_line": "#CC3333",  # red         — fault injection marker
    "pi_ref":     "#2CA02C",  # green       — π reference / respawn marker
    "worker_bg":  "#F4F7FB",  # very light blue — worker lane background
}

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchEvent:
    batch_id:    int
    worker_id:   int
    start_s:     float
    end_s:       float = None
    is_recovery: bool  = False


@dataclass
class FaultEvent:
    fault_index:            int
    worker_id:              int
    batch_id:               int
    detected_s:             float
    dispatch_s:             float = None
    recover_s:              float = None
    detect_to_dispatch_ms:  float = None
    dispatch_to_complete_ms: float = None
    total_recovery_ms:      float = None


@dataclass
class ProgressPoint:
    batch_done:    int
    total_batches: int
    pi:            float
    time_s:        float
    rate:          float   # cumulative batches/s at this point


@dataclass
class WorkerRespawn:
    worker_id: int
    time_s:    float


@dataclass
class RunSummary:
    mode:             str   = "unknown"
    num_workers:      int   = 0
    total_batches:    int   = 0
    total_samples:    int   = 0
    samples_per_batch: int  = 0
    pi:               float = 0.0
    error_pct:        float = 0.0
    elapsed_s:        float = 0.0
    throughput:       float = 0.0   # batches/s
    faults_injected:  int   = 0


@dataclass
class RunData:
    batches:  list = field(default_factory=list)   # list[BatchEvent]
    faults:   list = field(default_factory=list)   # list[FaultEvent]
    progress: list = field(default_factory=list)   # list[ProgressPoint]
    respawns: list = field(default_factory=list)   # list[WorkerRespawn]
    summary:  RunSummary = field(default_factory=RunSummary)


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(text: str) -> RunData:
    """Parse all structured lines emitted by the SC26 fault-tolerance binary."""
    data = RunData()
    fault_map: dict[int, FaultEvent] = {}
    batch_seen: dict[int, bool] = {}   # batch_id -> True once we've seen first BATCH_START

    for line in text.splitlines():
        line = line.strip()

        # [BATCH_START] batch=X worker=Y time_s=Z
        m = re.match(r"\[BATCH_START\] batch=(\d+) worker=(\d+) time_s=([\d.]+)", line)
        if m:
            bid, wid, t = int(m.group(1)), int(m.group(2)), float(m.group(3))
            is_rec = bid in batch_seen
            batch_seen[bid] = True
            data.batches.append(BatchEvent(bid, wid, t, is_recovery=is_rec))
            continue

        # [BATCH_END] batch=X worker=Y time_s=Z
        m = re.match(r"\[BATCH_END\] batch=(\d+) worker=(\d+) time_s=([\d.]+)", line)
        if m:
            bid, wid, t = int(m.group(1)), int(m.group(2)), float(m.group(3))
            # Match to latest unfinished event for this (batch_id, worker_id) pair
            for ev in reversed(data.batches):
                if ev.batch_id == bid and ev.worker_id == wid and ev.end_s is None:
                    ev.end_s = t
                    break
            continue

        # [WORKER_RESPAWN] worker=X time_s=Z
        m = re.match(r"\[WORKER_RESPAWN\] worker=(\d+) time_s=([\d.]+)", line)
        if m:
            data.respawns.append(WorkerRespawn(int(m.group(1)), float(m.group(2))))
            continue

        # [FAULT_DETECTED] fault=F worker=W batch=B time_ms=T
        m = re.match(
            r"\[FAULT_DETECTED\] fault=(-?\d+) worker=(\d+) batch=(\d+) time_ms=([\d.]+)",
            line,
        )
        if m:
            fi, wid, bid, t_ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
            fe = FaultEvent(fi, wid, bid, t_ms / 1000.0)
            fault_map[fi] = fe
            data.faults.append(fe)
            continue

        # [FAULT_DISPATCH] fault=F worker=W batch=B detect_to_dispatch_ms=D
        m = re.match(
            r"\[FAULT_DISPATCH\] fault=(\d+) worker=\d+ batch=\d+ detect_to_dispatch_ms=([\d.]+)",
            line,
        )
        if m:
            fi, d2d = int(m.group(1)), float(m.group(2))
            if fi in fault_map:
                fault_map[fi].detect_to_dispatch_ms = d2d
                fault_map[fi].dispatch_s = fault_map[fi].detected_s + d2d / 1000.0
            continue

        # [FAULT_RECOVERED] fault=F worker=W batch=B detect_to_dispatch_ms=A
        #                   dispatch_to_complete_ms=B total_recovery_ms=C
        m = re.match(
            r"\[FAULT_RECOVERED\] fault=(\d+) worker=\d+ batch=\d+ "
            r"detect_to_dispatch_ms=([\d.]+) dispatch_to_complete_ms=([\d.]+) "
            r"total_recovery_ms=([\d.]+)",
            line,
        )
        if m:
            fi = int(m.group(1))
            d2d, d2c, tot = float(m.group(2)), float(m.group(3)), float(m.group(4))
            if fi in fault_map:
                fault_map[fi].detect_to_dispatch_ms  = d2d
                fault_map[fi].dispatch_to_complete_ms = d2c
                fault_map[fi].total_recovery_ms       = tot
                base_s = fault_map[fi].dispatch_s or fault_map[fi].detected_s
                fault_map[fi].recover_s = base_s + d2c / 1000.0
            continue

        # [PROGRESS] batch=X/Y pi=Z time_s=W rate=R batches/s
        m = re.match(
            r"\[PROGRESS\] batch=(\d+)/(\d+) pi=([\d.]+) time_s=([\d.]+) rate=([\d.]+) batches/s",
            line,
        )
        if m:
            data.progress.append(ProgressPoint(
                int(m.group(1)), int(m.group(2)),
                float(m.group(3)), float(m.group(4)), float(m.group(5)),
            ))
            continue

        # [RESULT] mode=X workers=Y batches=Z total_samples=W pi=V error_pct=U
        #          elapsed_s=T throughput=S faults=R
        m = re.match(
            r"\[RESULT\] mode=(\S+) workers=(\d+) batches=(\d+) total_samples=(\d+) "
            r"pi=([\d.]+) error_pct=([\d.]+) elapsed_s=([\d.]+) "
            r"throughput=([\d.]+) faults=(\d+)",
            line,
        )
        if m:
            s = data.summary
            s.mode             = m.group(1)
            s.num_workers      = int(m.group(2))
            s.total_batches    = int(m.group(3))
            s.total_samples    = int(m.group(4))
            s.pi               = float(m.group(5))
            s.error_pct        = float(m.group(6))
            s.elapsed_s        = float(m.group(7))
            s.throughput       = float(m.group(8))
            s.faults_injected  = int(m.group(9))
            if s.total_batches > 0:
                s.samples_per_batch = s.total_samples // s.total_batches
            continue

        # Header line: "Workers: X  Batches: Y  Samples/batch: Z  ..."
        m = re.match(r"Workers:\s*(\d+)\s+Batches:\s*(\d+)\s+Samples/batch:\s*(\d+)", line)
        if m and data.summary.samples_per_batch == 0:
            data.summary.samples_per_batch = int(m.group(3))
            if data.summary.num_workers == 0:
                data.summary.num_workers = int(m.group(1))

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Recovery Timeline (Gantt chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gantt(run: RunData, path: Path) -> None:
    """Horizontal Gantt chart: one lane per worker, bars for each batch.

    Normal batches: steel blue.
    Recovery (re-dispatched) batches: orange with hatching.
    Worker-killed marker: red ×.
    Worker-respawned marker: green ▲.
    Dashed red vertical line: fault detected instant.
    """
    if not run.batches:
        print("  [gantt]      SKIP — no [BATCH_START]/[BATCH_END] data.\n"
              "               Recompile Fault-Tolerance/main.cpp and re-run.")
        return

    s        = run.summary
    workers  = sorted({b.worker_id for b in run.batches})
    y_map    = {w: i for i, w in enumerate(workers)}
    bar_h    = 0.55
    fig_h    = max(3.0, len(workers) * 0.9 + 1.8)
    fig, ax  = plt.subplots(figsize=(12, fig_h))

    # Worker lane backgrounds
    for w in workers:
        ax.axhspan(y_map[w] - 0.5, y_map[w] + 0.5,
                   color=COLORS["worker_bg"], zorder=0, linewidth=0)

    # Batch bars
    for b in run.batches:
        y  = y_map[b.worker_id]
        x0 = b.start_s
        x1 = b.end_s
        if x1 is None:
            # Worker was killed before this batch completed — clip bar to fault time
            clipped = [f.detected_s for f in run.faults
                       if f.worker_id == b.worker_id and f.detected_s >= x0]
            x1 = min(clipped) if clipped else x0 + 1e-4
        color = COLORS["recovery"] if b.is_recovery else COLORS["normal"]
        hatch = "///" if b.is_recovery else None
        ax.barh(y, x1 - x0, left=x0, height=bar_h,
                color=color, hatch=hatch, edgecolor="white",
                linewidth=0.4, zorder=2, alpha=0.85)

    # Fault: red × at detection time and dashed vertical line
    for f in run.faults:
        y = y_map.get(f.worker_id, -1)
        if y < 0:
            continue
        ax.scatter(f.detected_s, y, marker="x", color=COLORS["fault_line"],
                   s=140, zorder=6, linewidths=2.5)
        ax.axvline(f.detected_s, color=COLORS["fault_line"],
                   linestyle="--", linewidth=0.9, alpha=0.5, zorder=1)

    # Respawn: green ▲ just below the lane centre
    for r in run.respawns:
        y = y_map.get(r.worker_id, -1)
        if y < 0:
            continue
        ax.scatter(r.time_s, y - bar_h * 0.35, marker="^",
                   color=COLORS["pi_ref"], s=90, zorder=6)

    # Axis labels
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Worker")
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels([f"Worker {w}" for w in workers])
    ax.set_xlim(left=0)

    # Legend
    handles = [
        mpatches.Patch(color=COLORS["normal"], label="Active computation"),
        mpatches.Patch(color=COLORS["recovery"], hatch="///", label="Recovery batch"),
        Line2D([0], [0], marker="x", color="w",
               markeredgecolor=COLORS["fault_line"], markeredgewidth=2.5,
               markersize=10, linewidth=0, label="Worker killed"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor=COLORS["pi_ref"], markersize=8,
               linewidth=0, label="Worker respawned"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.92)

    n_faults = s.faults_injected
    ax.set_title(
        f"Recovery Timeline — {s.num_workers} workers, "
        f"{n_faults} fault{'s' if n_faults != 1 else ''}, "
        f"{s.total_batches} batches"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  [gantt]      → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Throughput Under Stress
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_throughput(
    completion_times: list[float], window: int = 5
) -> tuple[list[float], list[float]]:
    """Sliding-window throughput from sorted batch completion timestamps.

    Returns (time_points, batches_per_second) with len = len(times) - window + 1.
    """
    times = sorted(completion_times)
    ts, vs = [], []
    for i in range(window - 1, len(times)):
        dt = times[i] - times[i - (window - 1)]
        if dt > 1e-9:
            ts.append(times[i])
            vs.append(window / dt)
    return ts, vs


def plot_throughput(
    run: RunData,
    baseline: Optional[RunData],
    path: Path,
    window: int = 5,
) -> None:
    """Throughput over time for a fault-injection run, with optional baseline overlay.

    If BATCH_END data is present uses rolling-window throughput (precise).
    Falls back to the cumulative-average `rate` field from [PROGRESS] lines.
    Vertical dashed red lines mark each fault injection.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    s = run.summary

    def _plot_run(rd: RunData, label: str, color: str, ls: str = "-") -> None:
        end_times = [b.end_s for b in rd.batches if b.end_s is not None]
        if len(end_times) >= window:
            ts, vs = _rolling_throughput(end_times, window)
            ax.plot(ts, vs, color=color, linewidth=1.6, linestyle=ls,
                    label=f"{label} (rolling w={window})")
        elif rd.progress:
            pts = rd.progress
            ax.plot([p.time_s for p in pts], [p.rate for p in pts],
                    color=color, linewidth=1.6, linestyle=ls,
                    label=f"{label} (cumul. avg)")

    _plot_run(run, "Fault run", COLORS["normal"])

    if baseline:
        # Try rolling throughput from baseline; fall back to flat reference line
        bt = [b.end_s for b in baseline.batches if b.end_s is not None]
        if len(bt) >= window:
            _plot_run(baseline, "Baseline", COLORS["baseline"], ls="--")
        elif baseline.summary.throughput > 0:
            ax.axhline(baseline.summary.throughput, color=COLORS["baseline"],
                       linestyle="--", linewidth=1.5,
                       label=f"Baseline: {baseline.summary.throughput:.1f} batches/s",
                       alpha=0.75)

    # Fault injection vertical lines
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    for i, f in enumerate(run.faults):
        lbl = "Fault injection" if i == 0 else "_nolegend_"
        ax.axvline(f.detected_s, color=COLORS["fault_line"],
                   linestyle="--", linewidth=1.2, alpha=0.85, label=lbl, zorder=3)

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Throughput (batches / s)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.92)
    n_faults = s.faults_injected
    ax.set_title(
        f"Throughput Under Stress — {s.num_workers} workers, "
        f"{n_faults} fault{'s' if n_faults != 1 else ''}"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  [throughput] → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: π Convergence
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_samples(x: float, _pos) -> str:
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    return f"{x/1e3:.0f}K"


def plot_pi_convergence(run: RunData, path: Path) -> None:
    """π estimate vs cumulative samples.

    The curve should plateau briefly after each fault injection (batch
    stalled), then resume converging. Vertical dashed lines show where faults
    were detected (mapped to the nearest [PROGRESS] sample count).
    """
    if not run.progress:
        print("  [pi_conv]    SKIP — no [PROGRESS] data.")
        return

    s   = run.summary
    spb = s.samples_per_batch or 1

    xs = [p.batch_done * spb for p in run.progress]   # cumulative samples
    ys = [p.pi for p in run.progress]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ys, color=COLORS["normal"], linewidth=1.6,
            label="Estimated π", zorder=3)
    ax.axhline(math.pi, color=COLORS["pi_ref"], linewidth=1.3, linestyle="-",
               label=f"π = {math.pi:.6f}", zorder=2)

    # Map each fault detection time to the nearest cumulative sample count
    prog_times   = [p.time_s   for p in run.progress]
    prog_samples = [p.batch_done * spb for p in run.progress]

    for i, f in enumerate(run.faults):
        sample_at_fault = None
        for j, t in enumerate(prog_times):
            if t >= f.detected_s:
                sample_at_fault = prog_samples[j]
                break
        if sample_at_fault is None and prog_samples:
            sample_at_fault = prog_samples[-1]
        if sample_at_fault is not None:
            lbl = "Fault injection" if i == 0 else "_nolegend_"
            ax.axvline(sample_at_fault, color=COLORS["fault_line"],
                       linestyle="--", linewidth=1.2, alpha=0.85, label=lbl)

    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_fmt_samples))
    ax.set_xlabel("Cumulative Monte Carlo samples")
    ax.set_ylabel("π estimate")
    ax.legend(framealpha=0.92)
    ax.set_title(
        f"π Convergence — {s.num_workers} workers, "
        f"{s.total_batches} batches × {_fmt_samples(spb, None)} samples"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  [pi_conv]    → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Recovery Latency CDF
# ─────────────────────────────────────────────────────────────────────────────

def plot_recovery_cdf(all_runs: list[RunData], path: Path) -> None:
    """Empirical CDF of total recovery latency across all FAULT_RECOVERED events.

    Annotates the median and 95th-percentile with vertical dashed lines.
    Useful for claiming bounded recovery in the paper.
    """
    latencies = [
        f.total_recovery_ms
        for run in all_runs
        for f in run.faults
        if f.total_recovery_ms is not None
    ]

    if not latencies:
        print("  [cdf]        SKIP — no [FAULT_RECOVERED] data.")
        return

    lat = sorted(latencies)
    n   = len(lat)
    cdf = [(i + 1) / n for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(lat, cdf, where="post", color=COLORS["normal"], linewidth=2,
            label=f"Empirical CDF (n={n})")
    ax.scatter(lat, cdf, color=COLORS["normal"], s=30, zorder=5)

    p50 = float(np.percentile(lat, 50))
    p95 = float(np.percentile(lat, 95))
    ax.axvline(p50, color=COLORS["pi_ref"], linestyle="--", linewidth=1.3,
               label=f"Median: {p50:.1f} ms")
    ax.axvline(p95, color=COLORS["fault_line"], linestyle="--", linewidth=1.3,
               label=f"P95: {p95:.1f} ms")

    if n > 1:
        mean_ms = statistics.mean(lat)
        std_ms  = statistics.stdev(lat)
        ax.text(0.97, 0.07,
                f"mean = {mean_ms:.1f} ms\nσ = {std_ms:.1f} ms",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xlabel("Recovery latency (ms)")
    ax.set_ylabel("CDF   P(recovery ≤ t)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.07)
    ax.legend(framealpha=0.92)
    ax.set_title(f"Recovery Latency CDF  ({n} fault events, {len(all_runs)} run{'s' if len(all_runs)>1 else ''})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  [cdf]        → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Throughput Degradation
# ─────────────────────────────────────────────────────────────────────────────

def plot_degradation(runs_by_faults: dict[int, list[RunData]], path: Path) -> None:
    """Grouped bar chart of mean throughput (batches/s) per fault count.

    Shows percentage of baseline above each bar.  Error bars show ±1σ when
    multiple trials are provided per configuration.
    """
    fault_counts = sorted(runs_by_faults.keys())
    means, stds, counts = [], [], []
    baseline_mean = None

    for nf in fault_counts:
        tp_vals = [r.summary.throughput for r in runs_by_faults[nf]
                   if r.summary.throughput > 0]
        if not tp_vals:
            means.append(0.0); stds.append(0.0); counts.append(0)
            continue
        m = statistics.mean(tp_vals)
        σ = statistics.stdev(tp_vals) if len(tp_vals) > 1 else 0.0
        means.append(m); stds.append(σ); counts.append(len(tp_vals))
        if nf == 0:
            baseline_mean = m

    if not any(m > 0 for m in means):
        print("  [degradation] SKIP — no throughput data found.")
        return

    x      = np.arange(len(fault_counts))
    colors = [COLORS["baseline"] if fc == 0 else COLORS["normal"] for fc in fault_counts]

    fig, ax = plt.subplots(figsize=(max(6, len(fault_counts) * 1.5 + 2), 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                  error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5})

    # Annotate with percentage of baseline
    if baseline_mean and baseline_mean > 0:
        for i, (m, σ) in enumerate(zip(means, stds)):
            if m <= 0:
                continue
            pct = 100.0 * m / baseline_mean
            offset = σ + max(baseline_mean * 0.02, 0.5)
            ax.text(i, m + offset, f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.axhline(baseline_mean, color=COLORS["baseline"],
                   linestyle="--", linewidth=1.2, alpha=0.65,
                   label=f"Baseline: {baseline_mean:.1f} batches/s")
        ax.legend(framealpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{fc} fault{'s' if fc != 1 else ''}" for fc in fault_counts])
    ax.set_xlabel("Faults injected per run")
    ax.set_ylabel("Throughput (batches / s)")
    ax.set_ylim(bottom=0)
    ax.set_title("Throughput Degradation vs. Fault Count")

    if any(c > 1 for c in counts):
        fig.text(0.99, 0.01, "Error bars: ±1σ across trials",
                 ha="right", va="bottom", fontsize=8, color="grey")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  [degradation] → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_log(p: Path) -> RunData:
    data = parse_log(p.read_text())
    s = data.summary
    print(f"  Loaded {p.name}: mode={s.mode} workers={s.num_workers} "
          f"batches={s.total_batches} faults={s.faults_injected} "
          f"throughput={s.throughput:.1f} batches/s")
    return data


def run_binary(binary: Path, extra_args: list, cwd: Path) -> RunData:
    cmd = [str(binary.resolve())] + extra_args
    print(f"  Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True,
                             cwd=str(cwd.resolve()), timeout=7200)
    if result.returncode != 0:
        print(f"  WARNING: binary exited {result.returncode}", file=sys.stderr)
        print(result.stderr[:500], file=sys.stderr)
    return parse_log(result.stdout)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SC26 fault-tolerance paper plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--log",      type=Path, metavar="FILE",
                        help="Fault-injection run log file")
    parser.add_argument("--baseline", type=Path, metavar="FILE",
                        help="No-fault baseline run log file")
    parser.add_argument("--log-dir",  type=Path, metavar="DIR",
                        help="Directory of fault-injection .log/.txt files (for CDF)")
    parser.add_argument("--run",      type=Path, metavar="BINARY",
                        help="Path to the fault-tolerance binary (run it directly)")
    parser.add_argument("--args",     type=str,  default="", metavar="ARGS",
                        help="Extra CLI arguments forwarded to --run binary")
    parser.add_argument("--fault1",   type=Path, metavar="FILE",
                        help="Log for 1-fault run  (degradation plot)")
    parser.add_argument("--fault2",   type=Path, metavar="FILE",
                        help="Log for 2-fault run  (degradation plot)")
    parser.add_argument("--fault3",   type=Path, metavar="FILE",
                        help="Log for 3-fault run  (degradation plot)")
    parser.add_argument("--out-dir",  type=Path, default=Path("plots"),
                        help="Output directory (default: ./plots/)")
    parser.add_argument("--no-pdf",   action="store_true",
                        help="Save as PNG instead of PDF (useful for quick previews)")
    parser.add_argument("--window",   type=int, default=5, metavar="W",
                        help="Rolling window size for throughput plot (default: 5)")
    args = parser.parse_args()

    if not any([args.log, args.baseline, args.log_dir, args.run,
                args.fault1, args.fault2, args.fault3]):
        parser.error(
            "No input provided. Pass --log, --baseline, --run, --log-dir, "
            "or --fault1/2/3."
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".png" if args.no_pdf else ".pdf"

    plt.rcParams.update(STYLE)

    # ── Collect fault-injection run(s) ───────────────────────────────────────
    fault_runs: list[RunData] = []

    if args.run:
        cwd   = args.run.parent
        extra = args.args.split() if args.args else []
        fault_runs.append(run_binary(args.run, extra, cwd))

    if args.log:
        fault_runs.append(load_log(args.log))

    if args.log_dir:
        log_files = sorted(args.log_dir.glob("*.log")) + sorted(args.log_dir.glob("*.txt"))
        if not log_files:
            print(f"  WARNING: no .log/.txt files found in {args.log_dir}", file=sys.stderr)
        for lf in log_files:
            fault_runs.append(load_log(lf))

    # ── Baseline ─────────────────────────────────────────────────────────────
    baseline: Optional[RunData] = None
    if args.baseline:
        if not args.baseline.exists():
            if args.run:
                print(f"  Baseline log {args.baseline} not found. Generating it now...")
                cwd = args.run.parent
                base_extra = args.args.split() if args.args else []
                # Remove any existing fault injection flags and force baseline mode.
                filtered = []
                skip = False
                for a in base_extra:
                    if skip:
                        skip = False
                        continue
                    if a in ("--num-faults", "-F"):
                        skip = True
                        continue
                    if a in ("--no-fault", "-n"):
                        continue
                    filtered.append(a)
                filtered.append("--no-fault")

                cmd = [str(args.run.resolve())] + filtered
                print(f"  Executing baseline run: {' '.join(cmd)}", flush=True)
                res = subprocess.run(cmd, capture_output=True, text=True,
                                     cwd=str(cwd.resolve()), timeout=7200)
                if res.returncode == 0:
                    args.baseline.parent.mkdir(parents=True, exist_ok=True)
                    args.baseline.write_text(res.stdout)
                    print(f"  Baseline log saved to {args.baseline}")
                else:
                    print(f"  FATAL: Baseline generation failed (exit {res.returncode})", file=sys.stderr)
                    sys.exit(res.returncode)
            else:
                print(f"  ERROR: Baseline file {args.baseline} not found and no --run binary provided to create it.",
                      file=sys.stderr)
                sys.exit(1)

        baseline = load_log(args.baseline)

    # ── Primary run ──────────────────────────────────────────────────────────
    primary = fault_runs[0] if fault_runs else baseline
    if primary is None:
        parser.error("No primary run available. Pass at least one of --log / --run / --baseline.")

    s = primary.summary
    print(f"\nPrimary: mode={s.mode} workers={s.num_workers} "
          f"batches={s.total_batches} faults={s.faults_injected} "
          f"throughput={s.throughput:.1f} batches/s\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"Generating plots → {out_dir}/\n")

    plot_gantt(primary, out_dir / f"gantt{ext}")
    plot_throughput(primary, baseline, out_dir / f"throughput{ext}", window=args.window)
    plot_pi_convergence(primary, out_dir / f"pi_convergence{ext}")
    plot_recovery_cdf(fault_runs if fault_runs else [primary], out_dir / f"recovery_cdf{ext}")

    # Degradation: assemble {num_faults → [RunData]}
    runs_by_faults: dict[int, list[RunData]] = defaultdict(list)
    if baseline:
        runs_by_faults[0].append(baseline)
    for r in fault_runs:
        runs_by_faults[r.summary.faults_injected].append(r)
    for nf, path_arg in [(1, args.fault1), (2, args.fault2), (3, args.fault3)]:
        if path_arg:
            runs_by_faults[nf].append(load_log(path_arg))

    if len(runs_by_faults) >= 2:
        plot_degradation(dict(runs_by_faults), out_dir / f"degradation{ext}")
    else:
        print("  [degradation] SKIP — need logs for ≥2 fault counts "
              "(pass --baseline and --log, or explicit --fault1/2/3).")

    print("\nDone.")


if __name__ == "__main__":
    main()
