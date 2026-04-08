#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Only match NO scheduler sections
NO_SCHED_PATTERN = re.compile(
    r"Random Scaling NO scheduler \| actors=(\d+)\s+"
    r"===== SUPERVISOR TOTAL TIME spawn =====.*?"
    r"===== SUPERVISOR TOTAL TIME =====\s*"
    r"Total runtime:\s*([0-9]*\.?[0-9]+)\s*s",
    re.DOTALL,
)


def parse_file(path: Path):
    text = path.read_text(errors="replace")
    results = []
    for actors_str, runtime_str in NO_SCHED_PATTERN.findall(text):
        results.append((int(actors_str), float(runtime_str)))
    return results


def collect_data(base_dir: Path):
    """
    data[actors][gpu_count] = [runtime1, runtime2, ...]
    """
    data = defaultdict(lambda: defaultdict(list))

    for gpu_dir in sorted(base_dir.glob("gpus_*")):
        if not gpu_dir.is_dir():
            continue

        m = re.match(r"gpus_(\d+)$", gpu_dir.name)
        if not m:
            continue
        gpu_count = int(m.group(1))

        for txt_file in sorted(gpu_dir.glob("*.txt")):
            for actors, runtime in parse_file(txt_file):
                data[actors][gpu_count].append(runtime)

    return data


def make_plots(data, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for actors in sorted(data.keys()):
        gpu_map = data[actors]
        gpu_counts = sorted(gpu_map.keys())
        if not gpu_counts:
            continue

        means = [mean(gpu_map[g]) for g in gpu_counts]

        # 🔥 FIX: use categorical spacing instead of numeric spacing
        x_pos = list(range(len(gpu_counts)))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(x_pos, means)

        # Add value labels on top of bars
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.suptitle("multi-gpu scaling test on gpufarm7")
        ax.set_title(f"actors = {actors}")
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel("Runtime (s)")

        # Show actual GPU counts as labels (1,2,4,7) but evenly spaced
        ax.set_xticks(x_pos)
        ax.set_xticklabels(gpu_counts)

        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_file = out_dir / f"multi_gpu_scaling_actors_{actors}.png"
        fig.savefig(out_file, dpi=200)
        plt.close(fig)

        print(f"Saved {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPU scaling bar charts (NO scheduler only)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Directory containing gpus_* folders",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("gpu_scaling_plots"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    data = collect_data(args.base_dir)

    if not data:
        raise SystemExit("No data found.")

    make_plots(data, args.out_dir)


if __name__ == "__main__":
    main()
