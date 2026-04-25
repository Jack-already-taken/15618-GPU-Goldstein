#!/usr/bin/env python3
"""
plot_sweep.py — visualize size scaling sweep results across all 3 unwrap stages.

Reads the CSV produced by sweep_size.sh and emits:
  1. runtime_vs_size.png    — per-stage absolute runtime, one subplot per stage
  2. speedup_vs_size.png    — CUDA speedup over serial for each stage
  3. breakdown_vs_size.png  — stacked stage-time breakdown per (size, backend)
  4. summary.csv            — averaged-over-repeats pivot, one row per size
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAGE_COLS = [
    "load_ms",
    "init_bitflags_ms",
    "gradxy_ms",
    "cuda_init_ms",
    "residues_ms",
    "branch_cuts_ms",
    "unwrap_ms",
    "kernel_total_ms",
    "write_ms",
    "total_ms",
]

# (column_name, human_label) — drives every per-stage plot.
STAGES = [
    ("residues_ms",    "Stage 1 (residues)"),
    ("branch_cuts_ms", "Stage 2 (branch cuts)"),
    ("unwrap_ms",      "Stage 3 (unwrap)"),
]

STAGE_COLORS = {
    "residues_ms":    "#1f77b4",
    "branch_cuts_ms": "#ff7f0e",
    "unwrap_ms":      "#2ca02c",
}


def load_and_average(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    for c in STAGE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = (
        df.groupby(["size", "backend"], as_index=False)[STAGE_COLS]
        .mean(numeric_only=True)
        .sort_values(["backend", "size"])
        .reset_index(drop=True)
    )
    return agg


def pivot_by_backend(agg: pd.DataFrame, col: str) -> pd.DataFrame:
    return agg.pivot(index="size", columns="backend", values=col).sort_index()


def plot_runtime_all_stages(agg: pd.DataFrame, out_path: Path) -> None:
    """One subplot per stage, each comparing all backends."""
    fig, axes = plt.subplots(1, len(STAGES), figsize=(15, 5), sharey=False)
    if len(STAGES) == 1:
        axes = [axes]

    backends = sorted(agg["backend"].unique())

    for ax, (col, label) in zip(axes, STAGES):
        for backend in backends:
            sub = agg[agg["backend"] == backend].sort_values("size")
            if col not in sub.columns:
                continue
            ax.plot(
                sub["size"],
                sub[col],
                marker="o",
                linewidth=2,
                label=backend,
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("image side length (pixels)")
        ax.set_ylabel("wall time (ms)")
        ax.set_title(label)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="best")

    fig.suptitle("Per-stage runtime vs image size", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_speedup_all_stages(agg: pd.DataFrame, out_path: Path) -> None:
    """Serial / CUDA speedup, one line per stage."""
    fig, ax = plt.subplots(figsize=(9, 6))

    plotted = 0
    for col, label in STAGES:
        pv = pivot_by_backend(agg, col)
        if "serial" not in pv.columns or "cuda" not in pv.columns:
            print(f"warn: missing serial or cuda backend for {col}", file=sys.stderr)
            continue
        mask = (pv["serial"] > 0) & (pv["cuda"] > 0)
        pv = pv[mask]
        if pv.empty:
            print(f"warn: no valid serial/cuda rows for {col}", file=sys.stderr)
            continue

        speedup = pv["serial"] / pv["cuda"]
        ax.plot(
            pv.index,
            speedup.values,
            marker="o",
            linewidth=2,
            color=STAGE_COLORS.get(col),
            label=label,
        )
        plotted += 1

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="parity")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("image side length (pixels)")
    ax.set_ylabel("speedup (serial / cuda)")
    ax.set_title("CUDA speedup over serial CPU — per stage")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    if plotted == 0:
        print("warn: speedup plot has no series", file=sys.stderr)


def plot_breakdown(agg: pd.DataFrame, out_path: Path) -> None:
    """Stacked bar chart: per (size, backend), bar segments are the 3 stages."""
    sizes = sorted(agg["size"].unique())
    backends = sorted(agg["backend"].unique())

    if not sizes or not backends:
        print("warn: empty breakdown data", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(sizes) * len(backends)), 6))

    n_back = len(backends)
    bar_w = 0.8 / n_back
    x_base = np.arange(len(sizes))
    tick_labels = []

    for bi, backend in enumerate(backends):
        offsets = x_base + (bi - (n_back - 1) / 2.0) * bar_w
        bottom = np.zeros(len(sizes))

        for col, label in STAGES:
            heights = np.zeros(len(sizes))
            for si, sz in enumerate(sizes):
                row = agg[(agg["backend"] == backend) & (agg["size"] == sz)]
                if not row.empty and col in row.columns:
                    val = row[col].iloc[0]
                    if pd.notna(val):
                        heights[si] = float(val)
            ax.bar(
                offsets,
                heights,
                width=bar_w,
                bottom=bottom,
                color=STAGE_COLORS.get(col),
                edgecolor="white",
                linewidth=0.5,
                label=label if bi == 0 else None,
            )
            bottom += heights

        for si, sz in enumerate(sizes):
            ax.text(
                offsets[si],
                bottom[si],
                backend,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("image side length (pixels)")
    ax.set_ylabel("wall time (ms)")
    ax.set_title("Stage time breakdown per (size, backend)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_summary(agg: pd.DataFrame, out_path: Path) -> None:
    wide = agg.pivot(index="size", columns="backend", values=STAGE_COLS)
    wide.columns = [f"{b}_{c}" for c, b in wide.columns]
    wide = wide.sort_index()

    for col, _ in STAGES:
        s_col = f"serial_{col}"
        c_col = f"cuda_{col}"
        if s_col in wide.columns and c_col in wide.columns:
            wide[f"speedup_{col}"] = wide[s_col] / wide[c_col]

    wide.to_csv(out_path, float_format="%.3f")
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="sweep_results.csv",
                    help="input CSV from sweep_size.sh")
    ap.add_argument("-o", "--out-dir", default="sweep_plots",
                    help="output directory for PNGs and summary CSV")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"error: {csv_path} not found", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = load_and_average(csv_path)
    if agg.empty:
        print("error: no rows after aggregation", file=sys.stderr)
        return 1

    plot_runtime_all_stages(agg, out_dir / "runtime_vs_size.png")
    plot_speedup_all_stages(agg, out_dir / "speedup_vs_size.png")
    plot_breakdown(agg, out_dir / "breakdown_vs_size.png")
    write_summary(agg, out_dir / "summary.csv")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
