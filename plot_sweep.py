#!/usr/bin/env python3
"""
plot_sweep.py — visualize size scaling sweep results for Stage 2 only.

Reads the CSV produced by sweep_size.sh and emits:
  1. stage2_speedup_vs_size.png  — CUDA speedup over serial for branch cut stage
  2. stage2_runtime_vs_size.png  — absolute runtime for branch cut stage
  3. summary.csv                 — averaged-over-repeats pivot, one row per size
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


def plot_stage2_speedup(agg: pd.DataFrame, out_path: Path) -> None:
    """CUDA speedup = serial / cuda for Stage 2 (branch cuts) only."""
    pv = pivot_by_backend(agg, "branch_cuts_ms")

    fig, ax = plt.subplots(figsize=(9, 6))

    if "serial" in pv.columns and "cuda" in pv.columns:
        mask = (pv["serial"] > 0) & (pv["cuda"] > 0)
        pv = pv[mask]

        if not pv.empty:
            speedup = pv["serial"] / pv["cuda"]
            ax.plot(
                pv.index,
                speedup.values,
                marker="o",
                linewidth=2,
                label="Stage 2 (branch cuts)",
            )
        else:
            print("warn: no valid serial/cuda rows for branch_cuts_ms", file=sys.stderr)
    else:
        print("warn: missing serial or cuda backend for branch_cuts_ms", file=sys.stderr)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="parity")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("image side length (pixels)")
    ax.set_ylabel("speedup (serial / cuda)")
    ax.set_title("CUDA speedup over serial CPU — Stage 2 (branch cuts)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_stage2_runtime(agg: pd.DataFrame, out_path: Path) -> None:
    """Absolute runtime for Stage 2 (branch cuts) only."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for backend in sorted(agg["backend"].unique()):
        sub = agg[agg["backend"] == backend].sort_values("size")
        ax.plot(
            sub["size"],
            sub["branch_cuts_ms"],
            marker="o",
            linewidth=2,
            label=f"{backend} — Stage 2",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("image side length (pixels)")
    ax.set_ylabel("wall time (ms)")
    ax.set_title("Stage 2 (branch cuts) runtime vs image size")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_summary(agg: pd.DataFrame, out_path: Path) -> None:
    wide = agg.pivot(index="size", columns="backend", values=STAGE_COLS)
    wide.columns = [f"{b}_{c}" for c, b in wide.columns]
    wide = wide.sort_index()

    s_col = "serial_branch_cuts_ms"
    c_col = "cuda_branch_cuts_ms"
    if s_col in wide.columns and c_col in wide.columns:
        wide["speedup_branch_cuts_ms"] = wide[s_col] / wide[c_col]

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

    plot_stage2_speedup(agg, out_dir / "stage2_speedup_vs_size.png")
    plot_stage2_runtime(agg, out_dir / "stage2_runtime_vs_size.png")
    write_summary(agg, out_dir / "summary.csv")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())