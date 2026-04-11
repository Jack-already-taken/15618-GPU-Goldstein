"""
Generate full test suites for Goldstein branch-cut scaling experiments.

Produces three suites:

  1. size_scaling/     -- fixed residue density, varying image size
                          (for measuring speedup vs problem size)
  2. density_scaling/  -- fixed image size, varying residue density
                          (for measuring robustness of irregular stages)
  3. smooth/           -- smooth fields with zero or near-zero residues
                          (correctness sanity checks; integration-only stress)

Each case saves:
  <name>_wrapped.npy   -- float32 wrapped phase input
  <name>_true.npy      -- float32 ground-truth unwrapped phase
  <name>_wrapped.tiff  -- same as .npy but TIFF (OpenCV-friendly)
  *_vis.png            -- 8-bit visualizations (do NOT use numerically)

A manifest.csv lists every case with dimensions, residue count, and density.

Usage:
    python generate_test_suite.py --outdir phase_data
    python generate_test_suite.py --outdir phase_data --quick   # smaller suite
"""

import argparse
import csv
import os
import numpy as np

from generate_phase import (
    field_ramp, field_quadratic, field_peaks,
    wrap, count_residues,
    make_noisy, make_shear,
    save_case,
)


FORMATS = {"npy", "tiff", "png"}


def suite_size_scaling(outdir, sizes, noise_sigma=1.0, seed=0):
    """Fix residue density (roughly), sweep image size."""
    cases = []
    for n in sizes:
        true, wrapped = make_noisy(n, n, base="quadratic",
                                   noise_sigma=noise_sigma, seed=seed)
        n_res, res_map = count_residues(wrapped)
        name = f"size_{n}x{n}"
        save_case(outdir, name, true, wrapped, res_map, FORMATS)
        density = n_res / ((n - 1) * (n - 1))
        cases.append((name, n, n, n_res, density, "noisy", noise_sigma, seed))
        print(f"  [size]    {name}: {n_res} residues ({density*100:.3f}%)")
    return cases


def suite_density_scaling(outdir, size, sigmas, seed=0):
    """Fix image size, sweep noise level -> residue density."""
    cases = []
    h = w = size
    for sigma in sigmas:
        true, wrapped = make_noisy(h, w, base="quadratic",
                                   noise_sigma=sigma, seed=seed)
        n_res, res_map = count_residues(wrapped)
        name = f"density_{size}x{size}_sigma{sigma:.2f}"
        save_case(outdir, name, true, wrapped, res_map, FORMATS)
        density = n_res / ((h - 1) * (w - 1))
        cases.append((name, h, w, n_res, density, "noisy", sigma, seed))
        print(f"  [density] {name}: {n_res} residues ({density*100:.3f}%)")
    return cases


def suite_shear_scaling(outdir, size, shear_counts, seed=0):
    """Discontinuity-driven residues. Useful because residues come in structured
    pairs rather than random scatter, which stresses branch-cut placement."""
    cases = []
    h = w = size
    for n_shears in shear_counts:
        true, wrapped = make_shear(h, w, n_shears=n_shears, seed=seed)
        n_res, res_map = count_residues(wrapped)
        name = f"shear_{size}x{size}_n{n_shears}"
        save_case(outdir, name, true, wrapped, res_map, FORMATS)
        density = n_res / ((h - 1) * (w - 1))
        cases.append((name, h, w, n_res, density, "shear", float(n_shears), seed))
        print(f"  [shear]   {name}: {n_res} residues ({density*100:.3f}%)")
    return cases


def suite_smooth(outdir, size):
    """Smooth fields: zero residues in exact arithmetic. Correctness baselines."""
    cases = []
    h = w = size

    true = field_ramp(h, w)
    wrapped = wrap(true)
    n_res, res_map = count_residues(wrapped)
    save_case(outdir, f"smooth_ramp_{size}x{size}", true, wrapped, res_map, FORMATS)
    cases.append((f"smooth_ramp_{size}x{size}", h, w, n_res,
                  n_res / ((h - 1) * (w - 1)), "ramp", 0.0, 0))
    print(f"  [smooth]  ramp: {n_res} residues")

    true = field_quadratic(h, w)
    wrapped = wrap(true)
    n_res, res_map = count_residues(wrapped)
    save_case(outdir, f"smooth_quad_{size}x{size}", true, wrapped, res_map, FORMATS)
    cases.append((f"smooth_quad_{size}x{size}", h, w, n_res,
                  n_res / ((h - 1) * (w - 1)), "quadratic", 0.0, 0))
    print(f"  [smooth]  quadratic: {n_res} residues")

    true = field_peaks(h, w, n_peaks=6, seed=0)
    wrapped = wrap(true)
    n_res, res_map = count_residues(wrapped)
    save_case(outdir, f"smooth_peaks_{size}x{size}", true, wrapped, res_map, FORMATS)
    cases.append((f"smooth_peaks_{size}x{size}", h, w, n_res,
                  n_res / ((h - 1) * (w - 1)), "peaks", 0.0, 0))
    print(f"  [smooth]  peaks: {n_res} residues")

    return cases


def write_manifest(outdir, cases):
    path = os.path.join(outdir, "manifest.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "height", "width", "n_residues", "residue_density",
                    "field_type", "param", "seed"])
        for c in cases:
            w.writerow(c)
    print(f"\nmanifest: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="phase_data")
    p.add_argument("--quick", action="store_true",
                   help="smaller suite for fast iteration")
    args = p.parse_args()

    if args.quick:
        sizes = [256, 512, 1024]
        sigmas = [0.6, 0.9, 1.3]
        shear_counts = [20, 100]
        smooth_size = 512
    else:
        # full suite: sizes go up to 8k for the scaling sweep
        sizes = [256, 512, 1024, 2048, 4096, 8192]
        # sigmas chosen to span sparse -> dense -> saturated (see generate_phase.py docstring)
        sigmas = [0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0]
        shear_counts = [10, 25, 50, 100, 200, 400]
        smooth_size = 1024

    all_cases = []

    print("\n=== size scaling (fixed density) ===")
    size_dir = os.path.join(args.outdir, "size_scaling")
    all_cases += suite_size_scaling(size_dir, sizes)

    print("\n=== density scaling (fixed size) ===")
    dens_dir = os.path.join(args.outdir, "density_scaling")
    all_cases += suite_density_scaling(dens_dir, 1024, sigmas)

    print("\n=== shear scaling (structured residues) ===")
    shear_dir = os.path.join(args.outdir, "shear_scaling")
    all_cases += suite_shear_scaling(shear_dir, 1024, shear_counts)

    print("\n=== smooth (correctness baselines) ===")
    smooth_dir = os.path.join(args.outdir, "smooth")
    all_cases += suite_smooth(smooth_dir, smooth_size)

    write_manifest(args.outdir, all_cases)

    total_bytes = sum(
        h * w * 4 * 2  # wrapped + true, float32
        for (_, h, w, *_rest) in all_cases
    )
    print(f"total float32 payload: {total_bytes / 1e9:.2f} GB "
          f"(×~2 on disk with TIFF+PNG)")


if __name__ == "__main__":
    main()
