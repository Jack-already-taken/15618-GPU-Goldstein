"""
Generate TIFF test suites for Goldstein scaling (float32 rad; see generate_phase.py).

Suites:
  1. size_scaling/     quadratic + light additive noise (few % residues); vary N
  2. density_scaling/    fixed 1024², vary noise_sigma (residue density scaling)
  3. shear_scaling/      fixed size, vary n_shears
  4. smooth/             noiseless baselines (ramp / quadratic / peaks)

manifest.csv: name, height, width, n_residues, residue_density, field_type, param, seed
  • param = noise_sigma for noisy/density cases, n_shears for shear, 0 for smooth
"""

from __future__ import annotations

import argparse
import csv
import os

from generate_phase import (
    build_case,
    field_peaks,
    field_quadratic,
    field_ramp,
    make_noisy,
    save_case,
    wrap_principal_rad,
)


def suite_size_scaling(outdir: str, sizes: list[int], noise_sigma: float, seed: int) -> list[tuple]:
    """Vary image size; same mild noise → a few percent residues (not noiseless)."""
    cases = []
    os.makedirs(outdir, exist_ok=True)
    for n in sizes:
        true, wrapped = make_noisy(n, n, base="quadratic", noise_sigma=noise_sigma, seed=seed)
        name = f"size_{n}x{n}"
        n_res = save_case(
            outdir,
            name,
            true,
            wrapped,
            n_shears=0,
            noise_sigma=noise_sigma,
            field_type="noisy",
            seed=seed,
        )
        density = n_res / float((n - 1) * (n - 1)) if n > 1 else 0.0
        cases.append((name, n, n, n_res, density, "noisy", noise_sigma, seed))
        print(f"  [size] {name}: σ={noise_sigma:.2f} → {n_res} residues ({density * 100:.4f}%)")
    return cases


def suite_density_scaling(outdir: str, size: int, sigmas: list[float], seed: int) -> list[tuple]:
    cases = []
    h = w = size
    for sigma in sigmas:
        true, wrapped = make_noisy(h, w, base="quadratic", noise_sigma=sigma, seed=seed)
        name = f"density_{size}x{size}_sigma{sigma:.2f}"
        n_res = save_case(
            outdir,
            name,
            true,
            wrapped,
            n_shears=0,
            noise_sigma=sigma,
            field_type="noisy",
            seed=seed,
        )
        density = n_res / float((h - 1) * (w - 1)) if h > 1 else 0.0
        cases.append((name, h, w, n_res, density, "noisy", sigma, seed))
        print(f"  [density] {name}: {n_res} residues ({density * 100:.4f}%)")
    return cases


def suite_shear_scaling(outdir: str, size: int, shear_counts: list[int], seed: int) -> list[tuple]:
    cases = []
    h = w = size
    for n_shears in shear_counts:
        true, wrapped = build_case(h, w, n_shears)
        name = f"shear_{size}x{size}_n{n_shears}"
        n_res = save_case(
            outdir, name, true, wrapped, n_shears=n_shears, field_type="shear", seed=seed
        )
        density = n_res / float((h - 1) * (w - 1)) if h > 1 else 0.0
        cases.append((name, h, w, n_res, density, "shear", float(n_shears), seed))
        print(f"  [shear] {name}: {n_res} residues ({density * 100:.4f}%)")
    return cases


def suite_smooth(outdir: str, size: int) -> list[tuple]:
    cases = []
    h = w = size

    true = field_ramp(h, w)
    wrapped = wrap_principal_rad(true)
    n_res = save_case(outdir, f"smooth_ramp_{size}x{size}", true, wrapped, field_type="ramp")
    cases.append(
        (f"smooth_ramp_{size}x{size}", h, w, n_res, n_res / float((h - 1) * (w - 1)), "ramp", 0.0, 0)
    )
    print(f"  [smooth] ramp: {n_res} residues")

    true = field_quadratic(h, w)
    wrapped = wrap_principal_rad(true)
    n_res = save_case(outdir, f"smooth_quad_{size}x{size}", true, wrapped, field_type="quadratic")
    cases.append(
        (f"smooth_quad_{size}x{size}", h, w, n_res, n_res / float((h - 1) * (w - 1)), "quadratic", 0.0, 0)
    )
    print(f"  [smooth] quadratic: {n_res} residues")

    true = field_peaks(h, w, n_peaks=6, seed=0)
    wrapped = wrap_principal_rad(true)
    n_res = save_case(outdir, f"smooth_peaks_{size}x{size}", true, wrapped, field_type="peaks", seed=0)
    cases.append(
        (f"smooth_peaks_{size}x{size}", h, w, n_res, n_res / float((h - 1) * (w - 1)), "peaks", 0.0, 0)
    )
    print(f"  [smooth] peaks: {n_res} residues")

    return cases


def write_manifest(outdir: str, cases: list[tuple]) -> None:
    path = os.path.join(outdir, "manifest.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["name", "height", "width", "n_residues", "residue_density", "field_type", "param", "seed"]
        )
        for c in cases:
            w.writerow(c)
    print(f"\nmanifest: {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="phase_data")
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--size-noise-sigma",
        type=float,
        default=0.72,
        help="noise (radians) on raw phase for size_scaling (~1–3%% residues; raise for more)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for noisy/shear suites")
    args = p.parse_args()

    if args.quick:
        sizes = [256, 512, 1024]
        sigmas = [0.6, 0.9, 1.2]
        shear_counts = [0, 4, 16]
        smooth_size = 512
        fixed_size = 512
    else:
        sizes = [256, 512, 1024, 2048, 4096, 8192]
        sigmas = [0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0]
        shear_counts = [0, 2, 4, 8, 16, 32, 64, 128, 256]
        smooth_size = 1024
        fixed_size = 1024

    all_cases: list[tuple] = []

    print("\n=== size scaling (noisy quadratic, few residues) ===")
    all_cases += suite_size_scaling(
        os.path.join(args.outdir, "size_scaling"),
        sizes,
        noise_sigma=args.size_noise_sigma,
        seed=args.seed,
    )

    print("\n=== density scaling (fixed size, vary noise_sigma) ===")
    all_cases += suite_density_scaling(
        os.path.join(args.outdir, "density_scaling"), fixed_size, sigmas, args.seed
    )

    print("\n=== shear scaling ===")
    all_cases += suite_shear_scaling(
        os.path.join(args.outdir, "shear_scaling"), fixed_size, shear_counts, args.seed
    )

    print("\n=== smooth (noiseless baselines) ===")
    all_cases += suite_smooth(os.path.join(args.outdir, "smooth"), smooth_size)

    write_manifest(args.outdir, all_cases)


if __name__ == "__main__":
    main()
