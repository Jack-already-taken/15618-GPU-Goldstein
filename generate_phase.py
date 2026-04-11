"""
Synthetic wrapped phase image generator for Goldstein branch-cut unwrapping tests.

Generates true phase fields, wraps them to [-pi, pi], and saves both the wrapped
phase (input to unwrapping) and the true unwrapped phase (ground truth).

Output formats:
  - .npy   : float32 arrays (recommended for CUDA I/O; load with cnpy or custom reader)
  - .tiff  : 32-bit float TIFF (OpenCV can read with IMREAD_UNCHANGED)
  - .png   : 8-bit visualization only (NOT for numerical use)

Phase field types:
  - ramp       : smooth linear gradient, zero residues (sanity baseline)
  - quadratic  : smooth paraboloid, zero residues
  - peaks      : sum of Gaussians, zero residues in noise-free case
  - noisy      : smooth field + additive Gaussian noise on wrapped phase
                 (residue density controlled by noise sigma)
  - shear      : discontinuity-injected field producing many residues
                 (residue density controlled by number of shear segments)

Residues arise wherever the 2x2 wrapped-phase circulation is non-zero. Smooth
fields have zero residues; noise and discontinuities create them.
"""

import argparse
import os
import numpy as np


# ---------- phase field generators ----------

def field_ramp(h, w, slope_y=4.0, slope_x=6.0):
    """Linear ramp. Total phase range = slope_y*2pi vertically, slope_x*2pi horizontally."""
    y = np.linspace(0, slope_y * 2 * np.pi, h, dtype=np.float32)
    x = np.linspace(0, slope_x * 2 * np.pi, w, dtype=np.float32)
    return y[:, None] + x[None, :]


def field_quadratic(h, w, scale=8.0):
    """Paraboloid: scale*2pi at the corners."""
    y = np.linspace(-1, 1, h, dtype=np.float32)
    x = np.linspace(-1, 1, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return (scale * 2 * np.pi) * (xx * xx + yy * yy)


def field_peaks(h, w, n_peaks=5, amplitude=6.0, seed=0):
    """Sum of Gaussian bumps. Smooth, zero residues in the noise-free limit."""
    rng = np.random.default_rng(seed)
    y = np.linspace(-1, 1, h, dtype=np.float32)
    x = np.linspace(-1, 1, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    field = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_peaks):
        cy = rng.uniform(-0.7, 0.7)
        cx = rng.uniform(-0.7, 0.7)
        sigma = rng.uniform(0.1, 0.35)
        amp = rng.uniform(-amplitude, amplitude) * 2 * np.pi
        field += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return field


# ---------- wrapping and residue utilities ----------

def wrap(phi):
    """Wrap phase to [-pi, pi)."""
    return np.angle(np.exp(1j * phi)).astype(np.float32)


def count_residues(wrapped):
    """
    Count Goldstein residues on 2x2 cells.

    A residue exists where the sum of wrapped phase differences around a 2x2 loop
    is not zero. Returns (n_residues, residue_map) where residue_map has +1 / -1
    at residue cells and 0 elsewhere. Shape is (h-1, w-1).
    """
    def wdiff(a, b):
        return wrap(a - b)

    # corners of the 2x2 loop
    p00 = wrapped[:-1, :-1]
    p01 = wrapped[:-1, 1:]
    p11 = wrapped[1:, 1:]
    p10 = wrapped[1:, :-1]

    s = wdiff(p01, p00) + wdiff(p11, p01) + wdiff(p10, p11) + wdiff(p00, p10)
    # sum will be approximately 0, +2pi, or -2pi
    residue = np.round(s / (2 * np.pi)).astype(np.int8)
    n = int(np.count_nonzero(residue))
    return n, residue


# ---------- noise- and discontinuity-driven residue generation ----------

def make_noisy(h, w, base="quadratic", noise_sigma=0.5, seed=0, **kwargs):
    """
    Smooth base field + additive noise applied to the wrapped phase.
    Higher noise_sigma -> higher residue density.
    Measured on a 512x512 quadratic base:
        0.50 -> ~0.01% residues (very sparse)
        0.70 -> ~0.7%           (sparse)
        1.00 -> ~8%             (moderate)
        1.30 -> ~19%            (dense)
        1.60 -> ~27%            (very dense, stress test)
        2.00+-> ~32% (saturated; residue field approaches random)
    """
    if base == "ramp":
        true = field_ramp(h, w, **kwargs)
    elif base == "quadratic":
        true = field_quadratic(h, w, **kwargs)
    elif base == "peaks":
        true = field_peaks(h, w, seed=seed, **kwargs)
    else:
        raise ValueError(f"unknown base field {base}")

    rng = np.random.default_rng(seed)
    wrapped = wrap(true + rng.normal(0.0, noise_sigma, size=true.shape).astype(np.float32))
    return true, wrapped


def make_shear(h, w, n_shears=8, shear_strength=np.pi, seed=0):
    """
    Smooth quadratic base with injected line discontinuities that create
    many residues. n_shears controls residue density more directly than noise.
    """
    true = field_quadratic(h, w, scale=4.0)
    rng = np.random.default_rng(seed)
    out = true.copy()

    for _ in range(n_shears):
        horizontal = rng.random() < 0.5
        if horizontal:
            row = rng.integers(h // 8, 7 * h // 8)
            col0 = rng.integers(0, w // 2)
            col1 = rng.integers(w // 2, w)
            out[row:, col0:col1] += shear_strength
        else:
            col = rng.integers(w // 8, 7 * w // 8)
            row0 = rng.integers(0, h // 2)
            row1 = rng.integers(h // 2, h)
            out[row0:row1, col:] += shear_strength

    wrapped = wrap(out)
    return out, wrapped


# ---------- saving ----------

def save_npy(path, arr):
    np.save(path, arr.astype(np.float32))


def save_tiff(path, arr):
    """32-bit float TIFF. Uses tifffile if present, otherwise falls back to OpenCV."""
    arr = arr.astype(np.float32)
    try:
        import tifffile
        tifffile.imwrite(path, arr)
        return
    except ImportError:
        pass
    try:
        import cv2
        cv2.imwrite(path, arr)
    except ImportError:
        print(f"[warn] could not save {path}: install tifffile or opencv-python")


def save_png_visual(path, arr):
    """8-bit PNG for visual inspection only. Normalizes to full range."""
    try:
        import cv2
    except ImportError:
        return
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        vis = np.zeros_like(arr, dtype=np.uint8)
    else:
        vis = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
    cv2.imwrite(path, vis)


def save_case(outdir, name, true, wrapped, residue_map, formats):
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, name)
    if "npy" in formats:
        save_npy(stem + "_wrapped.npy", wrapped)
        save_npy(stem + "_true.npy", true)
    if "tiff" in formats:
        save_tiff(stem + "_wrapped.tiff", wrapped)
        save_tiff(stem + "_true.tiff", true)
    if "png" in formats:
        save_png_visual(stem + "_wrapped_vis.png", wrapped)
        save_png_visual(stem + "_true_vis.png", true)
        save_png_visual(stem + "_residues_vis.png", residue_map.astype(np.float32))


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="phase_data", help="output directory")
    p.add_argument("--type", default="noisy",
                   choices=["ramp", "quadratic", "peaks", "noisy", "shear"],
                   help="phase field type")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--noise-sigma", type=float, default=0.5,
                   help="noise level for --type noisy (controls residue density)")
    p.add_argument("--n-shears", type=int, default=8,
                   help="number of discontinuities for --type shear")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="base filename (default auto)")
    p.add_argument("--formats", default="npy,tiff,png",
                   help="comma-separated: npy,tiff,png")
    return p.parse_args()


def main():
    args = parse_args()
    h, w = args.height, args.width
    formats = set(f.strip() for f in args.formats.split(","))

    if args.type == "ramp":
        true = field_ramp(h, w)
        wrapped = wrap(true)
    elif args.type == "quadratic":
        true = field_quadratic(h, w)
        wrapped = wrap(true)
    elif args.type == "peaks":
        true = field_peaks(h, w, seed=args.seed)
        wrapped = wrap(true)
    elif args.type == "noisy":
        true, wrapped = make_noisy(h, w, base="quadratic",
                                   noise_sigma=args.noise_sigma, seed=args.seed)
    elif args.type == "shear":
        true, wrapped = make_shear(h, w, n_shears=args.n_shears, seed=args.seed)
    else:
        raise ValueError(args.type)

    n_res, res_map = count_residues(wrapped)
    total_cells = (h - 1) * (w - 1)
    density = n_res / total_cells

    name = args.name or f"{args.type}_{h}x{w}_seed{args.seed}"
    save_case(args.outdir, name, true, wrapped, res_map, formats)

    print(f"[ok] {name}")
    print(f"     size         : {h} x {w}")
    print(f"     residues     : {n_res} ({density*100:.4f}% of 2x2 cells)")
    print(f"     wrapped range: [{wrapped.min():.3f}, {wrapped.max():.3f}]")
    print(f"     true range   : [{true.min():.3f}, {true.max():.3f}]")
    print(f"     saved to     : {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
