"""
Synthetic wrapped phase for the Goldstein unwrapping benchmark.

Writes float32 TIFF consumed by src/main.c (wrapped phase in radians,
principal value (-π, π]; true phase in radians).

Includes residue / noise control via additive Gaussian noise on the raw
phase before wrapping (``make_noisy``), plus ramp / quadratic / peaks /
deterministic shear cases — similar to the original PNG-era suite, but TIFF.

Python: ``pip install tifffile`` (or imageio).
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


# ---------- phase field generators (raw radians) ----------


def field_ramp(h: int, w: int, slope_y: float = 4.0, slope_x: float = 6.0) -> np.ndarray:
    y = np.linspace(0, slope_y * 2 * np.pi, h, dtype=np.float32)
    x = np.linspace(0, slope_x * 2 * np.pi, w, dtype=np.float32)
    return y[:, None] + x[None, :]


def field_quadratic(h: int, w: int, scale: float = 8.0) -> np.ndarray:
    y = np.linspace(-1, 1, h, dtype=np.float64)
    x = np.linspace(-1, 1, w, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    v = scale * 2.0 * math.pi * (xx * xx + yy * yy)
    return v.astype(np.float32)


def field_peaks(h: int, w: int, n_peaks: int = 5, amplitude: float = 6.0, seed: int = 0) -> np.ndarray:
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


def wrap_principal_rad(phi: np.ndarray) -> np.ndarray:
    p = np.asarray(phi, dtype=np.float64)
    return np.arctan2(np.sin(p), np.cos(p)).astype(np.float32)


def count_residues_from_wrapped_rad(wrapped_rad: np.ndarray) -> tuple[int, np.ndarray]:
    """2×2 residue count; map wrapped radians to [0,1) per Goldstein convention."""

    def wdiff(a, b):
        d = a - b
        return d - np.round(d)

    u = np.mod(np.asarray(wrapped_rad, dtype=np.float64) / (2.0 * np.pi), 1.0)
    p00 = u[:-1, :-1]
    p01 = u[:-1, 1:]
    p11 = u[1:, 1:]
    p10 = u[1:, :-1]
    s = wdiff(p01, p00) + wdiff(p11, p01) + wdiff(p10, p11) + wdiff(p00, p10)
    residue = np.round(s).astype(np.int8)
    return int(np.count_nonzero(residue)), residue


def make_noisy(
    h: int,
    w: int,
    base: str = "quadratic",
    noise_sigma: float = 0.5,
    seed: int = 0,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    if base == "ramp":
        true = field_ramp(h, w, **kwargs)  # type: ignore[arg-type]
    elif base == "quadratic":
        true = field_quadratic(h, w, **kwargs)  # type: ignore[arg-type]
    elif base == "peaks":
        true = field_peaks(h, w, seed=seed, **kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"unknown base field {base}")

    rng = np.random.default_rng(seed)
    noisy_raw = true + rng.normal(0.0, noise_sigma, size=true.shape).astype(np.float32)
    wrapped = wrap_principal_rad(noisy_raw)
    return true, wrapped


def make_shear(h: int, w: int, n_shears: int = 8, shear_strength: float = np.pi, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    true = field_quadratic(h, w, scale=4.0)
    rng = np.random.default_rng(seed)
    out = true.copy()

    for _ in range(n_shears):
        if rng.random() < 0.5:
            row = rng.integers(h // 8, 7 * h // 8)
            col0 = rng.integers(0, w // 2)
            col1 = rng.integers(w // 2, w)
            out[row:, col0:col1] += np.float32(shear_strength)
        else:
            col = rng.integers(w // 8, 7 * w // 8)
            row0 = rng.integers(0, h // 2)
            row1 = rng.integers(h // 2, h)
            out[row0:row1, col:] += np.float32(shear_strength)

    wrapped = wrap_principal_rad(out)
    return out, wrapped


def apply_deterministic_shears(
    phi: np.ndarray, n_shears: int, shear_strength: float = math.pi
) -> np.ndarray:
    out = np.asarray(phi, dtype=np.float32).copy()
    h, w = out.shape
    if n_shears <= 0:
        return out
    for k in range(n_shears):
        if k % 2 == 0:
            row = max(1, min(h - 2, (k + 1) * h // (n_shears + 2)))
            col0, col1 = w // 4, 3 * w // 4
            if col1 > col0:
                out[row:, col0:col1] += np.float32(shear_strength)
        else:
            col = max(1, min(w - 2, (k + 1) * w // (n_shears + 2)))
            row0, row1 = h // 4, 3 * h // 4
            if row1 > row0:
                out[row0:row1, col:] += np.float32(shear_strength)
    return out


def build_case(h: int, w: int, n_shears: int) -> tuple[np.ndarray, np.ndarray]:
    base = field_quadratic(h, w, scale=4.0)
    true_unwrapped = apply_deterministic_shears(base, n_shears)
    wrapped_rad = wrap_principal_rad(true_unwrapped)
    return true_unwrapped, wrapped_rad


def save_tiff_float32(path: str, arr: np.ndarray) -> None:
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("save_tiff_float32 expects a 2-D array")
    try:
        import tifffile

        tifffile.imwrite(path, data, photometric="minisblack")
        return
    except ImportError:
        pass
    try:
        import imageio.v3 as iio

        iio.imwrite(path, data, extension=".tif")
        return
    except ImportError:
        pass
    raise RuntimeError(
        "Install tifffile (pip install tifffile) or imageio to write TIFF"
    )


def save_case(
    outdir: str,
    name: str,
    true_rad: np.ndarray,
    wrapped_rad: np.ndarray,
    *,
    n_shears: int = 0,
    noise_sigma: float | None = None,
    field_type: str = "",
    seed: int | None = None,
) -> int:
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, name)

    save_tiff_float32(stem + "_wrapped.tif", wrapped_rad.astype(np.float32))
    save_tiff_float32(stem + "_true.tif", true_rad.astype(np.float32))

    lo = float(true_rad.min())
    hi = float(true_rad.max())
    n_res, _ = count_residues_from_wrapped_rad(wrapped_rad)
    hh, ww = wrapped_rad.shape

    meta: dict = {
        "name": name,
        "height": int(hh),
        "width": int(ww),
        "n_shears": int(n_shears),
        "n_residues": n_res,
        "residue_density": n_res / float((hh - 1) * (ww - 1)) if hh > 1 and ww > 1 else 0.0,
        "wrapped_format": "float32; principal wrapped phase in radians (-π, π]",
        "true_format": "float32; unwrapped absolute phase in radians",
        "true_lo": lo,
        "true_hi": hi,
        "field_type": field_type,
    }
    if noise_sigma is not None:
        meta["noise_sigma"] = float(noise_sigma)
    if seed is not None:
        meta["seed"] = int(seed)

    with open(stem + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return n_res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--outdir", default="phase_data", help="output directory")
    p.add_argument(
        "--type",
        default="noisy",
        choices=["ramp", "quadratic", "peaks", "noisy", "shear"],
    )
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument(
        "--noise-sigma",
        type=float,
        default=0.72,
        help="Gaussian noise on raw phase (radians) for --type noisy",
    )
    p.add_argument("--n-shears", type=int, default=8, help="for --type shear")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    h, w = args.height, args.width

    if args.type == "ramp":
        true = field_ramp(h, w)
        wrapped = wrap_principal_rad(true)
        n_shears = 0
        noise_sigma = None
    elif args.type == "quadratic":
        true = field_quadratic(h, w)
        wrapped = wrap_principal_rad(true)
        n_shears = 0
        noise_sigma = None
    elif args.type == "peaks":
        true = field_peaks(h, w, seed=args.seed)
        wrapped = wrap_principal_rad(true)
        n_shears = 0
        noise_sigma = None
    elif args.type == "noisy":
        true, wrapped = make_noisy(
            h, w, base="quadratic", noise_sigma=args.noise_sigma, seed=args.seed
        )
        n_shears = 0
        noise_sigma = args.noise_sigma
    elif args.type == "shear":
        true, wrapped = make_shear(h, w, n_shears=args.n_shears, seed=args.seed)
        n_shears = args.n_shears
        noise_sigma = None
    else:
        raise ValueError(args.type)

    name = args.name or f"{args.type}_{h}x{w}_seed{args.seed}"
    n_res = save_case(
        args.outdir,
        name,
        true,
        wrapped,
        n_shears=n_shears,
        noise_sigma=noise_sigma,
        field_type=args.type,
        seed=args.seed,
    )

    density = n_res / float((h - 1) * (w - 1)) if h > 1 and w > 1 else 0.0
    print(f"[ok] {name}")
    print(f"     size          : {h} x {w}")
    print(f"     field         : {args.type}")
    print(f"     residues      : {n_res} ({density * 100:.4f}% of 2x2 cells)")
    print(f"     wrapped (rad) : [{wrapped.min():.4f}, {wrapped.max():.4f}]")
    print(f"     true (rad)    : [{true.min():.3f}, {true.max():.3f}]")
    print(f"     saved to      : {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
