# Parallel Goldstein phase unwrapping

This project implements a **parallel Goldstein** phase-unwrapping algorithm based on breadth-first propagation and branch cuts. The numerical approach follows the work published in *Optics and Lasers in Engineering*:

> W. De la Cruz, “Parallel Goldstein’s algorithm for two-dimensional phase unwrapping,” *Opt. Lasers Eng.*, vol. 121, 2019, Art. no. 105827.  
> DOI: [10.1016/j.optlaseng.2019.105827](https://doi.org/10.1016/j.optlaseng.2019.105827)

## Acknowledgment

The **original C implementation and dataset layout** are due to **William De la Cruz** and are distributed under the **MIT License** (see `LICENSE`). This repository retains that core algorithm and directory spirit (`include/`, `src/`, `data/`, `matlab/`) while extending it for **float32 TIFF** I/O, **OpenMP** throughput, optional **ground-truth RMS**, and **Python** tools to synthesize benchmark phases compatible with the native `main` executable.

If you use the original method or code ideas, please cite the paper above and respect the license terms.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `include/` | Headers (`pi.h`, `grad.h`, `util.h`, `tiff_io.h`, bundled **CImg** for TIFF helpers). |
| `src/` | C/C++ sources; run **`make`** here to build `main`. |
| `data/` | Sample binary phase data used by the legacy built-in **peaks** self-test. |
| `matlab/` | Original MATLAB helpers for simulated phase (optional). |
| `generate_phase.py` | Generate a **single** synthetic case as TIFF + JSON sidecar. |
| `generate_test_suite.py` | Generate **scaling suites** (size / density / shear / smooth) + `manifest.csv`. |
| `requirements.txt` | Python dependencies (`tifffile`, etc.). |

---

## Build (`main`)

From **`src/`**:

```bash
cd src
make
```

### Dependencies

- **GCC** (or compatible) with **OpenMP** (`-fopenmp`).
- **libtiff** development package (e.g. `libtiff-devel` on RHEL, `libtiff-dev` on Debian/Ubuntu).
- **g++** with **C++17** for `tiff_io.cpp` (wraps CImg + libtiff; display disabled).

The default make target produces **`./main`**.

---

## Python: synthetic TIFF datasets

Install dependencies once (from the repository root):

```bash
pip install -r requirements.txt
```

Generators write **single-channel float32 TIFF**:

- **Wrapped** phase: principal value in **radians** (approximately **(−π, π]**).
- **True** phase: **unwrapped** radians (for RMS against `-g`).

Each case also writes a small **JSON** metadata file (`*_true` range, residue count, etc.) suitable for `main`’s optional `-j` logging.

### `generate_phase.py` — one case at a time

Typical usage:

```bash
python generate_phase.py --outdir phase_data --type noisy --height 1024 --width 1024 --seed 0
```

Useful options:

| Option | Meaning |
|--------|---------|
| `--outdir DIR` | Output directory (default `phase_data`). |
| `--type TYPE` | `ramp`, `quadratic`, `peaks`, `noisy` (quadratic + Gaussian noise), or `shear`. |
| `--height`, `--width` | Image size (default 1024). |
| `--noise-sigma` | Noise std-dev in **radians** for `--type noisy` (default `0.72`). |
| `--n-shears` | Number of shear steps for `--type shear`. |
| `--seed` | RNG seed. |
| `--name STEM` | Base filename; default is derived from type/size/seed. |

**Output files** (with stem `NAME`):

- `NAME_wrapped.tif` — wrapped phase (float32).
- `NAME_true.tif` — unwrapped truth (float32).
- `NAME.json` — metadata (`true_lo`, `true_hi`, residue stats, …).

### `generate_test_suite.py` — benchmark suites

Builds four subfolders under `--outdir` (default `phase_data`):

1. **`size_scaling/`** — noisy quadratic fields at several resolutions.  
2. **`density_scaling/`** — fixed size, varying noise level (residue density).  
3. **`shear_scaling/`** — fixed size, varying deterministic shear count.  
4. **`smooth/`** — noiseless ramp / quadratic / peaks baselines.

Also writes **`manifest.csv`** at the suite root summarizing each case.

```bash
# Full suite (large; many large TIFFs)
python generate_test_suite.py --outdir phase_data

# Smaller grids for a quick smoke test
python generate_test_suite.py --outdir phase_data --quick

# Tune mild noise for the size-scaling suite
python generate_test_suite.py --outdir phase_data --size-noise-sigma 0.72 --seed 1
```

| Option | Meaning |
|--------|---------|
| `--outdir DIR` | Root output directory. |
| `--quick` | Smaller size lists and fewer parameter points. |
| `--size-noise-sigma` | Noise (rad) for the size-scaling noisy quadratic cases. |
| `--seed` | RNG seed for noisy/shear suites. |

---

## Running `main`

Invoke **`./main`** from `src/` (or pass the full path). **TIFF mode** is selected with **`-i`**.

### TIFF workflow (recommended)

```bash
./main -i ../phase_data/my_case_wrapped.tif
```

With optional ground truth for **mean-offset RMS** and JSON metadata for console logging:

```bash
./main -i ../phase_data/my_case_wrapped.tif \
       -g ../phase_data/my_case_true.tif \
       -j ../phase_data/my_case.json
```

Redirect outputs to a directory (`mkdir -p` is applied automatically):

```bash
./main -i ../phase_data/my_case_wrapped.tif -o ../runs/exp1
```

This writes `<stem>_unwrapped.tif` into `../runs/exp1/`, where `<stem>` is the input **basename without extension**.

### Options summary

| Flag | Long form | Description |
|------|-----------|-------------|
| `-i` | `--input` | Wrapped phase: **float32** grayscale TIFF, radians **~[−π, π]**. |
| `-g` | `--ground` | Optional float32 **unwrapped** truth TIFF (same size as input) for RMS. |
| `-j` | `--json` | Optional JSON with `true_lo` / `true_hi` (metadata only; RMS uses TIFF samples). |
| `-o` | `--output` | Output **directory** for `*_unwrapped.tif` (default: same folder as input). With **`-v`**, also writes `*_residues.tif` and `*_branchcuts.tif`. |
| `-m` | `--mask` | Load mask from `<output-prefix>.mask` (same prefix rules as outputs). |
| `-t` | `--threads` | OpenMP thread count (default: heuristic). |
| `-v` | `--verify-serial` | Extra serial comparisons + debug TIFFs; more time and memory. |
| `-h` | `--help` | Print usage. |

**Outputs**

- Always: **`<prefix>_unwrapped.tif`** (float32, radians).  
- With **`-v`**: **`<prefix>_residues.tif`**, **`<prefix>_branchcuts.tif`** (uint8 debug views).

### Built-in binary test (no `-i`)

Running **`./main`** with **no** `-i` runs the legacy **peaks** benchmark from **`../data/`** (paths relative to the current working directory; the program `chdir("..")` from `src/` in the default path). This path is mainly for regression / timing on the original binary layout.

---

## Licensing

See **`LICENSE`** for the MIT terms and disclaimer from the original author. Extensions in this tree (TIFF I/O, Python generators, build tweaks) are provided under the same spirit; keep the original copyright and license notice in distributions.

---

## Platform notes

The code is developed on **Linux** with **GCC**. **macOS** builds are generally possible with Homebrew `libtiff` and a suitable OpenMP toolchain; adjust include/library paths if needed.
