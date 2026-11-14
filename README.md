# Diff-FVR: Differentiable Fourier Volume Rendering

[![PyPI version](https://img.shields.io/pypi/v/diff-fvr.svg)](https://pypi.org/project/diff-fvr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/pareelamre/diff-fvr/blob/master/LICENSE)

**Dual-Domain Differentiable Fourier Volume Rendering (`Diff-FVR` & perspective `PoU-FVR`)** for high-speed digitally reconstructed radiographs (DRRs) and biplanar 3D–2D spine/pelvis registration on CPU and GPU.

Instead of stepping `O(N)` points per ray (`O(N^3)` per projection), Diff-FVR precomputes a padded 3D FFT once and evaluates candidate views via `O(N^2)` Fourier slicing—pairing Parseval Zero-IFFT Spectral-NCC (parallel beam) with Partition-of-Unity Perspective Slab FVR (`PoU-FVR`) and fused Triton/`CUDAGraph` kernels for true cone-beam C-arm geometry.

> **Research software.** Clinical accuracy and suitability for surgical navigation have not been established beyond retrospective phantom and cadaveric benchmark datasets.

Author: Pareel Amre · TU Dresden · `pareel.amre@gmail.com` · License: [MIT](https://github.com/pareelamre/diff-fvr/blob/master/LICENSE)

---

## Benchmarks vs. Spatial DRR Renderers (`DiffDRR` & `nanodrr`)

### 1. Clinical & Multi-Start 6-DoF Registration (NVIDIA A100-SXM4-40GB)

#### Real Paired Clinical DeepFluoro Cohort (`exp31`, All 6 Subjects, 18 Trials, `H=256`, `B=64`, Mean Init `12.002 mm`)

Evaluated under true cone-beam geometry against DiffDRR (`0.9782–0.9986` true-pose forward NCC across `subject01`–`subject06`):

| Cohort Subset | Method | Cases | Mean mTRE | Median mTRE | `< 2 mm` | `< 5 mm` | Mean Time | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All 6 Subjects** | **Hybrid PoU-Triton + DiffDRR polish** | **18** | **1.069 mm** | **0.938 mm** | **100.0%** | **100.0%** | **1.818 s** | 5,490.9 MB |
| All 6 Subjects | PoU-FVR Triton Multi-Start (`B=64`) | 18 | 1.786 mm | 1.907 mm | 50.0% | **100.0%** | 2.110 s | **1,348.0 MB** (`4.07x` less) |
| All 6 Subjects | DiffDRR Single-Start (`f32 B=1`) | 18 | 7.216 mm | 3.130 mm | 50.0% | 61.1% | 3.259 s | 5,490.8 MB |
| **Wide-Angle (`> 15°`)** | **Hybrid PoU-Triton + DiffDRR polish** | **15** | **1.034 mm** | **0.873 mm** | **100.0%** | **100.0%** | **1.811 s** | 5,454.2 MB |
| Wide-Angle (`> 15°`) | PoU-FVR Triton Multi-Start (`B=64`) | 15 | 1.729 mm | 1.800 mm | 60.0% | **100.0%** | 2.106 s | **1,348.0 MB** (`4.05x` less) |
| Wide-Angle (`> 15°`) | DiffDRR Single-Start (`f32 B=1`) | 15 | 5.968 mm | 1.774 mm | 53.3% | 66.7% | 3.261 s | 5,454.0 MB |

#### Bi-Planar `B=64` Multi-Start Registration vs. `nanodrr` (`exp30`, 10 Trials, 50 Steps, Mean Init `15.789 mm`)

With `warmup_pinhole` and dynamic Triton block loops eliminating Trial-1 JIT cold-start recompilation (`2.129 s` on Trial 1):

| Method | Trials | Mean mTRE | Median mTRE | `< 2 mm` | Mean Time | Speedup vs. `nanodrr triton` | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PoU-FVR Triton (`2x4x2@80 bf16`)** | **10** | **0.950 mm** | **0.848 mm** | **100.0%** | **2.161 s** | **`2.79x`** | 1,348.0 MB |
| **Hybrid PoU-Triton + `nanodrr` polish** | **10** | **1.495 mm** | **1.475 mm** | **100.0%** | **1.341 s** | **`4.50x`** | 464.0 MB |
| `nanodrr triton+compile (bf16 n=500)` | 10 | 1.503 mm | 1.503 mm | 100.0% | 6.035 s | `1.00x` | **355.4 MB** |
| `nanodrr torch (f32 n=500)` | 0 | `OOM` | — | — | `OOM` | — | `> 40 GB` (`OOM`) |

---

### 2. Batched Forward & Differentiable `fwd+bwd` Throughput on NVIDIA A100 (`exp29`)

Benchmarked on **NVIDIA A100-SXM4-40GB** (`0.394 s` one-time 3D FFT prep for `PoU-FVR 2x2x2`; zero OOM entries across all 135 `(H, B, method)` cells; `[mb=16]` sequential micro-batching is used where single-launch `nanodrr torch` at `512², B=64` would require `93.75 GiB`):

#### Cone-Beam Forward Pass Comparison

| Grid `(H², B)` | `PoU-FVR Triton+Graph (bf16)` | `PoU-FVR Triton eager (bf16)` | `nanodrr triton+comp (n=200)` | `nanodrr triton+comp (n=500)` | `nanodrr torch (n=500)` | `DiffDRR (f32)` | `PoU-FVR` Speedup vs. `triton (n=500)` / `torch` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `256², B=1` | **0.108 ms** (`9,254 FPS`) | 1.241 ms (`806 FPS`, 5 MB) | 0.570 ms (`1,753 FPS`) | 0.707 ms (`1,415 FPS`) | 2.860 ms (`350 FPS`, 503 MB) | 16.425 ms (`61 FPS`, 2,582 MB) | **`6.5x`** vs. Triton / **`152x`** vs. DiffDRR |
| `256², B=64` | **1.969 ms** (`32,501 FPS`) | 2.703 ms (`23,678 FPS`, 320 MB) | 6.034 ms (`10,606 FPS`) | 14.281 ms (`4,481 FPS`) | 115.527 ms (`554 FPS`, 32,144 MB) | — | **`7.25x`** vs. Triton / **`58.7x`** vs. Torch |
| `512², B=1` | **0.114 ms** (`8,738 FPS`) | 1.247 ms (`802 FPS`, 5 MB) | 0.835 ms (`1,197 FPS`) | 1.352 ms (`740 FPS`) | 7.440 ms (`134 FPS`, 2,010 MB) | — | **`11.8x`** vs. Triton / **`65.0x`** vs. Torch |
| `512², B=16` | **0.576 ms** (`27,773 FPS`) | 1.386 ms (`11,541 FPS`, 80 MB) | 6.028 ms (`2,654 FPS`) | 14.238 ms (`1,124 FPS`) | 112.421 ms (`142 FPS`, 32,145 MB) | — | **`24.7x`** vs. Triton / **`195x`** vs. Torch |
| `512², B=64` | **2.415 ms** (`26,506 FPS`) | 3.163 ms (`20,233 FPS`, 320 MB) | 22.650 ms (`2,826 FPS`) | 55.426 ms (`1,155 FPS`) | 446.134 ms (`143 FPS`, `[mb=16]`, 32,193 MB) | — | **`23.0x`** vs. Triton / **`185x`** vs. Torch |

> **Parallel-Beam Zero-IFFT Loss (`TorchDiffFVR Zero-IFFT NCC`):** Evaluates `256², B=64` in **1.294 ms** (**49,454 FPS**, `96 MB`) and `512², B=64` in **3.470 ms** (**18,445 FPS**, `384 MB`) with zero 2D IFFTs inside the optimization loop.

#### End-to-End Differentiable `fwd+bwd` Training Step (`PoU-FVR Triton 2x2x2` vs. `nanodrr triton n=500`)

| Grid `(H², B)` | `PoU-FVR Triton fwd+bwd` | `nanodrr triton (n=500) fwd+bwd` | Speedup | Working VRAM (`PoU` vs. `nanodrr`) |
| :---: | :---: | :---: | :---: | :---: |
| `256², B=16` | **4.270 ms** (`3,747 FPS`) | 10.318 ms (`1,551 FPS`) | **`2.42x`** | 80.0 MB vs. 22.3 MB |
| `256², B=64` | **7.074 ms** (`9,047 FPS`) | 38.836 ms (`1,648 FPS`) | **`5.49x`** | 320.0 MB vs. 88.3 MB |
| `512², B=16` | **6.137 ms** (`2,607 FPS`) | 38.628 ms (`414 FPS`) | **`6.29x`** | 80.0 MB vs. 89.0 MB |
| `512², B=64` | **21.843 ms** (`2,930 FPS`) | 150.999 ms (`424 FPS`) | **`6.91x`** | 320.0 MB vs. 353.0 MB |

---

### 3. PoU-FVR Block / Resolution Pareto Ablation on A100 (`exp29`, `256²`, `SDD=1020 mm`, `B=1`)

Fused-Triton forward wall latency remains flat at `1.20–1.26 ms` across 11 block-grid/spectral-resolution configurations:

| Blocks | Block Res. | Prep (s) | Wall (ms) | Intensity NCC vs. Raymarch | Edge-NCC vs. Raymarch | Cache VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1×1×1` | `64³` | 0.110 | 1.202 | 0.9213 | 0.4194 | 0.63 |
| `2×2×2` | `64³` | 0.399 | 1.222 | 0.9393 | 0.5231 | 5.00 |
| `2×4×2` | `80³` | 1.391 | 1.250 | 0.9512 | 0.6865 | 15.63 |
| `2×4×2` | `96³` | 1.773 | 1.241 | 0.9551 | **0.7456** | 22.50 |
| `4×4×2` | `80³` | 2.285 | 1.258 | **0.9633** | 0.7319 | 31.25 |

---

### 4. Geometry-Matched CPU & Real Clinical Benchmarks (`exp25`–`exp28`)

#### Strict Geometry-Matched CPU Comparison on DeepFluoro `subject01` (`exp27`)

| Detector & Geometry | Method | Prep (s) | Fwd (ms) | Fwd+Bwd (ms) | 40-Step 2-View Time | Speedup (Fwd / 40-Step) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `128²`, Parallel (`SDD=1e6 mm`) | **Diff-FVR (`1×1×1`)** | 0.323 | **27.31 ms** | **57.62 ms** | **4.93 s** | **`38.2x` / `31.1x`** vs. Siddon |
| `128²`, Parallel (`SDD=1e6 mm`) | DiffDRR Trilinear / Siddon | 0.000 | 466.49 / 1,042.68 ms | 1,182.29 / 1,918.18 ms | 94.58 / 153.45 s | `1.0x` |
| `128²`, Cone-Beam (`SDD=1020 mm`) | **PoU-FVR (`2×2×2`)** | 0.798 | **18.61 ms** | **40.25 ms** | **4.02 s** | **`59.8x` / `38.9x`** vs. Siddon |
| `128²`, Cone-Beam (`SDD=1020 mm`) | DiffDRR Trilinear / Siddon | 0.000 | 625.78 / 1,113.37 ms | 1,582.02 / 1,954.21 ms | 126.56 / 156.34 s | `1.0x` |
| `256²`, Parallel (`SDD=1e6 mm`) | **Diff-FVR (`1×1×1`)** | 0.323 | **28.16 ms** | **59.90 ms** | **5.12 s** | **`128.2x` / `112.9x`** vs. Siddon |
| `256²`, Parallel (`SDD=1e6 mm`) | DiffDRR Trilinear / Siddon | 0.000 | 1,762.42 / 3,609.73 ms | 4,513.23 / 7,225.14 ms | 361.06 / 578.01 s | `1.0x` |
| `256²`, Cone-Beam (`SDD=1020 mm`) | **PoU-FVR (`2×2×2`)** | 0.798 | **32.47 ms** | **90.32 ms** | **8.02 s** | **`125.3x` / `77.3x`** vs. Siddon |
| `256²`, Cone-Beam (`SDD=1020 mm`) | DiffDRR Trilinear / Siddon | 0.000 | 2,045.89 / 4,067.84 ms | 5,020.07 / 7,752.56 ms | 401.61 / 620.21 s | `1.0x` |

#### Real Paired CPU C-Arm & GE DICOM Magnification Cohorts (`exp25`, `exp26`, `exp28`)

| Cohort / Benchmark | Baseline Comparison | Measured Outcome |
| :--- | :--- | :--- |
| **`exp25` (`DeepFluoro subject01`, 10 views, CPU)** | Diff-FVR vs. DiffDRR `0.6.1` (Siddon) | **`106x`** forward speedup (`10.9 ms` vs. `1,153.8 ms`), `NCC = 0.921–0.986` (mean `0.9676`); Hybrid biplanar `5.93 mm` in `62.1 s` vs. DiffDRR `5.95 mm` in `103.3 s` |
| **`exp26` (`DeepFluoro subject01`–`06`, CPU)** | Multi-subject 6-DoF cohort (`N=6`) | **`88.3x`** forward speedup (`14.7 ms` vs. `1,248.1 ms`); Hybrid `3.41 mm` (`123.3 s`) vs. DiffDRR `3.73 mm` (`158.0 s`) |
| **`exp28` (GE DICOM C-Arms, `600/900`–`1000/1400 mm`)** | Uncalibrated Ortho vs. Calibrated vs. PoU-FVR | True-pose NCC `0.9912–0.9969`; PoU-FVR **`0.75–1.32 mm`** in `5.0–5.5 s` (vs. `39.7–41.0 mm` uncalibrated ortho, `5.9–10.2 mm` calibrated ortho) |

Full experiment CSVs: [exp29_gpu_benchmark.csv](https://github.com/pareelamre/diff-fvr/blob/master/outputs/exp29_gpu_benchmark.csv), [exp29_pou_ablation.csv](https://github.com/pareelamre/diff-fvr/blob/master/outputs/exp29_pou_ablation.csv), [exp30_gpu_multistart_reg.csv](https://github.com/pareelamre/diff-fvr/blob/master/outputs/exp30_gpu_multistart_reg.csv), [exp30_gpu_multistart_summary.csv](https://github.com/pareelamre/diff-fvr/blob/master/outputs/exp30_gpu_multistart_summary.csv), [exp31_deepfluoro_gpu_cohort.csv](https://github.com/pareelamre/diff-fvr/blob/master/outputs/exp31_deepfluoro_gpu_cohort.csv), [exp31_deepfluoro_gpu_summary.csv](https://github.com/pareelamre/diff-fvr/blob/master/outputs/exp31_deepfluoro_gpu_summary.csv), and [outputs/paper_figures/](https://github.com/pareelamre/diff-fvr/blob/master/outputs/paper_figures/README.md).

---

## What the Simulation & Unit Scope Covers

The `experiments/` directory contains 31 numbered simulation, real C-arm fluoroscopy, and NVIDIA A100 GPU benchmark studies (plus `9b`).

Experiments 7, 12, and 13 each use one source CT volume and multiple randomized poses with same-renderer Fourier targets and optional synthetic noise; `mTRE` in those early simulation scripts uses eight synthetic volume landmarks rather than anatomical fiducials. Historical CSVs/figures in `outputs/paper_figures/` numbered `<= 23` predate the physical-unit correction; see [result limitations](https://github.com/pareelamre/diff-fvr/blob/master/outputs/paper_figures/README.md). Experiments `25`–`31` evaluate against independent raycasting (`DiffDRR 0.6.1`, `nanodrr`) and paired real `DeepFluoro` pelvic CT + physical C-arm radiographs with anatomical fiducials.

---

## Projection Geometry and Physical Units

- `RobustDiffFVREngine` accepts a **cubic, isotropic** volume and spacing `(Z, Y, X)` in mm. Its poses use `(yaw, pitch, roll, tx, ty, tz)`, with rotations in degrees and translations/bounds in **mm**.
- The CLI preserves CT spacing while resampling to an isotropic cube that fits the longest physical dimension. NPY input contains HU values and defaults to `1 mm` per voxel (`--spacing-mm Z Y X` overrides this); DICOM spacing comes from metadata.
- AP projects along `Y` (`X` horizontal, `Z` vertical); Lateral projects along `X` (`Y` horizontal, `Z` vertical). CLI targets must match the rendered grid (`fvr-register` targets parallel-beam geometry and explicitly rejects uncalibrated `SDD`/`SAD` arguments; perspective cone-beam projection is implemented separately via `PerspectiveSlabFVR` / `PoU-FVR`).
- Low-level NumPy renderers and `gradients` retain their **voxel-coordinate** conventions. `compute_mtre_6dof` returns mm when passed measured `spacing` and `translation_units='mm'` for robust-core poses.

---

## Installation & Quick Start

```bash
# Base installation (CPU SciPy/NumPy engines + CLI)
pip install diff-fvr

# With PyTorch perspective PoU-FVR & Zero-IFFT Spectral-NCC
pip install "diff-fvr[torch]"

# With PyTorch + Triton kernels (Linux NVIDIA GPU)
pip install "diff-fvr[gpu]"
```

Run a synthetic biplanar CLI simulation or test suite from source:

```bash
git clone https://github.com/pareelamre/diff-fvr.git
cd diff-fvr
pip install -e ".[torch]"

# Generate synthetic AP & Lateral DRRs (parallel-beam CLI model)
fvr-render --ct data/dicom --res 96 --view ap --tx 4 --ty 2 --tz 3 --out outputs/example_ap.png
fvr-render --ct data/dicom --res 96 --view lateral --tx 4 --ty 2 --tz 3 --out outputs/example_lat.png

# Recover the 6-DoF pose
fvr-register --ct data/dicom --res 96 --ap outputs/example_ap.png --lat outputs/example_lat.png --out outputs/example_pose.json

# Run full unit & integration test suites
python -m unittest discover -s tests -v
python tests/test_core_engine.py
```

The [browser demonstration](https://pareelamre.github.io/diff-fvr/) is a scripted illustration with synthetic images and an interpolated convergence animation; it does not execute the Python registration engine.

---

## Repository Structure

| Path | Contents |
| :--- | :--- |
| `src/fvr/` | Core CPU Fourier engines, perspective `PoU-FVR`, fused Triton kernels, autograd, DICOM loader, and `fvr-render` / `fvr-register` CLI |
| `tests/` | Unit tests (`test_cone_beam_fvr.py`, `test_physical_units.py`, `test_torch_engine.py`) and end-to-end CT registration tests |
| `experiments/` | Simulation, real paired `DeepFluoro` / GE DICOM C-arm studies (`exp25`–`exp28`), and NVIDIA A100 GPU benchmarks (`exp29`–`exp31`) |
| `outputs/` | Benchmark CSVs (`exp29_*.csv`, `exp30_*.csv`, `exp31_*.csv`) and publication figures (`outputs/paper_figures/`) |
| `docs/` | Illustrative web demonstration |
