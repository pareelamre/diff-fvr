# Diff-FVR

**Differentiable Fourier volume rendering for rigid CT-to-X-ray registration.**

[![PyPI](https://img.shields.io/pypi/v/diff-fvr.svg)](https://pypi.org/project/diff-fvr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Diff-FVR reuses precomputed volume spectra to evaluate projections and optimize rigid poses. It combines direct spectral matching for parallel projections with **PoU-FVR**, a partition-of-unity approximation for perspective projection. Fourier candidate screening can precede spatial refinement with DiffDRR or nanodrr.

> **Research software.** Clinical accuracy and suitability for surgical navigation have not been established. The evidence below includes cadaveric development experiments and controlled simulations.

## Method

| Component | Geometry | Role |
| --- | --- | --- |
| **Diff-FVR-Spectral** | Parallel | Compare phase-sensitive Fourier slices with target spectra without reconstructing each candidate image. |
| **PoU-FVR** | Approximate perspective | Combine projections from overlapping weighted volume blocks on a pinhole detector. |
| **PoU-FVR + spatial refinement** | Perspective | Screen candidate poses with PoU-FVR, then refine selected candidates with a spatial projector. |

The workflow is **CT preparation → spectral representation → candidate projection and matching → pose optimization → aligned projections**. Hybrid results describe the complete screening-and-refinement pipeline.

See the [method identifiers](https://github.com/pareelamre/diff-fvr/blob/6968f16/docs/METHOD_NAMING.md) and [parallel geometry specification](https://github.com/pareelamre/diff-fvr/blob/6968f16/docs/PARALLEL_GEOMETRY.md) for definitions and conventions.

## Installation

Requires **Python 3.10 or later**. The CPU implementation uses NumPy and SciPy. PyTorch enables tensor implementations; Triton kernels require a supported Linux NVIDIA GPU environment.

For the geometry corrections and evaluation documented here, install the pinned research snapshot:

```bash
git clone https://github.com/pareelamre/diff-fvr.git
cd diff-fvr
git checkout 6968f16
python -m pip install -e .
```

Install the backend you need:

```bash
python -m pip install -e ".[torch]"   # PyTorch
python -m pip install -e ".[gpu]"     # PyTorch and Triton on Linux
```

Packaged releases are available on [PyPI](https://pypi.org/project/diff-fvr/). The pinned source snapshot above identifies the implementation and evidence used by this documentation; a packaged release can contain an earlier implementation.

## Quick start

The command-line tools use **parallel-beam geometry**. Supply a regular CT DICOM series directory in place of `path/to/ct-dicom`.

```bash
# Generate two synthetic targets at a shared pose.
fvr-render --ct path/to/ct-dicom --res 96 --view ap --tx 4 --ty 2 --tz 3 --out outputs/example_ap.png
fvr-render --ct path/to/ct-dicom --res 96 --view lateral --tx 4 --ty 2 --tz 3 --out outputs/example_lat.png

# Estimate the rigid pose from those targets.
fvr-register --ct path/to/ct-dicom --res 96 --ap outputs/example_ap.png --lat outputs/example_lat.png --out outputs/example_pose.json
```

CT arrays in `.npy` format are also supported. Set `--spacing-mm Z Y X` to the actual voxel spacing; NumPy input otherwise defaults to 1 mm per voxel. Rotations are in degrees and CLI translations are in millimetres.

DICOM input must be a single regular series with valid geometry and supported pixel encoding. The CLI does not register acquired cone-beam images directly. Perspective experiments use the separate PoU-FVR implementation and calibrated experiment protocols.

For the source test suite:

```bash
python -m unittest discover -s tests -v
```

## Evaluation

The complete acquired-image comparison contains **six methods, six cadaveric pelvic specimens and four shared initializations per specimen**: 144 registrations across 36 isolated measurement jobs. All six anatomies were used during development; repeated starts are not independent subjects.

| Method | Median mTRE (mm) | Trials below 2 mm |
| --- | ---: | ---: |
| DiffDRR, single start | 5.053 | 8/24 |
| DiffDRR, 64 candidates | 0.866 | 23/24 |
| nanodrr, 64 candidates | 1.095 | 22/24 |
| PoU-FVR | 1.450 | 20/24 |
| PoU-FVR + nanodrr refinement | 0.969 | 24/24 |
| PoU-FVR + DiffDRR refinement | 0.856 | 24/24 |

The PoU-FVR–nanodrr hybrid achieved **3.6–4.1× lower warm registration latency** among jointly successful pairs and **2.2–2.4× lower complete measurement-job time** than the 64-candidate nanodrr search. These descriptive comparisons use separate physical L4 cohorts containing three and two matched specimens. Complete job time includes startup, preparation, warm-up and four timed registrations; it is not single-acquisition latency.

There was **no general memory reduction**: the hybrid's maximum sampled GPU footprint was 4.77 GiB versus 4.66 GiB for nanodrr. Standalone PoU-FVR used more host memory and had fewer successful registrations.

A separate numerical study used six previously unused VerSe spine CTs and 24 registrations to independently integrated, noiseless parallel targets. Median mTRE was 0.086 mm. This tests unseen anatomy with matched CT attenuation and modest pose perturbations; it does not establish acquired-image generalization or clinical performance.

Research records and reproduction details:

- [Derived measurements, configurations and provenance](https://github.com/pareelamre/diff-fvr/tree/6968f16/outputs/journal_j34)
- [Device-stratified analysis and validation](https://github.com/pareelamre/diff-fvr/blob/6968f16/docs/JOURNAL_SESSION_ANALYSIS.md)
- [Measurement protocol and reproduction commands](https://github.com/pareelamre/diff-fvr/blob/6968f16/docs/JOURNAL_REPLICATION.md)
- [Historical development tables](benchmarks/historical-results.md)

## Repository

| Directory | Contents |
| --- | --- |
| `src/fvr/` | Fourier renderers, registration engines, physical-coordinate handling and command-line tools. |
| `experiments/` | Simulation and acquired-image evaluation scripts. |
| `tests/` | Geometry, gradient, registration and resource-accounting checks; available tests depend on the checked-out revision. |
| `outputs/` | Derived measurements and figures. |
| `benchmarks/` | Archived development reports with their original limitations. |
| `docs/` | Method documentation and browser demonstrations. |

The [browser demonstration](https://pareelamre.github.io/diff-fvr/) uses synthetic images and a scripted convergence animation. It does not execute the Python registration engine.

## Citation

If you use Diff-FVR, cite the software and identify the source revision used in your experiments. Cite the original datasets and comparison methods separately.

```bibtex
@software{amre2026difffvr,
  author  = {Amre, Pareel},
  title   = {Diff-FVR: Differentiable Fourier Volume Rendering for 3D-2D Registration},
  year    = {2026},
  version = {1.2.0},
  url     = {https://github.com/pareelamre/diff-fvr}
}
```

Author: **Pareel Amre** · TU Dresden (alumnus) · [ORCID](https://orcid.org/0000-0002-9657-8725)

## License and data

The software is distributed under the [MIT license](LICENSE). Original medical datasets retain their own terms: [DeepFluoro](https://github.com/rg2/DeepFluoroLabeling-IPCAI2020) uses CC BY-NC 4.0 and [VerSe](https://github.com/anjany/verse) uses CC BY-SA 4.0. The software license does not replace dataset or third-party image permissions.
