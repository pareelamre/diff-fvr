# Dual-Domain Differentiable Fourier Volume Rendering (Diff-FVR) for Real-Time Biplanar 3D–2D Spine Registration

[![PyPI version](https://img.shields.io/pypi/v/diff-fvr.svg)](https://pypi.org/project/diff-fvr/)
[![CI](https://github.com/pareelamre/diff-fvr/actions/workflows/ci.yml/badge.svg)](https://github.com/pareelamre/diff-fvr/actions/workflows/ci.yml)
[![Pages](https://github.com/pareelamre/diff-fvr/actions/workflows/pages.yml/badge.svg)](https://pareelamre.github.io/diff-fvr/)
[![Live Cockpit](https://img.shields.io/badge/Live%20Cockpit-GitHub%20Pages-brightgreen)](https://pareelamre.github.io/diff-fvr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**Author:** Pareel Amre  
**Affiliation:** Faculty of Computer Science, Technische Universität Dresden (TU Dresden), 01062 Dresden, Germany  
**Email:** `pareel.amre@gmail.com`  

---

## 🔬 Scientific Overview

Intensity-based 3D–2D registration between pre-operative volumetric Computed Tomography (pCT) and intra-operative 2D fluoroscopy is essential for computer-assisted spine surgery (e.g. pedicle screw navigation). However, conventional ray-casting DRR synthesis scales cubically ($\mathcal{O}(N^3)$), requiring $> 1.0\text{ s}$ per projection on CPU.

**Diff-FVR** formulates Digitally Reconstructed Radiographs (DRRs) via the **continuous Fourier Projection-Slice Theorem**, reducing rendering complexity to $\mathcal{O}(N^2)$ frequency slice extraction:

$$
\mathcal{F}_{2D}\{\mathcal{P}[f(x,y,z)]\} = \mathcal{S}_{2D}\{\mathcal{F}_{3D}[f(x,y,z)]\}
$$

1. **Mathematical Anti-Aliasing:** Enforces $2\times$ spatial zero-padding bound ($Z_p = 2.0 \ge \sqrt{3}$) and $\text{sinc}^2$ spatial pre-multiplication to eliminate boundary wrap-around and peripheral roll-off.
2. **Exact 1:1 Scale Preservation:** Implements full-bandwidth frequency slicing to preserve physical anatomical scale.
3. **Normalized Spectral Cross-Correlation (Spectral-NCC):** Computes similarity directly in 2D Fourier space via Parseval's identity ($< 1.0\text{ ms}$ per pose) with **zero inverse Fourier transforms** inside the optimization loop.
4. **Articulated Multi-Vertebra Kinematics:** Models inter-vertebral flexion (L3–L4–L5) on surgical tables using Fourier linearity:
$$
\mathcal{F}\left\{\sum_i I_i\right\} = \sum_i S_i(\mathbf{k}, \mathbf{T}_i)
$$
5. **Robustness to Physical Failure Limits:** Automatic radio-opaque metal hardware saturation masking + Masked Spatial Loss (achieving $0.000\text{ mm}$ error under $30\%$ occlusion) and isocenter geometric calibration ($1/\text{mag} = \text{SAD}/\text{SDD}$).

---

## 📊 Comprehensive Experimental Suite (23 Benchmarks)

| Experiment | Title | Key Quantitative Finding |
| :--- | :--- | :--- |
| **Exp 1** | O(N²) vs O(N³) Asymptotic Complexity | **11.5× to 72.5× speedup** over spatial raycasting (1.38 ms DRR). |
| **Exp 2** | Rotational Invariance (72 Angles) | PSNR = 23.96 dB, SSIM = 0.9329, NCC = 0.9943 ± 0.0007. |
| **Exp 3** | Zero-Padding & Sinc² Premultiplication | 2× padding eliminates high-frequency wrap-around; roll-off removed. |
| **Exp 4** | Spectral-NCC 50-Trial Benchmark | 0.84 mm mTRE (98.0% success < 2.0 mm) in 1.82 s (212× per-pose acceleration). |
| **Exp 5** | Energy Landscape & Capture Range | Smooth, strictly convex basin over ±25° / ±25 mm displacement. |
| **Exp 6** | Biplanar vs. Monoplanar 6-DoF | Depth error Δtx reduced from 3.59 mm down to **0.03 ± 0.04 mm**. |
| **Exp 7** | Real Hospital Patient DICOM CT Series | **mTRE = 0.020 mm** (Clean), **0.031 mm** (Low-Dose); 100% success. |
| **Exp 8** | Capture Basin Sensitivity Bounds | 100% success at ±5° / ±5 mm; 90% at ±16° / ±16 mm. |
| **Exp 9** | Cone-Beam Perspective Divergence | Identified 1.4× magnification distortion; isocenter calibration cuts error by 89%. |
| **Exp 10** | Surgical Metal Hardware Occlusion | Automatic saturation threshold + Masked Spatial Loss drops error from 49.15 mm to 0.000 mm. |
| **Exp 11** | Radiation Exposure Starvation Sweep | Sub-0.05 mm mTRE maintained across 6 orders of magnitude (10⁶ down to 10³ ph/px). |
| **Exp 12** | Multi-Center Kaggle Lumbar CT Cohort | **mTRE = 1.341 mm** (Clean), **1.351 mm** (Low-Dose); 85.0% success. |
| **Exp 13** | Multi-Center AMOS Cohort (NCT Dresden) | **mTRE = 1.118 mm** (Clean), **1.121 mm** (Low-Dose); 90.0% success. |
| **Exp 14** | Resolved Failure Limits Verification | Unit tests confirm 0.000 mm precision under 30% metal occlusion. |
| **Exp 15** | Articulated Multi-Vertebra Kinematics | Hierarchical L3-L4-L5 model recovers base pose (0.767 mm) and in-vivo flexion (Δloss = −54.7%). |
| **Exp 16** | Unified Multi-Center CED Benchmark | Standardized linear and logarithmic CED curves across all 3 independent cohorts. |
| **Exp 17** | Detector FOV Truncation & Clipping | 2D Tukey cosine tapering preserves 0.75 mm mTRE up to 30% clipping. |
| **Exp 18** | Non-Orthogonal C-Arm Angulations | Sub-millimeter (0.021–0.055 mm) down to Δϕ = 30° safe operating angle. |
| **Exp 19** | Polychromatic Beam Hardening & Soft Tissue | High-Pass Spectral-NCC achieves 100% sub-mm (0.028–0.358 mm) across all tiers. |
| **Exp 20** | Intraoperative Respiratory Motion Blur | Phase invariance preserves sub-millimeter tracking across 0–5 mm breathing smear. |
| **Exp 21** | Preoperative CT Anisotropic Slice Thickness | Biplanar decoupling preserves sub-millimeter precision (0.0008 mm) on 5.0 mm trauma slices. |
| **Exp 22** | Dynamic Streaming Fluoroscopy Tracking | Real-time video tracking across 60 frames with warm-start analytical gradients. |
| **Exp 23** | Joint Articulation + Metal Implants | Base mTRE 0.776 mm, cuts residual loss by 48.2%, recovers in-vivo flexion angles. |

---

## 🎮 Interactive Surgical Registration Cockpit

An interactive in-browser simulator demonstrating real-time 6-DoF biplanar registration, pedicle screw metal saturation, and physical breakdown regimes is live:

👉 **[Launch Interactive Cockpit](https://pareelamre.github.io/diff-fvr/)**

---

## 🚀 Installation & CLI Quick Start

```bash
# Install package in editable mode
pip install -e .

# 1. Synthesize a fast 2D DRR from 3D CT:
fvr-render --ct data/dicom --yaw 15.0 --pitch -5.0 --view ap --res 96 --out drr_ap.png

# 2. Run biplanar 6-DoF spine registration:
fvr-register --ct data/dicom --ap data/ap_xray.png --lat data/lat_xray.png --handle-metal --out pose_result.json

# 3. Run production integration test suite:
python tests/test_core_engine.py
```

---

## 📁 Repository Structure

```
fft-drr-project/
├── data/
│   ├── dicom/                 # Real clinical hospital patient CT DICOM series
│   ├── kaggle_lumbar/         # Multi-center Kaggle lumbar spine CT
│   └── amos/                  # AMOS multi-organ abdominal CT cohort (NCT Dresden)
├── docs/                      # GitHub Pages interactive surgical cockpit simulator
├── src/fvr/
│   ├── core_engine.py         # Production engine (calibration, metal masking, bounded search)
│   ├── diff_fvr.py            # Core Differentiable Fourier Volume Renderer
│   ├── articulated_spine.py   # Multi-body kinematic chain (L3-L4-L5)
│   ├── gradients.py           # Closed-form analytical Fourier phase & spectral gradients
│   ├── robust_registration.py # Masked spatial loss engine
│   └── dicom_loader.py        # DICOM series loader & Hounsfield-to-attenuation converter
├── experiments/               # Complete 23 benchmark experiment scripts
├── outputs/paper_figures/     # High-resolution 300 DPI publication figures & CSV tables
├── tests/                     # Verification and regression test suite
├── main.tex                   # Mathematical formulation and documentation
└── paper_manuscript.md        # Technical research report
```
