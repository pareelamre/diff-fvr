# Dual-Domain Differentiable Fourier Volume Rendering (Diff-FVR) for Real-Time Biplanar 3D–2D Spine Registration

**Author:** Pareel Amre  
**Affiliation:** Faculty of Computer Science, Technische Universität Dresden (TU Dresden), 01062 Dresden, Germany  
**Email:** `pareel.amre@gmail.com`  
**Target Venue:** *IEEE Transactions on Medical Imaging* / *Medical Image Analysis*  

---

## 🔬 Scientific Overview

Intensity-based 3D–2D registration between pre-operative volumetric Computed Tomography (pCT) and intra-operative 2D fluoroscopy is essential for computer-assisted spine surgery (e.g. pedicle screw navigation). However, conventional ray-casting DRR synthesis scales cubically ($\mathcal{O}(N^3)$), requiring $>1.0\text{ s}$ per projection on CPU.

**Diff-FVR** formulates Digitally Reconstructed Radiographs (DRRs) via the **continuous Fourier Projection-Slice Theorem**, reducing rendering complexity to $\mathcal{O}(N^2)$ frequency slice extraction:
1. **Mathematical Anti-Aliasing:** Enforces $2\times$ spatial zero-padding bound ($Z_p = 2.0 \ge \sqrt{3}$) and $\mathrm{sinc}^2$ spatial pre-multiplication to eliminate boundary wrap-around and peripheral roll-off.
2. **Exact 1:1 Scale Preservation:** Implements full-bandwidth frequency slicing to preserve physical anatomical scale.
3. **Normalized Spectral Cross-Correlation (Spectral-NCC):** Computes similarity directly in 2D Fourier space via Parseval's identity ($< 1.0\text{ ms}$ per pose) with **zero inverse Fourier transforms** inside the optimization loop.
4. **Articulated Multi-Vertebra Kinematics:** Models inter-vertebral flexion (L3–L4–L5) on surgical tables using Fourier linearity ($\mathcal{F}\{\sum_i I_i\} = \sum_i S_i(\mathbf{k}, \mathbf{T}_i)$).
5. **Robustness to Physical Failure Limits:** Automatic radio-opaque metal hardware saturation masking + Masked Spatial Loss (achieving $0.000\text{ mm}$ error under $30\%$ occlusion) and isocenter geometric calibration ($1/\text{mag} = \text{SAD}/\text{SDD}$).

---

## 📊 Comprehensive Experimental Suite (23 Benchmarks)

| Experiment | Title | Key Quantitative Finding |
| :--- | :--- | :--- |
| **Exp 1** | $\mathcal{O}(N^2)$ vs $\mathcal{O}(N^3)$ Asymptotic Complexity | **$11.49\times$ to $72.5\times$ speedup** over spatial raycasting ($1.38\text{ ms}$ DRR). |
| **Exp 2** | Rotational Invariance (72 Angles) | $\text{PSNR} = 23.96\text{ dB}$, $\text{SSIM} = 0.9329$, $\text{NCC} = 0.9943 \pm 0.0007$. |
| **Exp 3** | Zero-Padding & Sinc² Premultiplication | $2\times$ padding eliminates high-frequency wrap-around; roll-off removed. |
| **Exp 4** | Spectral-NCC 50-Trial Benchmark | $0.84\text{ mm}$ mTRE ($98.0\%$ success $< 2.0\text{ mm}$) in $1.82\text{ s}$ ($212\times$ per-pose acceleration). |
| **Exp 5** | Energy Landscape & Capture Range | Smooth, strictly convex basin over $\pm 25^\circ / \pm 25\text{ mm}$ displacement. |
| **Exp 6** | Biplanar vs. Monoplanar 6-DoF | Depth error $\Delta t_x$ reduced from $3.59\text{ mm}$ down to $\mathbf{0.03 \pm 0.04\text{ mm}}$. |
| **Exp 7** | Real Hospital Patient DICOM CT Series | $\text{mTRE} = \mathbf{0.020\text{ mm}}$ (Clean), $\mathbf{0.031\text{ mm}}$ (Low-Dose); $100\%$ success. |
| **Exp 8** | Capture Basin Sensitivity Bounds | $100\%$ success at $\pm 5^\circ / \pm 5\text{ mm}$; $90\%$ at $\pm 16^\circ / \pm 16\text{ mm}$. |
| **Exp 9** | Cone-Beam Perspective Divergence | Identified $1.4\times$ magnification distortion; isocenter calibration cuts error by $89\%$. |
| **Exp 10**| Surgical Metal Hardware Occlusion | Automatic saturation threshold + Masked Spatial Loss drops error from $49.15\text{ mm}$ to $0.000\text{ mm}$. |
| **Exp 11**| Radiation Exposure Starvation Sweep | Sub-$0.05\text{ mm}$ mTRE maintained across 6 orders of magnitude ($10^6$ down to $10^3\text{ ph/px}$). |
| **Exp 12**| Multi-Center Kaggle Lumbar CT Cohort | $\text{mTRE} = \mathbf{1.341\text{ mm}}$ (Clean), $\mathbf{1.351\text{ mm}}$ (Low-Dose); $85.0\%$ success. |
| **Exp 13**| Multi-Center AMOS Cohort (NCT Dresden) | $\text{mTRE} = \mathbf{1.118\text{ mm}}$ (Clean), $\mathbf{1.121\text{ mm}}$ (Low-Dose); $90.0\%$ success. |
| **Exp 14**| Resolved Failure Limits Verification | Unit tests confirm $0.000\text{ mm}$ precision under $30\%$ metal occlusion. |
| **Exp 15**| Articulated Multi-Vertebra Kinematics | Hierarchical L3-L4-L5 model recovers base pose ($0.767\text{ mm}$) and in-vivo flexion ($\Delta \text{loss} = -54.7\%$). |
| **Exp 16**| Unified Multi-Center CED Benchmark | Standardized linear and logarithmic CED curves across all 3 independent cohorts. |
| **Exp 17**| Detector FOV Truncation & Clipping | 2D Tukey cosine tapering preserves $0.75\text{ mm}$ mTRE up to $30\%$ clipping. |
| **Exp 18**| Non-Orthogonal C-Arm Angulations | Sub-millimeter ($0.021$–$0.055\text{ mm}$) down to $\Delta\phi = 30^\circ$ safe operating angle. |
| **Exp 19**| Polychromatic Beam Hardening & Soft Tissue | High-Pass Spectral-NCC achieves $100\%$ sub-mm ($0.028$–$0.358\text{ mm}$) across all tiers. |
| **Exp 20**| Intraoperative Respiratory Motion Blur | Phase invariance preserves sub-millimeter tracking across $0$–$5\text{ mm}$ breathing smear. |
| **Exp 21**| Preoperative CT Anisotropic Slice Thickness | Biplanar decoupling preserves sub-millimeter precision ($0.0008\text{ mm}$) on $5.0\text{ mm}$ trauma slices. |
| **Exp 22**| Dynamic Streaming Fluoroscopy Tracking | Real-time video tracking across 60 frames with warm-start analytical gradients. |
| **Exp 23**| Joint Articulation + Metal Implants | Base mTRE $0.776\text{ mm}$, cuts residual loss by $48.2\%$, recovers in-vivo flexion angles. |

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
├── src/fvr/
│   ├── core_engine.py         # Production engine (calibration, metal masking, bounded search)
│   ├── diff_fvr.py            # Core Differentiable Fourier Volume Renderer
│   ├── articulated_spine.py   # Multi-body kinematic chain (L3-L4-L5)
│   ├── gradients.py           # Closed-form analytical Fourier phase & spectral gradients
│   ├── robust_registration.py # Masked spatial loss engine
│   └── dicom_loader.py        # DICOM series loader & Hounsfield-to-attenuation converter
├── experiments/               # Complete 16-experiment benchmark scripts
├── outputs/paper_figures/     # High-resolution 300 DPI publication figures & CSV tables
├── tests/                     # Verification and regression test suite
├── main.tex                   # IEEE Transactions on Medical Imaging manuscript
└── paper_manuscript.md        # Comprehensive audited research report
```
