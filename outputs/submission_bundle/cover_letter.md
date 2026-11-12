# Cover Letter for Manuscript Submission

**To:** Editor-in-Chief, IEEE Transactions on Medical Imaging (TMI)  
**From:** Pareel Amre  
Faculty of Computer Science, Technische Universität Dresden (TU Dresden)  
01062 Dresden, Germany  
**Email:** pareel.amre@gmail.com  
**Date:** November 12, 2026  

**Title:** Dual-Domain Differentiable Fourier Volume Rendering for Ultra-Fast Biplanar 3D–2D Spine Registration  
**Authors:** Pareel Amre  

Dear Editor-in-Chief,

I am pleased to submit our original research article titled **"Dual-Domain Differentiable Fourier Volume Rendering for Ultra-Fast Biplanar 3D–2D Spine Registration"** for consideration for publication in *IEEE Transactions on Medical Imaging*.

### Summary and Key Contributions:
Intensity-based 3D–2D registration between preoperative 3D CT and intraoperative 2D X-ray fluoroscopy is the clinical backbone of computer-assisted orthopedic and spine surgery. However, spatial ray-casting DRR synthesis scales cubically at $\mathcal{O}(N^3)$, imposing a severe latency barrier (15–45 seconds per 6-DoF registration).

In this manuscript, we present **Dual-Domain Differentiable Fourier Volume Rendering (Diff-FVR)**, an exact $\mathcal{O}(N^2)$ framework based on the 3D Fourier Projection-Slice Theorem. We introduce:
1. **Normalized Spectral Cross-Correlation (Spectral-NCC):** Directly computes image similarity in the 2D Fourier domain in $< 1\text{ ms}$, eliminating inverse FFTs from the optimization loop.
2. **Closed-Form Analytical Fourier Gradients:** Exact spatial derivatives $\frac{\partial S}{\partial \mathbf{t}} = -2\pi i \mathbf{k} S(\mathbf{k})$ and spectral rotational gradients that accelerate 6-DoF optimization without discrete GPU dependencies.
3. **Multi-Center Clinical Generalizability:** Validated across 3 independent scanner cohorts (Hospital DICOM, AMOS Abdominal CT, Kaggle Lumbar CT), achieving sub-millimeter precision ($0.020$ to $1.118\text{ mm}$ mTRE) and $85.0\%$–$100.0\%$ clinical success ($< 2.0\text{ mm}$).
4. **Exhaustive Resolution of 8 Intraoperative Physical Boundaries:** Stress-tested across:
   - Surgical metal hardware occlusion ($0.0000\text{ mm}$ with saturation masking)
   - Cone-beam perspective magnification ($< 0.5\text{ mm}$ via isocenter calibration)
   - Quantum dose starvation (invariant down to $\Phi = 10^3\text{ photons/px}$)
   - In-vivo multi-vertebra articulation ($54.7\%$ residual reduction)
   - Detector FOV truncation ($0.75\text{ mm}$ with Tukey cosine tapering)
   - Oblique C-arm angulations (stable down to $\Delta\phi = 30^\circ$)
   - Polychromatic beam hardening & soft tissue ($100\%$ sub-mm via High-Pass Spectral-NCC)
   - Physiological respiratory motion blur ($0.001$ to $0.255\text{ mm}$ across $0$–$5\text{ mm}$ smear)
   - Anisotropic thick CT slices ($0.0008\text{ mm}$ on $5.0\text{ mm}$ trauma scans)

This manuscript represents original work that has not been published previously and is not under consideration elsewhere. All co-authors have approved the submission.

Thank you for your time and consideration of our work.

Sincerely,  
**Pareel Amre**  
M.Sc. in Computational Modeling and Simulation (CMS)  
Faculty of Computer Science, Technische Universität Dresden, Germany  
Email: pareel.amre@gmail.com  
