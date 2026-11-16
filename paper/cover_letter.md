# Cover Letter for Manuscript Submission

**To:** Editor-in-Chief, IEEE Transactions on Medical Imaging (TMI)  
**From:** Pareel Amre  
Faculty of Computer Science, Technische Universität Dresden (TU Dresden)  
01062 Dresden, Germany  
**Email:** pareel.amre@gmail.com  
**Date:** November 14, 2026  

**Title:** Dual-Domain Differentiable Fourier Volume Rendering for Fast Biplanar 3D--2D Spine and Pelvis Registration  
**Author:** Pareel Amre  

Dear Editor-in-Chief,

I am pleased to submit our original research article titled **"Dual-Domain Differentiable Fourier Volume Rendering for Fast Biplanar 3D--2D Spine and Pelvis Registration"** for consideration for publication in *IEEE Transactions on Medical Imaging*.

### Summary and Key Contributions
Iterative 3D--2D alignment between preoperative CT and intraoperative fluoroscopy remains limited by the $\mathcal{O}(N^3)$ complexity of spatial Digitally Reconstructed Radiograph (DRR) raycasting. In this work, we introduce **Dual-Domain Differentiable Fourier Volume Rendering (Diff-FVR)** and cone-beam **Partition-of-Unity Perspective Slab FVR (PoU-FVR)**:

1. **Exact Closed-Form Calibration & $\mathcal{O}(N^2)$ Parseval Scoring:** Solving wrap-around aliasing ($Z_p=2.0$), trilinear roll-off ($\mathrm{sinc}^2$ pre-compensation), and $1:1$ physical slice bandwidth yields `0.9943` rotational NCC against spatial line integration, while Parseval Spectral-NCC eliminates inner-loop 2D IFFTs (`38.2x`--`125.3x` matched CPU forward speedup over DiffDRR~0.6.1 Siddon, up to `77.3x` amortized 40-step speedup including FFT preparation).
2. **Seam-Free Cone-Beam PoU-FVR with Fused Triton + `CUDAGraph`:** Raised-cosine slab partitioning ($\sum_b w_b \equiv 1$) with shared optical slicing and exact pinhole ray-plane intersection reaches `32,501 FPS` at $256^2$ (`B=64`, `bf16`) and `26,506 FPS` at $512^2$ (`B=64`) on an NVIDIA~A100 (`23.0x` faster than compiled Triton `nanodrr` at `n=500`, `185x` faster than PyTorch `nanodrr`, and `6.91x` full `fwd+bwd` training speedup inside a `320 MB` static workspace).
3. **Matched Wide-Basin Real Clinical DeepFluoro & Multi-Center CT Validation (`exp32`--`exp33`):**
   - On all 6 real DeepFluoro clinical pelvic subjects (`N=24` including unanchored Tier~2 $\pm 16^\circ/\pm 22\text{ mm}$ wide-basin stress), Hybrid PoU-Triton + DiffDRR achieves `1.016 mm` median mTRE (`95.8% < 2 mm`) in `1.789 s` online time, outperforming single-start DiffDRR (`9.820 mm`, $p = 1.50 \times 10^{-4}$) and matched $B{=}64$ Triton `nanodrr` (`1.538 mm`, `5.302 s` reg, $p = 2.28 \times 10^{-4}$).
   - Resolving the remaining `exp32` soft spots (`exp33`), isotropic `4x4x4@64` slabs achieve `100.0% < 2 mm` standalone convergence across all four clinical Spine and Abdominal CT cohorts (`0.307`--`1.406 mm` median on Kaggle Lumbar $256^2$/$512^2$, AMOS~0001, and GE Clinical CT) and `91.7% < 2 mm` (`1.423 mm`) on DeepFluoro, while per-axis $\{-0.5,0,+0.5\}^3$ isocenter parity calibration paired with `K=4` multi-survivor `f32` `nanodrr` polish reaches **`0.835 mm` median mTRE (`100.0% < 2 mm`)** on DeepFluoro in **`1.368 s` online runtime** and **`464.1 MB` peak VRAM (`<0.5 GB`)**.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. Open-source code, CLI tools, and reproducible benchmark scripts are available at https://github.com/pareelamre/diff-fvr (MIT License).

Thank you for your time and consideration.

Sincerely,  
**Pareel Amre**  
Faculty of Computer Science, Technische Universität Dresden (TU Dresden)  
01062 Dresden, Germany  
Email: pareel.amre@gmail.com  
