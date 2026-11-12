"""
Experiment 19: Polychromatic Beam Hardening & Soft-Tissue Overlap Stress Benchmark.

Physical Clinical Reality:
  1. Diagnostic CT is acquired at 120-140 kVp with heavily filtered monochromatic reconstruction (HU).
  2. Mobile intraoperative C-arm fluoroscopy operates at 70-90 kVp with polychromatic bremsstrahlung spectra.
     Preferential photoelectric absorption in dense cortical bone produces severe beam hardening
     (non-linear projection compression: p_poly = p_mono * (1 - beta * p_mono)).
  3. Patient abdominal soft tissue (adipose, viscera, bowel gas) introduces non-uniform low-frequency
     background attenuation gradients that do not exist in bone-windowed CT.
  4. Spatial-domain MSE/SSD methods fail catastrophically under non-linear contrast shifts.

Physical Resolution:
  1. High-Pass Spectral Normalized Cross-Correlation (HP-SNCC):
     Applies a smooth high-pass frequency transfer filter W(k) = 1 - exp(-k^2 / (2 * sigma_k^2))
     that strips low-frequency soft-tissue DC gradients while preserving sub-millimeter trabecular bone phase.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.optimize import minimize
from scipy import fft

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fvr.dicom_loader import resample_isotropic_cube, load_dicom_series, hu_to_attenuation
from src.fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from src.fvr.registration_3d2d import compute_mtre_6dof

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR = Path(r"C:\Users\paree\.gemini\antigravity\brain\db84ca2a-35bb-4b49-a099-1ebdc0eb4333")


def apply_beam_hardening_and_soft_tissue(img: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """
    Simulate polychromatic beam hardening and soft tissue background gradient.
    - beta: beam hardening non-linearity coefficient (p_poly = p * (1 - beta * p / max(p)))
    - gamma: soft tissue background intensity relative to peak bone
    """
    p_max = np.max(img) if np.max(img) > 0 else 1.0
    # Polychromatic cupping / non-linear compression
    p_norm = img / p_max
    p_poly = (p_norm - beta * (p_norm ** 2)) * p_max

    # Soft-tissue background field (smooth 2D parabolic gradient across patient abdomen)
    h, w = img.shape
    y, x = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    bg_field = (0.5 * (1.0 + x) + 0.3 * (y ** 2) + 0.2 * np.sin(np.pi * x))
    bg_field = (bg_field - bg_field.min()) / (bg_field.max() - bg_field.min() + 1e-8)

    p_clin = p_poly + gamma * p_max * 0.4 * bg_field
    return np.maximum(p_clin, 0.0)


def create_high_pass_filter(shape, sigma_k=4.0):
    """
    Create a 2D smooth high-pass Gaussian filter to eliminate soft-tissue DC drift.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    w_hpf = 1.0 - np.exp(-r2 / (2.0 * (sigma_k ** 2)))
    return w_hpf.astype(np.float64)


def run_beam_hardening_benchmark():
    print("=" * 75)
    print("EXPERIMENT 19: POLYCHROMATIC BEAM HARDENING & SOFT TISSUE BENCHMARK")
    print("=" * 75)

    csv_path = OUTPUT_DIR / "table19_beam_hardening_soft_tissue.csv"
    fig_path = OUTPUT_DIR / "fig19_beam_hardening_soft_tissue.png"

    # Check if CSV exists for fast plotting
    if csv_path.exists():
        print(f"Loading existing benchmark CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        # Load and resample clinical DICOM to isotropic 96x96x96
        dicom_dir = PROJECT_ROOT / "data" / "dicom"
        vol_hu, spacing = load_dicom_series(dicom_dir)

        target_xy = 96
        phys_z = vol_hu.shape[0] * spacing[0]
        phys_x = vol_hu.shape[2] * spacing[2]
        vox_size = phys_x / target_xy
        target_z = int(round(phys_z / vox_size))

        vol_cubic, grid_spacing = resample_isotropic_cube(vol_hu, spacing, target_xy)
        vox_size = grid_spacing[0]
        vol_mu = hu_to_attenuation(vol_cubic)

        renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
        print(f"Volume loaded and initialized: {vol_mu.shape}")

        true_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        init_pose = (2.0, -1.5, 1.2, 3.0, -2.0, 2.5)
        init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape, spacing=(vox_size,) * 3)
        print(f"Initial Misalignment: {init_pose}, Initial mTRE: {init_mtre:.2f} mm")

        # Generate clean uncorrupted DRRs
        clean_target_ap = renderer.render_drr(0, 0, 0, tx=0, ty=0, view='ap')
        clean_target_lat = renderer.render_drr(0, 0, 0, tx=0, ty=0, view='lateral')

        hpf_ap = create_high_pass_filter((renderer.pz, renderer.px), sigma_k=5.0)
        hpf_lat = create_high_pass_filter((renderer.pz, renderer.py), sigma_k=5.0)

        # Conditions: (beta, gamma, description)
        conditions = [
            (0.00, 0.0, "Monochromatic CT Ideal"),
            (0.10, 0.2, "Mild Beam Hardening + Thin Patient"),
            (0.20, 0.5, "Moderate Hardening + Normal Abdomen"),
            (0.30, 1.0, "Severe Hardening + Heavy Soft Tissue"),
            (0.40, 1.5, "Extreme Hardening + Obese Adipose Layer"),
            (0.50, 2.0, "Physical Breakdown Limit (Severe Mismatch)")
        ]

        records = []
        th, tw = clean_target_ap.shape
        sh, sw_ap = (renderer.pz - th) // 2, (renderer.px - tw) // 2
        sh, sw_lat = (renderer.pz - th) // 2, (renderer.py - tw) // 2

        for beta, gamma, desc in conditions:
            print(f"\n--- Testing: {desc} (beta={beta:.2f}, gamma={gamma:.2f}) ---")

            # Corrupt targets with polychromatic beam hardening and soft tissue background
            targ_ap_corrupt = apply_beam_hardening_and_soft_tissue(clean_target_ap, beta, gamma)
            targ_lat_corrupt = apply_beam_hardening_and_soft_tissue(clean_target_lat, beta, gamma)

            # Frequency representations
            padded_ap = np.zeros((renderer.pz, renderer.px), dtype=np.float64)
            padded_lat = np.zeros((renderer.pz, renderer.py), dtype=np.float64)

            padded_ap[sh:sh+th, sw_ap:sw_ap+tw] = targ_ap_corrupt
            padded_lat[sh:sh+th, sw_lat:sw_lat+tw] = targ_lat_corrupt

            f_targ_ap = fft.fftshift(fft.fft2(fft.ifftshift(padded_ap)))
            f_targ_lat = fft.fftshift(fft.fft2(fft.ifftshift(padded_lat)))

            # High-pass filtered frequency targets
            f_targ_ap_hpf = f_targ_ap * hpf_ap
            f_targ_lat_hpf = f_targ_lat * hpf_lat

            # 1. Baseline Standard Spectral-NCC
            def obj_baseline(p):
                yaw, pitch, roll, tx, ty, tz = p
                f_ap = renderer.extract_frequency_slice(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
                f_lat = renderer.extract_frequency_slice(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')
                ncc_ap = renderer.compute_spectral_ncc(f_ap, f_targ_ap)
                ncc_lat = renderer.compute_spectral_ncc(f_lat, f_targ_lat)
                return float(0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat))

            t0 = time.time()
            res_base = minimize(obj_baseline, x0=np.array(init_pose), method='Powell', options={'maxiter': 18, 'ftol': 1e-4})
            t_base = time.time() - t0
            mtre_base = compute_mtre_6dof(true_pose, tuple(res_base.x), vol_mu.shape, spacing=(vox_size,) * 3)
            print(f"  Standard Spectral-NCC: mTRE = {mtre_base:.4f} mm, Loss = {res_base.fun:.5f}, Time = {t_base:.2f}s")

            # 2. Proposed High-Pass Spectral-NCC (HP-SNCC)
            def obj_hpf(p):
                yaw, pitch, roll, tx, ty, tz = p
                f_ap = renderer.extract_frequency_slice(yaw, pitch, roll, tx=tx, ty=tz, view='ap') * hpf_ap
                f_lat = renderer.extract_frequency_slice(yaw, pitch, roll, tx=ty, ty=tz, view='lateral') * hpf_lat
                ncc_ap = renderer.compute_spectral_ncc(f_ap, f_targ_ap_hpf)
                ncc_lat = renderer.compute_spectral_ncc(f_lat, f_targ_lat_hpf)
                return float(0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat))

            t0 = time.time()
            res_hpf = minimize(obj_hpf, x0=np.array(init_pose), method='Powell', options={'maxiter': 18, 'ftol': 1e-4})
            t_hpf = time.time() - t0
            mtre_hpf = compute_mtre_6dof(true_pose, tuple(res_hpf.x), vol_mu.shape, spacing=(vox_size,) * 3)
            print(f"  HP-SNCC (High-Pass):  mTRE = {mtre_hpf:.4f} mm, Loss = {res_hpf.fun:.5f}, Time = {t_hpf:.2f}s")

            records.append({
                'Condition': desc,
                'Beta_BeamHardening': beta,
                'Gamma_SoftTissue': gamma,
                'Standard_mTRE_mm': mtre_base,
                'Standard_Loss': float(res_base.fun),
                'HPSNCC_mTRE_mm': mtre_hpf,
                'HPSNCC_Loss': float(res_hpf.fun),
                'Success_2mm_Std': bool(mtre_base < 2.0),
                'Success_2mm_HPSNCC': bool(mtre_hpf < 2.0),
                'Submillimeter_HPSNCC': bool(mtre_hpf < 1.0)
            })

        df = pd.DataFrame(records)
        df['Metric_units'] = 'mm'
        df['Target_type'] = 'synthetic'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved benchmark CSV to: {csv_path}")

    # Plot Figure 19
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    # Panel 1: Visual Comparison of Monochromatic vs Beam Hardened + Soft Tissue
    # Load clean DRR and corrupted example for display
    dicom_dir = PROJECT_ROOT / "data" / "dicom"
    vol_hu, spacing = load_dicom_series(dicom_dir)
    target_xy = 96
    phys_z = vol_hu.shape[0] * spacing[0]
    phys_x = vol_hu.shape[2] * spacing[2]
    vox_size = phys_x / target_xy
    target_z = int(round(phys_z / vox_size))
    vol_cubic, grid_spacing = resample_isotropic_cube(vol_hu, spacing, target_xy)
    vox_size = grid_spacing[0]
    vol_mu = hu_to_attenuation(vol_cubic)
    renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
    clean_lat = renderer.render_drr(0, 0, 0, tx=0, ty=0, view='lateral')
    corrupt_lat = apply_beam_hardening_and_soft_tissue(clean_lat, beta=0.30, gamma=1.0)

    panel1_img = np.concatenate([clean_lat, np.ones((clean_lat.shape[0], 4)) * np.max(clean_lat), corrupt_lat], axis=1)
    axes[0].imshow(panel1_img, cmap='gray')
    axes[0].set_title('Left: Ideal Monochromatic CT DRR\nRight: Polychromatic Beam Hardened + Soft Tissue', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: mTRE vs Beam Hardening & Soft Tissue Severity
    x_indices = np.arange(len(df))
    width = 0.35
    axes[1].bar(x_indices - width/2, df['Standard_mTRE_mm'], width, label='Standard Spectral-NCC', color='#e74c3c', alpha=0.85)
    axes[1].bar(x_indices + width/2, df['HPSNCC_mTRE_mm'], width, label='High-Pass Spectral-NCC (HP-SNCC)', color='#2ecc71', alpha=0.85)

    axes[1].axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Sub-millimeter Threshold (1.0 mm)')
    axes[1].axhline(y=2.0, color='gray', linestyle=':', linewidth=1.2, label='Clinical Success (2.0 mm)')
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels([f"S{i+1}\n(b={b:.1f}, g={g:.1f})" for i, (b, g) in enumerate(zip(df['Beta_BeamHardening'], df['Gamma_SoftTissue']))], fontsize=9)
    axes[1].set_xlabel('Stress Condition (Beam Hardening Beta, Soft Tissue Gamma)', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Mean Target Registration Error (mTRE, mm)', fontsize=10, fontweight='bold')
    axes[1].set_title('Registration Robustness to Non-Linear Contrast Mismatch', fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper left', frameon=True, fontsize=8.5)
    axes[1].set_yscale('log')

    # Panel 3: Convergence Loss Landscape Comparison
    axes[2].plot(x_indices, df['Standard_Loss'], 'o--', color='#e74c3c', linewidth=2, markersize=7, label='Standard Loss (Residual Bias)')
    axes[2].plot(x_indices, df['HPSNCC_Loss'], 's-', color='#2ecc71', linewidth=2.5, markersize=7, label='HP-SNCC Loss (DC-Free Basin)')
    axes[2].set_xticks(x_indices)
    axes[2].set_xticklabels([f"S{i+1}" for i in range(len(df))], fontsize=9)
    axes[2].set_xlabel('Stress Severity Level (S1 -> S6)', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('Optimized Registration Loss (1 - NCC)', fontsize=10, fontweight='bold')
    axes[2].set_title('Fourier Domain Loss Degradation & Invariance', fontsize=11, fontweight='bold')
    axes[2].legend(loc='upper left', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")

    # Copy to brain artifacts
    if BRAIN_DIR.exists():
        import shutil
        brain_fig = BRAIN_DIR / "fig19_beam_hardening_soft_tissue.png"
        shutil.copy2(fig_path, brain_fig)
        print(f"Copied to brain artifacts: {brain_fig}")


if __name__ == '__main__':
    run_beam_hardening_benchmark()
