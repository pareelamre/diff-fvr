"""
Experiment 20: Intraoperative Motion Blur Stress Benchmark (Breathing & Table Micro-Motion).

Physical Clinical Reality:
  1. Surgical C-arm exposure times range from 50 to 150 ms.
  2. Patient respiratory motion (diaphragmatic excursion) causes periodic cranial-caudal
     displacement of lumbar/thoracic vertebrae by 0.5 to 2.5 mm.
  3. Operating table compliance and surgical manipulation introduce micro-vibrations (0.2 to 1.0 mm).
  4. Spatial edge matching / gradient correlation degrades severely as fine cortical bone edges
     are smeared out across multiple millimeters.

Physical Mathematical Invariance:
  In the frequency domain, linear motion blur corresponds to multiplication by a real-valued sinc function:
    H(u, v) = sinc(pi * (u * cos(theta) + v * sin(theta)) * L_pix)
  Because H(u, v) is real and symmetric, the Fourier phase angle angle(F(u, v)) is perfectly preserved
  within the central lobe, enabling Spectral-NCC to maintain sub-millimeter precision despite visible spatial blurring.
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


def apply_linear_motion_blur_fourier(img: np.ndarray, blur_mm: float, pixel_size_mm: float, angle_deg: float = 90.0) -> np.ndarray:
    """
    Apply physical linear motion blur via exact frequency-domain sinc modulation.
    - blur_mm: extent of motion in millimeters
    - pixel_size_mm: physical pixel pitch in mm
    - angle_deg: motion angle (90 deg = cranial-caudal along vertical axis)
    """
    if blur_mm <= 1e-4:
        return np.copy(img)

    l_pix = blur_mm / pixel_size_mm
    h, w = img.shape
    theta = np.deg2rad(angle_deg)

    # Frequency coordinates normalized to [-0.5, 0.5]
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    FX, FY = np.meshgrid(fx, fy)

    # Projection along motion vector
    proj_freq = FX * np.cos(theta) + FY * np.sin(theta)

    # Sinc transfer function H(u, v) = sinc(pi * proj * L)
    arg = np.pi * proj_freq * l_pix
    # np.sinc in numpy is sinc(x) = sin(pi * x) / (pi * x), so pass proj_freq * l_pix
    H = np.sinc(proj_freq * l_pix)

    # Apply in Fourier domain
    F_img = np.fft.fft2(img)
    F_blurred = F_img * H
    img_blurred = np.real(np.fft.ifft2(F_blurred))
    return np.maximum(img_blurred, 0.0)


def run_motion_blur_benchmark():
    print("=" * 75)
    print("EXPERIMENT 20: INTRAOPERATIVE MOTION BLUR STRESS BENCHMARK")
    print("=" * 75)

    csv_path = OUTPUT_DIR / "table20_motion_blur_stress.csv"
    fig_path = OUTPUT_DIR / "fig20_motion_blur_stress.png"

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
        pixel_size_mm = vox_size
        print(f"Volume loaded: {vol_mu.shape}, Pixel pitch: {pixel_size_mm:.3f} mm")

        true_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        init_pose = (2.0, -1.5, 1.2, 3.0, -2.0, 2.5)
        init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape, spacing=(vox_size,) * 3)
        print(f"Initial Misalignment: {init_pose}, Initial mTRE: {init_mtre:.2f} mm")

        clean_target_ap = renderer.render_drr(0, 0, 0, tx=0, ty=0, view='ap')
        clean_target_lat = renderer.render_drr(0, 0, 0, tx=0, ty=0, view='lateral')

        blur_levels_mm = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        records = []
        th, tw = clean_target_ap.shape
        sh, sw_ap = (renderer.pz - th) // 2, (renderer.px - tw) // 2
        sh, sw_lat = (renderer.pz - th) // 2, (renderer.py - tw) // 2

        for blur_mm in blur_levels_mm:
            print(f"\n--- Testing Motion Blur: {blur_mm:.1f} mm (Cranial-Caudal Excursion) ---")

            # Apply motion blur along cranial-caudal axis (90 deg)
            blur_ap = apply_linear_motion_blur_fourier(clean_target_ap, blur_mm, pixel_size_mm, angle_deg=90.0)
            blur_lat = apply_linear_motion_blur_fourier(clean_target_lat, blur_mm, pixel_size_mm, angle_deg=90.0)

            # Frequency targets
            padded_ap = np.zeros((renderer.pz, renderer.px), dtype=np.float64)
            padded_lat = np.zeros((renderer.pz, renderer.py), dtype=np.float64)
            padded_ap[sh:sh+th, sw_ap:sw_ap+tw] = blur_ap
            padded_lat[sh:sh+th, sw_lat:sw_lat+tw] = blur_lat

            f_targ_ap = fft.fftshift(fft.fft2(fft.ifftshift(padded_ap)))
            f_targ_lat = fft.fftshift(fft.fft2(fft.ifftshift(padded_lat)))

            def obj_sncc(p):
                yaw, pitch, roll, tx, ty, tz = p
                f_ap = renderer.extract_frequency_slice(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
                f_lat = renderer.extract_frequency_slice(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')
                ncc_ap = renderer.compute_spectral_ncc(f_ap, f_targ_ap)
                ncc_lat = renderer.compute_spectral_ncc(f_lat, f_targ_lat)
                return float(0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat))

            t0 = time.time()
            res = minimize(obj_sncc, x0=np.array(init_pose), method='Powell', options={'maxiter': 20, 'ftol': 1e-4})
            t_run = time.time() - t0
            mtre = compute_mtre_6dof(true_pose, tuple(res.x), vol_mu.shape, spacing=(vox_size,) * 3)

            # Extract translation errors
            tx_err = abs(res.x[3] - true_pose[3])
            ty_err = abs(res.x[4] - true_pose[4])
            tz_err = abs(res.x[5] - true_pose[5])

            print(f"  Result: mTRE = {mtre:.4f} mm, Tz_err (axial) = {tz_err:.4f} mm, Loss = {res.fun:.6f}, Time = {t_run:.2f}s")

            records.append({
                'Motion_Blur_mm': blur_mm,
                'Blur_Pixel_Equivalent': blur_mm / pixel_size_mm,
                'Final_mTRE_mm': mtre,
                'Axial_Error_Tz_mm': tz_err,
                'Depth_Error_Tx_mm': tx_err,
                'InPlane_Error_Ty_mm': ty_err,
                'Final_Loss': float(res.fun),
                'Time_sec': t_run,
                'Success_2mm': bool(mtre < 2.0),
                'Submillimeter_1mm': bool(mtre < 1.0)
            })

        df = pd.DataFrame(records)
        df['Metric_units'] = 'mm'
        df['Target_type'] = 'synthetic'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved benchmark CSV to: {csv_path}")

    # Plot Figure 20
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    # Panel 1: Visual comparison of 0 mm vs 2.0 mm vs 5.0 mm blur
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
    blur2_lat = apply_linear_motion_blur_fourier(clean_lat, 2.0, vox_size, angle_deg=90.0)
    blur5_lat = apply_linear_motion_blur_fourier(clean_lat, 5.0, vox_size, angle_deg=90.0)

    div = np.ones((clean_lat.shape[0], 3)) * np.max(clean_lat)
    triptych = np.concatenate([clean_lat, div, blur2_lat, div, blur5_lat], axis=1)
    axes[0].imshow(triptych, cmap='gray')
    axes[0].set_title('Lateral Fluoro Blur: 0 mm | 2.0 mm | 5.0 mm\n(Cranial-Caudal Respiratory Smear)', fontsize=10.5, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: mTRE vs Motion Blur Extent
    axes[1].plot(df['Motion_Blur_mm'], df['Final_mTRE_mm'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='Diff-FVR (Spectral-NCC)')
    axes[1].axhline(y=1.0, color='forestgreen', linestyle='--', linewidth=1.5, label='Sub-millimeter Target (1.0 mm)')
    axes[1].axhline(y=2.0, color='crimson', linestyle=':', linewidth=1.5, label='Clinical Breakdown Threshold (2.0 mm)')
    axes[1].axvspan(0.0, 2.0, color='forestgreen', alpha=0.10, label='Clinically Safe Zone (<= 2.0 mm)')
    axes[1].set_xlabel('Intraoperative Motion Blur Extent (mm)', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Mean Target Registration Error (mTRE, mm)', fontsize=10, fontweight='bold')
    axes[1].set_title('Accuracy vs Respiratory / Table Motion Blur', fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper left', frameon=True, fontsize=9)
    axes[1].set_ylim(-0.1, max(3.0, df['Final_mTRE_mm'].max() * 1.2))

    # Panel 3: Directional Axis Error Decomposition (Tx, Ty, Tz)
    axes[2].plot(df['Motion_Blur_mm'], df['Axial_Error_Tz_mm'], 's-', color='#e74c3c', linewidth=2, label='Cranial-Caudal Error (Tz, along blur)')
    axes[2].plot(df['Motion_Blur_mm'], df['Depth_Error_Tx_mm'], '^-', color='#2ecc71', linewidth=2, label='Lateral Depth Error (Tx)')
    axes[2].plot(df['Motion_Blur_mm'], df['InPlane_Error_Ty_mm'], 'd-', color='#9b59b6', linewidth=2, label='AP Transverse Error (Ty)')
    axes[2].set_xlabel('Motion Blur Extent (mm)', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('Absolute Coordinate Error (mm)', fontsize=10, fontweight='bold')
    axes[2].set_title('Coordinate Error Decomposition Under Motion Blur', fontsize=11, fontweight='bold')
    axes[2].legend(loc='upper left', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")

    if BRAIN_DIR.exists():
        import shutil
        brain_fig = BRAIN_DIR / "fig20_motion_blur_stress.png"
        shutil.copy2(fig_path, brain_fig)
        print(f"Copied to brain artifacts: {brain_fig}")


if __name__ == '__main__':
    run_motion_blur_benchmark()
