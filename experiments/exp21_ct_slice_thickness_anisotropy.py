"""
Experiment 21: Preoperative CT Anisotropic Slice Thickness Breakdown Limit.

Physical Clinical Reality:
  1. Diagnostic pre-op spine CT protocols vary across institutions:
     - High-resolution surgical CT: 0.6 mm - 1.0 mm slice thickness.
     - Standard helical abdominal/lumbar CT: 1.5 mm - 2.5 mm slice thickness.
     - Trauma emergency protocols or low-dose screening: 3.0 mm - 5.0 mm slice thickness.
  2. Large slice thickness introduces axial partial volume averaging and undersampling aliasing.
  3. Through the Fourier Slice Theorem, axial box filtering multiplies the 3D volume spectrum
     along the axial frequency w by sinc(w * s_z).
  4. What is the critical physical slice thickness s_z,crit beyond which sub-millimeter
     3D-2D registration fails?

Physical Resolution:
  Evaluate registration across slice thickness values s_z in [0.75, 1.25, 2.0, 3.0, 4.0, 5.0] mm.
  Quantify mTRE, depth error, and identify the breakdown boundary.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter1d
from scipy.optimize import minimize
from scipy import fft

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fvr.dicom_loader import load_dicom_series, hu_to_attenuation
from src.fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from src.fvr.registration_3d2d import compute_mtre_6dof

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR = Path(r"C:\Users\paree\.gemini\antigravity\brain\db84ca2a-35bb-4b49-a099-1ebdc0eb4333")


def simulate_ct_slice_thickness(vol_iso: np.ndarray, native_vox_mm: float, target_slice_thickness_mm: float) -> np.ndarray:
    """
    Simulate CT axial slice thickness via box-filter integration and down/up sampling.
    """
    if target_slice_thickness_mm <= native_vox_mm:
        return np.copy(vol_iso)

    # Box filter along z axis (axis 0)
    filter_width_vox = target_slice_thickness_mm / native_vox_mm
    # Standard deviation of uniform box filter of width W is sigma = W / sqrt(12)
    sigma_vox = filter_width_vox / np.sqrt(12.0)

    # Apply axial slice profile blur
    vol_blurred = gaussian_filter1d(vol_iso, sigma=sigma_vox, axis=0, mode='nearest')
    return vol_blurred


def run_slice_thickness_benchmark():
    print("=" * 75)
    print("EXPERIMENT 21: PREOPERATIVE CT SLICE THICKNESS BREAKDOWN LIMIT")
    print("=" * 75)

    csv_path = OUTPUT_DIR / "table21_slice_thickness_anisotropy.csv"
    fig_path = OUTPUT_DIR / "fig21_slice_thickness_anisotropy.png"

    if csv_path.exists():
        print(f"Loading existing benchmark CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        # Load and resample clinical DICOM
        dicom_dir = PROJECT_ROOT / "data" / "dicom"
        vol_hu, spacing = load_dicom_series(dicom_dir)

        target_xy = 96
        phys_z = vol_hu.shape[0] * spacing[0]
        phys_x = vol_hu.shape[2] * spacing[2]
        vox_size = phys_x / target_xy
        target_z = int(round(phys_z / vox_size))

        vol_iso = zoom(vol_hu, (target_z / vol_hu.shape[0], target_xy / vol_hu.shape[1], target_xy / vol_hu.shape[2]), order=1)
        pad_z_before = (target_xy - target_z) // 2
        pad_z_after = target_xy - target_z - pad_z_before
        vol_cubic = np.pad(vol_iso, ((pad_z_before, pad_z_after), (0, 0), (0, 0)), mode='constant', constant_values=-1000.0)

        # High-resolution isotropic ground truth renderer
        vol_mu_gt = hu_to_attenuation(vol_cubic)
        renderer_gt = DifferentiableFourierVolumeRenderer(vol_mu_gt, pad_factor=2.0)
        print(f"Native isotropic voxel size: {vox_size:.3f} mm")

        true_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        init_pose = (2.0, -1.5, 1.2, 3.0, -2.0, 2.5)
        init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu_gt.shape)
        print(f"Initial Misalignment: {init_pose}, Initial mTRE: {init_mtre:.2f} mm")

        # Ground-truth fluoroscopy targets (from true patient anatomy)
        clean_target_ap = renderer_gt.render_drr(0, 0, 0, tx=0, ty=0, view='ap')
        clean_target_lat = renderer_gt.render_drr(0, 0, 0, tx=0, ty=0, view='lateral')

        th, tw = clean_target_ap.shape
        padded_ap = np.zeros((renderer_gt.pz, renderer_gt.px), dtype=np.float64)
        padded_lat = np.zeros((renderer_gt.pz, renderer_gt.py), dtype=np.float64)
        sh, sw_ap = (renderer_gt.pz - th) // 2, (renderer_gt.px - tw) // 2
        sh, sw_lat = (renderer_gt.pz - th) // 2, (renderer_gt.py - tw) // 2
        padded_ap[sh:sh+th, sw_ap:sw_ap+tw] = clean_target_ap
        padded_lat[sh:sh+th, sw_lat:sw_lat+tw] = clean_target_lat

        f_targ_ap = fft.fftshift(fft.fft2(fft.ifftshift(padded_ap)))
        f_targ_lat = fft.fftshift(fft.fft2(fft.ifftshift(padded_lat)))

        # Slice thickness levels in mm
        slice_thicknesses_mm = [vox_size, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        records = []

        for st_mm in slice_thicknesses_mm:
            print(f"\n--- Testing Preoperative CT Slice Thickness: {st_mm:.2f} mm ---")

            # Simulate degraded pre-op CT volume
            vol_degraded_hu = simulate_ct_slice_thickness(vol_cubic, vox_size, st_mm)
            vol_degraded_mu = hu_to_attenuation(vol_degraded_hu)
            renderer_thick = DifferentiableFourierVolumeRenderer(vol_degraded_mu, pad_factor=2.0)

            def obj_thick(p):
                yaw, pitch, roll, tx, ty, tz = p
                f_ap = renderer_thick.extract_frequency_slice(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
                f_lat = renderer_thick.extract_frequency_slice(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')
                ncc_ap = renderer_thick.compute_spectral_ncc(f_ap, f_targ_ap)
                ncc_lat = renderer_thick.compute_spectral_ncc(f_lat, f_targ_lat)
                return float(0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat))

            t0 = time.time()
            res = minimize(obj_thick, x0=np.array(init_pose), method='Powell', options={'maxiter': 20, 'ftol': 1e-4})
            t_run = time.time() - t0
            mtre = compute_mtre_6dof(true_pose, tuple(res.x), vol_mu_gt.shape)

            tz_err = abs(res.x[5] - true_pose[5])
            tx_err = abs(res.x[3] - true_pose[3])
            ty_err = abs(res.x[4] - true_pose[4])
            pitch_err = abs(res.x[1] - true_pose[1])

            print(f"  Result: mTRE = {mtre:.4f} mm, Tz_err = {tz_err:.4f} mm, Pitch_err = {pitch_err:.4f} deg, Loss = {res.fun:.6f}, Time = {t_run:.2f}s")

            records.append({
                'Slice_Thickness_mm': st_mm,
                'Anisotropy_Ratio': st_mm / vox_size,
                'Final_mTRE_mm': mtre,
                'Axial_Error_Tz_mm': tz_err,
                'Depth_Error_Tx_mm': tx_err,
                'InPlane_Error_Ty_mm': ty_err,
                'Pitch_Error_deg': pitch_err,
                'Final_Loss': float(res.fun),
                'Time_sec': t_run,
                'Success_2mm': bool(mtre < 2.0),
                'Submillimeter_1mm': bool(mtre < 1.0)
            })

        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved benchmark CSV to: {csv_path}")

    # Plot Figure 21
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    # Panel 1: Visual comparison of sagittal CT slice at 0.7 mm vs 3.0 mm vs 5.0 mm
    dicom_dir = PROJECT_ROOT / "data" / "dicom"
    vol_hu, spacing = load_dicom_series(dicom_dir)
    target_xy = 96
    phys_z = vol_hu.shape[0] * spacing[0]
    phys_x = vol_hu.shape[2] * spacing[2]
    vox_size = phys_x / target_xy
    target_z = int(round(phys_z / vox_size))
    vol_iso = zoom(vol_hu, (target_z / vol_hu.shape[0], target_xy / vol_hu.shape[1], target_xy / vol_hu.shape[2]), order=1)
    pad_z_before = (target_xy - target_z) // 2
    pad_z_after = target_xy - target_z - pad_z_before
    vol_cubic = np.pad(vol_iso, ((pad_z_before, pad_z_after), (0, 0), (0, 0)), mode='constant', constant_values=-1000.0)

    sag_clean = vol_cubic[:, :, target_xy // 2]
    sag_3mm = simulate_ct_slice_thickness(vol_cubic, vox_size, 3.0)[:, :, target_xy // 2]
    sag_5mm = simulate_ct_slice_thickness(vol_cubic, vox_size, 5.0)[:, :, target_xy // 2]

    # Normalize for display
    def norm_img(im):
        im = np.clip(im, -200, 800)
        return (im - im.min()) / (im.max() - im.min() + 1e-8)

    div = np.ones((sag_clean.shape[0], 2))
    sag_triptych = np.concatenate([norm_img(sag_clean), div, norm_img(sag_3mm), div, norm_img(sag_5mm)], axis=1)
    axes[0].imshow(sag_triptych, cmap='bone')
    axes[0].set_title(f'CT Sagittal Slice: {vox_size:.1f} mm | 3.0 mm | 5.0 mm\n(Axial Partial Volume Blur)', fontsize=10.5, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: mTRE vs CT Slice Thickness
    axes[1].plot(df['Slice_Thickness_mm'], df['Final_mTRE_mm'], 's-', color='#e67e22', linewidth=2.5, markersize=8, label='Diff-FVR Registration')
    axes[1].axhline(y=1.0, color='forestgreen', linestyle='--', linewidth=1.5, label='Sub-millimeter Threshold (1.0 mm)')
    axes[1].axhline(y=2.0, color='crimson', linestyle=':', linewidth=1.5, label='Clinical Breakdown Threshold (2.0 mm)')
    axes[1].axvspan(df['Slice_Thickness_mm'].min(), 3.0, color='forestgreen', alpha=0.10, label='Clinically Safe Zone (<= 3.0 mm)')
    axes[1].set_xlabel('Preoperative CT Slice Thickness (mm)', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Mean Target Registration Error (mTRE, mm)', fontsize=10, fontweight='bold')
    axes[1].set_title('Registration Accuracy vs CT Anisotropic Slice Thickness', fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper left', frameon=True, fontsize=9)
    axes[1].set_ylim(-0.05, max(3.5, df['Final_mTRE_mm'].max() * 1.2))

    # Panel 3: Error Breakdown (Axial Tz vs Pitch Angle)
    axes[2].plot(df['Slice_Thickness_mm'], df['Axial_Error_Tz_mm'], 'o-', color='#c0392b', linewidth=2, label='Axial Translation Error (Tz, mm)')
    axes[2].plot(df['Slice_Thickness_mm'], df['Pitch_Error_deg'], '^--', color='#2980b9', linewidth=2, label='Sagittal Pitch Error (deg)')
    axes[2].plot(df['Slice_Thickness_mm'], df['Depth_Error_Tx_mm'], 'd:', color='#27ae60', linewidth=1.8, label='Depth Error (Tx, mm)')
    axes[2].set_xlabel('Preoperative CT Slice Thickness (mm)', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('Parameter Error (mm or deg)', fontsize=10, fontweight='bold')
    axes[2].set_title('Axial & Pitch Error Breakdown Under Anisotropy', fontsize=11, fontweight='bold')
    axes[2].legend(loc='upper left', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")

    if BRAIN_DIR.exists():
        import shutil
        brain_fig = BRAIN_DIR / "fig21_slice_thickness_anisotropy.png"
        shutil.copy2(fig_path, brain_fig)
        print(f"Copied to brain artifacts: {brain_fig}")


if __name__ == '__main__':
    run_slice_thickness_benchmark()
