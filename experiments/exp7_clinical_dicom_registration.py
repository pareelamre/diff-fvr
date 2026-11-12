"""One CT source volume, 20 poses, same-renderer synthetic targets; no acquired fluoroscopy."""

import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fvr.dicom_loader import resample_isotropic_cube, load_dicom_series, hu_to_attenuation
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.registration_3d2d import BiplanarSpineRegistrationEngine, compute_mtre_6dof


def add_clinical_noise(img: np.ndarray, photon_count: float = 5e4, gaussian_sigma: float = 0.015) -> np.ndarray:
    """
    Simulate realistic intraoperative C-arm fluoroscopic degradation:
      1. Poisson photon starvation noise
      2. Gaussian electronic sensor read noise
    """
    norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    # Poisson noise scaled by incident photon flux
    noisy_photons = np.random.poisson(norm * photon_count) / photon_count
    # Gaussian electronic noise
    noise_gaussian = np.random.normal(0, gaussian_sigma, img.shape)
    degraded = np.clip(noisy_photons + noise_gaussian, 0.0, 1.0)
    return degraded * (img.max() - img.min()) + img.min()


def run_clinical_benchmark():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    dicom_dir = Path(__file__).resolve().parent.parent / "data" / "dicom"

    print("=" * 70)
    print("  EXPERIMENT 7: Real Clinical Patient DICOM CT Registration Benchmark")
    print("=" * 70)

    # 1. Load Real Clinical DICOM CT Series
    print(f"Loading real clinical DICOM series from: {dicom_dir}")
    vol_hu, spacing = load_dicom_series(dicom_dir)
    print(f"Loaded clinical volume: Shape = {vol_hu.shape}, Spacing = {[round(s, 2) for s in spacing]} mm")
    print(f"CT Attenuation range: [{vol_hu.min():.1f}, {vol_hu.max():.1f}] HU")

    # Physical dimensions of clinical patient scan
    phys_z = vol_hu.shape[0] * spacing[0]
    phys_y = vol_hu.shape[1] * spacing[1]
    phys_x = vol_hu.shape[2] * spacing[2]
    print(f"Physical CT FOV: Z={phys_z:.1f} mm, Y={phys_y:.1f} mm, X={phys_x:.1f} mm")

    target_xy = 96
    vol_resampled, grid_spacing = resample_isotropic_cube(vol_hu, spacing, target_xy)
    vox_size = grid_spacing[0]
    vol_mu = hu_to_attenuation(vol_resampled)

    renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
    engine = BiplanarSpineRegistrationEngine(renderer)

    # 2. Monte Carlo Cohort on Real Clinical Patient Data
    np.random.seed(101)
    n_trials = 20

    records = []

    print(f"Running {n_trials} trials under Clean vs Noisy Fluoroscopy...")
    for trial in range(n_trials):
        yaw = float(np.random.uniform(-5.0, 5.0))
        pitch = float(np.random.uniform(-5.0, 5.0))
        roll = float(np.random.uniform(-5.0, 5.0))
        tx = float(np.random.uniform(-6.0, 6.0))
        ty = float(np.random.uniform(-4.0, 4.0))
        tz = float(np.random.uniform(-4.0, 4.0))
        true_pose = (yaw, pitch, roll, tx, ty, tz)

        init_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape, spacing=(vox_size,) * 3)

        # Render ground-truth projections
        target_ap_clean = renderer.render_drr(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
        target_lat_clean = renderer.render_drr(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')

        # Add realistic clinical fluoroscopic noise
        target_ap_noisy = add_clinical_noise(target_ap_clean)
        target_lat_noisy = add_clinical_noise(target_lat_clean)

        # Case A: Standard Clinical Scan (Clean)
        t0 = time.time()
        res_clean = engine.register_biplanar(target_ap_clean, target_lat_clean, initial_pose=init_pose, max_iter=70)
        t_clean = time.time() - t0
        mtre_clean = compute_mtre_6dof(true_pose, res_clean['estimated_pose'], vol_mu.shape, spacing=(vox_size,) * 3)

        records.append({
            'Trial': trial,
            'Condition': 'Clean (Synthetic)',
            'Init_mTRE_mm': init_mtre,
            'Final_mTRE_mm': mtre_clean,
            'Success_2mm': mtre_clean < 2.0,
            'Success_1mm': mtre_clean < 1.0,
            'Time_sec': t_clean,
            'Time_per_eval_ms': res_clean['time_per_eval_ms'],
            'Function_evals': res_clean['function_evals'],
            'Error_Yaw': abs(res_clean['estimated_pose'][0] - yaw),
            'Error_Pitch': abs(res_clean['estimated_pose'][1] - pitch),
            'Error_Roll': abs(res_clean['estimated_pose'][2] - roll),
            'Error_Tx': abs(res_clean['estimated_pose'][3] - tx),
            'Error_Ty': abs(res_clean['estimated_pose'][4] - ty),
            'Error_Tz': abs(res_clean['estimated_pose'][5] - tz),
        })

        # Case B: Low-Dose Noisy Intraoperative Fluoroscopy
        t0 = time.time()
        res_noisy = engine.register_biplanar(target_ap_noisy, target_lat_noisy, initial_pose=init_pose, max_iter=70)
        t_noisy = time.time() - t0
        mtre_noisy = compute_mtre_6dof(true_pose, res_noisy['estimated_pose'], vol_mu.shape, spacing=(vox_size,) * 3)

        records.append({
            'Trial': trial,
            'Condition': 'Low-Dose Fluoroscopy (Poisson + Gaussian Noise)',
            'Init_mTRE_mm': init_mtre,
            'Final_mTRE_mm': mtre_noisy,
            'Success_2mm': mtre_noisy < 2.0,
            'Success_1mm': mtre_noisy < 1.0,
            'Time_sec': t_noisy,
            'Time_per_eval_ms': res_noisy['time_per_eval_ms'],
            'Function_evals': res_noisy['function_evals'],
            'Error_Yaw': abs(res_noisy['estimated_pose'][0] - yaw),
            'Error_Pitch': abs(res_noisy['estimated_pose'][1] - pitch),
            'Error_Roll': abs(res_noisy['estimated_pose'][2] - roll),
            'Error_Tx': abs(res_noisy['estimated_pose'][3] - tx),
            'Error_Ty': abs(res_noisy['estimated_pose'][4] - ty),
            'Error_Tz': abs(res_noisy['estimated_pose'][5] - tz),
        })

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1:2d}/{n_trials} | Clean mTRE: {mtre_clean:.2f} mm | Low-Dose mTRE: {mtre_noisy:.2f} mm")

    df = pd.DataFrame(records)
    df['Metric_units'] = 'mm'
    df['Target_type'] = 'synthetic_same_renderer'
    csv_path = out_dir / "table7_clinical_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved clinical registration results to: {csv_path}")

    for cond, grp in df.groupby('Condition'):
        print(f"\n--- {cond} ---")
        print(f"  Init mTRE:  {grp['Init_mTRE_mm'].mean():.2f} ± {grp['Init_mTRE_mm'].std():.2f} mm")
        print(f"  Final mTRE: {grp['Final_mTRE_mm'].mean():.2f} ± {grp['Final_mTRE_mm'].std():.2f} mm")
        print(f"  Success (<2.0 mm): {grp['Success_2mm'].mean() * 100:.1f}%")
        print(f"  Submillimeter (<1.0 mm): {grp['Success_1mm'].mean() * 100:.1f}%")
        print(f"  Registration Latency: {grp['Time_sec'].mean():.2f} ± {grp['Time_sec'].std():.2f} s")
        print(f"  Eval Latency: {grp['Time_per_eval_ms'].mean():.2f} ms")

    # 3. Publication Figure 7: Clinical Patient Validation
    fig = plt.figure(figsize=(15, 9), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.3, wspace=0.25)

    # High-quality isotropic visualization volume (1.0 mm isotropic spacing)
    scale_z = spacing[0] / 1.0
    scale_y = spacing[1] / 1.0
    scale_x = spacing[2] / 1.0
    vol_disp = zoom(vol_hu, (scale_z, scale_y, scale_x), order=1)
    
    # Pad along Z to cubic FOV so all slices (Axial, Coronal, Sagittal) share identical physical dimensions
    nz_d, ny_d, nx_d = vol_disp.shape
    max_d = max(nz_d, ny_d, nx_d)
    pad_d1 = (max_d - nz_d) // 2
    pad_d2 = max_d - nz_d - pad_d1
    vol_disp_cubic = np.pad(vol_disp, ((pad_d1, pad_d2), (0, 0), (0, 0)), mode='constant', constant_values=-1000.0)

    # Anatomical landmark centers: spine body in Y is ~0.544, midline in X is ~0.50
    mid_z = max_d // 2
    spine_y = int(round(max_d * 0.544))
    mid_x = max_d // 2

    # Top Row: Real Patient CT Volume Slices (Isotropic 1:1 Physical Aspect Ratio)
    ax_ax = fig.add_subplot(gs[0, 0])
    ax_ax.imshow(vol_disp_cubic[mid_z, :, :], cmap='bone')
    ax_ax.set_title('Clinical CT: Axial View', fontsize=11, fontweight='bold')
    ax_ax.axis('off')

    ax_cor = fig.add_subplot(gs[0, 1])
    ax_cor.imshow(vol_disp_cubic[:, spine_y, :], cmap='bone', origin='lower')
    ax_cor.set_title('Clinical CT: Coronal View (Isotropic)', fontsize=11, fontweight='bold')
    ax_cor.axis('off')

    ax_sag = fig.add_subplot(gs[0, 2])
    ax_sag.imshow(vol_disp_cubic[:, :, mid_x], cmap='bone', origin='lower')
    ax_sag.set_title('Clinical CT: Sagittal View (Isotropic)', fontsize=11, fontweight='bold')
    ax_sag.axis('off')

    # Bottom Row: Radiographic Registrations & CED
    ax_drr1 = fig.add_subplot(gs[1, 0])
    ax_drr1.imshow(target_ap_clean, cmap='gray')
    ax_drr1.set_title('Target AP Radiograph (Clean)', fontsize=11, fontweight='bold')
    ax_drr1.axis('off')

    ax_drr2 = fig.add_subplot(gs[1, 1])
    ax_drr2.imshow(target_lat_noisy, cmap='gray')
    ax_drr2.set_title('Target Lateral (Low-Dose Fluoroscopy)', fontsize=11, fontweight='bold')
    ax_drr2.axis('off')

    # Cumulative Error Distribution on Real Clinical CT
    ax_ced = fig.add_subplot(gs[1, 2])
    th_range = np.linspace(0.0, 8.0, 81)
    clean_grp = df[df['Condition'] == 'Clean (Synthetic)']
    noisy_grp = df[df['Condition'] == 'Low-Dose Fluoroscopy (Poisson + Gaussian Noise)']

    clean_rates = [np.mean(clean_grp['Final_mTRE_mm'] <= t) * 100 for t in th_range]
    noisy_rates = [np.mean(noisy_grp['Final_mTRE_mm'] <= t) * 100 for t in th_range]

    ax_ced.plot(th_range, clean_rates, color='#1f77b4', linewidth=2.5, label='Clean Standard-Dose')
    ax_ced.plot(th_range, noisy_rates, color='#ff7f0e', linewidth=2.5, linestyle='--', label='Low-Dose Fluoroscopy')
    ax_ced.axvline(2.0, color='grey', linestyle=':', label='Clinical Threshold (2.0 mm)')
    ax_ced.axvline(1.0, color='darkgrey', linestyle='-.', label='Submillimeter (1.0 mm)')

    ax_ced.set_xlabel('mTRE Threshold (mm)', fontsize=10, fontweight='bold')
    ax_ced.set_ylabel('Success Rate (%)', fontsize=10, fontweight='bold')
    ax_ced.set_title('Clinical CT Registration CED', fontsize=11, fontweight='bold')
    ax_ced.set_xlim(0, 6)
    ax_ced.set_ylim(0, 105)
    ax_ced.grid(True, linestyle='--', alpha=0.5)
    ax_ced.legend(loc='lower right', fontsize=8.5, frameon=True)

    fig7_path = out_dir / "fig7_clinical_dicom_registration.png"
    fig.savefig(fig7_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 7 to: {fig7_path}")


if __name__ == "__main__":
    run_clinical_benchmark()
