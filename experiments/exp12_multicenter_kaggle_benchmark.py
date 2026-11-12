import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import zoom

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fvr.dicom_loader import load_dicom_series, hu_to_attenuation
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.registration_3d2d import BiplanarSpineRegistrationEngine, compute_mtre_6dof


def add_clinical_noise(img: np.ndarray, photon_count: float = 5e4, gaussian_sigma: float = 0.015) -> np.ndarray:
    norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    noisy_photons = np.random.poisson(norm * photon_count) / photon_count
    noise_gaussian = np.random.normal(0, gaussian_sigma, img.shape)
    degraded = np.clip(noisy_photons + noise_gaussian, 0.0, 1.0)
    return degraded * (img.max() - img.min()) + img.min()


def run_kaggle_benchmark():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle_dir = Path(__file__).resolve().parent.parent / "data" / "kaggle_lumbar" / "DICOM"

    print("=" * 70)
    print("  EXPERIMENT 12: Independent Multi-Center Kaggle Lumbar Spine Benchmark")
    print("=" * 70)

    print(f"Loading independent clinical DICOM from: {kaggle_dir}")
    vol_hu, spacing = load_dicom_series(kaggle_dir)
    dz, dy, dx = spacing
    print(f"Loaded volume: Shape={vol_hu.shape}, Spacing=[{dz:.2f}, {dy:.3f}, {dx:.3f}] mm")

    phys_z = vol_hu.shape[0] * dz
    phys_x = vol_hu.shape[2] * dx

    target_xy = 96
    vox_size = phys_x / target_xy
    target_z = int(round(phys_z / vox_size))

    vol_iso = zoom(vol_hu, (target_z / vol_hu.shape[0], target_xy / vol_hu.shape[1], target_xy / vol_hu.shape[2]), order=1)
    pad_z_before = (target_xy - target_z) // 2
    pad_z_after = target_xy - target_z - pad_z_before
    vol_cubic = np.pad(vol_iso, ((pad_z_before, pad_z_after), (0, 0), (0, 0)), mode='constant', constant_values=-1000.0)

    vol_mu = hu_to_attenuation(vol_cubic)
    renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
    engine = BiplanarSpineRegistrationEngine(renderer)

    n_trials = 20
    records = []
    np.random.seed(505)

    print(f"\nRunning {n_trials} trials on Independent Kaggle Clinical Dataset...")
    for trial in range(n_trials):
        yaw = float(np.random.uniform(-5.0, 5.0))
        pitch = float(np.random.uniform(-5.0, 5.0))
        roll = float(np.random.uniform(-5.0, 5.0))
        tx = float(np.random.uniform(-6.0, 6.0))
        ty = float(np.random.uniform(-4.0, 4.0))
        tz = float(np.random.uniform(-4.0, 4.0))
        true_pose = (yaw, pitch, roll, tx, ty, tz)

        init_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape)

        target_ap_clean = renderer.render_drr(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
        target_lat_clean = renderer.render_drr(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')

        target_ap_noisy = add_clinical_noise(target_ap_clean)
        target_lat_noisy = add_clinical_noise(target_lat_clean)

        # Standard Dose
        t0 = time.time()
        res_clean = engine.register_biplanar(target_ap_clean, target_lat_clean, initial_pose=init_pose, max_iter=70)
        t_clean = time.time() - t0
        mtre_clean = compute_mtre_6dof(true_pose, res_clean['estimated_pose'], vol_mu.shape)

        records.append({
            'Dataset': 'Kaggle Clinical Lumbar CT',
            'Condition': 'Clean (Standard Dose)',
            'Trial': trial,
            'Init_mTRE_mm': init_mtre,
            'Final_mTRE_mm': mtre_clean,
            'Success_2mm': mtre_clean < 2.0,
            'Success_1mm': mtre_clean < 1.0,
            'Time_sec': t_clean,
            'Eval_ms': res_clean['time_per_eval_ms']
        })

        # Low-Dose Fluoroscopy
        t0 = time.time()
        res_noisy = engine.register_biplanar(target_ap_noisy, target_lat_noisy, initial_pose=init_pose, max_iter=70)
        t_noisy = time.time() - t0
        mtre_noisy = compute_mtre_6dof(true_pose, res_noisy['estimated_pose'], vol_mu.shape)

        records.append({
            'Dataset': 'Kaggle Clinical Lumbar CT',
            'Condition': 'Low-Dose Fluoroscopy',
            'Trial': trial,
            'Init_mTRE_mm': init_mtre,
            'Final_mTRE_mm': mtre_noisy,
            'Success_2mm': mtre_noisy < 2.0,
            'Success_1mm': mtre_noisy < 1.0,
            'Time_sec': t_noisy,
            'Eval_ms': res_noisy['time_per_eval_ms']
        })

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1:2d}/{n_trials} | Clean: {mtre_clean:.3f} mm | Low-Dose: {mtre_noisy:.3f} mm")

    df = pd.DataFrame(records)
    csv_path = out_dir / "table12_kaggle_clinical_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved Kaggle benchmark to: {csv_path}")

    for cond, grp in df.groupby('Condition'):
        print(f"\n--- {cond} ---")
        print(f"  Final mTRE: {grp['Final_mTRE_mm'].mean():.3f} ± {grp['Final_mTRE_mm'].std():.3f} mm")
        print(f"  Success (<2.0 mm): {grp['Success_2mm'].mean() * 100:.1f}%")
        print(f"  Submillimeter (<1.0 mm): {grp['Success_1mm'].mean() * 100:.1f}%")
        print(f"  Mean Time: {grp['Time_sec'].mean():.2f} s")

    # Publication Figure 12: Independent Dataset Multi-Center Validation
    fig = plt.figure(figsize=(15, 8), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # Top Row: Slices of the Kaggle Lumbar CT
    mid_z = target_xy // 2
    mid_y = int(target_xy * 0.50)
    mid_x = target_xy // 2

    ax_ax = fig.add_subplot(gs[0, 0])
    ax_ax.imshow(vol_cubic[mid_z, :, :], cmap='bone')
    ax_ax.set_title('Kaggle Lumbar CT: Axial Slice', fontsize=11, fontweight='bold')
    ax_ax.axis('off')

    ax_cor = fig.add_subplot(gs[0, 1])
    ax_cor.imshow(vol_cubic[:, mid_y, :], cmap='bone', origin='lower')
    ax_cor.set_title('Kaggle Lumbar CT: Coronal View', fontsize=11, fontweight='bold')
    ax_cor.axis('off')

    ax_sag = fig.add_subplot(gs[0, 2])
    ax_sag.imshow(vol_cubic[:, :, mid_x], cmap='bone', origin='lower')
    ax_sag.set_title('Kaggle Lumbar CT: Sagittal View', fontsize=11, fontweight='bold')
    ax_sag.axis('off')

    # Bottom Row: Target DRRs and CED Curve
    ax_ap = fig.add_subplot(gs[1, 0])
    ax_ap.imshow(target_ap_clean, cmap='gray')
    ax_ap.set_title('Simulated AP Fluoroscopy', fontsize=11, fontweight='bold')
    ax_ap.axis('off')

    ax_lat = fig.add_subplot(gs[1, 1])
    ax_lat.imshow(target_lat_noisy, cmap='gray')
    ax_lat.set_title('Low-Dose Lateral Fluoroscopy', fontsize=11, fontweight='bold')
    ax_lat.axis('off')

    ax_ced = fig.add_subplot(gs[1, 2])
    th_range = np.linspace(0.0, 4.0, 80)
    clean_grp = df[df['Condition'] == 'Clean (Standard Dose)']
    noisy_grp = df[df['Condition'] == 'Low-Dose Fluoroscopy']

    clean_rates = [np.mean(clean_grp['Final_mTRE_mm'] <= t) * 100 for t in th_range]
    noisy_rates = [np.mean(noisy_grp['Final_mTRE_mm'] <= t) * 100 for t in th_range]

    ax_ced.plot(th_range, clean_rates, color='#1f77b4', linewidth=2.5, label='Clean Standard Dose')
    ax_ced.plot(th_range, noisy_rates, color='#ff7f0e', linewidth=2.5, linestyle='--', label='Low-Dose Fluoroscopy')
    ax_ced.axvline(2.0, color='grey', linestyle=':', label='Clinical Threshold (2.0 mm)')
    ax_ced.axvline(1.0, color='darkgrey', linestyle='-.', label='Submillimeter (1.0 mm)')

    ax_ced.set_xlabel('mTRE Threshold (mm)', fontsize=10, fontweight='bold')
    ax_ced.set_ylabel('Success Rate (%)', fontsize=10, fontweight='bold')
    ax_ced.set_title('Multi-Center Kaggle CED Curves', fontsize=11, fontweight='bold')
    ax_ced.set_xlim(0, 3)
    ax_ced.set_ylim(0, 105)
    ax_ced.grid(True, linestyle='--', alpha=0.5)
    ax_ced.legend(loc='lower right', fontsize=8.5)

    fig_path = out_dir / "fig12_kaggle_clinical_benchmark.png"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 12 to: {fig_path}")


if __name__ == "__main__":
    run_kaggle_benchmark()
