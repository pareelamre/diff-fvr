import os
import sys
import time
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import zoom

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fvr.dicom_loader import hu_to_attenuation
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.registration_3d2d import BiplanarSpineRegistrationEngine, compute_mtre_6dof


def add_quantum_noise(img: np.ndarray, photon_count: float = 5e4, gaussian_sigma: float = 0.015) -> np.ndarray:
    norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    noisy_photons = np.random.poisson(norm * photon_count) / photon_count
    noise_gaussian = np.random.normal(0, gaussian_sigma, img.shape)
    degraded = np.clip(noisy_photons + noise_gaussian, 0.0, 1.0)
    return degraded * (img.max() - img.min()) + img.min()


def run_amos_benchmark():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    amos_file = Path(__file__).resolve().parent.parent / "data" / "amos" / "amos_0001.nii.gz"

    print("=" * 75)
    print("  EXPERIMENT 13: Multi-Center AMOS Benchmark (NCT Dresden Cohort)")
    print("=" * 75)

    print(f"Loading AMOS CT volume from: {amos_file}")
    itk_img = sitk.ReadImage(str(amos_file))
    vol_hu = sitk.GetArrayFromImage(itk_img).astype(np.float32)
    # AMOS array shape: (Z, Y, X)
    nz, ny, nx = vol_hu.shape
    sp_x, sp_y, sp_z = itk_img.GetSpacing()
    print(f"Loaded AMOS CT: Shape=({nz}, {ny}, {nx}), Spacing=[{sp_z:.2f}, {sp_y:.3f}, {sp_x:.3f}] mm")

    # Resample to uniform cubic isotropic volume (96, 96, 96)
    target_dim = 96
    phys_z = nz * sp_z
    phys_y = ny * sp_y
    phys_x = nx * sp_x

    vox_size = max(phys_z, phys_x) / target_dim
    scale_z = target_dim / nz
    scale_y = (phys_y / vox_size) / ny
    scale_x = (phys_x / vox_size) / nx

    vol_iso = zoom(vol_hu, (scale_z, scale_y, scale_x), order=1)
    
    pad_y_before = (target_dim - vol_iso.shape[1]) // 2
    pad_y_after = target_dim - vol_iso.shape[1] - pad_y_before
    pad_x_before = (target_dim - vol_iso.shape[2]) // 2
    pad_x_after = target_dim - vol_iso.shape[2] - pad_x_before

    vol_cubic = np.pad(vol_iso, ((0, 0), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after)), mode='constant', constant_values=-1000.0)
    print(f"Resampled isotropic cubic volume: {vol_cubic.shape}, Voxel size={vox_size:.2f} mm")

    vol_mu = hu_to_attenuation(vol_cubic)
    renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
    engine = BiplanarSpineRegistrationEngine(renderer)

    n_trials = 20
    records = []
    np.random.seed(606)

    print(f"\nRunning {n_trials} trials on Independent AMOS Patient CT...")
    for trial in range(n_trials):
        yaw = float(np.random.uniform(-4.0, 4.0))
        pitch = float(np.random.uniform(-4.0, 4.0))
        roll = float(np.random.uniform(-4.0, 4.0))
        tx = float(np.random.uniform(-5.0, 5.0))
        ty = float(np.random.uniform(-4.0, 4.0))
        tz = float(np.random.uniform(-4.0, 4.0))
        true_pose = (yaw, pitch, roll, tx, ty, tz)

        init_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape)

        target_ap_clean = renderer.render_drr(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
        target_lat_clean = renderer.render_drr(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')

        target_ap_noisy = add_quantum_noise(target_ap_clean)
        target_lat_noisy = add_quantum_noise(target_lat_clean)

        # Standard Dose
        t0 = time.time()
        res_clean = engine.register_biplanar(target_ap_clean, target_lat_clean, initial_pose=init_pose, max_iter=70)
        t_clean = time.time() - t0
        mtre_clean = compute_mtre_6dof(true_pose, res_clean['estimated_pose'], vol_mu.shape)

        records.append({
            'Cohort': 'AMOS (NCT Dresden Baseline)',
            'Condition': 'Clean (Standard Dose)',
            'Trial': trial,
            'Init_mTRE_mm': init_mtre,
            'Final_mTRE_mm': mtre_clean,
            'Success_2mm': mtre_clean < 2.0,
            'Success_1mm': mtre_clean < 1.0,
            'Time_sec': t_clean,
            'Evals': res_clean['function_evals']
        })

        # Low-Dose Fluoroscopy
        t0 = time.time()
        res_noisy = engine.register_biplanar(target_ap_noisy, target_lat_noisy, initial_pose=init_pose, max_iter=70)
        t_noisy = time.time() - t0
        mtre_noisy = compute_mtre_6dof(true_pose, res_noisy['estimated_pose'], vol_mu.shape)

        records.append({
            'Cohort': 'AMOS (NCT Dresden Baseline)',
            'Condition': 'Low-Dose Fluoroscopy',
            'Trial': trial,
            'Init_mTRE_mm': init_mtre,
            'Final_mTRE_mm': mtre_noisy,
            'Success_2mm': mtre_noisy < 2.0,
            'Success_1mm': mtre_noisy < 1.0,
            'Time_sec': t_noisy,
            'Evals': res_noisy['function_evals']
        })

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1:2d}/{n_trials} | Clean: {mtre_clean:.3f} mm | Low-Dose: {mtre_noisy:.3f} mm")

    df = pd.DataFrame(records)
    csv_path = out_dir / "table13_amos_clinical_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved AMOS clinical benchmark to: {csv_path}")

    for cond, grp in df.groupby("Condition"):
        print(f"\n--- {cond} ---")
        print(f"  Mean mTRE: {grp['Final_mTRE_mm'].mean():.3f} ± {grp['Final_mTRE_mm'].std():.3f} mm")
        print(f"  Success Rate (< 2.0 mm): {grp['Success_2mm'].mean() * 100:.1f}%")
        print(f"  Submillimeter Rate (< 1.0 mm): {grp['Success_1mm'].mean() * 100:.1f}%")
        print(f"  Mean Execution Time: {grp['Time_sec'].mean():.2f} s")

    # 3. Publication Figure 13: AMOS Clinical Cohort Benchmark
    fig = plt.figure(figsize=(15, 8), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    ap_clean = renderer.render_drr(0, 0, 0, view='ap')
    lat_clean = renderer.render_drr(0, 0, 0, view='lateral')
    ap_noisy = add_quantum_noise(ap_clean)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(ap_clean, cmap='bone')
    ax0.set_title('AMOS Standard-Dose AP DRR', fontsize=11, fontweight='bold')
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(lat_clean, cmap='bone')
    ax1.set_title('AMOS Standard-Dose Lateral DRR', fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(ap_noisy, cmap='bone')
    ax2.set_title('AMOS Low-Dose Fluoroscopy Simulation', fontsize=11, fontweight='bold')
    ax2.axis('off')

    # Bottom Row Left: Boxplots for Clean vs Noisy
    ax_box = fig.add_subplot(gs[1, 0:2])
    clean_vals = df[df["Condition"] == "Clean (Standard Dose)"]["Final_mTRE_mm"].values
    noisy_vals = df[df["Condition"] == "Low-Dose Fluoroscopy"]["Final_mTRE_mm"].values

    bp = ax_box.boxplot([clean_vals, noisy_vals], tick_labels=['Standard Dose', 'Low-Dose Fluoroscopy'], patch_artist=True)
    colors = ['#c7e9c0', '#bcbddc']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set(color='red', linewidth=2)

    ax_box.axhline(2.0, color='red', linestyle='--', label='Surgical Limit (2.0 mm)')
    ax_box.axhline(1.0, color='green', linestyle=':', label='Submillimeter Limit (1.0 mm)')
    ax_box.set_ylabel('Final mTRE (mm)', fontsize=11, fontweight='bold')
    ax_box.set_title('Registration Accuracy on AMOS Cohort (NCT Dresden Baseline)', fontsize=12, fontweight='bold')
    ax_box.grid(True, linestyle='--', alpha=0.5)
    ax_box.legend(loc='upper right', fontsize=9.5)

    # Bottom Row Right: Cumulative Error Distribution
    ax_ced = fig.add_subplot(gs[1, 2])
    th_range = np.linspace(0.0, 3.0, 50)
    ced_clean = [np.mean(clean_vals <= t) * 100 for t in th_range]
    ced_noisy = [np.mean(noisy_vals <= t) * 100 for t in th_range]

    ax_ced.plot(th_range, ced_clean, label='Standard Dose', color='#2ca02c', linewidth=2.5)
    ax_ced.plot(th_range, ced_noisy, label='Low-Dose', color='#756bb1', linewidth=2.5, linestyle='--')
    ax_ced.axvline(2.0, color='grey', linestyle=':', label='2.0 mm Threshold')
    ax_ced.set_xlabel('mTRE Threshold (mm)', fontsize=10, fontweight='bold')
    ax_ced.set_ylabel('Success Rate (%)', fontsize=10, fontweight='bold')
    ax_ced.set_title('AMOS Cohort CED Curves', fontsize=11, fontweight='bold')
    ax_ced.grid(True, linestyle='--', alpha=0.5)
    ax_ced.legend(loc='lower right', fontsize=8.5)

    fig_path = out_dir / "fig13_amos_clinical_benchmark.png"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 13 to: {fig_path}")


if __name__ == "__main__":
    run_amos_benchmark()
