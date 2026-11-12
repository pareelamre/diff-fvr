import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.dicom_loader import create_spine_phantom, hu_to_attenuation
from fvr.raycasting_drr import raycast_drr


def main():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  EXPERIMENT 2: 360° Rotational Invariance & Radiographic Fidelity")
    print("=" * 65)

    size = 128
    vol_hu, _ = create_spine_phantom(size=size)
    vol_mu = hu_to_attenuation(vol_hu)

    renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)

    angles = np.arange(0, 360, 5)
    records = []

    print(f"Sweeping 360° rotation across {len(angles)} projection angles...")
    for idx, angle in enumerate(angles):
        fvr_drr = renderer.render_drr(yaw_deg=float(angle), view='lateral', crop_to_roi=True)
        ray_drr = raycast_drr(vol_mu, yaw_deg=float(angle))

        # Dynamic range min-max normalization to [0, 1]
        r_min, r_max = ray_drr.min(), ray_drr.max()
        f_min, f_max = fvr_drr.min(), fvr_drr.max()
        ray_norm = (ray_drr - r_min) / (r_max - r_min + 1e-8)
        fvr_norm = (fvr_drr - f_min) / (f_max - f_min + 1e-8)

        # Image Quality Metrics
        psnr = compute_psnr(ray_norm, fvr_norm, data_range=1.0)
        ssim = compute_ssim(ray_norm, fvr_norm, data_range=1.0)
        rmse = np.sqrt(np.mean((ray_norm - fvr_norm) ** 2))
        
        # Normalized Cross Correlation
        r_c = (ray_norm - ray_norm.mean()) / (ray_norm.std() + 1e-8)
        f_c = (fvr_norm - fvr_norm.mean()) / (fvr_norm.std() + 1e-8)
        ncc = float(np.mean(r_c * f_c))

        records.append({
            'Angle_deg': angle,
            'PSNR_dB': psnr,
            'SSIM': ssim,
            'RMSE': rmse,
            'NCC': ncc
        })

        if idx % 15 == 0:
            print(f"  Angle {angle:3d}° | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | NCC: {ncc:.4f}")

    df = pd.DataFrame(records)
    csv_path = out_dir / "table2_rotational_fidelity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved fidelity table to: {csv_path}")
    print(f"Mean PSNR across 360°: {df['PSNR_dB'].mean():.2f} ± {df['PSNR_dB'].std():.2f} dB")
    print(f"Mean SSIM across 360°: {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
    print(f"Mean NCC across 360°:  {df['NCC'].mean():.4f} ± {df['NCC'].std():.4f}")

    # Publication Figure 2: Cartesian and Polar Fidelity Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Plot 1: Angle vs PSNR & SSIM
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(df['Angle_deg'], df['PSNR_dB'], color='#1f77b4', linewidth=2.2, label='PSNR (dB)')
    l2 = ax1_twin.plot(df['Angle_deg'], df['SSIM'], color='#2ca02c', linewidth=2.2, linestyle='--', label='SSIM')

    ax1.set_xlabel('Projection Viewing Angle (Degrees)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Peak Signal-to-Noise Ratio (dB)', color='#1f77b4', fontsize=12, fontweight='bold')
    ax1_twin.set_ylabel('Structural Similarity (SSIM)', color='#2ca02c', fontsize=12, fontweight='bold')
    ax1.set_title('Radiographic Fidelity across 360° Rotational Sweep', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 355)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Combined legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right', frameon=True)

    # Plot 2: Polar Plot of Normalized Cross-Correlation
    ax2 = plt.subplot(1, 2, 2, projection='polar')
    theta_rad = np.radians(df['Angle_deg'])
    ax2.plot(theta_rad, df['NCC'], color='#d62728', linewidth=2.5, label='NCC vs Raycast')
    ax2.set_rmin(0.90)
    ax2.set_rmax(1.00)
    ax2.set_title('Polar Distribution of NCC\n(Isotropic Angular Consistency)', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='lower left', bbox_to_anchor=(0.1, -0.15), frameon=True)

    plt.tight_layout()
    fig2_path = out_dir / "fig2_rotational_fidelity.png"
    fig.savefig(fig2_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 2 to: {fig2_path}")


if __name__ == "__main__":
    main()
