import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.dicom_loader import create_spine_phantom, hu_to_attenuation
from fvr.raycasting_drr import raycast_drr


def main():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  EXPERIMENT 3: Comprehensive Ablation Studies")
    print("  (Zero-Padding, Spatial Premultiplication, Interpolation Order)")
    print("=" * 65)

    size = 128
    vol_hu, _ = create_spine_phantom(size=size)
    vol_mu = hu_to_attenuation(vol_hu)
    gt_drr = raycast_drr(vol_mu, yaw_deg=45.0)
    p1, p99 = np.percentile(gt_drr, 0.5), np.percentile(gt_drr, 99.5)
    gt_norm = np.clip((gt_drr - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)

    # 1. Ablation A: Zero-Padding Factor
    pad_factors = [1.0, 1.25, 1.5, 2.0, 2.5]
    pad_results = []
    print("\n[Ablation 1] Evaluating Zero-Padding Factors...")
    for pf in pad_factors:
        r = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=pf, apply_premultiplication=True)
        img = r.render_drr(yaw_deg=45.0, view='lateral', crop_to_roi=True)
        img_norm = np.clip((img - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
        psnr = compute_psnr(gt_norm, img_norm, data_range=1.0)
        mae = np.mean(np.abs(gt_norm - img_norm))
        print(f"  Padding {pf:.2f}x | PSNR: {psnr:.2f} dB | MAE: {mae:.4f}")
        pad_results.append({'Pad_Factor': pf, 'PSNR_dB': psnr, 'MAE': mae})

    df_pad = pd.DataFrame(pad_results)

    # 2. Ablation B: Spatial Premultiplication (With vs Without)
    print("\n[Ablation 2] Evaluating Spatial Premultiplication...")
    r_with = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0, apply_premultiplication=True)
    r_without = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0, apply_premultiplication=False)

    img_with = r_with.render_drr(yaw_deg=90.0, view='lateral')
    img_without = r_without.render_drr(yaw_deg=90.0, view='lateral')

    img_with_n = np.clip((img_with - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    img_without_n = np.clip((img_without - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)

    # Horizontal profiles across spine center
    mid_row = size // 2
    prof_gt = gt_norm[mid_row, :]
    prof_with = img_with_n[mid_row, :]
    prof_without = img_without_n[mid_row, :]

    # 3. Ablation C: Interpolation Order
    print("\n[Ablation 3] Evaluating Interpolation Order (Nearest, Linear, Cubic)...")
    interp_results = []
    for order, name in [(0, 'Nearest (k=0)'), (1, 'Trilinear (k=1)'), (3, 'Tricubic Spline (k=3)')]:
        img_ord = r_with.render_drr(yaw_deg=45.0, view='lateral', interp_order=order)
        img_ord_n = np.clip((img_ord - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
        psnr_ord = compute_psnr(gt_norm, img_ord_n, data_range=1.0)
        interp_results.append({'Order': order, 'Name': name, 'PSNR': psnr_ord})
        print(f"  {name:<22} | PSNR: {psnr_ord:.2f} dB")

    df_interp = pd.DataFrame(interp_results)

    # Save summary table
    df_pad.to_csv(out_dir / "table3_padding_ablation.csv", index=False)

    # Publication Figure 3: Comprehensive Ablation Multi-Panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # Panel 1: Padding vs PSNR
    axes[0].plot(df_pad['Pad_Factor'], df_pad['PSNR_dB'], 'o-', color='#1f77b4', linewidth=2.5, markersize=8)
    axes[0].axvline(2.0, color='red', linestyle='--', label='Malzbender Optimum (2.0x)')
    axes[0].set_xlabel('Zero-Padding Factor Z_p', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('PSNR (dB)', fontsize=11, fontweight='bold')
    axes[0].set_title('(A) Aliasing vs. Zero-Padding Factor', fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Spatial Premultiplication Profiles
    axes[1].plot(prof_with, label='With Premultiplication (Proposed)', color='blue', linewidth=2)
    axes[1].plot(prof_without, label='Without Premultiplication (Dimmed Edges)', color='orange', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Transverse Pixel Position', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Normalized Intensity', fontsize=11, fontweight='bold')
    axes[1].set_title('(B) Sinc² Rolloff Compensation', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Panel 3: Interpolation Order Bar Chart
    bars = axes[2].bar(df_interp['Name'], df_interp['PSNR'], color=['#9467bd', '#2ca02c', '#17becf'], width=0.55, edgecolor='black')
    axes[2].set_ylabel('PSNR vs Raycast GT (dB)', fontsize=11, fontweight='bold')
    axes[2].set_title('(C) Interpolation Kernel Order', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, max(df_interp['PSNR']) * 1.2)
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.5)
    for b in bars:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width() / 2.0, h + 0.6, f"{h:.2f} dB", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig3_path = out_dir / "fig3_ablation_analysis.png"
    fig.savefig(fig3_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 3 to: {fig3_path}")


if __name__ == "__main__":
    main()
