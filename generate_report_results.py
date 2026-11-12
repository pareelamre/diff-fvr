import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fvr import (
    FourierVolumeRenderer,
    raycast_drr,
    create_spine_phantom,
    hu_to_attenuation
)


def main():
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("Generating high-resolution publication-quality DRR results...")
    vol_size = 256
    volume_hu, spacing = create_spine_phantom(size=vol_size)
    volume_mu = hu_to_attenuation(volume_hu)

    # Precompute FVR (3D FFT)
    t0 = time.time()
    renderer = FourierVolumeRenderer(volume_mu, pad_factor=2.0, apply_premultiplication=True)
    t_fvr_prep = time.time() - t0

    # 1. Multi-Angle Projection Montage
    angles = [0, 30, 45, 60, 90]
    fig, axes = plt.subplots(2, len(angles), figsize=(16, 7))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    fvr_times = []
    ray_times = []

    for idx, angle in enumerate(angles):
        # FVR
        t_start = time.time()
        fvr_proj = renderer.render_drr(yaw_deg=float(angle), interp_order=1)
        fvr_time = time.time() - t_start
        fvr_times.append(fvr_time * 1000)

        # Raycasting
        t_start = time.time()
        ray_proj = raycast_drr(volume_mu, yaw_deg=float(angle))
        ray_time = time.time() - t_start
        ray_times.append(ray_time * 1000)

        # Invert intensity for standard X-ray appearance (attenuating bone = dark/white)
        def norm(img):
            p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
            return np.clip((img - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)

        fvr_norm = norm(fvr_proj)
        ray_norm = norm(ray_proj)

        label = f"AP (0°)" if angle == 0 else (f"Lateral (90°)" if angle == 90 else f"Oblique ({angle}°)")

        axes[0, idx].imshow(fvr_norm, cmap='bone', origin='lower')
        axes[0, idx].set_title(f"FVR: {label}\n{fvr_time*1000:.1f} ms", fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')

        axes[1, idx].imshow(ray_norm, cmap='bone', origin='lower')
        axes[1, idx].set_title(f"Raycasting (GT)\n{ray_time*1000:.1f} ms", fontsize=11)
        axes[1, idx].axis('off')

    # Row labels
    fig.text(0.08, 0.72, 'Fourier Volume\nRendering (FVR)', va='center', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.08, 0.28, 'Raycasting\n(Ground Truth)', va='center', ha='center', fontsize=12, fontweight='bold')

    montage_path = out_dir / "fvr_drr_multiangle_montage.png"
    fig.savefig(montage_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {montage_path}")

    # 2. Detailed Lateral Comparison with Line Profile
    lat_fvr = renderer.render_drr(yaw_deg=90.0, interp_order=1)
    lat_ray = raycast_drr(volume_mu, yaw_deg=90.0)

    p1, p99 = np.percentile(lat_fvr, 1), np.percentile(lat_fvr, 99)
    lat_fvr_norm = np.clip((lat_fvr - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    lat_ray_norm = np.clip((lat_ray - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    diff = np.abs(lat_fvr_norm - lat_ray_norm)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.8])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(lat_fvr_norm, cmap='bone', origin='lower')
    ax1.axhline(vol_size // 2, color='cyan', linestyle='--', alpha=0.7, label='Profile line')
    ax1.set_title("Lateral DRR (FVR Algorithm)", fontsize=12, fontweight='bold')
    ax1.axis('off')
    ax1.legend(loc='lower right')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(lat_ray_norm, cmap='bone', origin='lower')
    ax2.set_title("Lateral DRR (Raycasting GT)", fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    im_diff = ax3.imshow(diff, cmap='inferno', origin='lower', vmin=0, vmax=0.15)
    ax3.set_title(f"Error Map (Mean AE: {np.mean(diff):.4f})", fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im_diff, ax=ax3, fraction=0.046, pad=0.04)

    # Line profile across central vertebra
    ax_profile = fig.add_subplot(gs[1, :])
    slice_idx = vol_size // 2
    ax_profile.plot(lat_fvr_norm[slice_idx, :], label='FVR Profile', color='blue', linewidth=2)
    ax_profile.plot(lat_ray_norm[slice_idx, :], label='Raycasting Profile', color='orange', linestyle='--', linewidth=2)
    ax_profile.set_title("Intensity Line Profile Across Vertebral Body and Processes", fontsize=12)
    ax_profile.set_xlabel("Pixel Column (Lateral Depth)", fontsize=11)
    ax_profile.set_ylabel("Normalized Attenuation", fontsize=11)
    ax_profile.legend(loc='upper right')
    ax_profile.grid(True, alpha=0.3)

    plt.tight_layout()
    analysis_path = out_dir / "lateral_drr_fidelity_analysis.png"
    fig.savefig(analysis_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {analysis_path}")

    # Print summary metrics
    print("\n" + "=" * 60)
    print("  SUMMARY BENCHMARK REPORT")
    print("=" * 60)
    print(f"Volume Dimension:       {vol_size} x {vol_size} x {vol_size}")
    print(f"3D FFT Preprocessing:   {t_fvr_prep:.3f} s (one-time)")
    print(f"Average FVR Time:       {np.mean(fvr_times):.2f} ms ({1000/np.mean(fvr_times):.1f} fps)")
    print(f"Average Raycast Time:   {np.mean(ray_times):.2f} ms ({1000/np.mean(ray_times):.1f} fps)")
    print(f"Speedup Factor:         {np.mean(ray_times)/np.mean(fvr_times):.1f}x faster")
    print(f"Mean Absolute Error:    {np.mean(diff):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
