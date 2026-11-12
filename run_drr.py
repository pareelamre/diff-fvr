import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fvr import (
    FourierVolumeRenderer,
    raycast_drr,
    load_dicom_series,
    create_spine_phantom,
    hu_to_attenuation
)
from fvr.raycasting_drr import raycast_view


def resample_to_isotropic(volume: np.ndarray, spacing: tuple, target_spacing: float = 1.0) -> np.ndarray:
    """
    Resample anisotropic CT volume (e.g. thick slices dz >> dx) to isotropic voxel spacing
    and center-pad along Z to ensure a uniform cubic field of view without anatomical distortion.
    """
    dz, dy, dx = spacing
    zoom_factors = (dz / target_spacing, dy / target_spacing, dx / target_spacing)
    # If dimensions become too large (e.g. > 256), scale target spacing
    current_out_shape = [int(round(s * z)) for s, z in zip(volume.shape, zoom_factors)]
    max_dim = max(current_out_shape)
    if max_dim > 256:
        scale = 256.0 / max_dim
        zoom_factors = tuple(z * scale for z in zoom_factors)

    print(f"      Resampling volume from shape {volume.shape} with zoom factors {zoom_factors}...")
    resampled = zoom(volume, zoom_factors, order=1)
    
    # Center-pad along Z so that coronal and sagittal views match the transverse field of view
    nz, ny, nx = resampled.shape
    max_side = max(nz, ny, nx)
    pad_z1 = (max_side - nz) // 2
    pad_z2 = max_side - nz - pad_z1
    padded = np.pad(resampled, ((pad_z1, pad_z2), (0, 0), (0, 0)), mode='constant', constant_values=-1000.0)
    print(f"      Isotropic cubic volume shape: {padded.shape}")
    return padded.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate DRRs from Real DICOM or Synthetic CT via FVR")
    parser.add_argument("--dicom_dir", type=str, default="data/dicom", help="Path to directory containing DICOM series")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print("=" * 65)
    print("  AUDITED RUN: Real Clinical DICOM CT Evaluation via FVR")
    print("=" * 65)

    is_real_dicom = False
    if os.path.isdir(args.dicom_dir) and any(f.endswith('.dcm') for f in os.listdir(args.dicom_dir)):
        print(f"\n[1/4] Found real DICOM files in: '{args.dicom_dir}'")
        volume_hu, spacing = load_dicom_series(args.dicom_dir)
        is_real_dicom = True
        print(f"      Source: REAL CLINICAL DICOM SERIES")
        print(f"      Original Dimensions: {volume_hu.shape} (Z, Y, X)")
        print(f"      Voxel Spacing (mm):   dz={spacing[0]:.2f}, dy={spacing[1]:.2f}, dx={spacing[2]:.2f}")
        print(f"      CT Hounsfield Units:  min={volume_hu.min():.1f}, max={volume_hu.max():.1f}")
        
        # Resample to isotropic grid for accurate geometric projections
        volume_hu = resample_to_isotropic(volume_hu, spacing)
    else:
        print(f"\n[1/4] No DICOM files found in '{args.dicom_dir}'. Using synthetic spine phantom.")
        volume_hu, spacing = create_spine_phantom(size=128)

    volume_mu = hu_to_attenuation(volume_hu)

    # 2. Precompute 3D FFT
    print(f"\n[2/4] Precomputing 3D FFT with 2x zero-padding and spatial premultiplication...")
    t0 = time.time()
    renderer = FourierVolumeRenderer(volume_mu, pad_factor=2.0, apply_premultiplication=True)
    t_fvr_prep = time.time() - t0
    print(f"      3D FFT preprocessing completed in {t_fvr_prep:.3f} s.")

    # 3. Render Views
    views = [
        ("Coronal (AP View)", "ap", 0.0),
        ("Oblique View (30°)", "ap", 30.0),
        ("Sagittal (Lateral View)", "lateral", 0.0),
        ("Axial (Transverse View)", "axial", 0.0)
    ]

    fig, axes = plt.subplots(2, len(views), figsize=(16, 8))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    print("\n[3/4] Rendering projections (FVR vs Raycasting Ground Truth)...")
    for idx, (title, view_name, rot_deg) in enumerate(views):
        # FVR
        t_start = time.time()
        fvr_drr = renderer.render_view(view=view_name, rotation_deg=rot_deg)
        t_fvr = time.time() - t_start

        # Raycasting Baseline (Ground Truth)
        t_start = time.time()
        ray_drr = raycast_view(volume_mu, view=view_name, rotation_deg=rot_deg)
        t_ray = time.time() - t_start

        # Normalize for visualization
        def norm(img):
            p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
            return np.clip((img - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)

        fvr_n = norm(fvr_drr)
        ray_n = norm(ray_drr)

        axes[0, idx].imshow(fvr_n, cmap='bone', origin='lower')
        axes[0, idx].set_title(f"FVR: {title}\n{t_fvr*1000:.1f} ms", fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')

        axes[1, idx].imshow(ray_n, cmap='bone', origin='lower')
        axes[1, idx].set_title(f"Raycast GT: {title}\n{t_ray*1000:.1f} ms", fontsize=11)
        axes[1, idx].axis('off')

        print(f"      {title:<24} | FVR: {t_fvr*1000:5.1f} ms | Raycast: {t_ray*1000:5.1f} ms")

    fig.text(0.08, 0.72, 'Fourier Volume\nRendering (FVR)', va='center', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.08, 0.28, 'Raycasting\n(Ground Truth)', va='center', ha='center', fontsize=12, fontweight='bold')

    dataset_tag = "real_clinical_dicom" if is_real_dicom else "synthetic_phantom"
    out_file = os.path.join(args.save_dir, f"{dataset_tag}_drr_results.png")
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[4/4] Saved audited result figure to: {out_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
