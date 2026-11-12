import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.dicom_loader import create_spine_phantom, hu_to_attenuation
from fvr.registration_3d2d import Spine3D2DRegistrationEngine


def main():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  EXPERIMENT 5: Optimization Landscape & Capture Range Analysis")
    print("=" * 65)

    size = 128
    vol_hu, _ = create_spine_phantom(size=size)
    vol_mu = hu_to_attenuation(vol_hu)

    renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
    engine = Spine3D2DRegistrationEngine(renderer, view='lateral')

    # Target pose: zero
    target_drr = renderer.render_drr(yaw_deg=0.0, tx=0.0, ty=0.0, view='lateral', crop_to_roi=True)
    f_target = engine.prepare_target_spectrum(target_drr)

    # Grid of yaw angles and tx translations
    yaw_range = np.linspace(-25.0, 25.0, 41)
    tx_range = np.linspace(-25.0, 25.0, 41)
    YAW, TX = np.meshgrid(yaw_range, tx_range)
    cost_surface = np.zeros_like(YAW)

    print(f"Sampling {cost_surface.size} points on the 2D energy landscape...")
    for i in range(YAW.shape[0]):
        for j in range(YAW.shape[1]):
            f_cand = renderer.extract_frequency_slice(
                yaw_deg=float(YAW[i, j]), tx=float(TX[i, j]), view='lateral'
            )
            ncc = renderer.compute_spectral_ncc(f_cand, f_target)
            cost_surface[i, j] = 1.0 - ncc

    print("Energy landscape computation complete.")

    # Publication Figure 5: 2D Contour and 3D Surface
    fig = plt.figure(figsize=(15, 6), dpi=300)

    # Panel A: 2D Contour with Optimization Gradient Vector Flow
    ax1 = fig.add_subplot(1, 2, 1)
    cp = ax1.contourf(YAW, TX, cost_surface, levels=30, cmap='viridis')
    cbar = plt.colorbar(cp, ax=ax1)
    cbar.set_label('Objective Loss: 1 - NCC', fontsize=11, fontweight='bold')
    ax1.plot(0, 0, 'r*', markersize=14, label='Global Optimum (0°, 0 mm)')
    ax1.set_xlabel('Yaw Rotation Error Δθ (Degrees)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('In-Plane Translation Error Δt (mm)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) 2D Capture Range Contour Landscape', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Panel B: 3D Surface Plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(YAW, TX, cost_surface, cmap='plasma', edgecolor='none', alpha=0.85)
    ax2.set_xlabel('Δθ (deg)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Δt (mm)', fontsize=10, fontweight='bold')
    ax2.set_zlabel('1 - NCC', fontsize=10, fontweight='bold')
    ax2.set_title('(B) 3D Convex Basin of Attraction', fontsize=12, fontweight='bold')
    ax2.view_init(elev=28, azim=-125)

    plt.tight_layout()
    fig5_path = out_dir / "fig5_energy_landscape.png"
    fig.savefig(fig5_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 5 to: {fig5_path}")


if __name__ == "__main__":
    main()
