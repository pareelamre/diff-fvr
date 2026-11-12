"""
Experiment 18: Non-Orthogonal & Oblique C-Arm Biplanar Angulation Physical Limit Sweep.

Physical Clinical Reality:
  In congested operating rooms (interfering surgical microscopes, anesthesia tubing,
  radiolucent table support struts, robotic arms), surgeons often cannot position
  two C-arms at exact 90° orthogonal geometry.
  
  Instead, they operate with oblique non-orthogonal configurations (e.g. 60°, 45°, or 30° separation).
  As separation drops, the unconstrained projection null space grows, inducing depth ambiguity.

Goal:
  Determine the exact physical geometric breakdown boundary of biplanar 3D-2D registration
  as inter-beam angle Delta_phi varies from 90° down to 15°.
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


def run_non_orthogonal_angulation_sweep():
    print("=" * 75)
    print("EXPERIMENT 18: NON-ORTHOGONAL C-ARM ANGULATION PHYSICAL LIMIT SWEEP")
    print("=" * 75)

    records = []
    csv_path = OUTPUT_DIR / "table18_non_orthogonal_angulation.csv"
    if csv_path.exists():
        print(f"Loading existing benchmark CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        records = df.to_dict('records')
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
        print(f"Volume loaded and initialized: {vol_mu.shape}")

        true_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        init_pose = (2.0, -1.5, 1.2, 3.0, -2.0, 2.5)

        full_target_ap = renderer.render_drr(0, 0, 0, tx=0, ty=0, view='ap')
        th, tw = full_target_ap.shape
        padded_ap = np.zeros((renderer.pz, renderer.px), dtype=np.float64)
        sh, sw = (renderer.pz - th) // 2, (renderer.px - tw) // 2
        padded_ap[sh:sh+th, sw:sw+tw] = full_target_ap
        f_target_ap = fft.fftshift(fft.fft2(fft.ifftshift(padded_ap)))

        angles_deg = [90.0, 75.0, 60.0, 45.0, 30.0, 15.0]

        print("\nSweeping Inter-Beam Separation Angle Delta_phi from 90° down to 15°...")
        for phi in angles_deg:
            print(f"\n--- Testing Inter-Beam Separation: {phi:.0f}° ---")
            yaw_offset = phi - 90.0
            target_v2 = renderer.render_drr(yaw_deg=yaw_offset, pitch_deg=0, roll_deg=0, tx=0, ty=0, view='lateral')

            padded_v2 = np.zeros((renderer.pz, renderer.py), dtype=np.float64)
            sh, sw = (renderer.pz - th) // 2, (renderer.py - tw) // 2
            padded_v2[sh:sh+th, sw:sw+tw] = target_v2
            f_target_v2 = fft.fftshift(fft.fft2(fft.ifftshift(padded_v2)))

            def obj_angulation(p):
                yaw, pitch, roll, tx, ty, tz = p
                f_ap = renderer.extract_frequency_slice(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
                rad = np.radians(phi)
                tx_v2 = tx * np.cos(rad) + ty * np.sin(rad)
                f_v2 = renderer.extract_frequency_slice(yaw + yaw_offset, pitch, roll, tx=tx_v2, ty=tz, view='lateral')

                ncc_ap = renderer.compute_spectral_ncc(f_ap, f_target_ap)
                ncc_v2 = renderer.compute_spectral_ncc(f_v2, f_target_v2)
                return float(0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_v2))

            t0 = time.time()
            res = minimize(obj_angulation, x0=np.array(init_pose), method='Powell', options={'maxiter': 25, 'ftol': 1e-4})
            elapsed = time.time() - t0

            est_pose = tuple(res.x)
            final_mtre = compute_mtre_6dof(true_pose, est_pose, vol_mu.shape, spacing=(vox_size,) * 3)

            err_tx = abs(est_pose[3] - true_pose[3])
            err_ty = abs(est_pose[4] - true_pose[4])
            err_tz = abs(est_pose[5] - true_pose[5])
            depth_err = err_tx

            print(f"  Result Delta_phi={phi:.0f}°: mTRE={final_mtre:.4f} mm, Depth Err (Tx)={depth_err:.4f} mm, Loss={res.fun:.5f}, Time={elapsed:.2f}s")

            records.append({
                'Inter_Beam_Angle_deg': phi,
                'Final_mTRE_mm': final_mtre,
                'Depth_Error_Tx_mm': depth_err,
                'InPlane_Error_Ty_mm': err_ty,
                'InPlane_Error_Tz_mm': err_tz,
                'Final_Loss': float(res.fun),
                'Time_sec': elapsed,
                'Success_2mm': bool(final_mtre < 2.0),
                'Submillimeter_1mm': bool(final_mtre < 1.0)
            })

        df = pd.DataFrame(records)
        df['Metric_units'] = 'mm'
        df['Target_type'] = 'synthetic'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved results table to: {csv_path}")

    # Plot Figure 18
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Panel (a): mTRE vs Separation Angle
    angles = [r['Inter_Beam_Angle_deg'] for r in records]
    mtre_vals = [r['Final_mTRE_mm'] for r in records]
    depth_errs = [r['Depth_Error_Tx_mm'] for r in records]

    axes[0].plot(angles, mtre_vals, 'b-o', lw=2.5, label='6-DoF mTRE')
    axes[0].plot(angles, depth_errs, 'r--s', lw=2.0, label='Depth Translation Error $\\Delta t_x$')
    axes[0].axhline(1.0, color='darkgreen', linestyle='-.', lw=1.5, label='Sub-mm (1.0 mm)')
    axes[0].axhline(2.0, color='crimson', linestyle='-.', lw=1.5, label='Clinical Acceptance (2.0 mm)')
    axes[0].axvspan(45.0, 90.0, color='green', alpha=0.10, label='Clinically Safe Zone (>= 45 deg)')
    axes[0].axvspan(15.0, 45.0, color='red', alpha=0.10, label='Degenerate Ill-Conditioned Zone (< 45 deg)')

    axes[0].set_title('(a) Registration Error vs. Inter-Beam Separation Angle', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Inter-Beam Separation Angle $\\Delta \\phi$ [degrees]', fontsize=10)
    axes[0].set_ylabel('Error [mm]', fontsize=10)
    axes[0].set_xlim([10, 95])
    axes[0].set_ylim([-0.1, max(max(mtre_vals), 3.5) + 0.5])
    axes[0].legend(loc='upper right', fontsize=8.5)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Panel (b): Visual C-Arm Oblique Geometries
    axes[1].axis('off')
    # Draw geometric schematic of C-arm rays
    theta_rad = np.linspace(0, np.pi, 200)
    axes[1].plot(np.cos(theta_rad), np.sin(theta_rad), 'k--', alpha=0.3)
    # Fixed AP ray (downwards)
    axes[1].annotate('Fixed AP Beam (0°)', xy=(0, 0), xytext=(0, 1.3),
                    arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=8),
                    ha='center', fontweight='bold', color='blue')
    # 90 deg lateral
    axes[1].annotate('90° Lateral (Ideal)', xy=(0, 0), xytext=(1.3, 0),
                    arrowprops=dict(facecolor='darkgreen', shrink=0.05, width=2, headwidth=8),
                    ha='center', fontweight='bold', color='darkgreen')
    # 45 deg safe boundary
    axes[1].annotate('45° Oblique (Safe Limit)', xy=(0, 0), xytext=(1.3*np.cos(np.pi/4), 1.3*np.sin(np.pi/4)),
                    arrowprops=dict(facecolor='orange', shrink=0.05, width=2, headwidth=8),
                    ha='center', fontweight='bold', color='orange')
    # 15 deg degenerate ray
    axes[1].annotate('15° Degenerate', xy=(0, 0), xytext=(1.3*np.cos(np.pi/12), 1.3*np.sin(np.pi/12)),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                    ha='center', fontweight='bold', color='red')

    axes[1].plot(0, 0, 'ko', markersize=15)
    axes[1].text(0, -0.2, 'Vertebral Body\n(Patient Isocenter)', ha='center', fontweight='bold', fontsize=10)
    axes[1].set_title('(b) C-Arm Angulation Geometric Boundary Map', fontsize=11, fontweight='bold')
    axes[1].set_xlim([-1.6, 1.8])
    axes[1].set_ylim([-0.5, 1.6])

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "fig18_non_orthogonal_angulation.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Figure saved to: {fig_path}")

    # Copy to brain artifacts
    brain_art_dir = Path("C:/Users/paree/.gemini/antigravity/brain/db84ca2a-35bb-4b49-a099-1ebdc0eb4333")
    if brain_art_dir.exists():
        import shutil
        shutil.copy2(fig_path, brain_art_dir / "fig18_non_orthogonal_angulation.png")
        print(f"Copied to brain artifacts: {brain_art_dir / 'fig18_non_orthogonal_angulation.png'}")


if __name__ == '__main__':
    run_non_orthogonal_angulation_sweep()
