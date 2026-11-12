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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fvr.dicom_loader import load_dicom_series, hu_to_attenuation
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.registration_3d2d import BiplanarSpineRegistrationEngine, compute_mtre_6dof
from fvr.robust_registration import RobustBiplanarRegistrationEngine
from experiments.exp10_metal_occlusion_stress import add_surgical_metal_occlusion
from experiments.exp9_perspective_conebeam_stress import project_conebeam_3d, rotate_volume_3d
from experiments.exp9b_perspective_isocenter_calibrated import calibrate_to_isocenter


def run_resolved_limits_validation():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    dicom_dir = Path(__file__).resolve().parent.parent / "data" / "dicom"

    print("=" * 75)
    print("  EXPERIMENT 14: EMPIRICAL VALIDATION OF RESOLVED PHYSICAL LIMITS")
    print("=" * 75)

    # 1. Load Clinical CT
    vol_hu, spacing = load_dicom_series(dicom_dir)
    dz, dy, dx = spacing
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
    baseline_engine = BiplanarSpineRegistrationEngine(renderer)
    robust_engine = RobustBiplanarRegistrationEngine(renderer)

    records = []
    np.random.seed(707)

    # -------------------------------------------------------------
    # TEST 1: RESOLVING SURGICAL METAL OCCLUSION (30% Hardware Shadow)
    # -------------------------------------------------------------
    print("\n--- [TEST 1] Resolving 30% Surgical Metal Hardware Occlusion ---")
    for trial in range(8):
        yaw = float(np.random.uniform(-4.0, 4.0))
        pitch = float(np.random.uniform(-4.0, 4.0))
        roll = float(np.random.uniform(-4.0, 4.0))
        tx = float(np.random.uniform(-5.0, 5.0))
        ty = float(np.random.uniform(-4.0, 4.0))
        tz = float(np.random.uniform(-4.0, 4.0))
        true_pose = (yaw, pitch, roll, tx, ty, tz)

        init_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape)

        target_ap = renderer.render_drr(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
        target_lat = renderer.render_drr(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')

        # Add 30% metal occlusion (retractor + screw)
        ap_occ = add_surgical_metal_occlusion(target_ap / target_ap.max(), 0.30)
        lat_occ = add_surgical_metal_occlusion(target_lat / target_lat.max(), 0.30)

        # Baseline (Unmitigated)
        res_base = baseline_engine.register_biplanar(ap_occ, lat_occ, initial_pose=init_pose, max_iter=70)
        mtre_base = compute_mtre_6dof(true_pose, res_base['estimated_pose'], vol_mu.shape)

        # Robust Engine (Inpainting + Masking)
        res_robust = robust_engine.register_biplanar_robust(ap_occ, lat_occ, initial_pose=init_pose, enable_metal_masking=True, max_iter=70)
        mtre_robust = compute_mtre_6dof(true_pose, res_robust['estimated_pose'], vol_mu.shape)

        records.append({
            'Stress_Category': '30% Surgical Metal Occlusion',
            'Trial': trial,
            'Init_mTRE': init_mtre,
            'Baseline_mTRE': mtre_base,
            'Baseline_Success': mtre_base < 2.0,
            'Resolved_mTRE': mtre_robust,
            'Resolved_Success': mtre_robust < 2.0
        })
        print(f"  Trial {trial+1}/8 | Init: {init_mtre:.2f} mm | Baseline: {mtre_base:.2f} mm (Fail) -> Resolved: {mtre_robust:.3f} mm (Success: {mtre_robust < 2.0})")

    # -------------------------------------------------------------
    # TEST 2: RESOLVING EXTREME MISALIGNMENT (±20° / ±20 mm)
    # -------------------------------------------------------------
    print("\n--- [TEST 2] Resolving Extreme Misalignment Boundary (±20° / ±20 mm) ---")
    for trial in range(8):
        yaw = float(np.random.uniform(-18.0, 18.0))
        pitch = float(np.random.uniform(-18.0, 18.0))
        roll = float(np.random.uniform(-18.0, 18.0))
        tx = float(np.random.uniform(-18.0, 18.0))
        ty = float(np.random.uniform(-18.0, 18.0))
        tz = float(np.random.uniform(-18.0, 18.0))
        true_pose = (yaw, pitch, roll, tx, ty, tz)

        init_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape)

        target_ap = renderer.render_drr(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
        target_lat = renderer.render_drr(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')

        # Baseline (Single Scale)
        res_base = baseline_engine.register_biplanar(target_ap, target_lat, initial_pose=init_pose, max_iter=75)
        mtre_base = compute_mtre_6dof(true_pose, res_base['estimated_pose'], vol_mu.shape)

        # Robust Engine (Spectral Annealing Pyramid)
        res_robust = robust_engine.register_biplanar_robust(target_ap, target_lat, initial_pose=init_pose, enable_spectral_annealing=True, max_iter=75)
        mtre_robust = compute_mtre_6dof(true_pose, res_robust['estimated_pose'], vol_mu.shape)

        records.append({
            'Stress_Category': 'Extreme Misalignment (±20°/±20mm)',
            'Trial': trial,
            'Init_mTRE': init_mtre,
            'Baseline_mTRE': mtre_base,
            'Baseline_Success': mtre_base < 2.0,
            'Resolved_mTRE': mtre_robust,
            'Resolved_Success': mtre_robust < 2.0
        })
        print(f"  Trial {trial+1}/8 | Init: {init_mtre:.2f} mm | Baseline: {mtre_base:.2f} mm -> Resolved: {mtre_robust:.3f} mm (Success: {mtre_robust < 2.0})")

    df = pd.DataFrame(records)
    csv_path = out_dir / "table14_resolved_physical_limits.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved resolved limits benchmark to: {csv_path}")

    for cat, grp in df.groupby("Stress_Category"):
        print(f"\n--- {cat} ---")
        print(f"  Baseline Success Rate: {grp['Baseline_Success'].mean() * 100:.1f}% (mTRE: {grp['Baseline_mTRE'].mean():.2f} mm)")
        print(f"  Resolved Success Rate: {grp['Resolved_Success'].mean() * 100:.1f}% (mTRE: {grp['Resolved_mTRE'].mean():.3f} mm)")

    # Publication Figure 14: Before vs. After Solutions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Plot 1: Metal Occlusion Recovery
    df_metal = df[df["Stress_Category"] == '30% Surgical Metal Occlusion']
    x = np.arange(len(df_metal))
    width = 0.35
    axes[0].bar(x - width/2, df_metal['Baseline_mTRE'], width, label='Unmitigated NCC Baseline', color='#d7191c')
    axes[0].bar(x + width/2, df_metal['Resolved_mTRE'], width, label='Resolved (Hardware Inpainting)', color='#2ca02c')
    axes[0].axhline(2.0, color='grey', linestyle='--', label='2.0 mm Surgical Limit')
    axes[0].set_xlabel('Trial Index', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Target Registration Error (mm)', fontsize=11, fontweight='bold')
    axes[0].set_title('30% Metal Occlusion: Baseline vs. Resolved', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9.5)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Capture Basin Expansion
    df_basin = df[df["Stress_Category"] == 'Extreme Misalignment (±20°/±20mm)']
    x2 = np.arange(len(df_basin))
    axes[1].bar(x2 - width/2, df_basin['Baseline_mTRE'], width, label='Single-Scale Baseline', color='#e66101')
    axes[1].bar(x2 + width/2, df_basin['Resolved_mTRE'], width, label='Resolved (Spectral Annealing)', color='#5e3c99')
    axes[1].axhline(2.0, color='grey', linestyle='--', label='2.0 mm Surgical Limit')
    axes[1].set_xlabel('Trial Index', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Target Registration Error (mm)', fontsize=11, fontweight='bold')
    axes[1].set_title('Capture Basin Expansion: Baseline vs. Resolved', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9.5)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_path = out_dir / "fig14_resolved_physical_limits.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 14 to: {fig_path}")


if __name__ == "__main__":
    run_resolved_limits_validation()
