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


def run_capture_basin_sweep():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    dicom_dir = Path(__file__).resolve().parent.parent / "data" / "dicom"

    print("=" * 70)
    print("  EXPERIMENT 8: Capture Basin & Initial Misalignment Stress Test")
    print("=" * 70)

    # 1. Load Clinical CT and resample to isotropic cubic volume
    print(f"Loading clinical DICOM series from: {dicom_dir}")
    vol_hu, spacing = load_dicom_series(dicom_dir)
    dz, dy, dx = spacing

    phys_z = vol_hu.shape[0] * dz
    phys_y = vol_hu.shape[1] * dy
    phys_x = vol_hu.shape[2] * dx

    target_xy = 96
    vox_size = phys_x / target_xy
    target_z = int(round(phys_z / vox_size))

    zz = target_z / vol_hu.shape[0]
    zy = target_xy / vol_hu.shape[1]
    zx = target_xy / vol_hu.shape[2]
    vol_cubic, grid_spacing = resample_isotropic_cube(vol_hu, spacing, target_xy)
    vox_size = grid_spacing[0]

    vol_mu = hu_to_attenuation(vol_cubic)
    renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
    engine = BiplanarSpineRegistrationEngine(renderer)

    # 2. Perturbation Levels (Rotations in deg, Translations in mm)
    perturbation_levels = [
        {"name": "2° / 2 mm",   "rot": 2.0,  "trans": 2.0},
        {"name": "5° / 5 mm",   "rot": 5.0,  "trans": 5.0},
        {"name": "8° / 8 mm",   "rot": 8.0,  "trans": 8.0},
        {"name": "12° / 12 mm", "rot": 12.0, "trans": 12.0},
        {"name": "16° / 16 mm", "rot": 16.0, "trans": 16.0},
        {"name": "20° / 20 mm", "rot": 20.0, "trans": 20.0},
    ]

    trials_per_level = 10
    records = []
    np.random.seed(42)

    for level in perturbation_levels:
        rot_max = level["rot"]
        trans_max = level["trans"] / vox_size  # Low-level poses are in voxels.
        print(f"\n--- Testing Perturbation Level: {level['name']} ({trials_per_level} trials) ---")

        for trial in range(trials_per_level):
            yaw = float(np.random.uniform(-rot_max, rot_max))
            pitch = float(np.random.uniform(-rot_max, rot_max))
            roll = float(np.random.uniform(-rot_max, rot_max))
            tx = float(np.random.uniform(-trans_max, trans_max))
            ty = float(np.random.uniform(-trans_max, trans_max))
            tz = float(np.random.uniform(-trans_max, trans_max))
            true_pose = (yaw, pitch, roll, tx, ty, tz)

            init_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape, spacing=(vox_size,) * 3)

            target_ap = renderer.render_drr(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
            target_lat = renderer.render_drr(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')

            t0 = time.time()
            res = engine.register_biplanar(target_ap, target_lat, initial_pose=init_pose, max_iter=80)
            t_reg = time.time() - t0

            final_mtre = compute_mtre_6dof(true_pose, res['estimated_pose'], vol_mu.shape, spacing=(vox_size,) * 3)

            records.append({
                "Level": level["name"],
                "Max_Rot_deg": rot_max,
                "Max_Trans_mm": level["trans"],
                "Trial": trial,
                "Init_mTRE_mm": init_mtre,
                "Final_mTRE_mm": final_mtre,
                "Success_2mm": final_mtre < 2.0,
                "Success_1mm": final_mtre < 1.0,
                "Time_sec": t_reg,
                "Evals": res["function_evals"],
                "Time_per_eval_ms": res["time_per_eval_ms"]
            })

            print(f"  Trial {trial+1:2d}/{trials_per_level} | Init: {init_mtre:.2f} mm -> Final: {final_mtre:.3f} mm | Time: {t_reg:.2f} s")

    df = pd.DataFrame(records)
    df['Metric_units'] = 'mm'
    df['Target_type'] = 'synthetic'
    csv_path = out_dir / "table8_capture_basin.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved capture basin benchmark to: {csv_path}")

    # Summary Statistics
    summary = df.groupby("Level").agg({
        "Init_mTRE_mm": ["mean", "std"],
        "Final_mTRE_mm": ["mean", "median", "std"],
        "Success_2mm": lambda x: np.mean(x) * 100.0,
        "Success_1mm": lambda x: np.mean(x) * 100.0,
        "Time_sec": "mean"
    })
    print("\nSummary Table:")
    print(summary)

    # 3. Publication Figure 8: Capture Basin Analysis
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    levels_order = [lvl["name"] for lvl in perturbation_levels]
    rot_vals = [lvl["rot"] for lvl in perturbation_levels]

    success_2mm_rates = [df[df["Level"] == lvl]["Success_2mm"].mean() * 100 for lvl in levels_order]
    success_1mm_rates = [df[df["Level"] == lvl]["Success_1mm"].mean() * 100 for lvl in levels_order]

    # Panel A: Clinical Success Rate vs Initial Perturbation
    axes[0].plot(rot_vals, success_2mm_rates, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='Success (< 2.0 mm)')
    axes[0].plot(rot_vals, success_1mm_rates, 's--', color='#2ca02c', linewidth=2.5, markersize=8, label='Submillimeter (< 1.0 mm)')
    axes[0].axhline(80.0, color='grey', linestyle=':', label='80% Acceptable Threshold')
    axes[0].set_xlabel('Initial Perturbation Bound (±deg / ±mm)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('(A) Clinical Success vs. Capture Range', fontsize=12, fontweight='bold')
    axes[0].set_ylim(-5, 105)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='lower left', fontsize=9.5)

    # Panel B: Final mTRE Distribution Boxplot
    mtre_by_level = [df[df["Level"] == lvl]["Final_mTRE_mm"].values for lvl in levels_order]
    bp = axes[1].boxplot(mtre_by_level, tick_labels=[lvl["name"] for lvl in perturbation_levels], patch_artist=True)
    for box in bp['boxes']:
        box.set(facecolor='#aec7e8', edgecolor='#1f77b4', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='#d62728', linewidth=2)
    axes[1].axhline(2.0, color='red', linestyle='--', linewidth=1.5, label='Clinical Limit (2.0 mm)')
    axes[1].axhline(1.0, color='green', linestyle=':', linewidth=1.5, label='Sub-mm Limit (1.0 mm)')
    axes[1].set_xlabel('Initial Misalignment Level', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Final mTRE (mm)', fontsize=11, fontweight='bold')
    axes[1].set_title('(B) Registration Error Distribution', fontsize=12, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper left', fontsize=9.5)

    # Panel C: Initial vs Final mTRE Scatter with Identity Line
    scatter = axes[2].scatter(df["Init_mTRE_mm"], df["Final_mTRE_mm"], c=df["Max_Rot_deg"], cmap='viridis', s=45, alpha=0.85, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=axes[2])
    cbar.set_label('Initial Rotation Bound (±deg)', fontsize=10, fontweight='bold')
    axes[2].axhline(2.0, color='red', linestyle='--', label='2.0 mm Threshold')
    axes[2].set_xlabel('Initial mTRE (mm)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Final mTRE (mm)', fontsize=11, fontweight='bold')
    axes[2].set_title('(C) Error Reduction Convergence', fontsize=12, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[2].legend(loc='lower right', fontsize=9.5)

    fig_path = out_dir / "fig8_capture_basin_sweep.png"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 8 to: {fig_path}")


if __name__ == "__main__":
    run_capture_basin_sweep()
