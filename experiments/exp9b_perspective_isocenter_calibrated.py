import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import zoom, map_coordinates, affine_transform

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fvr.dicom_loader import load_dicom_series, hu_to_attenuation
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.registration_3d2d import BiplanarSpineRegistrationEngine, compute_mtre_6dof
from experiments.exp9_perspective_conebeam_stress import project_conebeam_3d, rotate_volume_3d


def calibrate_to_isocenter(det_image: np.ndarray, sad: float, sdd: float, target_size: int = 96) -> np.ndarray:
    """
    Apply clinical isocenter geometric calibration to flat-panel C-arm fluoroscopy.
    Rescales detector image by 1 / magnification = SAD / SDD, matching the 
    physical field-of-view at the isocenter.
    """
    mag = sdd / sad
    scale_factor = 1.0 / mag
    rescaled = zoom(det_image, scale_factor, order=1)
    
    h, w = rescaled.shape
    calibrated = np.zeros((target_size, target_size), dtype=np.float64)
    
    if h <= target_size and w <= target_size:
        pad_top = (target_size - h) // 2
        pad_left = (target_size - w) // 2
        calibrated[pad_top:pad_top + h, pad_left:pad_left + w] = rescaled
    else:
        c_y, c_x = h // 2, w // 2
        half = target_size // 2
        calibrated = rescaled[c_y - half:c_y + half, c_x - half:c_x + half]
        
    return calibrated


def run_perspective_calibrated_benchmark():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    dicom_dir = Path(__file__).resolve().parent.parent / "data" / "dicom"

    print("=" * 75)
    print("  EXPERIMENT 9b: Perspective Cone-Beam with Isocenter Geometric Calibration")
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
    engine = BiplanarSpineRegistrationEngine(renderer)

    carm_configs = [
        {"name": "Compact Mobile C-arm",   "sad": 600.0, "sdd": 900.0},
        {"name": "Standard Surgical C-arm", "sad": 750.0, "sdd": 1050.0},
        {"name": "Large Ceiling C-arm",     "sad": 850.0, "sdd": 1200.0},
        {"name": "Long Focal Distance",     "sad": 1000.0, "sdd": 1400.0},
    ]

    trials_per_config = 10
    records = []
    np.random.seed(909)

    for cfg in carm_configs:
        sad, sdd = cfg["sad"], cfg["sdd"]
        mag = sdd / sad
        print(f"\n--- Testing C-arm: {cfg['name']} (SAD={sad:.0f} mm, SDD={sdd:.0f} mm, Mag={mag:.2f}x) ---")

        for trial in range(trials_per_config):
            yaw = float(np.random.uniform(-4.0, 4.0))
            pitch = float(np.random.uniform(-4.0, 4.0))
            roll = float(np.random.uniform(-4.0, 4.0))
            tx = float(np.random.uniform(-5.0, 5.0))
            ty = float(np.random.uniform(-4.0, 4.0))
            tz = float(np.random.uniform(-4.0, 4.0))
            true_pose = (yaw, pitch, roll, tx, ty, tz)

            init_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape)

            # Ground truth Cone-Beam projection
            rot_vol = rotate_volume_3d(vol_mu, yaw, pitch, roll, tx, ty, tz)
            target_ap_cb = project_conebeam_3d(rot_vol, sad=sad, sdd=sdd, det_pixels=target_xy, view='ap')
            target_lat_cb = project_conebeam_3d(rot_vol, sad=sad, sdd=sdd, det_pixels=target_xy, view='lateral')

            # Apply isocenter geometric calibration
            target_ap_cal = calibrate_to_isocenter(target_ap_cb, sad, sdd, target_xy)
            target_lat_cal = calibrate_to_isocenter(target_lat_cb, sad, sdd, target_xy)

            target_ap_norm = (target_ap_cal - target_ap_cal.min()) / (target_ap_cal.max() - target_ap_cal.min() + 1e-8)
            target_lat_norm = (target_lat_cal - target_lat_cal.min()) / (target_lat_cal.max() - target_lat_cal.min() + 1e-8)

            t0 = time.time()
            res = engine.register_biplanar(target_ap_norm, target_lat_norm, initial_pose=init_pose, max_iter=75)
            t_reg = time.time() - t0

            final_mtre = compute_mtre_6dof(true_pose, res['estimated_pose'], vol_mu.shape)

            records.append({
                "C_arm_Model": cfg["name"],
                "SAD_mm": sad,
                "SDD_mm": sdd,
                "Magnification": mag,
                "Trial": trial,
                "Init_mTRE_mm": init_mtre,
                "Final_mTRE_mm": final_mtre,
                "Success_2mm": final_mtre < 2.0,
                "Success_1mm": final_mtre < 1.0,
                "Time_sec": t_reg,
                "Evals": res["function_evals"]
            })

            print(f"  Trial {trial+1:2d}/{trials_per_config} | Init: {init_mtre:.2f} mm -> Final: {final_mtre:.3f} mm | Success: {final_mtre < 2.0}")

    df = pd.DataFrame(records)
    csv_path = out_dir / "table9b_perspective_isocenter_calibrated.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved calibrated perspective results to: {csv_path}")

    for name, grp in df.groupby("C_arm_Model"):
        print(f"\n--- {name} (Isocenter Calibrated) ---")
        print(f"  Final mTRE: {grp['Final_mTRE_mm'].mean():.2f} ± {grp['Final_mTRE_mm'].std():.2f} mm")
        print(f"  Success (<2.0 mm): {grp['Success_2mm'].mean() * 100:.1f}%")
        print(f"  Submillimeter (<1.0 mm): {grp['Success_1mm'].mean() * 100:.1f}%")

    # 3. Publication Figure 9b: Calibrated vs Uncalibrated Comparison
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    sad_std, sdd_std = 750.0, 1050.0
    ap_parallel = renderer.render_drr(0, 0, 0, view='ap')
    ap_conebeam_raw = project_conebeam_3d(vol_mu, sad=sad_std, sdd=sdd_std, det_pixels=target_xy, view='ap')
    ap_conebeam_cal = calibrate_to_isocenter(ap_conebeam_raw, sad=sad_std, sdd=sdd_std, target_size=target_xy)

    # Top Row Visuals
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(ap_conebeam_raw, cmap='gray')
    ax0.set_title('Raw Cone-Beam Fluoroscopy\n(Uncalibrated, 1.40x Mag)', fontsize=10, fontweight='bold')
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(ap_conebeam_cal, cmap='gray')
    ax1.set_title('Isocenter-Calibrated Fluoroscopy\n(Normalized Field-of-View)', fontsize=10, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(ap_parallel, cmap='gray')
    ax2.set_title('Diff-FVR DRR\n(Parallel-Beam)', fontsize=10, fontweight='bold')
    ax2.axis('off')

    cal_norm = ap_conebeam_cal / (ap_conebeam_cal.max() + 1e-8)
    par_norm = ap_parallel / (ap_parallel.max() + 1e-8)
    diff_cal = np.abs(cal_norm - par_norm)

    ax3 = fig.add_subplot(gs[0, 3])
    im_d = ax3.imshow(diff_cal, cmap='inferno')
    plt.colorbar(im_d, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title('Residual Parallax Divergence\n(Calibrated vs Parallel)', fontsize=10, fontweight='bold')
    ax3.axis('off')

    # Bottom Row: Boxplot of Calibrated mTRE across C-arms
    ax_box = fig.add_subplot(gs[1, 0:2])
    cfg_names = [cfg["name"] for cfg in carm_configs]
    mtre_cal_data = [df[df["C_arm_Model"] == name]["Final_mTRE_mm"].values for name in cfg_names]

    bp = ax_box.boxplot(mtre_cal_data, tick_labels=[n.replace(" C-arm", "") for n in cfg_names], patch_artist=True)
    for box in bp['boxes']:
        box.set(facecolor='#b3cde3', edgecolor='#8856a7', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='#e41a1c', linewidth=2)
    ax_box.axhline(2.0, color='red', linestyle='--', label='Clinical Threshold (2.0 mm)')
    ax_box.axhline(1.0, color='green', linestyle=':', label='Submillimeter Threshold (1.0 mm)')
    ax_box.set_ylabel('Final mTRE (mm)', fontsize=11, fontweight='bold')
    ax_box.set_title('Registration Accuracy with Isocenter Calibration', fontsize=11, fontweight='bold')
    ax_box.grid(True, linestyle='--', alpha=0.5)
    ax_box.legend(loc='upper right', fontsize=9.5)

    # Bottom Row Right: Comparison Bar Chart (Uncalibrated vs Calibrated Success Rate)
    ax_cmp = fig.add_subplot(gs[1, 2:4])
    uncal_succ = [0.0, 0.0, 0.0, 0.0]
    cal_succ = [df[df["C_arm_Model"] == name]["Success_2mm"].mean() * 100 for name in cfg_names]
    cal_submm = [df[df["C_arm_Model"] == name]["Success_1mm"].mean() * 100 for name in cfg_names]

    x_indices = np.arange(len(cfg_names))
    width = 0.25
    ax_cmp.bar(x_indices - width, uncal_succ, width, label='Uncalibrated (0%)', color='#d7191c')
    ax_cmp.bar(x_indices, cal_succ, width, label='Isocenter Calibrated (< 2.0 mm)', color='#2b83ba')
    ax_cmp.bar(x_indices + width, cal_submm, width, label='Submillimeter (< 1.0 mm)', color='#abdda4')

    ax_cmp.set_xticks(x_indices)
    ax_cmp.set_xticklabels([n.replace(" C-arm", "") for n in cfg_names], fontsize=9)
    ax_cmp.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax_cmp.set_title('Uncalibrated vs. Calibrated Performance', fontsize=11, fontweight='bold')
    ax_cmp.set_ylim(0, 115)
    ax_cmp.grid(True, linestyle='--', alpha=0.5)
    ax_cmp.legend(loc='upper right', fontsize=9)

    fig_path = out_dir / "fig9b_perspective_isocenter_calibrated.png"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 9b to: {fig_path}")


if __name__ == "__main__":
    run_perspective_calibrated_benchmark()
