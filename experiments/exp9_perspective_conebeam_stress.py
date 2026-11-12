import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import zoom, map_coordinates, affine_transform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fvr.dicom_loader import resample_isotropic_cube, load_dicom_series, hu_to_attenuation
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.registration_3d2d import BiplanarSpineRegistrationEngine, compute_mtre_6dof


def project_conebeam_3d(volume: np.ndarray, sad: float, sdd: float, det_pixels: int = 96, view: str = 'ap') -> np.ndarray:
    """
    Simulate true C-arm cone-beam perspective line-integral radiograph.
    Source at distance SAD from isocenter; flat-panel detector at SDD from source.
    """
    nz, ny, nx = volume.shape
    u = np.linspace(-0.5, 0.5, det_pixels)
    v = np.linspace(-0.5, 0.5, det_pixels)
    uu, vv = np.meshgrid(u, v)

    proj = np.zeros((det_pixels, det_pixels), dtype=np.float64)

    if view == 'ap':
        # Ray along Y axis: depth steps through ny
        n_depth = ny
        depth_coords = np.linspace(-0.5, 0.5, n_depth)
        for iy in range(n_depth):
            # Physical depth y in mm (assuming isotropic 2.6 mm voxels)
            y_mm = depth_coords[iy] * (ny * 2.604)
            # Divergence factor at this depth plane
            scale = (y_mm + sad) / sdd
            x_idx = uu * scale * nx + nx / 2.0
            z_idx = vv * scale * nz + nz / 2.0

            coords = np.array([z_idx.ravel(), np.full(z_idx.size, iy), x_idx.ravel()])
            slice_vals = map_coordinates(volume, coords, order=1, mode='constant', cval=0.0).reshape((det_pixels, det_pixels))
            proj += slice_vals
    else:
        # Ray along X axis (Lateral): depth steps through nx
        n_depth = nx
        depth_coords = np.linspace(-0.5, 0.5, n_depth)
        for ix in range(n_depth):
            x_mm = depth_coords[ix] * (nx * 2.604)
            scale = (x_mm + sad) / sdd
            y_idx = uu * scale * ny + ny / 2.0
            z_idx = vv * scale * nz + nz / 2.0

            coords = np.array([z_idx.ravel(), y_idx.ravel(), np.full(z_idx.size, ix)])
            slice_vals = map_coordinates(volume, coords, order=1, mode='constant', cval=0.0).reshape((det_pixels, det_pixels))
            proj += slice_vals

    return proj


def rotate_volume_3d(volume: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float, tx: float, ty: float, tz: float) -> np.ndarray:
    """
    Rigidly rotate and translate 3D volume in spatial domain.
    """
    y_rad, p_rad, r_rad = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    rz = np.array([[1, 0, 0], [0, np.cos(y_rad), -np.sin(y_rad)], [0, np.sin(y_rad), np.cos(y_rad)]])
    ry = np.array([[np.cos(r_rad), 0, np.sin(r_rad)], [0, 1, 0], [-np.sin(r_rad), 0, np.cos(r_rad)]])
    rx = np.array([[np.cos(p_rad), -np.sin(p_rad), 0], [np.sin(p_rad), np.cos(p_rad), 0], [0, 0, 1]])
    rot = rz @ ry @ rx

    nz, ny, nx = volume.shape
    center = np.array([nz / 2.0, ny / 2.0, nx / 2.0])
    offset = center - rot @ center + np.array([-tz, -ty, -tx])

    return affine_transform(volume, matrix=rot, offset=offset, order=1, mode='constant', cval=0.0)


def run_perspective_stress_test():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    dicom_dir = Path(__file__).resolve().parent.parent / "data" / "dicom"

    print("=" * 70)
    print("  EXPERIMENT 9: Cone-Beam Perspective vs. Orthographic Stress Test")
    print("=" * 70)

    # 1. Load Clinical CT
    vol_hu, spacing = load_dicom_series(dicom_dir)
    dz, dy, dx = spacing
    phys_z = vol_hu.shape[0] * dz
    phys_x = vol_hu.shape[2] * dx

    target_xy = 96
    vox_size = phys_x / target_xy
    target_z = int(round(phys_z / vox_size))

    vol_cubic, grid_spacing = resample_isotropic_cube(vol_hu, spacing, target_xy)
    vox_size = grid_spacing[0]

    vol_mu = hu_to_attenuation(vol_cubic)
    renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
    engine = BiplanarSpineRegistrationEngine(renderer)

    # 2. C-arm Geometries
    carm_configs = [
        {"name": "Compact Mobile C-arm",   "sad": 600.0, "sdd": 900.0},
        {"name": "Standard Surgical C-arm", "sad": 750.0, "sdd": 1050.0},
        {"name": "Large Ceiling C-arm",     "sad": 850.0, "sdd": 1200.0},
        {"name": "Long Focal Distance",     "sad": 1000.0, "sdd": 1400.0},
    ]

    trials_per_config = 12
    records = []
    np.random.seed(202)

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
            init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape, spacing=(vox_size,) * 3)

            # Rotate volume in 3D and project with TRUE CONE-BEAM PERSPECTIVE
            rot_vol = rotate_volume_3d(vol_mu, yaw, pitch, roll, tx, ty, tz)
            target_ap_cb = project_conebeam_3d(rot_vol, sad=sad, sdd=sdd, det_pixels=target_xy, view='ap')
            target_lat_cb = project_conebeam_3d(rot_vol, sad=sad, sdd=sdd, det_pixels=target_xy, view='lateral')

            # Normalize to match fluoroscopic detector dynamic range
            target_ap_norm = (target_ap_cb - target_ap_cb.min()) / (target_ap_cb.max() - target_ap_cb.min() + 1e-8)
            target_lat_norm = (target_lat_cb - target_lat_cb.min()) / (target_lat_cb.max() - target_lat_cb.min() + 1e-8)

            t0 = time.time()
            res = engine.register_biplanar(target_ap_norm, target_lat_norm, initial_pose=init_pose, max_iter=70)
            t_reg = time.time() - t0

            final_mtre = compute_mtre_6dof(true_pose, res['estimated_pose'], vol_mu.shape, spacing=(vox_size,) * 3)

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
    df['Metric_units'] = 'mm'
    df['Target_type'] = 'synthetic'
    csv_path = out_dir / "table9_perspective_conebeam.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved perspective stress test results to: {csv_path}")

    # Summary
    for name, grp in df.groupby("C_arm_Model"):
        print(f"\n--- {name} ---")
        print(f"  Final mTRE: {grp['Final_mTRE_mm'].mean():.2f} ± {grp['Final_mTRE_mm'].std():.2f} mm")
        print(f"  Success (<2.0 mm): {grp['Success_2mm'].mean() * 100:.1f}%")
        print(f"  Submillimeter (<1.0 mm): {grp['Success_1mm'].mean() * 100:.1f}%")
        print(f"  Mean Time: {grp['Time_sec'].mean():.2f} s")

    # 3. Publication Figure 9: Cone-Beam Perspective Robustness
    fig = plt.figure(figsize=(15, 8), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # Top Row: Perspective vs Parallel Visual Comparison
    ap_parallel = renderer.render_drr(0, 0, 0, view='ap')
    ap_conebeam = project_conebeam_3d(vol_mu, sad=750.0, sdd=1050.0, det_pixels=target_xy, view='ap')

    ax_p = fig.add_subplot(gs[0, 0])
    ax_p.imshow(ap_parallel, cmap='gray')
    ax_p.set_title('Diff-FVR (Parallel-Beam)', fontsize=11, fontweight='bold')
    ax_p.axis('off')

    ax_cb = fig.add_subplot(gs[0, 1])
    ax_cb.imshow(ap_conebeam, cmap='gray')
    ax_cb.set_title('Ground Truth (Cone-Beam Perspective)', fontsize=11, fontweight='bold')
    ax_cb.axis('off')

    diff_map = np.abs(ap_parallel / ap_parallel.max() - ap_conebeam / ap_conebeam.max())
    ax_diff = fig.add_subplot(gs[0, 2])
    im_d = ax_diff.imshow(diff_map, cmap='inferno')
    plt.colorbar(im_d, ax=ax_diff, fraction=0.046, pad=0.04)
    ax_diff.set_title('Residual Geometric Divergence Map', fontsize=11, fontweight='bold')
    ax_diff.axis('off')

    # Bottom Row: Registration CED and Accuracy across C-arm Models
    ax_box = fig.add_subplot(gs[1, 0:2])
    cfg_names = [cfg["name"] for cfg in carm_configs]
    mtre_data = [df[df["C_arm_Model"] == name]["Final_mTRE_mm"].values for name in cfg_names]

    bp = ax_box.boxplot(mtre_data, tick_labels=cfg_names, patch_artist=True)
    for box in bp['boxes']:
        box.set(facecolor='#d9f0d3', edgecolor='#1b7837', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='#e41a1c', linewidth=2)
    ax_box.axhline(2.0, color='red', linestyle='--', label='Clinical Threshold (2.0 mm)')
    ax_box.axhline(1.0, color='green', linestyle=':', label='Submillimeter Threshold (1.0 mm)')
    ax_box.set_ylabel('Final mTRE (mm)', fontsize=11, fontweight='bold')
    ax_box.set_title('Registration Accuracy under Cone-Beam Distortion across C-Arm Geometries', fontsize=11, fontweight='bold')
    ax_box.grid(True, linestyle='--', alpha=0.5)
    ax_box.legend(loc='upper right', fontsize=9.5)

    ax_ced = fig.add_subplot(gs[1, 2])
    th_range = np.linspace(0.0, 4.0, 50)
    for name, grp in df.groupby("C_arm_Model"):
        rates = [np.mean(grp['Final_mTRE_mm'] <= t) * 100 for t in th_range]
        ax_ced.plot(th_range, rates, linewidth=2, label=name.split()[0])
    ax_ced.axvline(2.0, color='grey', linestyle=':')
    ax_ced.set_xlabel('mTRE Threshold (mm)', fontsize=10, fontweight='bold')
    ax_ced.set_ylabel('Success Rate (%)', fontsize=10, fontweight='bold')
    ax_ced.set_title('Cone-Beam CED Curves', fontsize=11, fontweight='bold')
    ax_ced.grid(True, linestyle='--', alpha=0.5)
    ax_ced.legend(loc='lower right', fontsize=8.5)

    fig_path = out_dir / "fig9_perspective_conebeam_stress.png"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 9 to: {fig_path}")


if __name__ == "__main__":
    run_perspective_stress_test()
