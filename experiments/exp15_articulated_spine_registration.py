"""
Experiment 15: Articulated Multi-Vertebra Kinematic Spine Registration Benchmark.

Evaluates registration of an articulated 3-level lumbar spine model (L3-L4-L5)
under intraoperative spinal flexion/bending on a surgical table.

Compares:
  1. Monolithic Rigid Registration (Standard assumption: treats spine as a single solid body).
  2. Proposed Articulated Kinematic Engine (Hierarchical 2-stage global-to-local joint refinement).
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fvr.registration_3d2d import compute_mtre_6dof
from src.fvr.articulated_spine import VertebralSegment, ArticulatedSpineModel
from src.fvr.dicom_loader import resample_isotropic_cube, load_dicom_series, hu_to_attenuation

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_articulated_spine_benchmark():
    print("=" * 75)
    print("EXPERIMENT 15: ARTICULATED MULTI-VERTEBRA SPINE REGISTRATION")
    print("=" * 75)

    # Load hospital DICOM and resample to isotropic 96x96x96 matrix
    dicom_dir = PROJECT_ROOT / "data" / "dicom"
    vol_hu, spacing = load_dicom_series(dicom_dir)
    print(f"Raw DICOM volume: {vol_hu.shape}, spacing: {spacing}")

    target_xy = 96
    phys_z = vol_hu.shape[0] * spacing[0]
    phys_x = vol_hu.shape[2] * spacing[2]
    vox_size = phys_x / target_xy
    target_z = int(round(phys_z / vox_size))

    vol_cubic, grid_spacing = resample_isotropic_cube(vol_hu, spacing, target_xy)
    vox_size = grid_spacing[0]
    volume = hu_to_attenuation(vol_cubic)
    print(f"Resampled isotropic cubic spine volume: {volume.shape}")

    nz, ny, nx = volume.shape

    # Segment into 3 vertebral levels along cranial-caudal axis (Z)
    z_step = nz // 3
    seg_l5_vol = np.zeros_like(volume)
    seg_l4_vol = np.zeros_like(volume)
    seg_l3_vol = np.zeros_like(volume)

    seg_l5_vol[:z_step] = volume[:z_step]
    seg_l4_vol[z_step:2*z_step] = volume[z_step:2*z_step]
    seg_l3_vol[2*z_step:] = volume[2*z_step:]

    seg_l5 = VertebralSegment("L5", seg_l5_vol, nominal_offset_z=0.0)
    seg_l4 = VertebralSegment("L4", seg_l4_vol, nominal_offset_z=float(z_step))
    seg_l3 = VertebralSegment("L3", seg_l3_vol, nominal_offset_z=float(2 * z_step))

    spine_model = ArticulatedSpineModel([seg_l5, seg_l4, seg_l3])
    print("Articulated 3-level spine model initialized (L5-L4-L3).")

    # True Ground Truth In-Vivo Deformation:
    # Base pose (L5): yaw=0, pitch=0, roll=0, tx=0, ty=0, tz=0
    gt_base_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Inter-vertebral flexion (patient positioned in kyphosis/flexion on surgical Wilson frame):
    # Joint 1 (L4 w.r.t L5): +3.5 deg pitch (flexion), +1.0 voxel tz
    # Joint 2 (L3 w.r.t L4): +4.0 deg pitch (flexion), +1.2 voxel tz
    gt_art_l4 = np.array([0.5, 3.5, -0.8, 0.5, -0.4, 1.0])
    gt_art_l3 = np.array([-0.6, 4.0, 1.0, -0.4, 0.5, 1.2])
    gt_articulations = [gt_art_l4, gt_art_l3]

    print("\nSynthesizing ground-truth articulated intraoperative biplanar fluoroscopy...")
    f_target_ap = spine_model.render_articulated_spectrum(gt_base_pose, gt_articulations, view='ap')
    f_target_lat = spine_model.render_articulated_spectrum(gt_base_pose, gt_articulations, view='lateral')

    drr_target_ap = spine_model.render_drr(gt_base_pose, gt_articulations, view='ap')
    drr_target_lat = spine_model.render_drr(gt_base_pose, gt_articulations, view='lateral')

    # Initial surgical misalignment on table:
    init_base_pose = np.array([2.5, -3.0, 1.5, 3.5, -2.5, 3.0])
    init_base_err = compute_mtre_6dof(gt_base_pose, init_base_pose, volume.shape, spacing=(vox_size,) * 3)
    print(f"Initial Base Misalignment: {init_base_pose}, Base mTRE: {init_base_err:.2f} mm")

    # --- 1. Classical Monolithic Rigid Registration ---
    print("\n[Method 1] Running Monolithic Rigid Registration (No Articulation Modeling)...")
    res_rigid = spine_model.register_articulated_biplanar(
        f_target_ap, f_target_lat, init_base_pose=init_base_pose,
        max_iter_global=40, max_iter_local=0
    )
    rigid_base_err = compute_mtre_6dof(gt_base_pose, res_rigid['base_pose'], volume.shape, spacing=(vox_size,) * 3)
    print(f"Rigid Result: Base mTRE={rigid_base_err:.3f} mm, Loss={res_rigid['final_loss']:.5f}, Time={res_rigid['time_sec']:.2f}s")

    # --- 2. Proposed Articulated Kinematic Registration ---
    print("\n[Method 2] Running Proposed Articulated Kinematic Engine (Hierarchical 2-Stage)...")
    res_articulated = spine_model.register_articulated_biplanar(
        f_target_ap, f_target_lat, init_base_pose=init_base_pose,
        max_iter_global=40, max_iter_local=30
    )
    art_base_err = compute_mtre_6dof(gt_base_pose, res_articulated['base_pose'], volume.shape, spacing=(vox_size,) * 3)
    # Compute error on reconstructed joint angles
    err_joint1 = np.linalg.norm(res_articulated['articulations'][0] - gt_art_l4)
    err_joint2 = np.linalg.norm(res_articulated['articulations'][1] - gt_art_l3)
    mean_joint_err = (err_joint1 + err_joint2) / 2.0

    print(f"Articulated Result: Base mTRE={art_base_err:.3f} mm, Joint Articulation Error={mean_joint_err:.3f}, Loss={res_articulated['final_loss']:.5f}, Time={res_articulated['time_sec']:.2f}s")
    print(f"Recovered Joint 1 (L4-L5): {np.round(res_articulated['articulations'][0], 2)} (GT: {gt_art_l4})")
    print(f"Recovered Joint 2 (L3-L4): {np.round(res_articulated['articulations'][1], 2)} (GT: {gt_art_l3})")

    # Render converged DRRs
    drr_rigid_lat = spine_model.render_drr(res_rigid['base_pose'], [], view='lateral')
    drr_art_lat = spine_model.render_drr(res_articulated['base_pose'], res_articulated['articulations'], view='lateral')

    # Save benchmark table
    records = [
        {
            'Framework': 'Monolithic Rigid Registration',
            'Base_mTRE_mm': rigid_base_err,
            'Joint_Art_Err': np.nan,
            'Final_Loss': res_rigid['final_loss'],
            'Time_sec': res_rigid['time_sec'],
            'Success_2mm': bool(rigid_base_err < 2.0)
        },
        {
            'Framework': 'Proposed Articulated Kinematic Registration',
            'Base_mTRE_mm': art_base_err,
            'Joint_Art_Err': mean_joint_err,
            'Final_Loss': res_articulated['final_loss'],
            'Time_sec': res_articulated['time_sec'],
            'Success_2mm': bool(art_base_err < 2.0)
        }
    ]
    df_res = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "table15_articulated_spine_benchmark.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"Saved results table to: {csv_path}")

    # Plot Figure 15
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=300)

    axes[0].imshow(drr_target_lat, cmap='gray')
    axes[0].set_title('Ground-Truth Deformed Spine\n(Wilson Frame Kyphosis)', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    diff_init = np.abs(spine_model.render_drr(init_base_pose, [], view='lateral') - drr_target_lat)
    axes[1].imshow(diff_init, cmap='magma')
    axes[1].set_title(f'Initial Misalignment Difference\n(Init Err: {init_base_err:.2f} mm)', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    diff_rigid = np.abs(drr_rigid_lat - drr_target_lat)
    axes[2].imshow(diff_rigid, cmap='magma')
    axes[2].set_title(f'Monolithic Rigid Residual\n(Residual Err: {rigid_base_err:.2f} mm)', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    diff_art = np.abs(drr_art_lat - drr_target_lat)
    axes[3].imshow(diff_art, cmap='magma')
    axes[3].set_title(f'Proposed Articulated Residual\n(Base Err: {art_base_err:.2f} mm, Joint: {mean_joint_err:.2f})', fontsize=11, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "fig15_articulated_spine_registration.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Figure saved to: {fig_path}")

    # Copy to brain artifacts
    brain_art_dir = Path("C:/Users/paree/.gemini/antigravity/brain/db84ca2a-35bb-4b49-a099-1ebdc0eb4333")
    if brain_art_dir.exists():
        import shutil
        shutil.copy2(fig_path, brain_art_dir / "fig15_articulated_spine_registration.png")
        print(f"Copied to brain artifacts: {brain_art_dir / 'fig15_articulated_spine_registration.png'}")


if __name__ == '__main__':
    run_articulated_spine_benchmark()
