import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fvr.dicom_loader import load_dicom_series, hu_to_attenuation
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.robust_registration import RobustBiplanarRegistrationEngine
from fvr.registration_3d2d import compute_mtre_6dof
from experiments.exp10_metal_occlusion_stress import add_surgical_metal_occlusion

def main():
    vol_hu, spacing = load_dicom_series(Path(__file__).resolve().parent.parent / "data" / "dicom")
    target_xy = 96
    phys_z = vol_hu.shape[0] * spacing[0]
    phys_x = vol_hu.shape[2] * spacing[2]
    vox_size = phys_x / target_xy
    target_z = int(round(phys_z / vox_size))
    vol_iso = zoom(vol_hu, (target_z / vol_hu.shape[0], target_xy / vol_hu.shape[1], target_xy / vol_hu.shape[2]), order=1)
    pad_z_before = (target_xy - target_z) // 2
    pad_z_after = target_xy - target_z - pad_z_before
    vol_cubic = np.pad(vol_iso, ((pad_z_before, pad_z_after), (0, 0), (0, 0)), mode='constant', constant_values=-1000.0)
    vol_mu = hu_to_attenuation(vol_cubic)

    renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
    engine = RobustBiplanarRegistrationEngine(renderer)

    true_pose = (3.5, -2.0, 1.8, 3.2, -2.5, 1.5)
    init_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape)

    target_ap = renderer.render_drr(*true_pose[:3], tx=true_pose[3], ty=true_pose[5], view='ap')
    target_lat = renderer.render_drr(*true_pose[:3], tx=true_pose[4], ty=true_pose[5], view='lateral')

    # 1. Clean Fluoroscopy
    res_clean = engine.register_biplanar_robust(target_ap, target_lat, initial_pose=init_pose)
    mtre_clean = compute_mtre_6dof(true_pose, res_clean['estimated_pose'], vol_mu.shape)
    print("=" * 60)
    print("1. Clean Fluoroscopy Benchmark:")
    print(f"   Initial mTRE: {init_mtre:.2f} mm -> Final mTRE: {mtre_clean:.4f} mm")
    print(f"   Metal Detected: {res_clean['metal_detected']} | Time: {res_clean['total_time_seconds']:.2f} s")

    # 2. 30% Surgical Metal Hardware Occlusion
    ap_occ = add_surgical_metal_occlusion(target_ap / target_ap.max(), 0.30)
    lat_occ = add_surgical_metal_occlusion(target_lat / target_lat.max(), 0.30)
    res_metal = engine.register_biplanar_robust(ap_occ, lat_occ, initial_pose=init_pose)
    mtre_metal = compute_mtre_6dof(true_pose, res_metal['estimated_pose'], vol_mu.shape)
    print("=" * 60)
    print("2. 30% Surgical Metal Hardware Occlusion Benchmark:")
    print(f"   Initial mTRE: {init_mtre:.2f} mm -> Final mTRE: {mtre_metal:.4f} mm")
    print(f"   Metal Detected: {res_metal['metal_detected']} | Time: {res_metal['total_time_seconds']:.2f} s")
    print(f"   Success (< 2.0 mm): {mtre_metal < 2.0} | Submillimeter (< 1.0 mm): {mtre_metal < 1.0}")
    print("=" * 60)

if __name__ == "__main__":
    main()
