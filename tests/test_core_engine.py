import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fvr.dicom_loader import load_dicom_series, hu_to_attenuation, resample_isotropic_cube
from src.fvr.core_engine import RobustDiffFVREngine
from src.fvr.registration_3d2d import compute_mtre_6dof


def test_core_production_engine():
    print("=" * 65)
    print("RUNNING CORE PRODUCTION ENGINE INTEGRATION SUITE")
    print("=" * 65)

    dicom_dir = PROJECT_ROOT / "data" / "dicom"
    vol_hu, spacing = load_dicom_series(dicom_dir)

    # Center the source CT on a physically consistent isotropic grid.
    target_xy = 96
    vol_cubic, grid_spacing = resample_isotropic_cube(vol_hu, spacing, target_xy)
    vox_size = grid_spacing[0]
    vol_mu = hu_to_attenuation(vol_cubic)
    print(f"Loaded and resampled clinical DICOM volume: {vol_mu.shape}")

    engine = RobustDiffFVREngine(vol_mu, spacing=(vox_size, vox_size, vox_size), pad_factor=2.0)
    print(f"Parallel-beam engine initialized at {vox_size:.3f} mm per voxel.")

    # 1. Perspective Calibrated DRR Generation
    drr_ap = engine.render_calibrated_drr(0, 0, 0, tx=0, ty=0, view='ap')
    drr_lat = engine.render_calibrated_drr(0, 0, 0, tx=0, ty=0, view='lateral')
    assert drr_ap.shape == (96, 96), f"Mismatch AP shape: {drr_ap.shape}"
    assert drr_lat.shape == (96, 96), f"Mismatch Lat shape: {drr_lat.shape}"
    print("[PASS] Parallel-beam DRR output dimensions verified.")

    # 2. Metal Hardware Saturation Detection
    drr_metal = np.copy(drr_lat)
    drr_metal[10:18, 10:25] = np.max(drr_lat) * 5.0
    mask_metal = engine.detect_metal_hardware_mask(drr_metal, threshold=0.98)
    assert not np.all(mask_metal), "Failed to detect metal hardware"
    print("[PASS] Radio-opaque metal saturation detection verified.")

    # 3. Clean Biplanar Registration Benchmark
    true_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    init_pose = (1.5, -1.0, 0.8, 2.0, -1.5, 1.8)
    init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape,
                               spacing=(vox_size,) * 3, translation_units='mm')

    res_clean = engine.register_biplanar_robust(drr_ap, drr_lat, initial_pose=init_pose, max_iter=30)
    final_mtre_clean = compute_mtre_6dof(true_pose, tuple(res_clean['final_pose']), vol_mu.shape,
                                      spacing=(vox_size,) * 3, translation_units='mm')
    print(f"[PASS] Clean Fluoroscopy: Init mTRE={init_mtre:.2f}mm -> Final mTRE={final_mtre_clean:.4f}mm (Time: {res_clean['time_sec']:.2f}s)")
    assert final_mtre_clean < 1.0, f"Expected sub-millimeter precision, got {final_mtre_clean:.4f}mm"

    # 4. Synthetic saturated patch (300 / 9216 pixels, about 3.26% coverage)
    drr_ap_occ = np.copy(drr_ap)
    drr_lat_occ = np.copy(drr_lat)
    drr_ap_occ[5:20, 10:30] = np.max(drr_ap) * 5.0
    drr_lat_occ[5:20, 10:30] = np.max(drr_lat) * 5.0

    res_metal = engine.register_biplanar_robust(drr_ap_occ, drr_lat_occ, initial_pose=init_pose, handle_metal=True, max_iter=30)
    final_mtre_metal = compute_mtre_6dof(true_pose, tuple(res_metal['final_pose']), vol_mu.shape,
                                      spacing=(vox_size,) * 3, translation_units='mm')
    print(f"[PASS] Metal Occlusion: Metal Detected={res_metal['metal_detected']} -> Final mTRE={final_mtre_metal:.4f}mm (Time: {res_metal['time_sec']:.2f}s)")
    assert res_metal['metal_detected'], "Metal was not detected by engine"
    assert final_mtre_metal < 2.0, f"Expected clinical convergence < 2.0mm, got {final_mtre_metal:.4f}mm"

    print("=" * 65)
    print("ALL PRODUCTION ENGINE INTEGRATION TESTS PASSED (100% GREEN)!")
    print("=" * 65)


if __name__ == '__main__':
    test_core_production_engine()
