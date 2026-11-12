"""
Command-Line Interface (CLI) for Diff-FVR.
Provides:
  - fvr-render: Ultra-fast DRR projection synthesis from 3D CT.
  - fvr-register: Biplanar 3D-2D 6-DoF registration against intraoperative fluoroscopy.
"""

import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from .dicom_loader import load_dicom_series, hu_to_attenuation
from .diff_fvr import DifferentiableFourierVolumeRenderer
from .core_engine import RobustDiffFVREngine


def load_volume(ct_path: Path, target_res: int = 96):
    """
    Load CT from DICOM directory or numpy array, with isotropic cubic resampling.
    """
    if ct_path.is_dir():
        vol_hu, spacing = load_dicom_series(ct_path)
    elif ct_path.suffix in ['.npy', '.npz']:
        vol_hu = np.load(ct_path)
        spacing = (1.0, 1.0, 1.0)
    else:
        raise ValueError(f"Unsupported CT path: {ct_path}. Must be a DICOM directory or .npy file.")

    if target_res is not None and vol_hu.shape[1] != target_res:
        target_xy = target_res
        phys_z = vol_hu.shape[0] * spacing[0]
        phys_x = vol_hu.shape[2] * spacing[2]
        vox_size = phys_x / target_xy
        target_z = int(round(phys_z / vox_size))

        vol_iso = zoom(vol_hu, (target_z / vol_hu.shape[0], target_xy / vol_hu.shape[1], target_xy / vol_hu.shape[2]), order=1)
        pad_z_before = (target_xy - target_z) // 2
        pad_z_after = target_xy - target_z - pad_z_before
        vol_cubic = np.pad(vol_iso, ((pad_z_before, pad_z_after), (0, 0), (0, 0)), mode='constant', constant_values=-1000.0)
    else:
        vol_cubic = vol_hu

    return hu_to_attenuation(vol_cubic)


def main_render():
    parser = argparse.ArgumentParser(description="Diff-FVR: Ultra-fast Digitally Reconstructed Radiograph (DRR) Renderer.")
    parser.add_argument("--ct", type=str, required=True, help="Path to pre-op CT DICOM directory or .npy file.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Rotation yaw in degrees (default: 0.0).")
    parser.add_argument("--pitch", type=float, default=0.0, help="Rotation pitch in degrees (default: 0.0).")
    parser.add_argument("--roll", type=float, default=0.0, help="Rotation roll in degrees (default: 0.0).")
    parser.add_argument("--tx", type=float, default=0.0, help="Translation Tx in mm (default: 0.0).")
    parser.add_argument("--ty", type=float, default=0.0, help="Translation Ty in mm (default: 0.0).")
    parser.add_argument("--tz", type=float, default=0.0, help="Translation Tz in mm (default: 0.0).")
    parser.add_argument("--view", type=str, choices=['ap', 'lateral'], default='ap', help="View angle ('ap' or 'lateral').")
    parser.add_argument("--res", type=int, default=128, help="Isotropic cubic grid dimension (default: 128).")
    parser.add_argument("--out", type=str, default="drr_output.png", help="Output PNG path (default: drr_output.png).")
    
    args = parser.parse_args()

    ct_path = Path(args.ct)
    if not ct_path.exists():
        print(f"Error: CT path not found: {ct_path}")
        return

    print(f"Loading CT from: {ct_path} (resampling to {args.res}^3)...")
    t0 = time.time()
    vol_mu = load_volume(ct_path, target_res=args.res)
    renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
    print(f"Initialized 3D FFT volume in {time.time() - t0:.2f} s.")

    t_render = time.perf_counter()
    drr = renderer.render_drr(args.yaw, args.pitch, args.roll, tx=args.tx, ty=args.ty, view=args.view)
    render_ms = (time.perf_counter() - t_render) * 1000.0

    print(f"Rendered {args.view.upper()} DRR in {render_ms:.2f} ms ({1000.0 / render_ms:.1f} FPS)!")

    # Normalize and save image
    drr_norm = (drr - drr.min()) / (drr.max() - drr.min() + 1e-8)
    img_uint8 = (drr_norm * 255.0).astype(np.uint8)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_uint8).save(out_path)
    print(f"Saved rendered DRR to: {out_path.resolve()}")


def main_register():
    parser = argparse.ArgumentParser(description="Diff-FVR: Biplanar 3D-2D 6-DoF Rigid Spine Registration.")
    parser.add_argument("--ct", type=str, required=True, help="Path to pre-op CT DICOM directory or .npy file.")
    parser.add_argument("--ap", type=str, required=True, help="Path to target intraoperative AP X-ray (PNG/TIFF/NPY).")
    parser.add_argument("--lat", type=str, required=True, help="Path to target intraoperative Lateral X-ray.")
    parser.add_argument("--res", type=int, default=96, help="Isotropic CT grid dimension (default: 96).")
    parser.add_argument("--max-iter", type=int, default=30, help="Maximum optimizer iterations (default: 30).")
    parser.add_argument("--handle-metal", action="store_true", help="Enable automatic radio-opaque surgical metal saturation masking.")
    parser.add_argument("--out", type=str, default="registration_result.json", help="Output JSON results path.")

    args = parser.parse_args()

    ct_path = Path(args.ct)
    ap_path = Path(args.ap)
    lat_path = Path(args.lat)

    if not ct_path.exists() or not ap_path.exists() or not lat_path.exists():
        print(f"Error: One or more input paths do not exist:\n  CT: {ct_path}\n  AP: {ap_path}\n  LAT: {lat_path}")
        return

    # Load targets
    def load_img(p: Path):
        if p.suffix == '.npy':
            return np.load(p)
        im = Image.open(p).convert('L')
        return np.array(im, dtype=np.float64)

    targ_ap = load_img(ap_path)
    targ_lat = load_img(lat_path)

    print(f"Loading and initializing CT from: {ct_path}...")
    vol_mu = load_volume(ct_path, target_res=args.res)
    engine = RobustDiffFVREngine(vol_mu, pad_factor=2.0)

    print(f"Running Biplanar 6-DoF Registration (handle_metal={args.handle_metal}, max_iter={args.max_iter})...")
    res = engine.register_biplanar_robust(
        target_ap=targ_ap,
        target_lat=targ_lat,
        handle_metal=args.handle_metal,
        max_iter=args.max_iter
    )

    pose = res['final_pose']
    print(f"\nREGISTRATION CONVERGED in {res['time_sec']:.2f} s ({res['evaluations']} evaluations):")
    print(f"  Rotation:    Yaw={pose[0]:.3f}°, Pitch={pose[1]:.3f}°, Roll={pose[2]:.3f}°")
    print(f"  Translation: Tx={pose[3]:.3f} mm, Ty={pose[4]:.3f} mm, Tz={pose[5]:.3f} mm")
    print(f"  Final Loss:  {res['final_loss']:.6f}")

    out_data = {
        'yaw_deg': float(pose[0]),
        'pitch_deg': float(pose[1]),
        'roll_deg': float(pose[2]),
        'tx_mm': float(pose[3]),
        'ty_mm': float(pose[4]),
        'tz_mm': float(pose[5]),
        'final_loss': float(res['final_loss']),
        'time_sec': float(res['time_sec']),
        'evaluations': int(res['evaluations']),
        'metal_detected': bool(res['metal_detected'])
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved results to: {out_path.resolve()}")


if __name__ == '__main__':
    main_render()
