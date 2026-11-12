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
from .dicom_loader import load_dicom_series, hu_to_attenuation, resample_isotropic_cube
from .core_engine import RobustDiffFVREngine


def load_volume(ct_path: Path, target_res: int = 96, spacing_mm=None):
    """
    Return (attenuation volume, spacing_zyx_mm) on an isotropic cubic grid.
    DICOM spacing comes from the series; NPY defaults to one mm per voxel.
    """
    if ct_path.is_dir():
        if spacing_mm is not None:
            raise ValueError('--spacing-mm is for NPY input; DICOM uses its metadata.')
        vol_hu, spacing = load_dicom_series(ct_path)
    elif ct_path.suffix == '.npy':
        vol_hu = np.load(ct_path, allow_pickle=False)
        spacing = (1.0, 1.0, 1.0) if spacing_mm is None else spacing_mm
    else:
        raise ValueError(f"Unsupported CT path: {ct_path}. Must be a DICOM directory or .npy file.")

    vol_cubic, spacing = resample_isotropic_cube(vol_hu, spacing, target_res)
    return hu_to_attenuation(vol_cubic), spacing


def main_render():
    parser = argparse.ArgumentParser(description="Diff-FVR: research parallel-beam DRR renderer (no cone-beam calibration).")
    parser.add_argument("--ct", type=str, required=True, help="Path to pre-op CT DICOM directory or .npy file.")
    parser.add_argument('--spacing-mm', type=float, nargs=3, metavar=('Z', 'Y', 'X'),
                        help='NPY voxel spacing in mm (default: 1 1 1). DICOM uses metadata.')
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
        parser.error(f'CT path not found: {ct_path}')

    print(f"Loading CT from: {ct_path} (resampling to {args.res}^3)...")
    t0 = time.time()
    try:
        vol_mu, spacing = load_volume(ct_path, target_res=args.res, spacing_mm=args.spacing_mm)
    except ValueError as exc:
        parser.error(str(exc))
    engine = RobustDiffFVREngine(vol_mu, spacing=spacing, pad_factor=2.0)
    print(f'Parallel-beam grid: {spacing[0]:.6g} mm/pixel; rotations in degrees, translations in mm.')
    print(f"Initialized 3D FFT volume in {time.time() - t0:.2f} s.")

    t_render = time.perf_counter()
    horizontal_mm = args.tx if args.view == 'ap' else args.ty
    drr = engine.render_calibrated_drr(args.yaw, args.pitch, args.roll,
                                       tx=horizontal_mm, ty=args.tz, view=args.view)
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
    parser = argparse.ArgumentParser(description='Diff-FVR: research parallel-beam biplanar registration. '
                                    'Targets must match the rendered grid; acquired cone-beam images are unsupported.')
    parser.add_argument("--ct", type=str, required=True, help="Path to pre-op CT DICOM directory or .npy file.")
    parser.add_argument('--spacing-mm', type=float, nargs=3, metavar=('Z', 'Y', 'X'),
                        help='NPY voxel spacing in mm (default: 1 1 1). DICOM uses metadata.')
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
        parser.error(f'One or more input paths do not exist: {ct_path}, {ap_path}, {lat_path}')

    # Load targets
    def load_img(p: Path):
        if p.suffix == '.npy':
            return np.load(p, allow_pickle=False)
        im = Image.open(p).convert('L')
        return np.array(im, dtype=np.float64)

    targ_ap = load_img(ap_path)
    targ_lat = load_img(lat_path)

    print(f"Loading and initializing CT from: {ct_path}...")
    try:
        vol_mu, spacing = load_volume(ct_path, target_res=args.res, spacing_mm=args.spacing_mm)
    except ValueError as exc:
        parser.error(str(exc))
    engine = RobustDiffFVREngine(vol_mu, spacing=spacing, pad_factor=2.0)
    print(f'Parallel-beam target grid: {spacing[0]:.6g} mm/pixel; expected image shape {vol_mu.shape[:2]}.')

    print(f"Running Biplanar 6-DoF Registration (handle_metal={args.handle_metal}, max_iter={args.max_iter})...")
    res = engine.register_biplanar_robust(
        target_ap=targ_ap,
        target_lat=targ_lat,
        handle_metal=args.handle_metal,
        max_iter=args.max_iter
    )

    pose = res['final_pose']
    status = 'CONVERGED' if res['success'] else 'DID NOT CONVERGE'
    print(f"\nREGISTRATION {status} in {res['time_sec']:.2f} s ({res['evaluations']} evaluations):")
    print(f"  Rotation:    Yaw={pose[0]:.3f}°, Pitch={pose[1]:.3f}°, Roll={pose[2]:.3f}°")
    print(f"  Translation: Tx={pose[3]:.3f} mm, Ty={pose[4]:.3f} mm, Tz={pose[5]:.3f} mm")
    print(f"  Final Loss:  {res['final_loss']:.6f}")

    out_data = {
        'success': bool(res['success']),
        'projection_model': 'parallel_beam',
        'voxel_spacing_zyx_mm': list(spacing),
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
    if not res['success']:
        parser.exit(1, 'Optimizer did not converge; result saved with success=false.\n')


if __name__ == '__main__':
    main_render()
