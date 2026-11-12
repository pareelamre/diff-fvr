"""
Experiment 17: Detector Field-of-View (FOV) Truncation & Anatomical Clipping Stress Test.

Physical Clinical Reality:
  Mobile surgical C-arm flat panel detectors have limited dimensions (e.g. 20x20 cm or 15x15 cm).
  In large or obese patients, anatomical structures (transverse processes, rib heads, soft tissue)
  extend outside the detector boundary, causing sharp intensity truncation.
  
  In the frequency domain, sharp rectangular step edges induce severe cross-shaped Gibbs ringing,
  corrupting Spectral-NCC and causing conventional registration to fail.

Physical Resolution:
  1. Differentiable Smooth Boundary Tapering (Tukey/Hann Windowing) on detector edges.
  2. Suppresses boundary Gibbs ringing while preserving full high-frequency bone internal trabeculae.
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

from src.fvr.dicom_loader import load_dicom_series, hu_to_attenuation
from src.fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from src.fvr.registration_3d2d import compute_mtre_6dof

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_tukey_window_2d(shape, alpha=0.15):
    """
    Create a 2D Tukey (tapered cosine) window to suppress boundary step discontinuities.
    alpha: fraction of window inside which cosine taper occurs (e.g., 0.15 = outer 15%).
    """
    h, w = shape
    def tukey_1d(n, a):
        x = np.linspace(0, 1, n)
        w_1d = np.ones(n)
        edge = int(a * n / 2)
        if edge > 0:
            ramp = 0.5 * (1 + np.cos(np.pi * (-1 + 2 * np.arange(edge) / (a * n))))
            w_1d[:edge] = ramp
            w_1d[-edge:] = ramp[::-1]
        return w_1d

    w_y = tukey_1d(h, alpha)
    w_x = tukey_1d(w, alpha)
    return np.outer(w_y, w_x)


def apply_fov_truncation(img: np.ndarray, clip_fraction: float) -> np.ndarray:
    """
    Simulate limited C-arm detector active area by zeroing outer margins.
    clip_fraction: e.g. 0.20 = 20% of width and height clipped (clipped from margins).
    """
    clipped = np.copy(img)
    h, w = img.shape
    pad_h = int(round(h * clip_fraction / 2.0))
    pad_w = int(round(w * clip_fraction / 2.0))

    if pad_h > 0:
        clipped[:pad_h, :] = 0.0
        clipped[-pad_h:, :] = 0.0
    if pad_w > 0:
        clipped[:, :pad_w] = 0.0
        clipped[:, -pad_w:] = 0.0
    return clipped


def run_fov_truncation_benchmark():
    print("=" * 75)
    print("EXPERIMENT 17: DETECTOR FOV TRUNCATION & ANATOMICAL CLIPPING BENCHMARK")
    print("=" * 75)

    # Load and resample clinical DICOM to isotropic 96x96x96
    dicom_dir = PROJECT_ROOT / "data" / "dicom"
    vol_hu, spacing = load_dicom_series(dicom_dir)

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
    print(f"Volume loaded and initialized: {vol_mu.shape}")

    true_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    init_pose = (2.0, -1.5, 1.2, 3.0, -2.0, 2.5)
    init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape)
    print(f"Ground-truth pose: {true_pose}")
    print(f"Initial Misalignment: {init_pose}, Initial mTRE: {init_mtre:.2f} mm")

    # Generate full clean unclipped targets
    full_target_ap = renderer.render_drr(0, 0, 0, tx=0, ty=0, view='ap')
    full_target_lat = renderer.render_drr(0, 0, 0, tx=0, ty=0, view='lateral')

    truncation_fractions = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    records = []

    tukey_window = create_tukey_window_2d(full_target_ap.shape, alpha=0.20)

    print("\nRunning Truncation Sweep across 0% to 50% Anatomical Clipping...")
    for frac in truncation_fractions:
        print(f"\n--- Testing FOV Truncation: {frac*100:.0f}% ---")

        # 1. Simulate truncated intraoperative fluoroscopy
        trunc_ap = apply_fov_truncation(full_target_ap, frac)
        trunc_lat = apply_fov_truncation(full_target_lat, frac)

        # Baseline: Unwindowed Spectral-NCC
        # Zero-pad and take FFT
        th, tw = trunc_ap.shape
        padded_ap = np.zeros((renderer.pz, renderer.px), dtype=np.float64)
        padded_lat = np.zeros((renderer.pz, renderer.py), dtype=np.float64)
        sh, sw = (renderer.pz - th) // 2, (renderer.px - tw) // 2
        padded_ap[sh:sh+th, sw:sw+tw] = trunc_ap
        sh, sw = (renderer.pz - th) // 2, (renderer.py - tw) // 2
        padded_lat[sh:sh+th, sw:sw+tw] = trunc_lat
        f_targ_ap_base = fft.fftshift(fft.fft2(fft.ifftshift(padded_ap)))
        f_targ_lat_base = fft.fftshift(fft.fft2(fft.ifftshift(padded_lat)))

        def obj_baseline(p):
            yaw, pitch, roll, tx, ty, tz = p
            f_ap = renderer.extract_frequency_slice(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
            f_lat = renderer.extract_frequency_slice(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')
            ncc_ap = renderer.compute_spectral_ncc(f_ap, f_targ_ap_base)
            ncc_lat = renderer.compute_spectral_ncc(f_lat, f_targ_lat_base)
            return float(0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat))

        t0 = time.time()
        res_base = minimize(obj_baseline, x0=np.array(init_pose), method='Powell', options={'maxiter': 20, 'ftol': 1e-4})
        t_base = time.time() - t0
        mtre_base = compute_mtre_6dof(true_pose, tuple(res_base.x), vol_mu.shape)
        print(f"  Unwindowed Baseline: mTRE = {mtre_base:.4f} mm, Loss = {res_base.fun:.5f}, Time = {t_base:.2f}s")

        # Proposed: Tukey Tapered Boundary Anti-Aliasing
        tapered_ap = trunc_ap * tukey_window
        tapered_lat = trunc_lat * tukey_window
        padded_ap[:] = 0.0; padded_lat[:] = 0.0
        sh, sw = (renderer.pz - th) // 2, (renderer.px - tw) // 2
        padded_ap[sh:sh+th, sw:sw+tw] = tapered_ap
        sh, sw = (renderer.pz - th) // 2, (renderer.py - tw) // 2
        padded_lat[sh:sh+th, sw:sw+tw] = tapered_lat
        f_targ_ap_taper = fft.fftshift(fft.fft2(fft.ifftshift(padded_ap)))
        f_targ_lat_taper = fft.fftshift(fft.fft2(fft.ifftshift(padded_lat)))

        def obj_tapered(p):
            yaw, pitch, roll, tx, ty, tz = p
            f_ap = renderer.extract_frequency_slice(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
            f_lat = renderer.extract_frequency_slice(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')
            ncc_ap = renderer.compute_spectral_ncc(f_ap, f_targ_ap_taper)
            ncc_lat = renderer.compute_spectral_ncc(f_lat, f_targ_lat_taper)
            return float(0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat))

        t0 = time.time()
        res_taper = minimize(obj_tapered, x0=np.array(init_pose), method='Powell', options={'maxiter': 20, 'ftol': 1e-4})
        t_taper = time.time() - t0
        mtre_taper = compute_mtre_6dof(true_pose, tuple(res_taper.x), vol_mu.shape)
        print(f"  Tukey Tapered Engine: mTRE = {mtre_taper:.4f} mm, Loss = {res_taper.fun:.5f}, Time = {t_taper:.2f}s")

        records.append({
            'Truncation_Fraction': frac,
            'Clipping_Percent': f"{frac*100:.0f}%",
            'Baseline_mTRE_mm': mtre_base,
            'Baseline_Loss': float(res_base.fun),
            'Tapered_mTRE_mm': mtre_taper,
            'Tapered_Loss': float(res_taper.fun),
            'Success_2mm_Base': bool(mtre_base < 2.0),
            'Success_2mm_Taper': bool(mtre_taper < 2.0),
            'Submillimeter_Taper': bool(mtre_taper < 1.0)
        })

    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "table17_fov_truncation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark CSV to: {csv_path}")

    # Plot Figure 17
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Panel 1: Visual Truncation Example
    axes[0].imshow(apply_fov_truncation(full_target_lat, 0.30), cmap='gray')
    axes[0].set_title('30% Detector FOV Truncation\n(Clipped Transverse Anatomy)', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Tukey Windowed Tapering
    axes[1].imshow(apply_fov_truncation(full_target_lat, 0.30) * tukey_window, cmap='gray')
    axes[1].set_title('Tukey Smooth Boundary Tapering\n(Suppressed Spectral Gibbs Ringing)', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Error vs Truncation Curve
    clipping_pct = [r['Truncation_Fraction'] * 100 for r in records]
    base_errors = [r['Baseline_mTRE_mm'] for r in records]
    taper_errors = [r['Tapered_mTRE_mm'] for r in records]

    axes[2].plot(clipping_pct, base_errors, 'r-o', lw=2.2, label='Unwindowed Baseline')
    axes[2].plot(clipping_pct, taper_errors, 'g-s', lw=2.5, label='Proposed Tukey Tapered')
    axes[2].axhline(1.0, color='darkgreen', linestyle='-.', lw=1.5, label='Sub-mm (1.0 mm)')
    axes[2].axhline(2.0, color='crimson', linestyle='-.', lw=1.5, label='Clinical Acceptability (2.0 mm)')

    axes[2].set_title('mTRE vs. Anatomical Clipping %', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Detector Truncation / Clipping [%]', fontsize=10)
    axes[2].set_ylabel('Final mTRE [mm]', fontsize=10)
    axes[2].set_ylim([-0.2, max(max(base_errors), 5.0) + 1.0])
    axes[2].legend(loc='upper left', fontsize=9)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "fig17_fov_truncation_stress.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Figure saved to: {fig_path}")

    # Copy to brain artifacts
    brain_art_dir = Path("C:/Users/paree/.gemini/antigravity/brain/db84ca2a-35bb-4b49-a099-1ebdc0eb4333")
    if brain_art_dir.exists():
        import shutil
        shutil.copy2(fig_path, brain_art_dir / "fig17_fov_truncation_stress.png")
        print(f"Copied to brain artifacts: {brain_art_dir / 'fig17_fov_truncation_stress.png'}")


if __name__ == '__main__':
    run_fov_truncation_benchmark()
