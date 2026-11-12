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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fvr.dicom_loader import load_dicom_series, hu_to_attenuation
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.registration_3d2d import BiplanarSpineRegistrationEngine, compute_mtre_6dof


def add_surgical_metal_occlusion(img: np.ndarray, occlusion_ratio: float) -> np.ndarray:
    """
    Simulate realistic radio-opaque surgical hardware:
      - Steel retractor blade (rectangular opaque block)
      - Titanium pedicle screw / K-wire (high-density diagonal shaft)
    """
    if occlusion_ratio <= 0.0:
        return img.copy()

    h, w = img.shape
    total_pixels = h * w
    target_occluded_pixels = int(total_pixels * occlusion_ratio)

    occluded = img.copy()

    # 1. Surgical Retractor Blade (Rectangular metal shadow)
    # Area ~ 70% of target occluded pixels
    retractor_pixels = int(target_occluded_pixels * 0.70)
    rw = max(3, int(np.sqrt(retractor_pixels * 1.5)))
    rh = max(3, retractor_pixels // rw)
    rx = np.random.randint(0, max(1, w - rw))
    ry = np.random.randint(0, max(1, h - rh))
    # Full radio-opacity: high attenuation shadow (value = 1.0)
    occluded[ry:ry + rh, rx:rx + rw] = 1.0

    # 2. Pedicle Screw / K-wire (Diagonal dense shaft)
    wire_pixels = target_occluded_pixels - retractor_pixels
    if wire_pixels > 0:
        thickness = max(2, int(w * 0.03))
        # Draw a line across the image
        xx = np.arange(w)
        slope = float(np.random.uniform(-0.8, 0.8))
        intercept = float(np.random.uniform(h * 0.2, h * 0.8))
        for t in range(-thickness // 2, thickness // 2 + 1):
            yy = np.clip((slope * xx + intercept + t).astype(int), 0, h - 1)
            occluded[yy, xx] = 1.0

    return occluded


def run_metal_occlusion_stress_test():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    dicom_dir = Path(__file__).resolve().parent.parent / "data" / "dicom"

    print("=" * 70)
    print("  EXPERIMENT 10: Surgical Metal Implant & Foreign Body Stress Test")
    print("=" * 70)

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

    # 2. Occlusion levels
    occlusion_levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    trials_per_level = 10
    records = []
    np.random.seed(303)

    for occ in occlusion_levels:
        occ_pct = int(occ * 100)
        print(f"\n--- Testing Surgical Occlusion: {occ_pct}% Hardware Coverage ({trials_per_level} trials) ---")

        for trial in range(trials_per_level):
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

            # Normalize to [0, 1]
            ap_norm = (target_ap - target_ap.min()) / (target_ap.max() - target_ap.min() + 1e-8)
            lat_norm = (target_lat - target_lat.min()) / (target_lat.max() - target_lat.min() + 1e-8)

            # Apply realistic surgical metal occlusion
            ap_occ = add_surgical_metal_occlusion(ap_norm, occ)
            lat_occ = add_surgical_metal_occlusion(lat_norm, occ)

            t0 = time.time()
            res = engine.register_biplanar(ap_occ, lat_occ, initial_pose=init_pose, max_iter=75)
            t_reg = time.time() - t0

            final_mtre = compute_mtre_6dof(true_pose, res['estimated_pose'], vol_mu.shape)

            records.append({
                "Occlusion_Fraction": occ,
                "Occlusion_Pct": occ_pct,
                "Trial": trial,
                "Init_mTRE_mm": init_mtre,
                "Final_mTRE_mm": final_mtre,
                "Success_2mm": final_mtre < 2.0,
                "Success_1mm": final_mtre < 1.0,
                "Time_sec": t_reg,
                "Evals": res["function_evals"]
            })

            print(f"  Trial {trial+1:2d}/{trials_per_level} | Init: {init_mtre:.2f} mm -> Final: {final_mtre:.3f} mm | Success: {final_mtre < 2.0}")

    df = pd.DataFrame(records)
    csv_path = out_dir / "table10_metal_occlusion.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metal occlusion benchmark to: {csv_path}")

    # Summary
    for occ_pct, grp in df.groupby("Occlusion_Pct"):
        print(f"\n--- Occlusion: {occ_pct}% ---")
        print(f"  Final mTRE: {grp['Final_mTRE_mm'].mean():.2f} ± {grp['Final_mTRE_mm'].std():.2f} mm")
        print(f"  Success (<2.0 mm): {grp['Success_2mm'].mean() * 100:.1f}%")
        print(f"  Submillimeter (<1.0 mm): {grp['Success_1mm'].mean() * 100:.1f}%")

    # 3. Publication Figure 10: Surgical Hardware Occlusion Analysis
    fig = plt.figure(figsize=(16, 8), dpi=300)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # Top Row: Visual Fluoroscopy with Increasing Occlusion (0%, 10%, 30%, 50%)
    sample_ap = renderer.render_drr(0, 0, 0, view='ap')
    sample_norm = (sample_ap - sample_ap.min()) / (sample_ap.max() - sample_ap.min() + 1e-8)
    sample_occs = [0.0, 0.10, 0.30, 0.50]

    for idx, s_occ in enumerate(sample_occs):
        ax_img = fig.add_subplot(gs[0, idx])
        occ_img = add_surgical_metal_occlusion(sample_norm, s_occ)
        ax_img.imshow(occ_img, cmap='bone')
        ax_img.set_title(f'Metal Occlusion: {int(s_occ*100)}%', fontsize=11, fontweight='bold')
        ax_img.axis('off')

    # Bottom Row Left: mTRE vs Occlusion Degradation Curve
    ax_curve = fig.add_subplot(gs[1, 0:2])
    occ_x = [int(o * 100) for o in occlusion_levels]
    mean_mtre = [df[df["Occlusion_Pct"] == o]["Final_mTRE_mm"].mean() for o in occ_x]
    std_mtre = [df[df["Occlusion_Pct"] == o]["Final_mTRE_mm"].std() for o in occ_x]

    ax_curve.errorbar(occ_x, mean_mtre, yerr=std_mtre, fmt='o-', color='#d62728', linewidth=2.5, markersize=8, capsize=5, label='Mean mTRE ± Std')
    ax_curve.axhline(2.0, color='grey', linestyle='--', label='Surgical Limit (2.0 mm)')
    ax_curve.axhline(1.0, color='green', linestyle=':', label='Sub-mm Limit (1.0 mm)')
    ax_curve.set_xlabel('Radio-Opaque Hardware Occlusion (%)', fontsize=11, fontweight='bold')
    ax_curve.set_ylabel('Registration Error (mm)', fontsize=11, fontweight='bold')
    ax_curve.set_title('Robustness to Surgical Hardware & Occlusion', fontsize=12, fontweight='bold')
    ax_curve.grid(True, linestyle='--', alpha=0.5)
    ax_curve.legend(loc='upper left', fontsize=9.5)

    # Bottom Row Right: Success Rate Breakdown
    ax_bar = fig.add_subplot(gs[1, 2:4])
    succ_2mm = [df[df["Occlusion_Pct"] == o]["Success_2mm"].mean() * 100 for o in occ_x]
    succ_1mm = [df[df["Occlusion_Pct"] == o]["Success_1mm"].mean() * 100 for o in occ_x]

    bar_width = 3.5
    ax_bar.bar(np.array(occ_x) - bar_width/2, succ_2mm, width=bar_width, color='#1f77b4', label='Clinical Success (< 2.0 mm)')
    ax_bar.bar(np.array(occ_x) + bar_width/2, succ_1mm, width=bar_width, color='#2ca02c', label='Submillimeter (< 1.0 mm)')
    ax_bar.set_xlabel('Occlusion Percentage (%)', fontsize=11, fontweight='bold')
    ax_bar.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax_bar.set_title('Clinical Success Rate under Hardware Occlusion', fontsize=12, fontweight='bold')
    ax_bar.set_ylim(0, 110)
    ax_bar.grid(True, linestyle='--', alpha=0.5)
    ax_bar.legend(loc='lower left', fontsize=9.5)

    fig_path = out_dir / "fig10_metal_occlusion_stress.png"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 10 to: {fig_path}")


if __name__ == "__main__":
    run_metal_occlusion_stress_test()
