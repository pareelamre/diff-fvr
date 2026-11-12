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


def add_quantum_fluoroscopy_noise(img: np.ndarray, photon_flux: float, gaussian_sigma: float = 0.015) -> np.ndarray:
    """
    Simulate physical intraoperative fluoroscopy degradation:
      1. Poisson photon arrival process (quantum mottle scaled by incident photon flux)
      2. Gaussian electronic detector read noise
    """
    norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    # Poisson noise scaled by incident photon flux
    noisy_photons = np.random.poisson(norm * photon_flux) / photon_flux
    # Electronic sensor noise
    noise_gaussian = np.random.normal(0, gaussian_sigma, img.shape)
    degraded = np.clip(noisy_photons + noise_gaussian, 0.0, 1.0)
    return degraded * (img.max() - img.min()) + img.min()


def run_dose_sweep():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    dicom_dir = Path(__file__).resolve().parent.parent / "data" / "dicom"

    print("=" * 70)
    print("  EXPERIMENT 11: Radiation Dose & Quantum Mottle Starvation Stress Test")
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

    # 2. Dose regimes
    dose_regimes = [
        {"name": "Standard Dose (10⁶)",     "flux": 1e6,  "desc": "Full diagnostic dose"},
        {"name": "Half Dose (2×10⁵)",       "flux": 2e5,  "desc": "50% ALARA reduction"},
        {"name": "Low Dose (5×10⁴)",        "flux": 5e4,  "desc": "Standard surgical fluoroscopy"},
        {"name": "Very Low Dose (10⁴)",     "flux": 1e4,  "desc": "Aggressive dose reduction"},
        {"name": "Ultra-Low Dose (5×10³)",  "flux": 5e3,  "desc": "Severe quantum mottle"},
        {"name": "Extreme Starvation (10³)", "flux": 1e3, "desc": "1000x photon starvation"}
    ]

    trials_per_dose = 10
    records = []
    np.random.seed(404)

    for regime in dose_regimes:
        flux = regime["flux"]
        print(f"\n--- Testing Dose Regime: {regime['name']} (Flux={flux:.0e} photons/px) ---")

        for trial in range(trials_per_dose):
            yaw = float(np.random.uniform(-4.0, 4.0))
            pitch = float(np.random.uniform(-4.0, 4.0))
            roll = float(np.random.uniform(-4.0, 4.0))
            tx = float(np.random.uniform(-5.0, 5.0))
            ty = float(np.random.uniform(-4.0, 4.0))
            tz = float(np.random.uniform(-4.0, 4.0))
            true_pose = (yaw, pitch, roll, tx, ty, tz)

            init_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape)

            target_ap_clean = renderer.render_drr(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
            target_lat_clean = renderer.render_drr(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')

            # Degrade with Poisson quantum flux and Gaussian electronic read noise
            target_ap_noisy = add_quantum_fluoroscopy_noise(target_ap_clean, flux)
            target_lat_noisy = add_quantum_fluoroscopy_noise(target_lat_clean, flux)

            t0 = time.time()
            res = engine.register_biplanar(target_ap_noisy, target_lat_noisy, initial_pose=init_pose, max_iter=75)
            t_reg = time.time() - t0

            final_mtre = compute_mtre_6dof(true_pose, res['estimated_pose'], vol_mu.shape)

            records.append({
                "Regime": regime["name"],
                "Photon_Flux": flux,
                "Trial": trial,
                "Init_mTRE_mm": init_mtre,
                "Final_mTRE_mm": final_mtre,
                "Success_2mm": final_mtre < 2.0,
                "Success_1mm": final_mtre < 1.0,
                "Time_sec": t_reg,
                "Evals": res["function_evals"]
            })

            print(f"  Trial {trial+1:2d}/{trials_per_dose} | Init: {init_mtre:.2f} mm -> Final: {final_mtre:.3f} mm | Success: {final_mtre < 2.0}")

    df = pd.DataFrame(records)
    csv_path = out_dir / "table11_radiation_dose_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved radiation dose benchmark to: {csv_path}")

    for name, grp in df.groupby("Regime"):
        print(f"\n--- {name} ---")
        print(f"  Final mTRE: {grp['Final_mTRE_mm'].mean():.3f} ± {grp['Final_mTRE_mm'].std():.3f} mm")
        print(f"  Success (<2.0 mm): {grp['Success_2mm'].mean() * 100:.1f}%")
        print(f"  Submillimeter (<1.0 mm): {grp['Success_1mm'].mean() * 100:.1f}%")

    # 3. Publication Figure 11: Radiation Dose Sensitivity Analysis
    fig = plt.figure(figsize=(16, 8), dpi=300)
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.35, wspace=0.25)

    # Top Row: Sample Radiographs across all 6 Radiation Doses
    sample_ap = renderer.render_drr(0, 0, 0, view='ap')
    for idx, regime in enumerate(dose_regimes):
        ax_img = fig.add_subplot(gs[0, idx])
        noisy_img = add_quantum_fluoroscopy_noise(sample_ap, regime["flux"])
        ax_img.imshow(noisy_img, cmap='gray')
        ax_img.set_title(f"Φ = {regime['flux']:.0e}\nphotons/px", fontsize=9.5, fontweight='bold')
        ax_img.axis('off')

    # Bottom Row Left: mTRE vs Dose Degradation Curve
    ax_curve = fig.add_subplot(gs[1, 0:3])
    regime_names = [r["name"] for r in dose_regimes]
    fluxes = [r["flux"] for r in dose_regimes]
    mean_mtre = [df[df["Regime"] == name]["Final_mTRE_mm"].mean() for name in regime_names]
    std_mtre = [df[df["Regime"] == name]["Final_mTRE_mm"].std() for name in regime_names]

    ax_curve.errorbar(fluxes, mean_mtre, yerr=std_mtre, fmt='o-', color='#1f77b4', linewidth=2.5, markersize=8, capsize=5, label='Mean mTRE ± Std')
    ax_curve.axhline(2.0, color='red', linestyle='--', label='Clinical Threshold (2.0 mm)')
    ax_curve.axhline(1.0, color='green', linestyle=':', label='Submillimeter (1.0 mm)')
    ax_curve.set_xscale('log')
    ax_curve.set_yscale('log')
    ax_curve.set_xlabel('Incident Photon Flux Φ (photons / pixel)', fontsize=11, fontweight='bold')
    ax_curve.set_ylabel('Registration Error mTRE (mm)', fontsize=11, fontweight='bold')
    ax_curve.set_title('Registration Robustness across 6 Orders of Radiation Dose', fontsize=11, fontweight='bold')
    ax_curve.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_curve.legend(loc='upper right', fontsize=9.5)

    # Bottom Row Right: Clinical Success Rates
    ax_bar = fig.add_subplot(gs[1, 3:6])
    succ_2mm = [df[df["Regime"] == name]["Success_2mm"].mean() * 100 for name in regime_names]
    succ_1mm = [df[df["Regime"] == name]["Success_1mm"].mean() * 100 for name in regime_names]

    x_idx = np.arange(len(dose_regimes))
    bar_w = 0.35
    ax_bar.bar(x_idx - bar_w/2, succ_2mm, width=bar_w, color='#2ca02c', label='Clinical Success (< 2.0 mm)')
    ax_bar.bar(x_idx + bar_w/2, succ_1mm, width=bar_w, color='#17becf', label='Submillimeter (< 1.0 mm)')
    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels([f"10^{int(np.log10(r['flux']))}" if r['flux'] in [1e6, 1e4, 1e3] else f"{r['flux']:.0e}" for r in dose_regimes], rotation=15, fontsize=9.5)
    ax_bar.set_xlabel('Photon Flux Regime', fontsize=11, fontweight='bold')
    ax_bar.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax_bar.set_title('Clinical Success Rate vs. Radiation Exposure', fontsize=11, fontweight='bold')
    ax_bar.set_ylim(0, 110)
    ax_bar.grid(True, linestyle='--', alpha=0.5)
    ax_bar.legend(loc='lower left', fontsize=9.5)

    fig_path = out_dir / "fig11_radiation_dose_sweep.png"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 11 to: {fig_path}")


if __name__ == "__main__":
    run_dose_sweep()
