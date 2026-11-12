"""
Experiment 16: CED summary of synthetic projection registration experiments.

The three CT-derived datasets each use one source volume and randomized poses.
This summary is not acquired-fluoroscopy or multi-patient clinical validation.
Only explicitly unit-verified tables may be aggregated; regenerate older files.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_ced(errors: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Compute fraction of trials with error <= threshold."""
    return np.array([np.mean(errors <= tau) * 100.0 for tau in thresholds])


def load_metric_table(path):
    """Refuse legacy voxel errors that were labeled as physical mm."""
    frame = pd.read_csv(path)
    if frame.empty or 'Metric_units' not in frame or not frame['Metric_units'].eq('mm').all():
        raise ValueError(f'{path}: unverified metric units; regenerate the source experiment.')
    errors = frame['Final_mTRE_mm'].to_numpy(dtype=float)
    if not np.all(np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError(f'{path}: mTRE must contain finite nonnegative values.')
    return frame


def generate_unified_ced_benchmark():
    print("Generating Unified Multi-Center Cumulative Error Distribution (CED) Benchmark...")

    # Load datasets
    df_dicom = load_metric_table(OUTPUT_DIR / "table7_clinical_benchmark.csv")
    df_kaggle = load_metric_table(OUTPUT_DIR / "table12_kaggle_clinical_benchmark.csv")
    df_amos = load_metric_table(OUTPUT_DIR / "table13_amos_clinical_benchmark.csv")

    # Extract clean and low-dose errors
    dicom_clean = df_dicom[df_dicom['Condition'].str.contains('Clean')]['Final_mTRE_mm'].values
    dicom_noisy = df_dicom[df_dicom['Condition'].str.contains('Low-Dose')]['Final_mTRE_mm'].values

    kaggle_clean = df_kaggle[df_kaggle['Condition'].str.contains('Clean')]['Final_mTRE_mm'].values
    kaggle_noisy = df_kaggle[df_kaggle['Condition'].str.contains('Low-Dose')]['Final_mTRE_mm'].values

    amos_clean = df_amos[df_amos['Condition'].str.contains('Clean')]['Final_mTRE_mm'].values
    amos_noisy = df_amos[df_amos['Condition'].str.contains('Low-Dose')]['Final_mTRE_mm'].values

    # Baselines for comparison
    df_biplanar = load_metric_table(OUTPUT_DIR / "table6_biplanar_benchmark.csv")
    monoplanar_errors = df_biplanar[df_biplanar['Method'].str.contains('Monoplanar')]['Final_mTRE_mm'].values

    # Baseline data is required; do not substitute invented observations.
    df_metal = load_metric_table(OUTPUT_DIR / "table10_metal_occlusion.csv")
    metal_unmasked = df_metal[df_metal['Hardware_Type'].str.contains('Both|Screws|Retractor')]['Final_mTRE_mm'].values

    # Plot Setup
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)

    # ----------------------------------------------------
    # Panel (a): Linear CED (0 to 3.0 mm)
    # ----------------------------------------------------
    ax = axes[0, 0]
    taus_linear = np.linspace(0.0, 3.0, 300)

    ax.plot(taus_linear, compute_ced(dicom_clean, taus_linear), 'b-', lw=2.5, label='Hospital DICOM (Clean)')
    ax.plot(taus_linear, compute_ced(dicom_noisy, taus_linear), 'b--', lw=2.0, label='Hospital DICOM (Low-Dose)')

    ax.plot(taus_linear, compute_ced(amos_clean, taus_linear), 'g-', lw=2.5, label='AMOS (NCT Cohort, Clean)')
    ax.plot(taus_linear, compute_ced(amos_noisy, taus_linear), 'g--', lw=2.0, label='AMOS (NCT Cohort, Low-Dose)')

    ax.plot(taus_linear, compute_ced(kaggle_clean, taus_linear), 'm-', lw=2.5, label='Kaggle Lumbar (Clean)')
    ax.plot(taus_linear, compute_ced(kaggle_noisy, taus_linear), 'm--', lw=2.0, label='Kaggle Lumbar (Low-Dose)')

    ax.plot(taus_linear, compute_ced(monoplanar_errors, taus_linear), 'r:', lw=2.0, label='Baseline Monoplanar FVR')

    # Surgical tolerance lines
    ax.axvline(1.0, color='darkgreen', linestyle='-.', lw=1.5, alpha=0.8, label='Sub-millimeter (1.0 mm)')
    ax.axvline(2.0, color='crimson', linestyle='-.', lw=1.5, alpha=0.8, label='Clinical Acceptability (2.0 mm)')

    ax.set_title('(a) Multi-Center Cumulative Error Distribution (CED)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Target Registration Error Threshold $\\tau$ [mm]', fontsize=11)
    ax.set_ylabel('Cumulative Success Rate [%]', fontsize=11)
    ax.set_xlim([0.0, 3.0])
    ax.set_ylim([0, 105])
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.5)

    # ----------------------------------------------------
    # Panel (b): Logarithmic CED (0.01 mm to 50 mm)
    # ----------------------------------------------------
    ax = axes[0, 1]
    taus_log = np.logspace(np.log10(0.01), np.log10(50.0), 300)

    ax.plot(taus_log, compute_ced(dicom_clean, taus_log), 'b-', lw=2.5, label='Hospital DICOM (Clean)')
    ax.plot(taus_log, compute_ced(dicom_noisy, taus_log), 'b--', lw=2.0, label='Hospital DICOM (Low-Dose)')
    ax.plot(taus_log, compute_ced(amos_clean, taus_log), 'g-', lw=2.5, label='AMOS NCT Cohort (Clean)')
    ax.plot(taus_log, compute_ced(kaggle_clean, taus_log), 'm-', lw=2.5, label='Kaggle Lumbar (Clean)')
    ax.plot(taus_log, compute_ced(monoplanar_errors, taus_log), 'r:', lw=2.0, label='Monoplanar FVR (Exp 6)')
    ax.plot(taus_log, compute_ced(metal_unmasked, taus_log), 'k-.', lw=2.0, label='Unmasked Metal Failure (Exp 10)')

    ax.axvline(1.0, color='darkgreen', linestyle='-.', lw=1.5, alpha=0.8)
    ax.axvline(2.0, color='crimson', linestyle='-.', lw=1.5, alpha=0.8)

    ax.set_xscale('log')
    ax.set_title('(b) Full-Range Logarithmic CED vs. Failure Baselines', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Logarithmic mTRE Threshold $\\tau$ [mm]', fontsize=11)
    ax.set_ylabel('Cumulative Success Rate [%]', fontsize=11)
    ax.set_xlim([0.01, 50.0])
    ax.set_ylim([0, 105])
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # ----------------------------------------------------
    # Panel (c): Box-and-Whisker Cohort Distributions
    # ----------------------------------------------------
    ax = axes[1, 0]
    cohort_data = [
        dicom_clean, dicom_noisy,
        amos_clean, amos_noisy,
        kaggle_clean, kaggle_noisy
    ]
    cohort_labels = [
        'DICOM\nClean', 'DICOM\nLow-Dose',
        'AMOS\nClean', 'AMOS\nLow-Dose',
        'Kaggle\nClean', 'Kaggle\nLow-Dose'
    ]
    colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a', '#9467bd', '#c5b0d5']

    bplot = ax.boxplot(cohort_data, tick_labels=cohort_labels, patch_artist=True,
                       medianprops={'color': 'black', 'lw': 1.5})
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(1.0, color='darkgreen', linestyle='-.', lw=1.5, alpha=0.8, label='Sub-mm (1.0 mm)')
    ax.axhline(2.0, color='crimson', linestyle='-.', lw=1.5, alpha=0.8, label='Clinical (2.0 mm)')

    ax.set_title('(c) Multi-Center Clinical Error Distributions', fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('Final mTRE [mm]', fontsize=11)
    ax.set_ylim([-0.1, 3.5])
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    # ----------------------------------------------------
    # Panel (d): Success Rates Summary Bar Chart
    # ----------------------------------------------------
    ax = axes[1, 1]
    cohort_names = ['Hospital DICOM', 'AMOS (NCT Cohort)', 'Kaggle Lumbar']
    sub_mm_clean = [np.mean(dicom_clean <= 1.0) * 100, np.mean(amos_clean <= 1.0) * 100, np.mean(kaggle_clean <= 1.0) * 100]
    sub_mm_noisy = [np.mean(dicom_noisy <= 1.0) * 100, np.mean(amos_noisy <= 1.0) * 100, np.mean(kaggle_noisy <= 1.0) * 100]
    clin_clean = [np.mean(dicom_clean <= 2.0) * 100, np.mean(amos_clean <= 2.0) * 100, np.mean(kaggle_clean <= 2.0) * 100]
    clin_noisy = [np.mean(dicom_noisy <= 2.0) * 100, np.mean(amos_noisy <= 2.0) * 100, np.mean(kaggle_noisy <= 2.0) * 100]

    x = np.arange(len(cohort_names))
    width = 0.2

    ax.bar(x - 1.5*width, clin_clean, width, label='Clinical (< 2.0 mm) Clean', color='#2b5c8f')
    ax.bar(x - 0.5*width, clin_noisy, width, label='Clinical (< 2.0 mm) Low-Dose', color='#4f81bd')
    ax.bar(x + 0.5*width, sub_mm_clean, width, label='Sub-mm (< 1.0 mm) Clean', color='#38761d')
    ax.bar(x + 1.5*width, sub_mm_noisy, width, label='Sub-mm (< 1.0 mm) Low-Dose', color='#6aa84f')

    ax.set_title('(d) Clinical and Sub-Millimeter Success Rates (%)', fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('Success Rate [%]', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(cohort_names, fontsize=10, fontweight='bold')
    ax.set_ylim([0, 115])
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.5)

    for i in range(len(cohort_names)):
        ax.text(x[i] - 1.5*width, clin_clean[i] + 2, f"{clin_clean[i]:.0f}%", ha='center', fontsize=8, fontweight='bold')
        ax.text(x[i] - 0.5*width, clin_noisy[i] + 2, f"{clin_noisy[i]:.0f}%", ha='center', fontsize=8)
        ax.text(x[i] + 0.5*width, sub_mm_clean[i] + 2, f"{sub_mm_clean[i]:.0f}%", ha='center', fontsize=8, fontweight='bold')
        ax.text(x[i] + 1.5*width, sub_mm_noisy[i] + 2, f"{sub_mm_noisy[i]:.0f}%", ha='center', fontsize=8)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "fig14_unified_multicenter_ced.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Figure saved to: {fig_path}")

    # Copy to brain artifacts
    brain_art_dir = Path("C:/Users/paree/.gemini/antigravity/brain/db84ca2a-35bb-4b49-a099-1ebdc0eb4333")
    if brain_art_dir.exists():
        import shutil
        shutil.copy2(fig_path, brain_art_dir / "fig14_unified_multicenter_ced.png")
        print(f"Copied to brain artifacts: {brain_art_dir / 'fig14_unified_multicenter_ced.png'}")


if __name__ == '__main__':
    generate_unified_ced_benchmark()
