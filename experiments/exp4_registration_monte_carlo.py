import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.dicom_loader import create_spine_phantom, hu_to_attenuation
from fvr.registration_3d2d import Spine3D2DRegistrationEngine, compute_mtre


def main():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  EXPERIMENT 4: Monte Carlo 3D-2D Spine Registration Benchmark")
    print("  (50 Stochastic Trials: Spectral FVR vs Spatial Baseline)")
    print("=" * 65)

    size = 128
    vol_hu, _ = create_spine_phantom(size=size)
    vol_mu = hu_to_attenuation(vol_hu)

    renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
    engine = Spine3D2DRegistrationEngine(renderer, view='lateral')

    # Fix random seed for exact scientific reproducibility
    np.random.seed(42)
    n_trials = 50

    results_fvr = []
    results_spatial = []

    print(f"Executing {n_trials} Monte Carlo registration trials...")
    for t_idx in range(n_trials):
        # Generate random ground-truth patient perturbation:
        # yaw, pitch, roll in [-8, 8] deg, tx, ty in [-8, 8] mm
        gt_yaw = np.random.uniform(-8.0, 8.0)
        gt_pitch = np.random.uniform(-4.0, 4.0)
        gt_roll = np.random.uniform(-4.0, 4.0)
        gt_tx = np.random.uniform(-6.0, 6.0)
        gt_ty = np.random.uniform(-6.0, 6.0)
        true_pose = (gt_yaw, gt_pitch, gt_roll, gt_tx, gt_ty)

        # Generate target intraoperative fluoroscopy image
        target_drr = renderer.render_drr(
            yaw_deg=gt_yaw, pitch_deg=gt_pitch, roll_deg=gt_roll, tx=gt_tx, ty=gt_ty, view='lateral', crop_to_roi=True
        )

        # Initial misaligned guess: identity (0, 0, 0, 0, 0)
        init_pose = (0.0, 0.0, 0.0, 0.0, 0.0)
        init_mtre = compute_mtre(true_pose, init_pose, vol_mu.shape)

        # Run 1: Proposed Spectral-NCC FVR Registration (Fast)
        res_fvr = engine.register(target_drr, initial_pose=init_pose, use_spectral_loss=True, max_iter=60)
        final_mtre_fvr = compute_mtre(true_pose, res_fvr['estimated_pose'], vol_mu.shape)
        results_fvr.append({
            'Trial': t_idx + 1,
            'Method': 'Spectral FVR (Proposed)',
            'Init_mTRE_mm': init_mtre,
            'Final_mTRE_mm': final_mtre_fvr,
            'Success': final_mtre_fvr < 2.0,
            'Time_s': res_fvr['total_time_seconds'],
            'Time_per_eval_ms': res_fvr['time_per_eval_ms'],
            'Iterations': res_fvr['iterations'],
            'Evals': res_fvr['function_evals']
        })

        # Run 2: Spatial-Domain Baseline (Conventional) on a subset of 15 trials for speed
        if t_idx < 15:
            res_sp = engine.register(target_drr, initial_pose=init_pose, use_spectral_loss=False, max_iter=60)
            final_mtre_sp = compute_mtre(true_pose, res_sp['estimated_pose'], vol_mu.shape)
            results_spatial.append({
                'Trial': t_idx + 1,
                'Method': 'Spatial Baseline',
                'Init_mTRE_mm': init_mtre,
                'Final_mTRE_mm': final_mtre_sp,
                'Success': final_mtre_sp < 2.0,
                'Time_s': res_sp['total_time_seconds'],
                'Time_per_eval_ms': res_sp['time_per_eval_ms'],
                'Iterations': res_sp['iterations'],
                'Evals': res_sp['function_evals']
            })

        if (t_idx + 1) % 10 == 0:
            print(f"  Completed {t_idx + 1:2d}/{n_trials} trials | Avg FVR mTRE: {np.mean([r['Final_mTRE_mm'] for r in results_fvr]):.2f} mm | Avg FVR Time: {np.mean([r['Time_s'] for r in results_fvr]):.2f} s")

    df_fvr = pd.DataFrame(results_fvr)
    df_sp = pd.DataFrame(results_spatial)
    df_all = pd.concat([df_fvr, df_sp], ignore_index=True)
    df_all.to_csv(out_dir / "table4_registration_benchmark.csv", index=False)

    # Print summary statistics
    print("\n" + "=" * 65)
    print("  MONTE CARLO REGISTRATION RESULTS")
    print("=" * 65)
    print(f"Initial Mean mTRE:        {df_fvr['Init_mTRE_mm'].mean():.2f} ± {df_fvr['Init_mTRE_mm'].std():.2f} mm")
    print(f"Proposed FVR Final mTRE:  {df_fvr['Final_mTRE_mm'].mean():.2f} ± {df_fvr['Final_mTRE_mm'].std():.2f} mm")
    print(f"Proposed FVR Success Rate: {df_fvr['Success'].mean() * 100:.1f}% (< 2.0 mm)")
    print(f"Proposed FVR Total Time:   {df_fvr['Time_s'].mean():.2f} ± {df_fvr['Time_s'].std():.2f} s")
    print(f"Spatial Baseline Time:     {df_sp['Time_s'].mean():.2f} ± {df_sp['Time_s'].std():.2f} s")
    print(f"Registration Acceleration: {df_sp['Time_s'].mean() / df_fvr['Time_s'].mean():.1f}x faster")
    print("=" * 65)

    # Publication Figure 4: Registration Accuracy and Runtime Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel A: Cumulative Error Distribution Function (CED)
    thresholds = np.linspace(0, 8, 100)
    ced_fvr = [np.mean(df_fvr['Final_mTRE_mm'] <= th) * 100 for th in thresholds]
    ced_sp = [np.mean(df_sp['Final_mTRE_mm'] <= th) * 100 for th in thresholds]
    ced_init = [np.mean(df_fvr['Init_mTRE_mm'] <= th) * 100 for th in thresholds]

    ax1.plot(thresholds, ced_fvr, 'b-', linewidth=2.5, label='Proposed Spectral FVR')
    ax1.plot(thresholds, ced_sp, 'r--', linewidth=2.0, label='Spatial Baseline')
    ax1.plot(thresholds, ced_init, 'k:', linewidth=1.5, label='Initial Misalignment')
    ax1.axvline(2.0, color='green', linestyle='--', alpha=0.7, label='Clinical Success Limit (2.0 mm)')
    ax1.set_xlabel('Target Registration Error mTRE (mm)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Cumulative Error Distribution (CED)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel B: Runtime Boxplot Comparison
    bp = ax2.boxplot(
        [df_fvr['Time_s'], df_sp['Time_s']],
        tick_labels=['Proposed Spectral FVR', 'Spatial Baseline'],
        patch_artist=True,
        widths=0.45
    )
    colors = ['#1f77b4', '#d62728']
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Convergence Time (Seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Optimization Runtime Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig4_path = out_dir / "fig4_registration_benchmark.png"
    fig.savefig(fig4_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 4 to: {fig4_path}")


if __name__ == "__main__":
    main()
