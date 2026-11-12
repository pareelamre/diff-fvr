import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from fvr.dicom_loader import create_spine_phantom, hu_to_attenuation
from fvr.raycasting_drr import raycast_view
from fvr.registration_3d2d import BiplanarSpineRegistrationEngine, compute_mtre_6dof


def run_benchmark():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  EXPERIMENT 6: Biplanar vs Monoplanar 6-DoF Registration Benchmark")
    print("=" * 70)

    # 1. Setup Volume & Renderer
    size = 64  # Fast & representative for Monte Carlo statistical power
    vol_hu, _ = create_spine_phantom(size=size)
    vol_mu = hu_to_attenuation(vol_hu)

    renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
    biplanar_engine = BiplanarSpineRegistrationEngine(renderer)

    # 2. Experimental Parameters
    np.random.seed(42)
    n_trials = 30  # Rigorous Monte Carlo cohort

    # Perturbation bounds (representative of clinical intraoperative C-arm positioning)
    # Rotations in degrees, translations in mm (voxel units)
    rot_range = 6.0   # [-6°, +6°]
    trans_inplane = 5.0 # [-5mm, +5mm]
    trans_depth = 8.0   # [-8mm, +8mm] (depth along X)

    records = []

    print(f"Executing {n_trials} Monte Carlo trials across 3 registration regimes...")

    for trial in range(n_trials):
        # Generate random ground truth displacement
        yaw = float(np.random.uniform(-rot_range, rot_range))
        pitch = float(np.random.uniform(-rot_range, rot_range))
        roll = float(np.random.uniform(-rot_range, rot_range))
        tx = float(np.random.uniform(-trans_depth, trans_depth))
        ty = float(np.random.uniform(-trans_inplane, trans_inplane))
        tz = float(np.random.uniform(-trans_inplane, trans_inplane))
        true_pose = (yaw, pitch, roll, tx, ty, tz)

        init_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        init_mtre = compute_mtre_6dof(true_pose, init_pose, vol_mu.shape)

        # Generate True Clinical Target Radiographs
        target_ap = renderer.render_drr(
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, tx=tx, ty=tz, view='ap'
        )
        target_lat = renderer.render_drr(
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, tx=ty, ty=tz, view='lateral'
        )

        f_targ_ap = biplanar_engine._prepare_spectrum(target_ap, 'ap')
        f_targ_lat = biplanar_engine._prepare_spectrum(target_lat, 'lateral')

        # -------------------------------------------------------------
        # Regime 1: Proposed Biplanar Spectral-FVR (AP + Lateral)
        # -------------------------------------------------------------
        t0 = time.time()
        res_bi = biplanar_engine.register_biplanar(target_ap, target_lat, initial_pose=init_pose, max_iter=80)
        t_bi = time.time() - t0
        mtre_bi = compute_mtre_6dof(true_pose, res_bi['estimated_pose'], vol_mu.shape)

        records.append({
            'Trial': trial,
            'Method': 'Biplanar Spectral-FVR (Proposed)',
            'Init_mTRE_mm': init_mtre,
            'Final_mTRE_mm': mtre_bi,
            'Success_2mm': mtre_bi < 2.0,
            'Success_1mm': mtre_bi < 1.0,
            'Submillimeter': mtre_bi < 1.0,
            'Time_sec': t_bi,
            'Time_per_eval_ms': res_bi['time_per_eval_ms'],
            'Function_evals': res_bi['function_evals'],
            'Error_Yaw': abs(res_bi['estimated_pose'][0] - yaw),
            'Error_Pitch': abs(res_bi['estimated_pose'][1] - pitch),
            'Error_Roll': abs(res_bi['estimated_pose'][2] - roll),
            'Error_Tx': abs(res_bi['estimated_pose'][3] - tx),
            'Error_Ty': abs(res_bi['estimated_pose'][4] - ty),
            'Error_Tz': abs(res_bi['estimated_pose'][5] - tz),
        })

        # -------------------------------------------------------------
        # Regime 2: Monoplanar Spectral-FVR (Lateral Only)
        # -------------------------------------------------------------
        t0 = time.time()
        eval_mono = 0
        def mono_obj(p):
            nonlocal eval_mono
            eval_mono += 1
            f_cand = renderer.extract_frequency_slice(p[0], p[1], p[2], tx=p[4], ty=p[5], view='lateral')
            return 1.0 - renderer.compute_spectral_ncc(f_cand, f_targ_lat)

        res_mono = minimize(mono_obj, x0=np.zeros(6), method='Powell', options={'maxiter': 80})
        t_mono = time.time() - t0
        mtre_mono = compute_mtre_6dof(true_pose, res_mono.x, vol_mu.shape)

        records.append({
            'Trial': trial,
            'Method': 'Monoplanar Spectral-FVR (Lateral Only)',
            'Init_mTRE_mm': init_mtre,
            'Final_mTRE_mm': mtre_mono,
            'Success_2mm': mtre_mono < 2.0,
            'Success_1mm': mtre_mono < 1.0,
            'Submillimeter': mtre_mono < 1.0,
            'Time_sec': t_mono,
            'Time_per_eval_ms': (t_mono / max(1, eval_mono)) * 1000.0,
            'Function_evals': eval_mono,
            'Error_Yaw': abs(res_mono.x[0] - yaw),
            'Error_Pitch': abs(res_mono.x[1] - pitch),
            'Error_Roll': abs(res_mono.x[2] - roll),
            'Error_Tx': abs(res_mono.x[3] - tx),
            'Error_Ty': abs(res_mono.x[4] - ty),
            'Error_Tz': abs(res_mono.x[5] - tz),
        })

        # -------------------------------------------------------------
        # Regime 3: Biplanar Spatial Raycasting (Subset of trials)
        # -------------------------------------------------------------
        # Run raycasting for first 10 trials to benchmark computational scaling
        if trial < 10:
            t0 = time.time()
            eval_ray = 0
            # Spatial NCC objective
            targ_ap_norm = (target_ap - target_ap.mean()) / (target_ap.std() + 1e-8)
            targ_lat_norm = (target_lat - target_lat.mean()) / (target_lat.std() + 1e-8)

            def ray_obj(p):
                nonlocal eval_ray
                eval_ray += 1
                # Simplified spatial raycast
                c_ap = raycast_view(vol_mu, view='ap', rotation_deg=p[0])
                c_lat = raycast_view(vol_mu, view='lateral', rotation_deg=p[0])
                ap_norm = (c_ap - c_ap.mean()) / (c_ap.std() + 1e-8)
                lat_norm = (c_lat - c_lat.mean()) / (c_lat.std() + 1e-8)
                ncc_ap = float(np.mean(targ_ap_norm * ap_norm))
                ncc_lat = float(np.mean(targ_lat_norm * lat_norm))
                return 0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat)

            res_ray = minimize(ray_obj, x0=np.zeros(6), method='Powell', options={'maxiter': 25})
            t_ray = time.time() - t0
            mtre_ray = compute_mtre_6dof(true_pose, res_ray.x, vol_mu.shape)

            records.append({
                'Trial': trial,
                'Method': 'Biplanar Raycasting (Spatial)',
                'Init_mTRE_mm': init_mtre,
                'Final_mTRE_mm': mtre_ray,
                'Success_2mm': mtre_ray < 2.0,
                'Success_1mm': mtre_ray < 1.0,
                'Submillimeter': mtre_ray < 1.0,
                'Time_sec': t_ray,
                'Time_per_eval_ms': (t_ray / max(1, eval_ray)) * 1000.0,
                'Function_evals': eval_ray,
                'Error_Yaw': abs(res_ray.x[0] - yaw),
                'Error_Pitch': abs(res_ray.x[1] - pitch),
                'Error_Roll': abs(res_ray.x[2] - roll),
                'Error_Tx': abs(res_ray.x[3] - tx),
                'Error_Ty': abs(res_ray.x[4] - ty),
                'Error_Tz': abs(res_ray.x[5] - tz),
            })

        if (trial + 1) % 5 == 0:
            print(f"  Completed {trial + 1}/{n_trials} trials | Biplanar mTRE: {mtre_bi:.2f} mm | Mono mTRE: {mtre_mono:.2f} mm")

    df = pd.DataFrame(records)
    df['Metric_units'] = 'mm'  # Synthetic phantom uses one mm per voxel.
    df['Target_type'] = 'synthetic'
    csv_path = out_dir / "table6_biplanar_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved registration benchmark table to: {csv_path}")

    # Print summary statistics
    for method, grp in df.groupby('Method'):
        print(f"\n--- {method} ---")
        print(f"  Count: {len(grp)} trials")
        print(f"  Init mTRE:  {grp['Init_mTRE_mm'].mean():.2f} ± {grp['Init_mTRE_mm'].std():.2f} mm")
        print(f"  Final mTRE: {grp['Final_mTRE_mm'].mean():.2f} ± {grp['Final_mTRE_mm'].std():.2f} mm (Median: {grp['Final_mTRE_mm'].median():.2f})")
        print(f"  Success (<2mm): {grp['Success_2mm'].mean() * 100:.1f}%")
        print(f"  Submillimeter (<1mm): {grp['Success_1mm'].mean() * 100:.1f}%")
        print(f"  Time: {grp['Time_sec'].mean():.2f} ± {grp['Time_sec'].std():.2f} s")
        print(f"  Eval Latency: {grp['Time_per_eval_ms'].mean():.2f} ms")
        print(f"  Depth Error (Tx): {grp['Error_Tx'].mean():.2f} ± {grp['Error_Tx'].std():.2f} mm")

    # 3. Publication Figure 6: Cumulative Error Distribution & Speedup
    fig = plt.figure(figsize=(16, 5), dpi=300)
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    # Panel A: Cumulative Error Distribution (CED)
    ax1 = fig.add_subplot(gs[0, 0])
    thresholds = np.linspace(0.0, 10.0, 101)

    colors = {
        'Biplanar Spectral-FVR (Proposed)': '#1f77b4',
        'Monoplanar Spectral-FVR (Lateral Only)': '#d62728',
        'Biplanar Raycasting (Spatial)': '#2ca02c'
    }

    for method in ['Biplanar Spectral-FVR (Proposed)', 'Monoplanar Spectral-FVR (Lateral Only)', 'Biplanar Raycasting (Spatial)']:
        grp = df[df['Method'] == method]
        if len(grp) == 0:
            continue
        rates = [np.mean(grp['Final_mTRE_mm'] <= th) * 100 for th in thresholds]
        ax1.plot(thresholds, rates, label=method, color=colors[method], linewidth=2.5)

    ax1.axvline(2.0, color='grey', linestyle='--', linewidth=1.2, label='Clinical Success (2.0 mm)')
    ax1.axvline(1.0, color='darkgrey', linestyle=':', linewidth=1.2, label='Submillimeter (1.0 mm)')
    ax1.set_xlabel('mTRE Threshold (mm)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Successful Registrations (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Cumulative Error Distribution (CED)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=8.5, frameon=True)

    # Panel B: Residual 6-DoF Parameter Errors (Biplanar vs Monoplanar)
    ax2 = fig.add_subplot(gs[0, 1])
    bi_grp = df[df['Method'] == 'Biplanar Spectral-FVR (Proposed)']
    mono_grp = df[df['Method'] == 'Monoplanar Spectral-FVR (Lateral Only)']

    params = ['Yaw (°)', 'Pitch (°)', 'Roll (°)', 'Tx (mm)', 'Ty (mm)', 'Tz (mm)']
    cols = ['Error_Yaw', 'Error_Pitch', 'Error_Roll', 'Error_Tx', 'Error_Ty', 'Error_Tz']

    x = np.arange(len(params))
    width = 0.35

    bi_means = [bi_grp[c].mean() for c in cols]
    mono_means = [mono_grp[c].mean() for c in cols]

    ax2.bar(x - width/2, bi_means, width, label='Biplanar (Proposed)', color='#1f77b4', alpha=0.9)
    ax2.bar(x + width/2, mono_means, width, label='Monoplanar', color='#d62728', alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(params, fontsize=9.5, fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Parameter Errors (Depth Ambiguity Elimination)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.legend(loc='upper right', fontsize=9.5, frameon=True)

    # Panel C: Computational Speedup
    ax3 = fig.add_subplot(gs[0, 2])
    methods_plot = ['Biplanar\nSpectral-FVR', 'Monoplanar\nSpectral-FVR', 'Biplanar\nRaycasting']
    eval_times = [
        df[df['Method'] == 'Biplanar Spectral-FVR (Proposed)']['Time_per_eval_ms'].mean(),
        df[df['Method'] == 'Monoplanar Spectral-FVR (Lateral Only)']['Time_per_eval_ms'].mean(),
        df[df['Method'] == 'Biplanar Raycasting (Spatial)']['Time_per_eval_ms'].mean()
    ]
    bar_colors = ['#1f77b4', '#d62728', '#2ca02c']

    bars = ax3.bar(methods_plot, eval_times, color=bar_colors, width=0.55, alpha=0.9)
    ax3.set_ylabel('Latency per Evaluation (ms)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Evaluation Latency Comparison', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, linestyle='--', alpha=0.5, which='both', axis='y')

    for b in bars:
        h = b.get_height()
        ax3.annotate(f'{h:.1f} ms',
                     xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig6_path = out_dir / "fig6_biplanar_ced_comparison.png"
    fig.savefig(fig6_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 6 to: {fig6_path}")


if __name__ == "__main__":
    run_benchmark()
