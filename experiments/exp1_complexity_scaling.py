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
from fvr.raycasting_drr import raycast_drr


def main():
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  EXPERIMENT 1: Computational Complexity & Scaling Analysis")
    print("=" * 65)

    grid_sizes = [64, 128, 192, 256]
    n_repeats = 10

    results = []

    for sz in grid_sizes:
        print(f"\nEvaluating Volume Size: {sz} x {sz} x {sz} ({sz**3:,} voxels)...")
        vol_hu, _ = create_spine_phantom(size=sz)
        vol_mu = hu_to_attenuation(vol_hu)

        # Measure Preprocessing Time (3D FFT)
        t_preps = []
        for _ in range(3):
            t0 = time.time()
            renderer = DifferentiableFourierVolumeRenderer(vol_mu, pad_factor=2.0)
            t_preps.append(time.time() - t0)
        t_prep_mean = float(np.mean(t_preps))

        # Measure FVR Rendering Time (Per-Projection)
        fvr_times = []
        angles = np.linspace(0, 360, n_repeats, endpoint=False)
        for ang in angles:
            t0 = time.time()
            _ = renderer.render_drr(yaw_deg=ang, view='lateral')
            fvr_times.append(time.time() - t0)
        t_fvr_mean = float(np.mean(fvr_times)) * 1000.0  # ms
        t_fvr_std = float(np.std(fvr_times)) * 1000.0

        # Measure Raycasting Ground Truth Rendering Time
        ray_times = []
        for ang in angles[:min(5, len(angles))]:
            t0 = time.time()
            _ = raycast_drr(vol_mu, yaw_deg=ang)
            ray_times.append(time.time() - t0)
        t_ray_mean = float(np.mean(ray_times)) * 1000.0  # ms
        t_ray_std = float(np.std(ray_times)) * 1000.0

        speedup = t_ray_mean / t_fvr_mean

        print(f"  3D FFT Preprocessing: {t_prep_mean:.3f} s")
        print(f"  FVR Latency:          {t_fvr_mean:.2f} ± {t_fvr_std:.2f} ms ({1000/t_fvr_mean:.1f} fps)")
        print(f"  Raycast Latency:      {t_ray_mean:.2f} ± {t_ray_std:.2f} ms ({1000/t_ray_mean:.1f} fps)")
        print(f"  Speedup Factor:       {speedup:.1f}x faster")

        results.append({
            'Volume_Size': sz,
            'Total_Voxels': sz**3,
            'Prep_Time_s': t_prep_mean,
            'FVR_ms': t_fvr_mean,
            'FVR_std_ms': t_fvr_std,
            'Raycast_ms': t_ray_mean,
            'Raycast_std_ms': t_ray_std,
            'Speedup': speedup
        })

    df = pd.DataFrame(results)
    csv_path = out_dir / "table1_complexity_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark table to: {csv_path}")

    # Publication Figure 1: Log-Log Complexity Plot & Speedup Curve
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Plot 1: Log-Log Runtime vs Volume Dimension
    ax1.plot(df['Volume_Size'], df['Raycast_ms'], 'r-o', linewidth=2.5, markersize=8, label='Spatial Raycasting (O(N³))')
    ax1.plot(df['Volume_Size'], df['FVR_ms'], 'b-s', linewidth=2.5, markersize=8, label='Fourier Volume Rendering (O(N²))')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xticks(grid_sizes)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel('Volume Dimension N (Voxel Grid N × N × N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Projection Latency (ms, log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Computational Complexity Scaling', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    # Plot 2: Speedup Factor Bar Chart
    bars = ax2.bar([str(s) for s in df['Volume_Size']], df['Speedup'], color='#2b5c8f', width=0.55, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Volume Dimension N', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup Factor (Raycast / FVR)', fontsize=12, fontweight='bold')
    ax2.set_title('Acceleration Factor over Spatial Raycasting', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', ls="--", alpha=0.4)

    # Add numeric labels atop bars
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}x", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig1_path = out_dir / "fig1_complexity_scaling.png"
    fig.savefig(fig1_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 1 to: {fig1_path}")


if __name__ == "__main__":
    main()
