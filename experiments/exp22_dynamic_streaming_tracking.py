"""
Experiment 22: Real-Time Dynamic Fluoroscopy Video Tracking Benchmark (30+ FPS).

Physical Clinical Reality:
  In live computer-assisted surgical navigation (e.g., navigated pedicle screw insertion,
  vertebroplasty, or needle biopsy), fluoroscopic video streams at 15-30 frames per second.
  Conventional 3D-2D registration takes 1.0 to 15.0 seconds per frame, inducing a prohibitive
  lag of 30 to 450 frames.
  
Algorithmic Solution:
  Leveraging Diff-FVR with closed-form analytical Fourier gradients and warm-start pose propagation:
  Between consecutive frames (Delta t = 33 ms), inter-frame displacement is small (|Delta t| < 0.2 mm, |Delta theta| < 0.2 deg).
  Warm-starting from the previous converged pose T_{t-1} enables analytical gradient steps to converge
  in 2-5 iterations per frame, achieving sustained video-rate (>30 FPS) tracking on CPU workstations.

Benchmark Scenario:
  60 consecutive fluoroscopy frames (2.0 seconds of 30 FPS video) incorporating:
  1. Physiological respiratory excursion: Sinusoidal cranial-caudal motion (amplitude 1.5 mm, frequency 0.25 Hz).
  2. Operating table micro-shift: Sudden 1.0 mm step translation at frame 30.
  3. Continuous measurement of latency (ms/frame), effective FPS, mTRE (mm), and drift accumulation.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fvr.dicom_loader import load_dicom_series, hu_to_attenuation
from src.fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from src.fvr.gradients import SpectralGradientEngine
from src.fvr.registration_3d2d import compute_mtre_6dof

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR = Path(r"C:\Users\paree\.gemini\antigravity\brain\db84ca2a-35bb-4b49-a099-1ebdc0eb4333")


def run_dynamic_video_tracking_benchmark():
    print("=" * 75)
    print("EXPERIMENT 22: REAL-TIME DYNAMIC FLUOROSCOPY VIDEO TRACKING (30+ FPS)")
    print("=" * 75)

    csv_path = OUTPUT_DIR / "table22_dynamic_video_tracking.csv"
    fig_path = OUTPUT_DIR / "fig22_dynamic_video_tracking.png"

    # Load and resample clinical DICOM
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
    grad_engine = SpectralGradientEngine(renderer)
    print(f"Volume loaded: {vol_mu.shape}, Voxel spacing: {vox_size:.3f} mm")

    # Simulation setup: 60 frames at 30 FPS (2.0 seconds)
    n_frames = 60
    fps_target = 30.0
    dt = 1.0 / fps_target

    # Trajectory generation:
    # Respiratory excursion along z (cranial-caudal): A * sin(2*pi*f*t)
    # A = 1.5 mm, f = 0.5 Hz (1 cycle per 2 seconds)
    # At frame 30: sudden table micro-shift of +1.0 mm along x (depth) and +0.5 deg pitch
    times = np.arange(n_frames) * dt
    true_poses = []
    for i, t in enumerate(times):
        yaw = 0.3 * np.sin(2 * np.pi * 0.5 * t)
        pitch = 0.5 * np.cos(2 * np.pi * 0.5 * t) + (0.8 if i >= 30 else 0.0)
        roll = 0.2 * np.sin(2 * np.pi * 0.5 * t)
        tx = 0.4 * np.cos(2 * np.pi * 0.5 * t) + (1.0 if i >= 30 else 0.0)
        ty = 0.3 * np.sin(2 * np.pi * 0.5 * t)
        tz = 1.5 * np.sin(2 * np.pi * 0.5 * t)
        true_poses.append(np.array([yaw, pitch, roll, tx, ty, tz], dtype=np.float64))

    print(f"Simulating {n_frames} dynamic fluoroscopy frames...")

    # Tracking parameters
    current_estimate = np.copy(true_poses[0]) + np.array([0.5, -0.5, 0.3, 0.8, -0.6, 0.7]) # Initial acquisition offset
    max_iters_per_frame = 5
    lr_rot = 0.08
    lr_trans = 0.25

    frame_records = []

    for frame_idx in range(n_frames):
        target_pose = true_poses[frame_idx]

        # Generate live fluoroscopy projections for current ground truth pose
        f_targ_ap = renderer.extract_frequency_slice(target_pose[0], target_pose[1], target_pose[2],
                                                     tx=target_pose[3], ty=target_pose[5], view='ap')
        f_targ_lat = renderer.extract_frequency_slice(target_pose[0], target_pose[1], target_pose[2],
                                                      tx=target_pose[4], ty=target_pose[5], view='lateral')

        t_frame_start = time.perf_counter()

        # Warm-start optimization loop: 2-5 fast analytical gradient updates
        for iter_step in range(max_iters_per_frame):
            loss, grad = grad_engine.compute_biplanar_loss_and_gradients(current_estimate, f_targ_ap, f_targ_lat)
            if loss < 1e-6:
                break

            # Gradient update with decoupled learning rates (rot vs trans)
            current_estimate[:3] -= lr_rot * grad[:3]
            current_estimate[3:] -= lr_trans * grad[3:]

        latency_ms = (time.perf_counter() - t_frame_start) * 1000.0
        effective_fps = 1000.0 / latency_ms if latency_ms > 0 else 999.0

        # Evaluate tracking accuracy
        mtre = compute_mtre_6dof(tuple(target_pose), tuple(current_estimate), vol_mu.shape)

        frame_records.append({
            'Frame': frame_idx,
            'Time_sec': times[frame_idx],
            'Latency_ms': latency_ms,
            'Effective_FPS': effective_fps,
            'mTRE_mm': mtre,
            'True_Tz_mm': target_pose[5],
            'Est_Tz_mm': current_estimate[5],
            'True_Tx_mm': target_pose[3],
            'Est_Tx_mm': current_estimate[3],
            'Loss': loss,
            'Submillimeter': bool(mtre < 1.0)
        })

        if frame_idx % 10 == 0 or frame_idx == 30:
            print(f"  Frame {frame_idx:02d} (t={times[frame_idx]:.2f}s): Latency = {latency_ms:.1f} ms ({effective_fps:.1f} FPS), mTRE = {mtre:.4f} mm")

    df = pd.DataFrame(frame_records)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved tracking CSV to: {csv_path}")

    avg_latency = df['Latency_ms'].mean()
    avg_fps = df['Effective_FPS'].mean()
    avg_mtre = df['mTRE_mm'].mean()
    max_mtre = df['mTRE_mm'].max()
    submm_rate = (df['mTRE_mm'] < 1.0).mean() * 100.0

    print(f"\nSUMMARY BENCHMARK:")
    print(f"  Average Frame Latency: {avg_latency:.2f} ms")
    print(f"  Average Frame Rate:    {avg_fps:.1f} FPS (Target >= 30 FPS: {avg_fps >= 30.0})")
    print(f"  Mean Tracking Error:   {avg_mtre:.4f} mm")
    print(f"  Max Tracking Error:    {max_mtre:.4f} mm")
    print(f"  Sub-millimeter Rate:   {submm_rate:.1f}%")

    # Plot Figure 22
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    # Panel 1: Real-time Trajectory Tracking (True vs Estimated Cranial-Caudal Excursion Tz)
    axes[0].plot(df['Time_sec'], df['True_Tz_mm'], 'k-', linewidth=2.5, label='Ground-Truth Respiratory Tz (mm)')
    axes[0].plot(df['Time_sec'], df['Est_Tz_mm'], 'r--', linewidth=2.0, label='Diff-FVR Estimated Tz (mm)')
    axes[0].axvline(x=times[30], color='blue', linestyle=':', linewidth=1.5, label='Sudden Table Shift (Frame 30)')
    axes[0].set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Cranial-Caudal Position Tz (mm)', fontsize=10, fontweight='bold')
    axes[0].set_title('Continuous Real-Time Trajectory Tracking', fontsize=11, fontweight='bold')
    axes[0].legend(loc='lower left', frameon=True, fontsize=9)

    # Panel 2: mTRE Tracking Error across 60 Frames
    axes[1].plot(df['Time_sec'], df['mTRE_mm'], 'o-', color='#27ae60', linewidth=2.0, markersize=5, label='Tracking mTRE (mm)')
    axes[1].axhline(y=1.0, color='crimson', linestyle='--', linewidth=1.5, label='Sub-millimeter Threshold (1.0 mm)')
    axes[1].axvline(x=times[30], color='blue', linestyle=':', linewidth=1.5)
    axes[1].set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Mean Target Registration Error (mm)', fontsize=10, fontweight='bold')
    axes[1].set_title(f'Tracking Precision (Mean mTRE = {avg_mtre:.3f} mm)', fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True, fontsize=9)
    axes[1].set_ylim(-0.05, max(1.5, max_mtre * 1.3))

    # Panel 3: Latency & Effective FPS Throughput
    color = '#2980b9'
    axes[2].plot(df['Time_sec'], df['Latency_ms'], 's-', color=color, linewidth=2.0, markersize=5, label='Frame Latency (ms)')
    axes[2].axhline(y=33.33, color='orange', linestyle='--', linewidth=1.5, label='30 FPS Video Threshold (33.3 ms)')
    axes[2].set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('Frame Processing Latency (ms)', color=color, fontsize=10, fontweight='bold')
    axes[2].tick_params(axis='y', labelcolor=color)

    ax2_right = axes[2].twinx()
    color2 = '#8e44ad'
    ax2_right.plot(df['Time_sec'], df['Effective_FPS'], '^-', color=color2, linewidth=1.5, markersize=4, alpha=0.7, label='Effective FPS')
    ax2_right.set_ylabel('Throughput (Frames Per Second)', color=color2, fontsize=10, fontweight='bold')
    ax2_right.tick_params(axis='y', labelcolor=color2)
    ax2_right.grid(False)

    axes[2].set_title(f'Video-Rate Performance (Avg: {avg_fps:.1f} FPS / {avg_latency:.1f} ms)', fontsize=11, fontweight='bold')
    axes[2].legend(loc='upper left', frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")

    if BRAIN_DIR.exists():
        import shutil
        brain_fig = BRAIN_DIR / "fig22_dynamic_video_tracking.png"
        shutil.copy2(fig_path, brain_fig)
        print(f"Copied to brain artifacts: {brain_fig}")


if __name__ == '__main__':
    run_dynamic_video_tracking_benchmark()
