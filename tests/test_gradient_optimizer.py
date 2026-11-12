import sys
from pathlib import Path
import numpy as np
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fvr.dicom_loader import load_dicom_series
from src.fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from src.fvr.registration_3d2d import BiplanarSpineRegistrationEngine
from src.fvr.gradient_optimizer import GradientRegistrationOptimizer


def run_benchmark():
    dicom_dir = PROJECT_ROOT / "data" / "dicom"
    volume, spacing = load_dicom_series(dicom_dir)
    print(f"Loaded DICOM volume: {volume.shape}, spacing: {spacing}")

    renderer = DifferentiableFourierVolumeRenderer(volume, pad_factor=2.0)
    print("Renderer initialized with 3D FFT.")

    # Target ground truth DRRs at pose: [0, 0, 0, 0, 0, 0]
    target_ap = renderer.render_drr(0, 0, 0, tx=0, ty=0, view='ap')
    target_lat = renderer.render_drr(0, 0, 0, tx=0, ty=0, view='lateral')

    # Initial misalignment: [2.5 deg, -3.0 deg, 1.5 deg, 3.5 mm, -2.0 mm, 4.0 mm]
    init_pose = np.array([2.5, -3.0, 1.5, 3.5, -2.0, 4.0])
    init_mtre = np.linalg.norm(init_pose[3:])
    print(f"Initial pose: {init_pose}, Initial translation error: {init_mtre:.3f} mm")

    # --- 1. Classical Powell (Registration3D2D) ---
    print("\nRunning Classical Powell Optimizer...")
    powell_engine = BiplanarSpineRegistrationEngine(renderer)
    t0 = time.time()
    res_powell = powell_engine.register_biplanar(target_ap, target_lat, initial_pose=tuple(init_pose), max_iter=30)
    t_powell = time.time() - t0
    mtre_powell = np.linalg.norm(res_powell['estimated_pose'][3:])
    print(f"Powell: Time={t_powell:.3f}s, Final mTRE={mtre_powell:.4f}mm, Evals={res_powell['function_evals']}")

    # --- 2. Gradient Adam Optimizer ---
    print("\nRunning Analytical Gradient Adam Optimizer...")
    f_targ_ap = powell_engine._prepare_spectrum(target_ap, 'ap')
    f_targ_lat = powell_engine._prepare_spectrum(target_lat, 'lateral')
    grad_engine = GradientRegistrationOptimizer(renderer)
    t0 = time.time()
    res_grad = grad_engine.register_adam(f_targ_ap, f_targ_lat, init_pose=init_pose, max_iter=30)
    t_grad = time.time() - t0
    mtre_grad = np.linalg.norm(res_grad['final_pose'][3:])
    print(f"Adam: Time={t_grad:.3f}s, Final mTRE={mtre_grad:.4f}mm, Iterations={res_grad['iterations']}, Final Loss={res_grad['final_loss']:.6f}")
    speedup = t_powell / t_grad
    print(f"\nSPEEDUP: {speedup:.2f}x faster with analytical gradients!")


if __name__ == '__main__':
    run_benchmark()
