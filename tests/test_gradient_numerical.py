import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fvr.diff_fvr import DifferentiableFourierVolumeRenderer
from src.fvr.gradients import SpectralGradientEngine


def test_numerical_gradients():
    # Synthetic test cube
    np.random.seed(42)
    cube = np.zeros((32, 32, 32), dtype=np.float64)
    cube[10:22, 10:22, 10:22] = 1.0

    renderer = DifferentiableFourierVolumeRenderer(cube, pad_factor=2.0)
    engine = SpectralGradientEngine(renderer)

    # Reference target
    f_targ_ap = renderer.extract_frequency_slice(0, 0, 0, tx=0, ty=0, view='ap')
    f_targ_lat = renderer.extract_frequency_slice(0, 0, 0, tx=0, ty=0, view='lateral')

    # Perturbed test pose
    test_pose = np.array([1.0, -1.0, 0.5, 1.5, -1.0, 2.0])

    loss0, grad_analytical = engine.compute_biplanar_loss_and_gradients(test_pose, f_targ_ap, f_targ_lat)

    # Numerical finite differences
    eps = 1e-4
    grad_numerical = np.zeros(6)
    for i in range(6):
        p_plus = np.copy(test_pose)
        p_minus = np.copy(test_pose)
        p_plus[i] += eps
        p_minus[i] -= eps

        def get_loss(p):
            yaw, pitch, roll, tx, ty, tz = p
            f_ap = renderer.extract_frequency_slice(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
            f_lat = renderer.extract_frequency_slice(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')
            ncc_ap = renderer.compute_spectral_ncc(f_ap, f_targ_ap)
            ncc_lat = renderer.compute_spectral_ncc(f_lat, f_targ_lat)
            return 0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat)

        l_plus = get_loss(p_plus)
        l_minus = get_loss(p_minus)
        grad_numerical[i] = (l_plus - l_minus) / (2.0 * eps)

    print("Parameter names: [yaw, pitch, roll, tx, ty, tz]")
    print(f"Analytical gradient: {np.round(grad_analytical, 6)}")
    print(f"Numerical gradient:  {np.round(grad_numerical, 6)}")
    diff = np.abs(grad_analytical - grad_numerical)
    print(f"Absolute diff:       {np.round(diff, 6)}")
    rel_err = diff / (np.abs(grad_numerical) + 1e-8)
    print(f"Relative error:      {np.round(rel_err, 4)}")


if __name__ == '__main__':
    test_numerical_gradients()
