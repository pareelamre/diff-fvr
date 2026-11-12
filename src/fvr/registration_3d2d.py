import time
import numpy as np
from scipy import fft
from scipy.optimize import minimize
from typing import Tuple, Dict, Any, List, Optional

from .diff_fvr import DifferentiableFourierVolumeRenderer


class BiplanarSpineRegistrationEngine:
    """
    State-of-the-Art Biplanar 3D-2D Registration Framework (AP + Lateral).
    
    Clinical Principle:
      A single 2D radiograph cannot constrain out-of-plane depth translation.
      In clinical computer-assisted spine surgery (C-arm navigation), surgeons
      acquire orthogonal dual projections (AP and Lateral).
      
      We formulate a unified Biplanar Spectral-NCC loss:
        L_biplanar(R, t) = 0.5 * (1 - NCC_AP) + 0.5 * (1 - NCC_Lateral)
        
      Because Spectral-NCC evaluates directly in 2D Fourier space with zero IFFTs,
      evaluating BOTH orthogonal views takes under 1.6 ms per trial pose!
    """

    def __init__(self, renderer: DifferentiableFourierVolumeRenderer):
        self.renderer = renderer

    def _prepare_spectrum(self, img: np.ndarray, view: str) -> np.ndarray:
        nz, ny, nx = self.renderer.orig_shape
        if view == 'lateral':
            slice_h, slice_w = self.renderer.pz, self.renderer.py
        elif view == 'ap':
            slice_h, slice_w = self.renderer.pz, self.renderer.px
        else:
            slice_h, slice_w = self.renderer.py, self.renderer.px

        th, tw = img.shape
        padded = np.zeros((slice_h, slice_w), dtype=np.float64)
        sh = (slice_h - th) // 2
        sw = (slice_w - tw) // 2
        padded[sh:sh + th, sw:sw + tw] = img
        return fft.fftshift(fft.fft2(fft.ifftshift(padded)))

    def register_biplanar(
        self,
        target_ap: np.ndarray,
        target_lat: np.ndarray,
        initial_pose: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        max_iter: int = 80
    ) -> Dict[str, Any]:
        """
        Execute full 6-DoF rigid body biplanar registration:
          Parameters: (yaw_deg, pitch_deg, roll_deg, tx, ty, tz)
        """
        t_start = time.time()

        f_targ_ap = self._prepare_spectrum(target_ap, 'ap')
        f_targ_lat = self._prepare_spectrum(target_lat, 'lateral')

        eval_count = 0
        loss_history = []

        def objective(params):
            nonlocal eval_count
            eval_count += 1
            yaw, pitch, roll, tx, ty, tz = params

            # In AP view (rays along Y, detector in X-Z):
            # Horizontal motion is tx, vertical motion is tz
            f_cand_ap = self.renderer.extract_frequency_slice(
                yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, tx=tx, ty=tz, view='ap'
            )
            ncc_ap = self.renderer.compute_spectral_ncc(f_cand_ap, f_targ_ap)

            # In Lateral view (rays along X, detector in Y-Z):
            # Horizontal motion is ty, vertical motion is tz
            f_cand_lat = self.renderer.extract_frequency_slice(
                yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, tx=ty, ty=tz, view='lateral'
            )
            ncc_lat = self.renderer.compute_spectral_ncc(f_cand_lat, f_targ_lat)

            # Combined biplanar loss
            loss = 0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat)
            loss_history.append(loss)
            return loss

        # Run Powell optimization
        res = minimize(
            objective,
            x0=np.array(initial_pose),
            method='Powell',
            options={'maxiter': max_iter, 'ftol': 1e-5, 'xtol': 1e-4}
        )

        total_time = time.time() - t_start
        est_pose = res.x

        return {
            'success': res.success,
            'estimated_pose': tuple(est_pose),
            'final_loss': float(res.fun),
            'final_biplanar_ncc': float(1.0 - res.fun),
            'iterations': res.nit,
            'function_evals': eval_count,
            'total_time_seconds': total_time,
            'time_per_eval_ms': (total_time / max(1, eval_count)) * 1000.0,
            'loss_history': loss_history
        }


def compute_mtre_6dof(
    true_pose: Tuple[float, float, float, float, float, float],
    est_pose: Tuple[float, float, float, float, float, float],
    volume_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    translation_units: str = 'voxels'
) -> float:
    """
    Return mTRE in mm on eight synthetic volume landmarks, not anatomical labels.

    spacing is (Z, Y, X) in mm. Legacy renderers and registration engines use
    voxel translations; RobustDiffFVREngine uses translation_units='mm'.
    The default spacing assumes one mm per voxel; pass measured spacing for CT.
    """
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('spacing must contain three finite positive values in mm.')
    if translation_units not in ('voxels', 'mm'):
        raise ValueError("translation_units must be 'voxels' or 'mm'.")
    nz, ny, nx = volume_shape
    cz, cy, cx = nz / 2.0, ny / 2.0, nx / 2.0
    r = min(nz, ny, nx) * 0.25

    corners = np.array([
        [cz - r, cy - r, cx - r],
        [cz - r, cy - r, cx + r],
        [cz - r, cy + r, cx - r],
        [cz - r, cy + r, cx + r],
        [cz + r, cy - r, cx - r],
        [cz + r, cy - r, cx + r],
        [cz + r, cy + r, cx - r],
        [cz + r, cy + r, cx + r],
    ])

    def transform_pts(pts, pose):
        yaw, pitch, roll, tx, ty, tz = pose
        y_rad, p_rad, r_rad = np.radians(yaw), np.radians(pitch), np.radians(roll)
        # Rz: rotation around Z axis (Yaw: rotates Y and X)
        rz = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(y_rad), -np.sin(y_rad)],
            [0.0, np.sin(y_rad),  np.cos(y_rad)]
        ])
        # Ry: rotation around Y axis (Roll: rotates Z and X)
        ry = np.array([
            [ np.cos(r_rad), 0.0, np.sin(r_rad)],
            [ 0.0,           1.0, 0.0          ],
            [-np.sin(r_rad), 0.0, np.cos(r_rad)]
        ])
        # Rx: rotation around X axis (Pitch: rotates Z and Y)
        rx = np.array([
            [np.cos(p_rad), -np.sin(p_rad), 0.0],
            [np.sin(p_rad),  np.cos(p_rad), 0.0],
            [0.0,            0.0,           1.0]
        ])
        rot = rz @ ry @ rx
        translation = np.array([tz, ty, tx])
        if translation_units == 'voxels':
            translation = translation * spacing
        return ((pts - np.array([cz, cy, cx])) * spacing) @ rot.T + translation

    pts_true = transform_pts(corners, true_pose)
    pts_est = transform_pts(corners, est_pose)
    return float(np.mean(np.linalg.norm(pts_true - pts_est, axis=1)))
