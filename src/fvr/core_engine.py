"""
Research engine for parallel-beam Differentiable Fourier Volume Rendering.

Integrates:
  1. 3D Anti-Aliased Fourier Projection-Slice Theorem with 1:1 Physical Scale.
  2. Physical translations on an isotropic voxel grid (millimetres at this API).
  3. Automatic Radio-Opaque Metal Saturated Hardware Detection (>3% threshold).
  4. Active Masked Spatial Loss on Unoccluded Bone Pixels during Metal Occlusion.
  5. Ultra-Fast Frequency-Domain Spectral-NCC under Clean/Low-Dose Fluoroscopy.
"""

import time
import numpy as np
from scipy import fft
from scipy.optimize import minimize
from typing import Tuple, Dict, Any, Optional

from .diff_fvr import DifferentiableFourierVolumeRenderer


def detect_radio_opaque_metal(img: np.ndarray, saturation_threshold: float = 0.98) -> np.ndarray:
    """
    Segment dense radio-opaque surgical hardware (titanium pedicle screws, steel retractor blades)
    by detecting saturated plateaus in normalized fluoroscopy.
    """
    norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return norm >= saturation_threshold


class RobustDiffFVREngine:
    """
    Parallel-beam registration on a cubic, isotropic volume.

    Spacing is (Z, Y, X) in mm; poses are (yaw, pitch, roll, tx, ty, tz),
    with rotations in degrees and translations in mm. Target pixels must have
    the same spacing, orientation and field of view as the rendered projections.
    Cone-beam calibration and arbitrary acquired fluoroscopy are not supported.
    """

    def __init__(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        pad_factor: float = 2.0,
        sdd_mm: Optional[float] = None,
        sad_mm: Optional[float] = None
    ):
        if sdd_mm is not None or sad_mm is not None:
            raise ValueError('Only parallel-beam geometry is supported; SDD/SAD calibration is not implemented.')
        spacing = np.asarray(spacing, dtype=float)
        if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError('spacing must contain three finite positive values in mm (Z, Y, X).')
        if not np.allclose(spacing, spacing[0], rtol=1e-6, atol=0):
            raise ValueError('Resample the volume to isotropic spacing before registration.')
        if volume.ndim != 3 or len(set(volume.shape)) != 1 or min(volume.shape) < 2:
            raise ValueError('Use a cubic 3D volume with at least two voxels per axis.')
        if not np.all(np.isfinite(volume)):
            raise ValueError('volume must contain only finite values.')
        self.volume = volume.astype(np.float64)
        self.spacing = tuple(spacing)

        # Initialize base Fourier Volume Renderer
        self.renderer = DifferentiableFourierVolumeRenderer(
            self.volume, pad_factor=pad_factor, apply_premultiplication=True
        )

    def detect_metal_hardware_mask(self, img: np.ndarray, threshold: float = 0.98) -> np.ndarray:
        """Returns boolean mask where True indicates UN-OCCLUDED pixels."""
        return ~detect_radio_opaque_metal(img, saturation_threshold=threshold)

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

    def render_calibrated_drr(
        self,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        tx: float = 0.0,
        ty: float = 0.0,
        view: str = 'lateral',
        crop_to_roi: bool = True
    ) -> np.ndarray:
        """Render a parallel-beam DRR; legacy name, no perspective calibration.

        tx and ty are horizontal and vertical translations in mm in this view.
        """
        if view not in ('ap', 'lateral', 'axial'):
            raise ValueError('view must be ap, lateral or axial.')
        return self.renderer.render_drr(
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
            tx=tx / self.spacing[0], ty=ty / self.spacing[0],
            view=view, crop_to_roi=crop_to_roi
        )

    def register_biplanar_robust(
        self,
        target_ap: np.ndarray,
        target_lat: np.ndarray,
        initial_pose: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        handle_metal: bool = True,
        max_iter: int = 50,
        bounds_trans_mm: float = 25.0,
        bounds_rot_deg: float = 20.0
    ) -> Dict[str, Any]:
        """
        Execute parallel-beam 6-DoF registration, translations and bounds in mm:
          - If metal hardware detected (>3% saturation), activates Masked Spatial Loss.
          - Otherwise, runs Spectral-NCC.
        """
        for name, target in (('AP', target_ap), ('lateral', target_lat)):
            if target.shape != self.volume.shape[:2] or not np.all(np.isfinite(target)):
                raise ValueError(f'{name} target must be finite with shape {self.volume.shape[:2]}; '
                                 'match the parallel-beam projection grid before registration.')
            if np.ptp(target) == 0:
                raise ValueError(f'{name} target must contain intensity variation for registration.')
        initial_pose = np.asarray(initial_pose, dtype=float)
        if initial_pose.shape != (6,) or not np.all(np.isfinite(initial_pose)):
            raise ValueError('initial_pose must contain six finite values.')
        if not np.all(np.isfinite([bounds_trans_mm, bounds_rot_deg])) or min(bounds_trans_mm, bounds_rot_deg) <= 0:
            raise ValueError('Translation and rotation bounds must be finite and positive.')
        if max_iter < 1:
            raise ValueError('max_iter must be positive.')
        t0 = time.time()
        eval_count = 0

        # Metal detection
        mask_ap = detect_radio_opaque_metal(target_ap) if handle_metal else np.zeros_like(target_ap, dtype=bool)
        mask_lat = detect_radio_opaque_metal(target_lat) if handle_metal else np.zeros_like(target_lat, dtype=bool)
        metal_present = bool((np.mean(mask_ap) > 0.03) or (np.mean(mask_lat) > 0.03))

        # Physical bounds
        bounds = [
            (-bounds_rot_deg, bounds_rot_deg),
            (-bounds_rot_deg, bounds_rot_deg),
            (-bounds_rot_deg, bounds_rot_deg),
            (-bounds_trans_mm, bounds_trans_mm),
            (-bounds_trans_mm, bounds_trans_mm),
            (-bounds_trans_mm, bounds_trans_mm)
        ]

        if metal_present:
            # Active Masked Spatial Loss
            valid_ap = ~mask_ap
            valid_lat = ~mask_lat

            ap_target_valid = target_ap[valid_ap]
            lat_target_valid = target_lat[valid_lat]
            t_ap_norm = (ap_target_valid - np.mean(ap_target_valid)) / (np.std(ap_target_valid) + 1e-8)
            t_lat_norm = (lat_target_valid - np.mean(lat_target_valid)) / (np.std(lat_target_valid) + 1e-8)

            def objective(p):
                nonlocal eval_count
                eval_count += 1
                yaw, pitch, roll, tx, ty, tz = p
                tx, ty, tz = np.array([tx, ty, tz]) / self.spacing[0]
                drr_ap = self.renderer.render_drr(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
                drr_lat = self.renderer.render_drr(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')

                d_ap = drr_ap[valid_ap]
                d_ap_norm = (d_ap - np.mean(d_ap)) / (np.std(d_ap) + 1e-8)
                ncc_ap = np.mean(d_ap_norm * t_ap_norm)

                d_lat = drr_lat[valid_lat]
                d_lat_norm = (d_lat - np.mean(d_lat)) / (np.std(d_lat) + 1e-8)
                ncc_lat = np.mean(d_lat_norm * t_lat_norm)

                return float(0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat))

            res = minimize(
                objective,
                x0=np.array(initial_pose),
                method='Powell',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-4}
            )
        else:
            # Clean / Low-dose Spectral-NCC in frequency space (ultra-fast)
            f_targ_ap = self._prepare_spectrum(target_ap, 'ap')
            f_targ_lat = self._prepare_spectrum(target_lat, 'lateral')

            def objective(p):
                nonlocal eval_count
                eval_count += 1
                yaw, pitch, roll, tx, ty, tz = p
                tx, ty, tz = np.array([tx, ty, tz]) / self.spacing[0]
                f_cand_ap = self.renderer.extract_frequency_slice(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
                f_cand_lat = self.renderer.extract_frequency_slice(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')

                ncc_ap = self.renderer.compute_spectral_ncc(f_cand_ap, f_targ_ap)
                ncc_lat = self.renderer.compute_spectral_ncc(f_cand_lat, f_targ_lat)
                return float(0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat))

            res = minimize(
                objective,
                x0=np.array(initial_pose),
                method='Powell',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-4}
            )

        elapsed = time.time() - t0

        return {
            'success': bool(res.success),
            'final_pose': res.x,
            'final_loss': float(res.fun),
            'evaluations': eval_count,
            'metal_detected': metal_present,
            'time_sec': elapsed
        }
