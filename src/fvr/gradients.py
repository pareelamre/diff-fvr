"""
Analytical and Differentiable Gradients for Spectral Fourier Volume Rendering.

Provides exact closed-form derivatives for in-plane and 3D translations,
along with slice-level rotational gradients for rapid 6-DoF biplanar registration.
"""

import numpy as np
from typing import Tuple, Dict
from scipy.ndimage import map_coordinates


class SpectralGradientEngine:
    """
    Computes analytical gradients of Spectral-NCC with respect to 6-DoF rigid parameters:
      theta = [yaw_deg, pitch_deg, roll_deg]
      t     = [tx_mm, ty_mm, tz_mm]
    """

    def __init__(self, renderer):
        self.renderer = renderer
        self.pz = renderer.pz
        self.py = renderer.py
        self.px = renderer.px
        self.orig_shape = renderer.orig_shape

    def compute_spectral_ncc_and_grad_translation(
        self,
        f_cand: np.ndarray,
        f_target: np.ndarray,
        view: str = 'lateral'
    ) -> Tuple[float, float, float]:
        """
        Compute Spectral-NCC and EXACT closed-form analytical gradients with respect
        to in-plane translations.
        
        For Lateral view: in-plane coordinates are (ty, tz)
        For AP view:      in-plane coordinates are (tx, tz)
        
        Returns:
            ncc: Normalized Cross Correlation
            d_ncc_du: gradient w.r.t horizontal translation (ty for lateral, tx for ap)
            d_ncc_dv: gradient w.r.t vertical translation (tz)
        """
        slice_h, slice_w = f_cand.shape

        norm_s = np.linalg.norm(f_cand)
        norm_t = np.linalg.norm(f_target)
        if norm_s * norm_t == 0:
            return 0.0, 0.0, 0.0

        inner_prod = np.real(np.sum(f_cand * np.conj(f_target)))
        ncc = float(inner_prod / (norm_s * norm_t))

        # Residual matrix in frequency space:
        # R_res = (1 / (||S|| * ||T||)) * [ T - ncc * (||T|| / ||S||) * S ]
        r_res = (f_target - ncc * (norm_t / norm_s) * f_cand) / (norm_s * norm_t)

        # Coordinate ramps
        u = np.linspace(-slice_w / 2.0, slice_w / 2.0, slice_w, endpoint=False)
        v = np.linspace(-slice_h / 2.0, slice_h / 2.0, slice_h, endpoint=False)
        uu, vv = np.meshgrid(u, v)

        # Translation Fourier derivative:
        # dS/du = -2pi*i * (u / slice_w) * S
        # dS/dv = -2pi*i * (v / slice_h) * S
        factor_u = -2j * np.pi * (uu / slice_w) * f_cand
        factor_v = -2j * np.pi * (vv / slice_h) * f_cand

        # d(NCC)/dp = Re(sum( dS/dp * conj(R_res) ))
        d_ncc_du = float(np.real(np.sum(factor_u * np.conj(r_res))))
        d_ncc_dv = float(np.real(np.sum(factor_v * np.conj(r_res))))

        return ncc, d_ncc_du, d_ncc_dv

    def compute_biplanar_loss_and_gradients(
        self,
        pose: np.ndarray,
        f_target_ap: np.ndarray,
        f_target_lat: np.ndarray,
        delta_rot_deg: float = 0.2
    ) -> Tuple[float, np.ndarray]:
        """
        Compute joint biplanar loss:
          L = 0.5 * (1 - NCC_AP) + 0.5 * (1 - NCC_LAT)
        and its 6-DoF gradient:
          grad = [dL/dyaw, dL/dpitch, dL/droll, dL/dtx, dL/dty, dL/dtz]
          
        Pose vector:
          pose = [yaw_deg, pitch_deg, roll_deg, tx_mm, ty_mm, tz_mm]
        """
        yaw, pitch, roll, tx, ty, tz = pose

        # --- 1. AP View (In-plane: tx, tz; Depth: ty) ---
        f_ap = self.renderer.extract_frequency_slice(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
        ncc_ap, d_ap_dtx, d_ap_dtz = self.compute_spectral_ncc_and_grad_translation(f_ap, f_target_ap, view='ap')

        # --- 2. Lateral View (In-plane: ty, tz; Depth: tx) ---
        f_lat = self.renderer.extract_frequency_slice(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')
        ncc_lat, d_lat_dty, d_lat_dtz = self.compute_spectral_ncc_and_grad_translation(f_lat, f_target_lat, view='lateral')

        loss = 0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat)

        # Translation gradients for total loss L = 0.5 * (1 - NCC_AP) + 0.5 * (1 - NCC_LAT)
        # dL/dt = -0.5 * d(NCC)/dt
        dL_dtx = -0.5 * d_ap_dtx
        dL_dty = -0.5 * d_lat_dty
        dL_dtz = -0.5 * (d_ap_dtz + d_lat_dtz)

        # --- 3. Rotational Gradients (yaw, pitch, roll) via Central Difference on Slices ---
        # Only 6 slice extractions total (3 angles x 2)
        grad_rot = np.zeros(3, dtype=np.float64)
        for i in range(3):
            p_plus = np.copy(pose[:3])
            p_minus = np.copy(pose[:3])
            p_plus[i] += delta_rot_deg
            p_minus[i] -= delta_rot_deg

            f_ap_p = self.renderer.extract_frequency_slice(p_plus[0], p_plus[1], p_plus[2], tx=tx, ty=tz, view='ap')
            f_ap_m = self.renderer.extract_frequency_slice(p_minus[0], p_minus[1], p_minus[2], tx=tx, ty=tz, view='ap')
            ncc_ap_p = self.renderer.compute_spectral_ncc(f_ap_p, f_target_ap)
            ncc_ap_m = self.renderer.compute_spectral_ncc(f_ap_m, f_target_ap)

            f_lat_p = self.renderer.extract_frequency_slice(p_plus[0], p_plus[1], p_plus[2], tx=ty, ty=tz, view='lateral')
            f_lat_m = self.renderer.extract_frequency_slice(p_minus[0], p_minus[1], p_minus[2], tx=ty, ty=tz, view='lateral')
            ncc_lat_p = self.renderer.compute_spectral_ncc(f_lat_p, f_target_lat)
            ncc_lat_m = self.renderer.compute_spectral_ncc(f_lat_m, f_target_lat)

            d_ap = (ncc_ap_p - ncc_ap_m) / (2.0 * delta_rot_deg)
            d_lat = (ncc_lat_p - ncc_lat_m) / (2.0 * delta_rot_deg)

            grad_rot[i] = -0.5 * (d_ap + d_lat)

        grad = np.array([grad_rot[0], grad_rot[1], grad_rot[2], dL_dtx, dL_dty, dL_dtz], dtype=np.float64)
        return float(loss), grad
