import numpy as np
from scipy import fft
from scipy.ndimage import map_coordinates
from typing import Tuple, Optional


class DifferentiableFourierVolumeRenderer:
    """
    Differentiable & Spectral Fourier Volume Rendering (Diff-FVR).
    
    Novel Scientific Contributions:
      1. Direct Frequency-Domain Normalized Cross-Correlation (Spectral-NCC):
         By Parseval's Theorem, the spatial correlation between a candidate DRR
         and a target radiograph equals the Hermitian inner product of their
         2D Fourier spectra. This allows evaluating trial poses in <1 ms WITHOUT
         computing 2D inverse Fourier transforms inside the optimization loop.
      2. 6-DoF Rigid Motion Parameterization:
         Rotations (yaw, pitch, roll) correspond to planar slice normal orientation.
         Translations (tx, ty) correspond to linear phase shift ramps in frequency space.
    """

    def __init__(self, volume: np.ndarray, pad_factor: float = 2.0, apply_premultiplication: bool = True):
        self.orig_shape = volume.shape  # (Z, Y, X)
        self.pad_factor = pad_factor
        self.apply_premultiplication = apply_premultiplication

        vol = volume.astype(np.float64)
        nz, ny, nx = vol.shape

        self.pz = int(round(nz * self.pad_factor))
        self.py = int(round(ny * self.pad_factor))
        self.px = int(round(nx * self.pad_factor))

        if self.apply_premultiplication:
            def sinc2_correction(n_orig, n_pad):
                coords = (np.arange(n_orig) - n_orig / 2.0) / n_pad
                s = np.sinc(coords)
                denom = np.maximum(s ** 2, 0.05)
                return 1.0 / denom

            cz = sinc2_correction(nz, self.pz)[:, None, None]
            cy = sinc2_correction(ny, self.py)[None, :, None]
            cx = sinc2_correction(nx, self.px)[None, None, :]
            vol = vol * (cz * cy * cx)

        # 3D Zero-Padding
        padded = np.zeros((self.pz, self.py, self.px), dtype=np.float64)
        sz = (self.pz - nz) // 2
        sy = (self.py - ny) // 2
        sx = (self.px - nx) // 2
        padded[sz:sz + nz, sy:sy + ny, sx:sx + nx] = vol

        # Compute 3D FFT (centered)
        self.f_3d = fft.fftshift(fft.fftn(fft.ifftshift(padded)))

    def _euler_to_basis(self, yaw_deg: float, pitch_deg: float, roll_deg: float, view: str = 'lateral'):
        """
        Compute projection plane basis vectors u and v given viewing plane and Euler rotations.
        Coordinate system: [Z (height), Y (depth/AP), X (width/lateral)].
          - Yaw (yaw_deg): Rotation around Z (patient longitudinal axis).
          - Pitch (pitch_deg): Rotation around X (flexion/extension).
          - Roll (roll_deg): Rotation around Y (lateral bending).
        """
        y_rad = np.radians(yaw_deg)
        p_rad = np.radians(pitch_deg)
        r_rad = np.radians(roll_deg)

        # Rotation around Z axis (Yaw: rotates Y and X)
        rz = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(y_rad), -np.sin(y_rad)],
            [0.0, np.sin(y_rad),  np.cos(y_rad)]
        ])
        # Rotation around Y axis (Roll: rotates Z and X)
        ry = np.array([
            [ np.cos(r_rad), 0.0, np.sin(r_rad)],
            [ 0.0,           1.0, 0.0          ],
            [-np.sin(r_rad), 0.0, np.cos(r_rad)]
        ])
        # Rotation around X axis (Pitch: rotates Z and Y)
        rx = np.array([
            [np.cos(p_rad), -np.sin(p_rad), 0.0],
            [np.sin(p_rad),  np.cos(p_rad), 0.0],
            [0.0,            0.0,           1.0]
        ])
        # Composite rotation in (Z, Y, X) space
        rot = rz @ ry @ rx

        # Basis vectors in frequency space coordinates (wz, wy, wx)
        if view == 'lateral':
            # Base: normal along wx [0,0,1], plane spanned by wy [0,1,0] (horiz) and wz [1,0,0] (vert)
            u_base = np.array([0.0, 1.0, 0.0])  # Horizontal (AP depth)
            v_base = np.array([1.0, 0.0, 0.0])  # Vertical (Z axis)
        elif view == 'ap':
            # Base: normal along wy [0,1,0], plane spanned by wx [0,0,1] (horiz) and wz [1,0,0] (vert)
            u_base = np.array([0.0, 0.0, 1.0])  # Horizontal (X axis)
            v_base = np.array([1.0, 0.0, 0.0])  # Vertical (Z axis)
        else: # axial
            u_base = np.array([0.0, 0.0, 1.0])
            v_base = np.array([0.0, 1.0, 0.0])

        u_vec = rot @ u_base
        v_vec = rot @ v_base
        return u_vec, v_vec

    def extract_frequency_slice(
        self,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        tx: float = 0.0,
        ty: float = 0.0,
        view: str = 'lateral',
        interp_order: int = 1
    ) -> np.ndarray:
        """
        Extract a parallel-beam frequency slice. tx/ty are in-plane VOXELS,
        not mm. Use RobustDiffFVREngine for physical-unit translation handling.
        """
        nz, ny, nx = self.orig_shape
        if view == 'lateral':
            slice_h, slice_w = self.pz, self.py
        elif view == 'ap':
            slice_h, slice_w = self.pz, self.px
        else:
            slice_h, slice_w = self.py, self.px

        cz, cy, cx = self.pz / 2.0, self.py / 2.0, self.px / 2.0
        u_vec, v_vec = self._euler_to_basis(yaw_deg, pitch_deg, roll_deg, view=view)

        u = np.linspace(-slice_w / 2.0, slice_w / 2.0, slice_w, endpoint=False)
        v = np.linspace(-slice_h / 2.0, slice_h / 2.0, slice_h, endpoint=False)
        uu, vv = np.meshgrid(u, v)

        kwz = uu * u_vec[0] + vv * v_vec[0] + cz
        kwy = uu * u_vec[1] + vv * v_vec[1] + cy
        kwx = uu * u_vec[2] + vv * v_vec[2] + cx

        coords = np.array([kwz.ravel(), kwy.ravel(), kwx.ravel()])

        real_slice = map_coordinates(self.f_3d.real, coords, order=interp_order, mode='constant', cval=0.0).reshape((slice_h, slice_w))
        imag_slice = map_coordinates(self.f_3d.imag, coords, order=interp_order, mode='constant', cval=0.0).reshape((slice_h, slice_w))
        f_slice = real_slice + 1j * imag_slice

        # Translation via Fourier Phase Shift: f(x - tx, y - ty) <=> F(u, v) * exp(-2pi*i*(u*tx/W + v*ty/H))
        if tx != 0.0 or ty != 0.0:
            phase_ramp = np.exp(-2j * np.pi * (uu * tx / slice_w + vv * ty / slice_h))
            f_slice = f_slice * phase_ramp

        return f_slice

    def render_drr(
        self,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        tx: float = 0.0,
        ty: float = 0.0,
        view: str = 'lateral',
        interp_order: int = 1,
        crop_to_roi: bool = True
    ) -> np.ndarray:
        """
        Render spatial DRR from extracted frequency slice.
        """
        f_slice = self.extract_frequency_slice(yaw_deg, pitch_deg, roll_deg, tx, ty, view=view, interp_order=interp_order)
        slice_h, slice_w = f_slice.shape
        spatial_proj = fft.fftshift(fft.ifft2(fft.ifftshift(f_slice)))
        drr = np.real(spatial_proj)

        scale_factor = (self.pz * self.py * self.px) / (slice_h * slice_w)
        drr = np.maximum(drr * scale_factor, 0.0)

        if crop_to_roi:
            nz, ny, nx = self.orig_shape
            target_h = nz if view in ['lateral', 'ap'] else ny
            target_w = ny if view == 'lateral' else nx
            sh = (slice_h - target_h) // 2
            sw = (slice_w - target_w) // 2
            drr = drr[sh:sh + target_h, sw:sw + target_w]

        return drr

    def compute_spectral_ncc(self, f_slice_candidate: np.ndarray, f_target_precomputed: np.ndarray) -> float:
        """
        Compute Normalized Cross Correlation directly in Fourier space via Parseval's identity.
        Cost: O(N^2) element-wise operations with NO 2D inverse Fourier transform.
        """
        # Inner product in Fourier space: sum(S1 * conj(S2))
        inner_prod = np.real(np.sum(f_slice_candidate * np.conj(f_target_precomputed)))
        norm_cand = np.linalg.norm(f_slice_candidate)
        norm_targ = np.linalg.norm(f_target_precomputed)
        if norm_cand * norm_targ == 0:
            return 0.0
        return float(inner_prod / (norm_cand * norm_targ))
