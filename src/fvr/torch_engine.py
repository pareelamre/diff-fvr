"""
Diff-FVR: PyTorch / CUDA GPU Acceleration Module.

Implements the continuous Fourier Projection-Slice Theorem using PyTorch tensors,
enabling batched, autograd-differentiable DRR synthesis on CPU and CUDA devices.
"""

from typing import Tuple, Union, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchDiffFVREngine(nn.Module):
    """
    Tensorized Differentiable Fourier Volume Renderer in PyTorch.

    Attributes
    ----------
    f_vol_real_imag : torch.Tensor
        Pre-computed 3D Fourier volume stored as (1, 2, D_pad, H_pad, W_pad)
        representing [real, imag] channels for grid_sample.
    """

    def __init__(
        self,
        volume: Union[torch.Tensor, "np.ndarray"],
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        pad_factor: float = 2.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()
        else:
            volume = volume.float()

        self.device = device or volume.device
        volume = volume.to(self.device)

        self.orig_shape = volume.shape  # (D, H, W)
        self.spacing = spacing
        self.pad_factor = pad_factor

        # 1. Zero-padding bound (Zp >= sqrt(3) ~ 1.73, default 2.0)
        d, h, w = self.orig_shape
        pd = int(round(d * pad_factor))
        ph = int(round(h * pad_factor))
        pw = int(round(w * pad_factor))

        pad_d0 = (pd - d) // 2
        pad_d1 = pd - d - pad_d0
        pad_h0 = (ph - h) // 2
        pad_h1 = ph - h - pad_h0
        pad_w0 = (pw - w) // 2
        pad_w1 = pw - w - pad_w0

        # F.pad expects (w0, w1, h0, h1, d0, d1)
        vol_pad = F.pad(
            volume.unsqueeze(0).unsqueeze(0),
            (pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1),
            mode="constant",
            value=0.0,
        ).squeeze()

        self.pad_shape = vol_pad.shape

        # 2. Pre-compute 3D Fourier Transform
        # Note: torch.fft.fftn followed by fftshift
        f_vol = torch.fft.fftn(vol_pad)
        f_vol_shifted = torch.fft.fftshift(f_vol)

        # 3. Store as (1, 2, D, H, W) real/imag channels for trilinear grid_sample
        real_part = torch.real(f_vol_shifted)
        imag_part = torch.imag(f_vol_shifted)
        f_vol_tensor = torch.stack([real_part, imag_part], dim=0).unsqueeze(0)  # (1, 2, D, H, W)

        self.register_buffer("f_vol_5d", f_vol_tensor)

    @staticmethod
    def _euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
        """Build batched 3x3 rotation matrices from Euler angles in degrees."""
        # yaw (Z), pitch (Y), roll (X)
        deg_to_rad = math.pi / 180.0
        rz = yaw_deg * deg_to_rad
        ry = pitch_deg * deg_to_rad
        rx = roll_deg * deg_to_rad

        cz, sz = torch.cos(rz), torch.sin(rz)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cx, sx = torch.cos(rx), torch.sin(rx)

        # Rz
        Rz = torch.stack([
            torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
            torch.stack([sz, cz, torch.zeros_like(cz)], dim=-1),
            torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1)
        ], dim=-2)

        # Ry
        Ry = torch.stack([
            torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
            torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1)
        ], dim=-2)

        # Rx
        Rx = torch.stack([
            torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
            torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
            torch.stack([torch.zeros_like(cx), sx, cx], dim=-1)
        ], dim=-2)

        # Full R = Rz @ Ry @ Rx
        R = torch.matmul(Rz, torch.matmul(Ry, Rx))
        return R

    def forward(
        self,
        poses: torch.Tensor,
        view: str = "ap",
        out_res: int = 128,
    ) -> torch.Tensor:
        """
        Batched differentiable forward pass.

        Parameters
        ----------
        poses : torch.Tensor of shape (B, 6)
            Columns: (yaw_deg, pitch_deg, roll_deg, tx_mm, ty_mm, tz_mm).
        view : str
            'ap' (Antero-Posterior coronal projection) or
            'lateral' (Sagittal projection).
        out_res : int
            Dimension of synthesized square DRR.

        Returns
        -------
        drrs : torch.Tensor of shape (B, 1, out_res, out_res)
            Synthesized DRR images. Fully differentiable w.r.t. `poses`.
        """
        B = poses.shape[0]
        device = poses.device

        yaw = poses[:, 0]
        pitch = poses[:, 1]
        roll = poses[:, 2]
        tx = poses[:, 3]
        ty = poses[:, 4]
        tz = poses[:, 5]

        # 1. Base viewing orientation matrix
        # For 'ap', central slice is k_z = 0 (viewing down Z)
        # For 'lateral', central slice is k_x = 0 (viewing down X)
        R_pose = self._euler_to_rotation_matrix(yaw, pitch, roll)  # (B, 3, 3)

        if view.lower() == "lateral":
            # 90-degree yaw rotation to view lateral
            yaw_lat = torch.full_like(yaw, 90.0)
            zeros = torch.zeros_like(yaw)
            R_lat = self._euler_to_rotation_matrix(yaw_lat, zeros, zeros)
            R_total = torch.matmul(R_pose, R_lat)
        else:
            R_total = R_pose

        # 2. Form normalized 2D frequency grid in [-1, 1]
        # In grid_sample, coordinates are (x, y, z) mapped to [-1, 1]
        coords_1d = torch.linspace(-1.0, 1.0, out_res, device=device)
        ku, kv = torch.meshgrid(coords_1d, coords_1d, indexing="ij")  # (out_res, out_res)

        # Central slice plane has normal vector in 3D: [ku, kv, 0]
        k_plane = torch.stack([ku, kv, torch.zeros_like(ku)], dim=-1)  # (N, N, 3)
        k_plane = k_plane.view(1, out_res * out_res, 3).expand(B, -1, -1)  # (B, N*N, 3)

        # 3. Rotate sampling coordinates via R_total^T
        # k_rotated = k_plane @ R_total
        grid_3d = torch.bmm(k_plane, R_total)  # (B, N*N, 3)
        grid_sample_coords = grid_3d.view(B, out_res, out_res, 1, 3)  # (B, H, W, D, 3)

        # 4. Trilinear sampling from 3D complex Fourier volume
        # f_vol_5d is (1, 2, D, H, W)
        f_vol_batched = self.f_vol_5d.expand(B, -1, -1, -1, -1)
        sampled_slice = F.grid_sample(
            f_vol_batched,
            grid_sample_coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B, 2, H, W, 1)

        real_slice = sampled_slice[:, 0, :, :, 0]  # (B, H, W)
        imag_slice = sampled_slice[:, 1, :, :, 0]  # (B, H, W)

        f_slice = torch.complex(real_slice, imag_slice)  # (B, H, W)

        # 5. Translation modulation via Fourier Phase Shift theorem:
        # exp(-i * 2*pi * (ku * tx + kv * ty) / physical_span)
        # Normalize translation to sampling domain
        voxel_size = self.spacing[0]
        phys_span = out_res * voxel_size
        kx = ku.unsqueeze(0)  # (1, H, W)
        ky = kv.unsqueeze(0)

        phase_angle = -2.0 * math.pi * (kx * tx.view(B, 1, 1) + ky * ty.view(B, 1, 1)) / (phys_span + 1e-6)
        modulation = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle))
        f_slice_modulated = f_slice * modulation

        # 6. Inverse 2D FFT to recover spatial DRR
        f_slice_shifted = torch.fft.ifftshift(f_slice_modulated, dim=(-2, -1))
        drr_complex = torch.fft.ifft2(f_slice_shifted, dim=(-2, -1))
        drr = torch.real(drr_complex)

        # Rectify non-physical negative lobes from sinc interpolation
        drr = F.relu(drr)

        return drr.unsqueeze(1)  # (B, 1, H, W)

    def spectral_ncc_loss(
        self,
        pred_drr: torch.Tensor,
        target_drr: torch.Tensor,
        high_pass: bool = True,
        cutoff_sigma: float = 4.0,
    ) -> torch.Tensor:
        """
        Differentiable Normalized Spectral Cross-Correlation (Spectral-NCC) Loss.

        loss = 1.0 - NCC(pred, target)
        """
        B, C, H, W = pred_drr.shape

        # 2D FFT of predicted and target DRRs
        F_pred = torch.fft.fftshift(torch.fft.fft2(pred_drr), dim=(-2, -1))
        F_targ = torch.fft.fftshift(torch.fft.fft2(target_drr), dim=(-2, -1))

        # Optional High-Pass filter for soft-tissue removal
        if high_pass:
            y = torch.linspace(-H / 2, H / 2, H, device=pred_drr.device)
            x = torch.linspace(-W / 2, W / 2, W, device=pred_drr.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            r_sq = xx ** 2 + yy ** 2
            hp_filter = (1.0 - torch.exp(-r_sq / (2.0 * cutoff_sigma ** 2))).unsqueeze(0).unsqueeze(0)
            F_pred = F_pred * hp_filter
            F_targ = F_targ * hp_filter

        # Parseval Cross-Correlation
        cross_spec = F_pred * torch.conj(F_targ)
        cross_corr = torch.sum(torch.real(cross_spec), dim=(-2, -1))

        energy_pred = torch.sum(torch.abs(F_pred) ** 2, dim=(-2, -1))
        energy_targ = torch.sum(torch.abs(F_targ) ** 2, dim=(-2, -1))

        ncc = cross_corr / (torch.sqrt(energy_pred * energy_targ) + 1e-8)
        loss = 1.0 - torch.mean(ncc)
        return loss
