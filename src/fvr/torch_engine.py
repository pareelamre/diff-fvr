"""
Diff-FVR: PyTorch / CUDA GPU Acceleration Module.

Implements the continuous Fourier Projection-Slice Theorem using PyTorch tensors,
enabling batched, autograd-differentiable DRR synthesis on CPU and CUDA devices.
"""

from typing import Dict, Tuple, Union, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_slice import fourier_slice_3d


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
        dtype: torch.dtype = torch.float32,
        use_triton: bool = True,
    ):
        super().__init__()
        self.use_triton = bool(use_triton)
        self._freq_cache: Dict[Tuple[int, str, torch.dtype], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._hp_cache: Dict[Tuple[int, int, float, str, torch.dtype], torch.Tensor] = {}
        self._compiled_forward_core = None
        self._compiled_spectral_ncc = None
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

        # 3. Store as (1, 2, D, H, W) real/imag channels for trilinear grid_sample / Triton
        real_part = torch.real(f_vol_shifted)
        imag_part = torch.imag(f_vol_shifted)
        f_vol_tensor = torch.stack([real_part, imag_part], dim=0).unsqueeze(0).to(dtype=dtype)

        self.register_buffer("f_vol_5d", f_vol_tensor)

    def to_precision(self, dtype: torch.dtype) -> "TorchDiffFVREngine":
        """Cast stored 3D Fourier volume ``f_vol_5d`` to reduced precision (e.g. ``bfloat16``/``float16``)."""
        self.f_vol_5d = self.f_vol_5d.to(dtype=dtype)
        self._compiled_forward_core = None
        self._compiled_spectral_ncc = None
        return self

    def compile_renderer(self, mode: str = "reduce-overhead", fullgraph: bool = False) -> "TorchDiffFVREngine":
        """Compile forward DRR and zero-IFFT Spectral-NCC kernels via ``torch.compile``."""
        if hasattr(torch, "compile"):
            try:
                self._compiled_forward_core = torch.compile(self._forward_core, mode=mode, fullgraph=fullgraph)
                self._compiled_spectral_ncc = torch.compile(self._spectral_ncc_core, mode=mode, fullgraph=fullgraph)
            except Exception:
                self._compiled_forward_core = None
                self._compiled_spectral_ncc = None
        return self

    def _get_freq_grid(
        self, out_res: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (int(out_res), str(device), dtype)
        cached = self._freq_cache.get(key)
        if cached is not None:
            return cached
        coords_1d = torch.linspace(-1.0, 1.0, out_res, device=device, dtype=dtype)
        ku, kv = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
        ku_flat = ku.reshape(-1).contiguous()
        kv_flat = kv.reshape(-1).contiguous()
        cached = (ku, kv, ku_flat, kv_flat)
        self._freq_cache[key] = cached
        return cached

    def _get_hp_filter(
        self, H: int, W: int, cutoff_sigma: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (int(H), int(W), float(cutoff_sigma), str(device), dtype)
        cached = self._hp_cache.get(key)
        if cached is not None:
            return cached
        y = torch.linspace(-H / 2, H / 2, H, device=device, dtype=dtype)
        x = torch.linspace(-W / 2, W / 2, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        r_sq = xx * xx + yy * yy
        hp = (1.0 - torch.exp(-r_sq / (2.0 * cutoff_sigma ** 2))).unsqueeze(0).unsqueeze(0)
        self._hp_cache[key] = hp
        return hp

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

    def _extract_slice_spectrum(
        self,
        poses: torch.Tensor,
        view: str = "ap",
        out_res: int = 128,
    ) -> torch.Tensor:
        """
        Extract and phase-modulate batched 2D Fourier slices ``(B, out_res, out_res)``
        directly from ``self.f_vol_5d`` without computing any 2D IFFT.
        """
        B = poses.shape[0]
        device = poses.device
        coord_dtype = torch.float32

        yaw, pitch, roll = poses[:, 0], poses[:, 1], poses[:, 2]
        tx, ty = poses[:, 3], poses[:, 4]

        R_pose = self._euler_to_rotation_matrix(yaw, pitch, roll)
        if view.lower() == "lateral":
            yaw_lat = torch.full_like(yaw, 90.0)
            zeros = torch.zeros_like(yaw)
            R_lat = self._euler_to_rotation_matrix(yaw_lat, zeros, zeros)
            R_total = torch.matmul(R_pose, R_lat)
        else:
            R_total = R_pose

        ku, kv, ku_flat, kv_flat = self._get_freq_grid(out_res, device, coord_dtype)

        # Map normalized [-1, 1] slice axes R[:, 0, :], R[:, 1, :] to voxel coords [0, W-1] x [0, H-1] x [0, D-1]
        D_pad, H_pad, W_pad = self.pad_shape
        cx = 0.5 * float(max(W_pad - 1, 1))
        cy = 0.5 * float(max(H_pad - 1, 1))
        cz = 0.5 * float(max(D_pad - 1, 1))
        scale_xyz = torch.tensor([cx, cy, cz], device=device, dtype=R_total.dtype).view(1, 3)
        u_xyz = R_total[:, 0, :] * scale_xyz
        v_xyz = R_total[:, 1, :] * scale_xyz

        s_ri = fourier_slice_3d(
            self.f_vol_5d,
            u_xyz,
            v_xyz,
            ku_flat,
            kv_flat,
            (cx, cy, cz),
            use_triton=self.use_triton,
        )  # (1, B, 2, out_res*out_res)
        if s_ri.dtype in (torch.float16, torch.bfloat16):
            s_ri = s_ri.float()
        s_ri = s_ri.squeeze(0).view(B, 2, out_res, out_res)
        f_slice = torch.complex(s_ri[:, 0], s_ri[:, 1])  # (B, H, W)

        voxel_size = self.spacing[0]
        phys_span = out_res * voxel_size
        phase_angle = (
            -2.0 * math.pi * (ku.unsqueeze(0) * tx.view(B, 1, 1) + kv.unsqueeze(0) * ty.view(B, 1, 1))
            / (phys_span + 1e-6)
        )
        modulation = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle))
        return f_slice * modulation

    def _forward_core(self, poses: torch.Tensor, view: str = "ap", out_res: int = 128) -> torch.Tensor:
        f_slice_modulated = self._extract_slice_spectrum(poses, view=view, out_res=out_res)
        f_slice_shifted = torch.fft.ifftshift(f_slice_modulated, dim=(-2, -1))
        drr_complex = torch.fft.ifft2(f_slice_shifted, dim=(-2, -1))
        drr = F.relu(torch.real(drr_complex))
        return drr.unsqueeze(1)

    def forward(
        self,
        poses: torch.Tensor,
        view: str = "ap",
        out_res: int = 128,
    ) -> torch.Tensor:
        """
        Batched differentiable forward pass returning ``(B, 1, out_res, out_res)``.
        """
        fn = self._compiled_forward_core if self._compiled_forward_core is not None else self._forward_core
        return fn(poses, view=view, out_res=out_res)

    def prepare_spectral_target(
        self,
        target_drr: torch.Tensor,
        high_pass: bool = True,
        cutoff_sigma: float = 4.0,
    ) -> torch.Tensor:
        """Precompute normalized/filtered 2D spectrum ``F_targ`` once before optimization."""
        _, _, H, W = target_drr.shape
        f_targ = torch.fft.fftshift(torch.fft.fft2(target_drr.float()), dim=(-2, -1))
        if high_pass:
            hp = self._get_hp_filter(H, W, cutoff_sigma, target_drr.device, torch.float32)
            f_targ = f_targ * hp
        return f_targ

    def _spectral_ncc_core(
        self,
        poses: torch.Tensor,
        f_targ_precomputed: torch.Tensor,
        view: str = "ap",
        high_pass: bool = True,
        cutoff_sigma: float = 4.0,
    ) -> torch.Tensor:
        out_res = int(f_targ_precomputed.shape[-1])
        f_pred = self._extract_slice_spectrum(poses, view=view, out_res=out_res).unsqueeze(1)
        if high_pass:
            hp = self._get_hp_filter(out_res, out_res, cutoff_sigma, poses.device, torch.float32)
            f_pred = f_pred * hp

        cross_spec = f_pred * torch.conj(f_targ_precomputed)
        cross_corr = torch.sum(torch.real(cross_spec), dim=(-2, -1))
        energy_pred = torch.sum(f_pred.real.square() + f_pred.imag.square(), dim=(-2, -1))
        energy_targ = torch.sum(
            f_targ_precomputed.real.square() + f_targ_precomputed.imag.square(), dim=(-2, -1)
        )
        ncc = cross_corr / (torch.sqrt(energy_pred * energy_targ) + 1e-8)
        return 1.0 - torch.mean(ncc)

    def forward_spectral_ncc(
        self,
        poses: torch.Tensor,
        f_targ_precomputed: torch.Tensor,
        view: str = "ap",
        high_pass: bool = True,
        cutoff_sigma: float = 4.0,
    ) -> torch.Tensor:
        """
        Zero-IFFT Parseval Spectral-NCC loss evaluated directly in Fourier space ``O(B * N^2)``.

        Skips both the predicted 2D ``ifft2`` and 2D ``fft2`` inside the pose optimization loop
        using a precomputed ``f_targ_precomputed`` from ``prepare_spectral_target``.
        """
        fn = self._compiled_spectral_ncc if self._compiled_spectral_ncc is not None else self._spectral_ncc_core
        return fn(poses, f_targ_precomputed, view=view, high_pass=high_pass, cutoff_sigma=cutoff_sigma)

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
        _, _, H, W = pred_drr.shape
        F_pred = torch.fft.fftshift(torch.fft.fft2(pred_drr), dim=(-2, -1))
        F_targ = torch.fft.fftshift(torch.fft.fft2(target_drr), dim=(-2, -1))

        if high_pass:
            hp_filter = self._get_hp_filter(H, W, cutoff_sigma, pred_drr.device, pred_drr.dtype)
            F_pred = F_pred * hp_filter
            F_targ = F_targ * hp_filter

        cross_spec = F_pred * torch.conj(F_targ)
        cross_corr = torch.sum(torch.real(cross_spec), dim=(-2, -1))

        energy_pred = torch.sum(F_pred.real.square() + F_pred.imag.square(), dim=(-2, -1))
        energy_targ = torch.sum(F_targ.real.square() + F_targ.imag.square(), dim=(-2, -1))

        ncc = cross_corr / (torch.sqrt(energy_pred * energy_targ) + 1e-8)
        return 1.0 - torch.mean(ncc)
