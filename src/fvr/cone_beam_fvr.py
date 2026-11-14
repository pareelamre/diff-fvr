"""
Piecewise-Perspective (Cone-Beam) Differentiable Fourier Volume Renderer.

Decomposes a 3D attenuation volume via an exact smooth Partition of Unity
into overlapping sub-volumes (sum_b w_b(x) = 1 everywhere). At any cone-beam
C-arm pose (source position src and detector lattice trg_pts), each active
sub-volume is evaluated by a single O(P^2) Fourier Central Slice along the shared
optical axis n_0 (eliminating inter-block seam discontinuities) and warped onto
the detector via exact pinhole ray-plane intersection and 1/cos(theta) obliquity.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_slice import fourier_slice_3d, pou_pinhole_warp_sum

_SOBEL_CACHE: Dict[Tuple[str, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_sobel_kernels(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), dtype)
    cached = _SOBEL_CACHE.get(key)
    if cached is not None:
        return cached
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    _SOBEL_CACHE[key] = (kx, ky)
    return kx, ky


def _make_pou_1d(length: int, n_blocks: int, overlap: float) -> Tuple[list, float]:
    """Build 1D raised-cosine Partition-of-Unity windows summing identically to 1.0."""
    if n_blocks <= 1:
        return [torch.ones(length, dtype=torch.float32)], float(length)
    step = length / float(n_blocks)
    ov = max(2.0, step * overlap)
    coords = torch.arange(length, dtype=torch.float32) + 0.5
    weights = []
    for i in range(n_blocks):
        w = torch.ones(length, dtype=torch.float32)
        if i > 0:
            x0, x1 = i * step - ov, i * step + ov
            t = torch.clamp((coords - x0) / (x1 - x0), 0.0, 1.0)
            w = w * 0.5 * (1.0 - torch.cos(torch.pi * t))
        if i < n_blocks - 1:
            x0, x1 = (i + 1) * step - ov, (i + 1) * step + ov
            t = torch.clamp((coords - x0) / (x1 - x0), 0.0, 1.0)
            w = w * 0.5 * (1.0 + torch.cos(torch.pi * t))
        weights.append(w)
    return weights, step


def _sinc2_precomp_1d(n: int, p: int) -> torch.Tensor:
    """Inverse sinc^2 trilinear pre-compensation window for Fourier slice lookup."""
    c = (torch.arange(n, dtype=torch.float32) - (n - 1) / 2.0) / float(p)
    s = torch.where(c == 0, torch.ones_like(c), torch.sin(torch.pi * c) / (torch.pi * c))
    return 1.0 / torch.clamp(s ** 2, min=0.05)


def edge_dominant_ncc(
    pred: torch.Tensor,
    target: torch.Tensor,
    border_crop: Optional[int] = None,
    multiscale_weight: float = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Edge-dominant Gradient-NCC similarity resistant to fluoroscopic diaphragm vignetting.

    Computes normalized cross-correlation over interior Sobel image gradients (with
    border cropping to ignore collimator edges) plus optional 3x3-smoothed Sobel
    gradients to widen the local convergence basin.
    """
    H = pred.shape[-1]
    crop = max(4, H // 8) if border_crop is None else int(border_crop)

    def _grad_ncc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        kx, ky = _get_sobel_kernels(a.device, a.dtype)
        sl = slice(crop, -crop if crop > 0 else None)
        ax = F.conv2d(a, kx, padding=1)[..., sl, sl]
        ay = F.conv2d(a, ky, padding=1)[..., sl, sl]
        bx = F.conv2d(b, kx, padding=1)[..., sl, sl]
        by = F.conv2d(b, ky, padding=1)[..., sl, sl]

        def _ncc(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            u0 = u - u.mean(dim=(-2, -1), keepdim=True)
            v0 = v - v.mean(dim=(-2, -1), keepdim=True)
            num = (u0 * v0).sum(dim=(-2, -1))
            den = (u0.square().sum(dim=(-2, -1)) * v0.square().sum(dim=(-2, -1))).clamp(min=1e-12).sqrt()
            val = (num / den).mean(dim=1)  # average channels -> shape (B,)
            return val.mean() if reduction == "mean" else val

        return 0.5 * (_ncc(ax, bx) + _ncc(ay, by))

    g_fine = _grad_ncc(pred, target)
    if multiscale_weight <= 0.0 or H < 96:
        return g_fine
    p_blur = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    t_blur = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    g_coarse = _grad_ncc(p_blur, t_blur)
    return (1.0 - multiscale_weight) * g_fine + multiscale_weight * g_coarse


class PerspectiveSlabFVR(nn.Module):
    """
    Differentiable Partition-of-Unity Cone-Beam Fourier Volume Renderer.

    Supports both:
    - Isotropic/metric 3D NumPy/Torch volumes via ``PerspectiveSlabFVR.from_volume(...)``
    - Calibrated DiffDRR subjects via ``PerspectiveSlabFVR.from_diffdrr_subject(...)``
    """

    def __init__(
        self,
        vol_3d: torch.Tensor,
        vox_mm: Union[float, Tuple[float, float, float]] = 1.0,
        origin_world_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis_signs: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        grid_blocks: Tuple[int, int, int] = (2, 2, 2),
        block_res: int = 80,
        pad_mult: int = 2,
        overlap: float = 0.25,
        energy_thresh: float = 1e-4,
        dtype: torch.dtype = torch.float32,
        use_triton: bool = True,
    ):
        super().__init__()
        self.use_triton = bool(use_triton)
        self._diffdrr_geom_cache: Dict[Tuple, Tuple[torch.Tensor, ...]] = {}
        self._compiled_project_core = None
        self._compiled_project_invariant = None
        vol = vol_3d.detach().float().cpu()
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume tensor, got shape {tuple(vol.shape)}")
        n0, n1, n2 = vol.shape

        if isinstance(vox_mm, (tuple, list)):
            sp = tuple(float(x) for x in vox_mm)
            if not (abs(sp[0] - sp[1]) < 1e-3 and abs(sp[1] - sp[2]) < 1e-3):
                raise ValueError(f"Volume spacing must be isotropic (got {sp})")
            self.vox_mm = sp[0]
        else:
            self.vox_mm = float(vox_mm)

        self.block_res = int(block_res)
        self.pad_mult = int(pad_mult)
        self.P = self.block_res * self.pad_mult

        k0, k1, k2 = grid_blocks
        w0_list, s0 = _make_pou_1d(n0, k0, overlap)
        w1_list, s1 = _make_pou_1d(n1, k1, overlap)
        w2_list, s2 = _make_pou_1d(n2, k2, overlap)

        ov0 = max(2, int(round(s0 * overlap))) if k0 > 1 else 0
        ov1 = max(2, int(round(s1 * overlap))) if k1 > 1 else 0
        ov2 = max(2, int(round(s2 * overlap))) if k2 > 1 else 0

        m0 = int(np.ceil(s0 + 2 * ov0))
        m1 = int(np.ceil(s1 + 2 * ov1))
        m2 = int(np.ceil(s2 + 2 * ov2))
        M = max(m0, m1, m2) + (1 if max(k0, k1, k2) > 1 else 0)
        self.ds_block = (M * self.vox_mm) / float(self.block_res)

        w_sinc = _sinc2_precomp_1d(self.block_res, self.P)
        w3 = w_sinc[:, None, None] * w_sinc[None, :, None] * w_sinc[None, None, :]

        pad_b = (self.P - self.block_res) // 2
        # Account for exact offset between even block midpoint and FFT zero-phase center P//2
        half_vox_mm = ((self.P // 2) - (pad_b + (self.block_res - 1) * 0.5)) * self.ds_block

        blocks, centers = [], []
        for w0 in w0_list:
            nz0 = torch.nonzero(w0 > 1e-6).squeeze(-1)
            a0, b0 = int(nz0[0]), int(nz0[-1]) + 1
            for w1 in w1_list:
                nz1 = torch.nonzero(w1 > 1e-6).squeeze(-1)
                a1, b1 = int(nz1[0]), int(nz1[-1]) + 1
                for w2 in w2_list:
                    nz2 = torch.nonzero(w2 > 1e-6).squeeze(-1)
                    a2, b2 = int(nz2[0]), int(nz2[-1]) + 1

                    sub = vol[a0:b0, a1:b1, a2:b2] * (
                        w0[a0:b0, None, None] * w1[None, a1:b1, None] * w2[None, None, a2:b2]
                    )
                    if float(sub.sum()) <= energy_thresh:
                        continue

                    l0, l1, l2 = sub.shape
                    m_0, m_1, m_2 = (M - l0) // 2, (M - l1) // 2, (M - l2) // 2
                    cube = F.pad(sub, (m_2, M - l2 - m_2, m_1, M - l1 - m_1, m_0, M - l0 - m_0))

                    # Exact volume index corresponding to padded cube center (M - 1)/2
                    c0_idx = a0 + (M - 1) * 0.5 - m_0
                    c1_idx = a1 + (M - 1) * 0.5 - m_1
                    c2_idx = a2 + (M - 1) * 0.5 - m_2
                    wc0 = origin_world_mm[0] + axis_signs[0] * c0_idx * self.vox_mm + axis_signs[0] * half_vox_mm
                    wc1 = origin_world_mm[1] + axis_signs[1] * c1_idx * self.vox_mm + axis_signs[1] * half_vox_mm
                    wc2 = origin_world_mm[2] + axis_signs[2] * c2_idx * self.vox_mm + axis_signs[2] * half_vox_mm

                    if M != self.block_res:
                        cube_r = (
                            F.interpolate(
                                cube[None, None],
                                size=(self.block_res, self.block_res, self.block_res),
                                mode="trilinear",
                                align_corners=False,
                            ).squeeze()
                            * self.ds_block
                        )
                    else:
                        cube_r = cube * self.ds_block

                    cube_pad = F.pad(
                        cube_r * w3,
                        (
                            pad_b,
                            self.P - self.block_res - pad_b,
                            pad_b,
                            self.P - self.block_res - pad_b,
                            pad_b,
                            self.P - self.block_res - pad_b,
                        ),
                    )
                    blocks.append(cube_pad)
                    centers.append([wc0, wc1, wc2])

        if not blocks:
            raise ValueError("Input volume contains no non-zero attenuation mass above threshold.")

        self.B = len(blocks)
        stack = torch.stack(blocks, dim=0)
        F_blk = torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(stack, dim=(-3, -2, -1)), dim=(-3, -2, -1)),
            dim=(-3, -2, -1),
        )
        self.register_buffer("F_blk_ri", torch.stack([F_blk.real, F_blk.imag], dim=1).to(dtype=dtype))
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))

        freqs = torch.fft.fftshift(torch.fft.fftfreq(self.P, d=self.ds_block))
        fy, fx = torch.meshgrid(freqs, freqs, indexing="ij")
        self.register_buffer("FX", fx)
        self.register_buffer("FY", fy)
        self.register_buffer("FX_flat", fx.reshape(-1).contiguous())
        self.register_buffer("FY_flat", fy.reshape(-1).contiguous())

    def to_precision(self, dtype: torch.dtype) -> "PerspectiveSlabFVR":
        """
        Cast 3D Fourier slab storage ``F_blk_ri`` to reduced precision (e.g. ``torch.bfloat16``
        or ``torch.float16`` on CUDA) while keeping geometric coordinates in ``float32``.
        """
        self.F_blk_ri = self.F_blk_ri.to(dtype=dtype)
        self._compiled_project_core = None
        self._compiled_project_invariant = None
        return self

    def compile_renderer(self, mode: str = "reduce-overhead", fullgraph: bool = False) -> "PerspectiveSlabFVR":
        """
        Compile the core tensor projection graph with ``torch.compile`` (CUDA Graphs when on GPU).
        Falls back gracefully if compilation is unavailable in the runtime environment.
        """
        if hasattr(torch, "compile"):
            try:
                self._compiled_project_core = torch.compile(self._project_core, mode=mode, fullgraph=fullgraph)
                self._compiled_project_invariant = torch.compile(
                    self._project_core_invariant, mode=mode, fullgraph=fullgraph
                )
            except Exception:
                self._compiled_project_core = None
                self._compiled_project_invariant = None
        return self

    @classmethod
    def from_volume(
        cls,
        vol_zyx: Union[np.ndarray, torch.Tensor],
        vox_mm: float = 1.0,
        grid_blocks: Tuple[int, int, int] = (2, 2, 2),
        block_res: int = 64,
        pad_mult: int = 2,
        overlap: float = 0.25,
        center_mode: str = "pixel_center",
    ) -> "PerspectiveSlabFVR":
        """
        Construct cone-beam PoU-FVR from an isotropic (Z, Y, X) attenuation volume.

        Args:
            center_mode: ``"pixel_center"`` aligns the middle of pixel indices ``(N-1)/2``
                with world ``(0, 0, 0)``; ``"grid_center"`` aligns ``N/2`` (matching
                ``scipy.ndimage`` rotation-center conventions in C-arm DICOM phantoms).
        """
        vt = torch.as_tensor(vol_zyx, dtype=torch.float32)
        nz, ny, nx = vt.shape
        v = float(vox_mm)
        if center_mode == "pixel_center":
            origin = (-0.5 * (nz - 1) * v, -0.5 * (ny - 1) * v, -0.5 * (nx - 1) * v)
        elif center_mode == "grid_center":
            origin = (-0.5 * nz * v, -0.5 * ny * v, -0.5 * nx * v)
        else:
            raise ValueError(f"Unknown center_mode={center_mode!r}; expected 'pixel_center' or 'grid_center'")
        return cls(
            vt,
            vox_mm=v,
            origin_world_mm=origin,
            axis_signs=(1.0, 1.0, 1.0),
            grid_blocks=grid_blocks,
            block_res=block_res,
            pad_mult=pad_mult,
            overlap=overlap,
        )

    @classmethod
    def from_diffdrr_subject(
        cls,
        subject,
        grid_blocks: Tuple[int, int, int] = (2, 2, 2),
        block_res: int = 80,
        pad_mult: int = 2,
        overlap: float = 0.25,
        **kwargs,
    ) -> "PerspectiveSlabFVR":
        """Construct cone-beam PoU-FVR matching DiffDRR Subject world RAS coordinates."""
        vol_raw = subject.density.data.squeeze(0).float()
        vol_world = vol_raw.flip(0, 1)  # align (+x, +y, +z) in RAS order
        nx, ny, nz = vol_world.shape
        aff = subject.density.affine
        sx = abs(float(aff[0, 0]))
        sy = abs(float(aff[1, 1]))
        sz = abs(float(aff[2, 2]))
        origin = (
            -(nx - 1) * sx + float(aff[0, 3]),
            -(ny - 1) * sy + float(aff[1, 3]),
            float(aff[2, 3]),
        )
        return cls(
            vol_world,
            vox_mm=(sx, sy, sz),
            origin_world_mm=origin,
            axis_signs=(1.0, 1.0, 1.0),
            grid_blocks=grid_blocks,
            block_res=block_res,
            pad_mult=pad_mult,
            overlap=overlap,
            energy_thresh=1e-3,
            **kwargs,
        )

    def _project_core(
        self,
        srcs: torch.Tensor,
        trg_pts: torch.Tensor,
        u_det: torch.Tensor,
        v_det: torch.Tensor,
        n_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Core batched cone-beam projection kernel for ``N_pose`` poses.

        Args:
            srcs: ``(N_pose, 3)`` X-ray source positions in world coordinates.
            trg_pts: ``(N_pose, H, W, 3)`` detector pixel lattice points.
            u_det: ``(N_pose, 3)`` unit detector horizontal axis.
            v_det: ``(N_pose, 3)`` unit detector vertical axis.
            n_0: ``(N_pose, 3)`` unit optical axis pointing source -> detector.

        Returns:
            ``(N_pose, H, W)`` rendered line-integral projections.
        """
        N_pose, H, W, _ = trg_pts.shape
        B_blk = self.B
        P = self.P

        imgs = self._slice_and_ifft(u_det, v_det)
        d_b_parallel, cu, cv = self._compute_block_offsets(srcs, u_det, v_det, n_0)

        rays = trg_pts - srcs.view(N_pose, 1, 1, 3)
        r_hat = rays * torch.rsqrt((rays * rays).sum(dim=-1, keepdim=True).clamp(min=1e-12))
        cos_theta = (r_hat * n_0.view(N_pose, 1, 1, 3)).sum(dim=-1).clamp(min=0.1)
        inv_cos = 1.0 / cos_theta

        ru = (r_hat * u_det.view(N_pose, 1, 1, 3)).sum(dim=-1)
        rv = (r_hat * v_det.view(N_pose, 1, 1, 3)).sum(dim=-1)

        tau = d_b_parallel.view(B_blk, N_pose, 1, 1) * inv_cos.unsqueeze(0)
        xi = tau * ru.unsqueeze(0) - cu.view(B_blk, N_pose, 1, 1)
        eta = tau * rv.unsqueeze(0) - cv.view(B_blk, N_pose, 1, 1)

        inv_ds = 1.0 / self.ds_block
        inv_pm1 = 2.0 / float(P - 1)
        c_idx = float(P // 2)
        gu = (xi * inv_ds + c_idx) * inv_pm1 - 1.0
        gv = (eta * inv_ds + c_idx) * inv_pm1 - 1.0
        grid_2d = torch.stack([gu, gv], dim=-1).view(B_blk * N_pose, H, W, 2)

        sampled = F.grid_sample(
            imgs.reshape(B_blk * N_pose, 1, P, P),
            grid_2d,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(B_blk, N_pose, H, W)

        return sampled.sum(dim=0) * inv_cos

    def _slice_and_ifft(self, u_det: torch.Tensor, v_det: torch.Tensor) -> torch.Tensor:
        """Evaluate O(P^2) Fourier central slices and invert to real-space slab images ``(B_blk, N_pose, P, P)``."""
        N_pose = u_det.shape[0]
        B_blk = self.B
        P = self.P
        vox_scale = float(P) * self.ds_block
        u_xyz = u_det.flip(-1) * vox_scale
        v_xyz = v_det.flip(-1) * vox_scale
        c_vox = float(P // 2)
        s_ri = fourier_slice_3d(
            self.F_blk_ri,
            u_xyz,
            v_xyz,
            self.FX_flat,
            self.FY_flat,
            (c_vox, c_vox, c_vox),
            use_triton=self.use_triton,
        )
        if s_ri.dtype in (torch.float16, torch.bfloat16):
            s_ri = s_ri.float()
        s_ri = s_ri.view(B_blk, N_pose, 2, P, P)
        slices = torch.complex(s_ri[:, :, 0], s_ri[:, :, 1])
        return torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(slices, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        ).real

    def _compute_block_offsets(
        self,
        srcs: torch.Tensor,
        u_det: torch.Tensor,
        v_det: torch.Tensor,
        n_0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-block, per-pose camera-frame center offsets ``(d_b_parallel, cu, cv)`` of shape ``(B_blk, N_pose)``."""
        B_blk = self.B
        N_pose = srcs.shape[0]
        diff_c = self.centers.view(B_blk, 1, 3) - srcs.view(1, N_pose, 3)
        d_b_parallel = (diff_c * n_0.view(1, N_pose, 3)).sum(dim=-1)
        cu = (diff_c * u_det.view(1, N_pose, 3)).sum(dim=-1)
        cv = (diff_c * v_det.view(1, N_pose, 3)).sum(dim=-1)
        return d_b_parallel, cu, cv

    @staticmethod
    def _make_ray_cache(
        src0: torch.Tensor,
        trg0: torch.Tensor,
        u0: torch.Tensor,
        v0: torch.Tensor,
        n0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute SE(3)-invariant camera-frame ray coefficients ``(rx_hw, ry_hw, inv_cos_hw, rx_flat, ry_flat, ic_flat)``."""
        rays0 = (trg0 - src0.view(1, 1, 3)).float()
        r0_hat = rays0 * torch.rsqrt((rays0 * rays0).sum(dim=-1, keepdim=True).clamp(min=1e-12))
        cos0 = (r0_hat * n0.float().view(1, 1, 3)).sum(dim=-1).clamp(min=0.1)
        inv_cos_hw = (1.0 / cos0).contiguous()
        rx_hw = ((r0_hat * u0.float().view(1, 1, 3)).sum(dim=-1) * inv_cos_hw).contiguous()
        ry_hw = ((r0_hat * v0.float().view(1, 1, 3)).sum(dim=-1) * inv_cos_hw).contiguous()
        return (
            rx_hw,
            ry_hw,
            inv_cos_hw,
            rx_hw.reshape(-1).contiguous(),
            ry_hw.reshape(-1).contiguous(),
            inv_cos_hw.reshape(-1).contiguous(),
        )

    def _project_core_invariant(
        self,
        srcs: torch.Tensor,
        u_det: torch.Tensor,
        v_det: torch.Tensor,
        n_0: torch.Tensor,
        ray_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Projection kernel exploiting SE(3) ray invariance with zero 4D/5D ray intermediates."""
        imgs = self._slice_and_ifft(u_det, v_det)
        d_b_parallel, cu, cv = self._compute_block_offsets(srcs, u_det, v_det, n_0)
        rx_hw, ry_hw, inv_cos_hw, rx_flat, ry_flat, ic_flat = ray_cache
        return pou_pinhole_warp_sum(
            imgs,
            d_b_parallel,
            cu,
            cv,
            rx_hw,
            ry_hw,
            inv_cos_hw,
            rx_flat,
            ry_flat,
            ic_flat,
            1.0 / self.ds_block,
            float(self.P // 2),
            use_triton=self.use_triton,
        )

    @staticmethod
    def _extract_detector_basis(
        srcs: torch.Tensor,
        trg_pts: torch.Tensor,
        u_det: Optional[torch.Tensor] = None,
        v_det: Optional[torch.Tensor] = None,
        n_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute orthonormal detector axes ``(u_det, v_det, n_0)`` of shape ``(N_pose, 3)``."""
        c_det = trg_pts.mean(dim=(1, 2))
        if u_det is None:
            u_raw = trg_pts[:, 0, -1] - trg_pts[:, 0, 0]
            u_det = u_raw / (u_raw.norm(dim=-1, keepdim=True) + 1e-12)
        if v_det is None:
            v_raw = trg_pts[:, -1, 0] - trg_pts[:, 0, 0]
            v_det = v_raw / (v_raw.norm(dim=-1, keepdim=True) + 1e-12)
        if n_0 is None:
            n_raw = torch.cross(v_det, u_det, dim=-1)
            n_0 = n_raw / (n_raw.norm(dim=-1, keepdim=True) + 1e-12)
            flip = ((c_det - srcs) * n_0).sum(dim=-1, keepdim=True) < 0
            n_0 = torch.where(flip, -n_0, n_0)
        return u_det, v_det, n_0

    def project_rays_batched(
        self,
        srcs: torch.Tensor,
        trg_pts: torch.Tensor,
        u_det: Optional[torch.Tensor] = None,
        v_det: Optional[torch.Tensor] = None,
        n_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Render batched cone-beam line integrals for ``srcs`` ``(N_pose, 3)`` and
        detector lattices ``trg_pts`` ``(N_pose, H, W, 3)``.
        """
        u_det, v_det, n_0 = self._extract_detector_basis(srcs, trg_pts, u_det, v_det, n_0)
        fn = self._compiled_project_core if self._compiled_project_core is not None else self._project_core
        return fn(srcs, trg_pts, u_det, v_det, n_0)

    def project_rays(
        self,
        src: torch.Tensor,
        trg_pts: torch.Tensor,
        u_det: Optional[torch.Tensor] = None,
        v_det: Optional[torch.Tensor] = None,
        n_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Render cone-beam line integrals for any source point `src` (3,) and
        detector lattice `trg_pts` (H, W, 3) in the volume's coordinate frame.
        """
        return self.project_rays_batched(
            src.unsqueeze(0),
            trg_pts.unsqueeze(0),
            None if u_det is None else u_det.unsqueeze(0),
            None if v_det is None else v_det.unsqueeze(0),
            None if n_0 is None else n_0.unsqueeze(0),
        ).squeeze(0)

    def cache_diffdrr_geometry(
        self, drr_mod, out_res: int
    ) -> Tuple[torch.Tensor, ...]:
        """Cache camera-frame detector lattice & SE(3)-invariant ray coefficients once per DiffDRR config."""
        det = drr_mod.detector
        device = self.centers.device
        key = (
            float(det.sdd),
            int(out_res),
            int(det.height),
            int(det.width),
            float(det.delx),
            float(det.dely),
            float(det.x0),
            float(det.y0),
            bool(getattr(det, "reverse_x_axis", False)),
            str(device),
        )
        cached = self._diffdrr_geom_cache.get(key)
        if cached is not None:
            return cached

        from diffdrr.pose import RigidTransform

        ident = RigidTransform(torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0))
        with torch.no_grad():
            src0, trg0 = det(ident, calibration=None)
            src0 = src0.squeeze(0).squeeze(0).to(device=device, dtype=torch.float32)
            trg0 = trg0.squeeze(0).view(out_res, out_res, 3).to(device=device, dtype=torch.float32)
            u0, v0, n0 = self._extract_detector_basis(src0.unsqueeze(0), trg0.unsqueeze(0))
            u0s, v0s, n0s = u0.squeeze(0), v0.squeeze(0), n0.squeeze(0)
            ray_cache = self._make_ray_cache(src0, trg0, u0s, v0s, n0s)
            cached = (src0, trg0, u0s, v0s, n0s, *ray_cache)
        self._diffdrr_geom_cache[key] = cached
        return cached

    def _forward_from_lattice(
        self,
        pose_mats: torch.Tensor,
        lattice: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Transform a precomputed camera-frame lattice by SE(3) matrices ``(N_pose, 4, 4)`` and project."""
        src0, trg0, u0, v0, n0 = lattice[:5]
        R = pose_mats[:, :3, :3]
        t = pose_mats[:, :3, 3]
        srcs = torch.einsum("nij,j->ni", R, src0) + t
        u_det = torch.einsum("nij,j->ni", R, u0)
        v_det = torch.einsum("nij,j->ni", R, v0)
        n_0 = torch.einsum("nij,j->ni", R, n0)
        if len(lattice) >= 11:
            ray_cache = lattice[5:11]
            fn = self._compiled_project_invariant if self._compiled_project_invariant is not None else self._project_core_invariant
            return fn(srcs, u_det, v_det, n_0, ray_cache).unsqueeze(1)
        trg_pts = torch.einsum("nij,hwj->nhwi", R, trg0) + t[:, None, None, :]
        return self.project_rays_batched(srcs, trg_pts, u_det=u_det, v_det=v_det, n_0=n_0).unsqueeze(1)

    def forward_diffdrr_batched(self, drr_mod, pose_mats: torch.Tensor, out_res: int) -> torch.Tensor:
        """
        Render ``(N_pose, 1, out_res, out_res)`` cone-beam DRRs from SE(3) ``pose_mats``
        ``(N_pose, 4, 4)`` using cached camera geometry (zero per-step detector setup).
        """
        lattice = self.cache_diffdrr_geometry(drr_mod, out_res)
        return self._forward_from_lattice(pose_mats, lattice)

    def forward_diffdrr(self, drr_mod, pose, out_res: int) -> torch.Tensor:
        """Render cone-beam DRR tensor ``(B, 1, out_res, out_res)`` for a DiffDRR ``RigidTransform``."""
        mat = pose.matrix
        if mat.ndim == 2:
            mat = mat.unsqueeze(0)
        return self.forward_diffdrr_batched(drr_mod, mat, out_res)

    @staticmethod
    def make_pinhole_lattice(
        k_inv: torch.Tensor,
        sdd_mm: float,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Precompute camera-space pinhole detector lattice & SE(3)-invariant ray cache
        matching Hartley-Zisserman ``k_inv`` (compatible with ``nanodrr.camera.make_k_inv``).
        """
        k0 = k_inv.squeeze(0) if k_inv.ndim == 3 else k_inv
        device, dtype = k0.device, torch.float32
        cols = torch.arange(width, device=device, dtype=dtype) + 0.5
        rows = torch.arange(height, device=device, dtype=dtype) + 0.5
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        pix = torch.stack([cc, rr, torch.ones_like(cc)], dim=-1)  # (H, W, 3)
        cam_dirs = torch.einsum("ij,hwj->hwi", k0.to(dtype=dtype), pix)
        trg0 = cam_dirs * (float(sdd_mm) / cam_dirs[..., 2:3].clamp(min=1e-6))
        src0 = torch.zeros(3, device=device, dtype=dtype)
        u0, v0, n0 = PerspectiveSlabFVR._extract_detector_basis(src0.unsqueeze(0), trg0.unsqueeze(0))
        u0s, v0s, n0s = u0.squeeze(0), v0.squeeze(0), n0.squeeze(0)
        ray_cache = PerspectiveSlabFVR._make_ray_cache(src0, trg0, u0s, v0s, n0s)
        return (src0, trg0, u0s, v0s, n0s, *ray_cache)

    def forward_pinhole(
        self,
        rt_inv: torch.Tensor,
        lattice: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        Zero-allocation batched pinhole projection from ``rt_inv`` ``(N_pose, 4, 4)``
        and a precomputed ``lattice`` from ``make_pinhole_lattice``.
        Returns ``(N_pose, 1, H, W)``.
        """
        return self._forward_from_lattice(rt_inv, lattice)

    def warmup_pinhole(
        self,
        lattice: Tuple[torch.Tensor, ...],
        batch_size: int = 1,
        backward: bool = True,
    ) -> None:
        """Pre-warm Triton/cuFFT kernels for the target ``batch_size`` to prevent first-trial JIT stalls."""
        dev = self.centers.device
        eye = torch.eye(4, device=dev, dtype=torch.float32).unsqueeze(0).repeat(max(1, int(batch_size)), 1, 1)
        if backward:
            eye = eye.requires_grad_(True)
            out = self.forward_pinhole(eye, lattice)
            out.sum().backward()
        else:
            with torch.no_grad():
                _ = self.forward_pinhole(eye, lattice)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def forward_carm_view(
        self,
        yaw_deg: torch.Tensor,
        pitch_deg: torch.Tensor,
        roll_deg: torch.Tensor,
        tx_vox: torch.Tensor,
        ty_vox: torch.Tensor,
        tz_vox: torch.Tensor,
        sad_mm: float,
        sdd_mm: float,
        det_px: int,
        det_fov_mm: float,
        view: str = "ap",
    ) -> torch.Tensor:
        """Render `(det_px, det_px)` cone-beam projection for C-arm AP or lateral geometry matching exp9."""
        device = self.centers.device
        y_r = torch.deg2rad(yaw_deg)
        p_r = torch.deg2rad(pitch_deg)
        r_r = torch.deg2rad(roll_deg)

        cy, sy = torch.cos(y_r), torch.sin(y_r)
        cp, sp = torch.cos(p_r), torch.sin(p_r)
        cr, sr = torch.cos(r_r), torch.sin(r_r)

        zero = torch.zeros((), device=device, dtype=torch.float32)
        one = torch.ones((), device=device, dtype=torch.float32)

        rz = torch.stack([
            torch.stack([one, zero, zero]),
            torch.stack([zero, cy, -sy]),
            torch.stack([zero, sy, cy]),
        ])
        ry = torch.stack([
            torch.stack([cr, zero, sr]),
            torch.stack([zero, one, zero]),
            torch.stack([-sr, zero, cr]),
        ])
        rx = torch.stack([
            torch.stack([cp, -sp, zero]),
            torch.stack([sp, cp, zero]),
            torch.stack([zero, zero, one]),
        ])
        rot = rz @ ry @ rx

        u_lin = torch.linspace(-0.5, 0.5, det_px, device=device) * float(det_fov_mm)
        v_lin = torch.linspace(-0.5, 0.5, det_px, device=device) * float(det_fov_mm)
        vv, uu = torch.meshgrid(v_lin, u_lin, indexing="ij")
        det_d = float(sdd_mm - sad_mm)

        if view == "ap":
            src0 = torch.tensor([0.0, -float(sad_mm), 0.0], device=device)
            trg0 = torch.stack([vv, torch.full_like(uu, det_d), uu], dim=-1)
            u_det0 = torch.tensor([0.0, 0.0, 1.0], device=device)
            v_det0 = torch.tensor([1.0, 0.0, 0.0], device=device)
            n0 = torch.tensor([0.0, 1.0, 0.0], device=device)
        else:
            src0 = torch.tensor([0.0, 0.0, -float(sad_mm)], device=device)
            trg0 = torch.stack([vv, uu, torch.full_like(uu, det_d)], dim=-1)
            u_det0 = torch.tensor([0.0, 1.0, 0.0], device=device)
            v_det0 = torch.tensor([1.0, 0.0, 0.0], device=device)
            n0 = torch.tensor([0.0, 0.0, 1.0], device=device)

        t_shift = torch.stack([tz_vox, ty_vox, tx_vox]) * self.vox_mm
        src = rot @ src0 - t_shift
        trg_pts = torch.einsum("ij,hwj->hwi", rot, trg0) - t_shift
        return self.project_rays(
            src,
            trg_pts,
            u_det=rot @ u_det0,
            v_det=rot @ v_det0,
            n_0=rot @ n0,
        )

    def register_diffdrr_multistart(
        self,
        drrA,
        drrB,
        xrayA: torch.Tensor,
        xrayB: torch.Tensor,
        pose_init,
        T_AB,
        drrA_coarse=None,
        drrB_coarse=None,
        xrayA_coarse: Optional[torch.Tensor] = None,
        xrayB_coarse: Optional[torch.Tensor] = None,
        n_candidates: int = 24,
        rot_rad: float = 0.055,
        trans_mm: float = 8.0,
        top_k: int = 4,
        coarse_steps: int = 15,
        fine_steps: int = 45,
        fine_drr_polish_steps: int = 0,
        seed: int = 42,
    ) -> Tuple[object, Dict[str, object]]:
        """
        Coarse-to-fine multi-start biplanar cone-beam 2D/3D registration under edge-dominant
        Gradient-NCC, enforcing rigid biplanar coupling ``pose_B = pose_A @ T_AB``.

        Returns:
            ``(best_poseA, info_dict)`` where ``info_dict`` records per-stage losses and
            candidate diagnostics.
        """
        from diffdrr.pose import RigidTransform, convert

        out_fine = int(xrayA.shape[-1])
        use_coarse = (
            drrA_coarse is not None
            and drrB_coarse is not None
            and xrayA_coarse is not None
            and xrayB_coarse is not None
        )
        out_coarse = int(xrayA_coarse.shape[-1]) if use_coarse else out_fine
        c_drrA = drrA_coarse if use_coarse else drrA
        c_drrB = drrB_coarse if use_coarse else drrB
        c_xrA = xrayA_coarse if use_coarse else xrayA
        c_xrB = xrayB_coarse if use_coarse else xrayB

        def _pose_pair(r: torch.Tensor, t: torch.Tensor):
            dT = convert(r, t, parameterization="euler_angles", convention="ZYX").matrix
            pa = RigidTransform(dT @ pose_init.matrix)
            pb = RigidTransform(pa.matrix @ T_AB.matrix)
            return pa, pb

        def _loss(imA: torch.Tensor, imB: torch.Tensor, refA: torch.Tensor, refB: torch.Tensor) -> torch.Tensor:
            return 1.0 - 0.5 * (edge_dominant_ncc(imA, refA) + edge_dominant_ncc(imB, refB))

        rng = np.random.default_rng(seed)
        cands: List[Tuple[torch.Tensor, torch.Tensor]] = [(torch.zeros(1, 3), torch.zeros(1, 3))]
        for _ in range(max(0, n_candidates - 1)):
            cands.append((
                torch.tensor(rng.uniform(-rot_rad, rot_rad, size=(1, 3)), dtype=torch.float32),
                torch.tensor(rng.uniform(-trans_mm, trans_mm, size=(1, 3)), dtype=torch.float32),
            ))

        # Stage 1: Batched coarse candidate screening
        scored = []
        r_stack = torch.cat([cr for cr, _ in cands], dim=0).to(device=self.centers.device)
        t_stack = torch.cat([ct for _, ct in cands], dim=0).to(device=self.centers.device)
        with torch.no_grad():
            dT_all = convert(r_stack, t_stack, parameterization="euler_angles", convention="ZYX").matrix
            pA_mats = dT_all @ pose_init.matrix.to(device=self.centers.device)
            pB_mats = pA_mats @ T_AB.matrix.to(device=self.centers.device)
            imA_all = self.forward_diffdrr_batched(c_drrA, pA_mats, out_coarse)
            imB_all = self.forward_diffdrr_batched(c_drrB, pB_mats, out_coarse)
            losses = 1.0 - 0.5 * (
                edge_dominant_ncc(imA_all, c_xrA, reduction="none")
                + edge_dominant_ncc(imB_all, c_xrB, reduction="none")
            )
            for idx, (cr, ct) in enumerate(cands):
                scored.append((float(losses[idx]), cr, ct))
        scored.sort(key=lambda x: x[0])

        # Stage 2: Short coarse-basin optimization on Top-K survivors, re-ranked at fine resolution
        refined = []
        for _, cr0, ct0 in scored[: max(1, top_k)]:
            r = cr0.clone().detach().requires_grad_(True)
            t = ct0.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([{"params": [r], "lr": 0.005}, {"params": [t], "lr": 1.0}])
            for _ in range(coarse_steps):
                opt.zero_grad()
                pa, pb = _pose_pair(r, t)
                l = _loss(
                    self.forward_diffdrr(c_drrA, pa, out_coarse),
                    self.forward_diffdrr(c_drrB, pb, out_coarse),
                    c_xrA,
                    c_xrB,
                )
                l.backward()
                torch.nn.utils.clip_grad_norm_([r], 0.25)
                torch.nn.utils.clip_grad_norm_([t], 2.0)
                opt.step()
            with torch.no_grad():
                pa, pb = _pose_pair(r, t)
                lf = float(
                    _loss(
                        self.forward_diffdrr(drrA, pa, out_fine),
                        self.forward_diffdrr(drrB, pb, out_fine),
                        xrayA,
                        xrayB,
                    )
                )
            refined.append((lf, r.detach(), t.detach()))
        refined.sort(key=lambda x: x[0])

        # Stage 3: Fine cosine-annealed PoU-FVR polish on winning basin
        r = refined[0][1].clone().detach().requires_grad_(True)
        t = refined[0][2].clone().detach().requires_grad_(True)
        if fine_steps > 0:
            opt_f = torch.optim.Adam([{"params": [r], "lr": 0.004}, {"params": [t], "lr": 0.8}])
            sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=fine_steps, eta_min=0.0001)
            for _ in range(fine_steps):
                opt_f.zero_grad()
                pa, pb = _pose_pair(r, t)
                l = _loss(
                    self.forward_diffdrr(drrA, pa, out_fine),
                    self.forward_diffdrr(drrB, pb, out_fine),
                    xrayA,
                    xrayB,
                )
                l.backward()
                torch.nn.utils.clip_grad_norm_([r], 0.2)
                torch.nn.utils.clip_grad_norm_([t], 1.5)
                opt_f.step()
                sched_f.step()

        # Optional Stage 4: Ray-cast DiffDRR fine polish under identical edge-dominant metric
        if fine_drr_polish_steps > 0:
            opt_h = torch.optim.Adam([{"params": [r], "lr": 0.002}, {"params": [t], "lr": 0.4}])
            for _ in range(fine_drr_polish_steps):
                opt_h.zero_grad()
                pa, pb = _pose_pair(r, t)
                l = _loss(drrA(pa), drrB(pb), xrayA, xrayB)
                l.backward()
                opt_h.step()

        with torch.no_grad():
            best_poseA, best_poseB = _pose_pair(r, t)
            final_loss = float(
                _loss(
                    self.forward_diffdrr(drrA, best_poseA, out_fine),
                    self.forward_diffdrr(drrB, best_poseB, out_fine),
                    xrayA,
                    xrayB,
                )
            )
        return best_poseA, {
            "coarse_best_loss": scored[0][0],
            "fine_prepolish_loss": refined[0][0],
            "final_loss": final_loss,
        }
