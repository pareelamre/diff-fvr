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
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=a.dtype,
            device=a.device,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=a.dtype,
            device=a.device,
        ).view(1, 1, 3, 3)
        sl = slice(crop, -crop if crop > 0 else None)
        ax = F.conv2d(a, kx, padding=1)[..., sl, sl]
        ay = F.conv2d(a, ky, padding=1)[..., sl, sl]
        bx = F.conv2d(b, kx, padding=1)[..., sl, sl]
        by = F.conv2d(b, ky, padding=1)[..., sl, sl]

        def _ncc(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            u0 = u - u.mean(dim=(-2, -1), keepdim=True)
            v0 = v - v.mean(dim=(-2, -1), keepdim=True)
            num = (u0 * v0).sum(dim=(-2, -1))
            den = (u0.square().sum(dim=(-2, -1)) * v0.square().sum(dim=(-2, -1))).sqrt() + 1e-8
            return (num / den).mean()

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
    ):
        super().__init__()
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
        self.register_buffer("F_blk_ri", torch.stack([F_blk.real, F_blk.imag], dim=1))
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))

        freqs = torch.fft.fftshift(torch.fft.fftfreq(self.P, d=self.ds_block))
        fy, fx = torch.meshgrid(freqs, freqs, indexing="ij")
        self.register_buffer("FX", fx)
        self.register_buffer("FY", fy)

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
        )

    def project_rays(
        self,
        src: torch.Tensor,
        trg_pts: torch.Tensor,
        u_det: torch.Tensor = None,
        v_det: torch.Tensor = None,
        n_0: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Render cone-beam line integrals for any source point `src` (3,) and
        detector lattice `trg_pts` (H, W, 3) in the volume's coordinate frame.
        """
        c_det = trg_pts.mean(dim=(0, 1))
        if u_det is None:
            u_det = trg_pts[0, -1] - trg_pts[0, 0]
            u_det = u_det / (u_det.norm() + 1e-12)
        if v_det is None:
            v_det = trg_pts[-1, 0] - trg_pts[0, 0]
            v_det = v_det / (v_det.norm() + 1e-12)
        if n_0 is None:
            n_0 = torch.cross(v_det, u_det, dim=0)
            n_0 = n_0 / (n_0.norm() + 1e-12)
            if torch.dot(c_det - src, n_0) < 0:
                n_0 = -n_0

        # 1. Shared O(P^2) Fourier Central Slice across all B sub-volumes
        k_grid = self.FX.unsqueeze(-1) * u_det + self.FY.unsqueeze(-1) * v_det
        idx = k_grid * (self.P * self.ds_block) + (self.P // 2)
        g = 2.0 * idx / float(self.P - 1) - 1.0
        grid_3d = torch.stack([g[..., 2], g[..., 1], g[..., 0]], dim=-1)[None, None].expand(
            self.B, -1, -1, -1, -1
        )

        s_ri = F.grid_sample(
            self.F_blk_ri, grid_3d, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(2)
        slices = torch.complex(s_ri[:, 0], s_ri[:, 1])
        imgs = torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(slices, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        ).real

        # 2. Exact pinhole ray-plane intersection + 1/cos(theta) path obliquity
        rays = trg_pts - src.view(1, 1, 3)
        r_hat = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cos_theta = (r_hat * n_0.view(1, 1, 3)).sum(dim=-1).clamp(min=0.1)

        d_b_parallel = ((self.centers - src.unsqueeze(0)) * n_0.unsqueeze(0)).sum(dim=-1)
        tau = d_b_parallel[:, None, None] / cos_theta.unsqueeze(0)
        pts_plane = src.view(1, 1, 1, 3) + tau.unsqueeze(-1) * r_hat.unsqueeze(0)

        rel = pts_plane - self.centers[:, None, None, :]
        xi = (rel * u_det.view(1, 1, 1, 3)).sum(dim=-1)
        eta = (rel * v_det.view(1, 1, 1, 3)).sum(dim=-1)

        gu = 2.0 * (xi / self.ds_block + (self.P // 2)) / float(self.P - 1) - 1.0
        gv = 2.0 * (eta / self.ds_block + (self.P // 2)) / float(self.P - 1) - 1.0
        grid_2d = torch.stack([gu, gv], dim=-1)
        sampled = F.grid_sample(
            imgs.unsqueeze(1), grid_2d, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(1)

        return (sampled / cos_theta.unsqueeze(0)).sum(dim=0)

    def forward_diffdrr(self, drr_mod, pose, out_res: int) -> torch.Tensor:
        """Render cone-beam DRR tensor `(1, 1, out_res, out_res)` for a DiffDRR `DRR` module & `RigidTransform`."""
        src, trg = drr_mod.detector(pose, calibration=None)
        src = src.squeeze(0).squeeze(0)
        trg_pts = trg.squeeze(0).view(out_res, out_res, 3)
        return self.project_rays(src, trg_pts).unsqueeze(0).unsqueeze(0)

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

        # Stage 1: Coarse candidate screening
        scored = []
        with torch.no_grad():
            for cr, ct in cands:
                pa, pb = _pose_pair(cr, ct)
                lv = float(
                    _loss(
                        self.forward_diffdrr(c_drrA, pa, out_coarse),
                        self.forward_diffdrr(c_drrB, pb, out_coarse),
                        c_xrA,
                        c_xrB,
                    )
                )
                scored.append((lv, cr, ct))
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
