"""
Fused 3D Fourier-Slice Kernel (CUDA Triton + Exact PyTorch Autograd Reference).

Extracts batched 2D complex Fourier slices ``slice[b_vol, n_pose, 2, K]`` directly
from a 5D real/imag 3D Fourier volume ``f_vol_ri`` of shape ``(B_vol, 2, D, H, W)``
along planes parameterized in ``(x, y, z) == (W, H, D)`` voxel coordinates:

    p(q) = center_xyz + fx[q] * u_xyz[n_pose] + fy[q] * v_xyz[n_pose]

matching ``torch.nn.functional.grid_sample(..., mode='bilinear', align_corners=True)``
without materializing intermediate ``(B_vol, N_pose, K, 3)`` coordinate buffers in
global memory or saving them in the autograd graph.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU or non-Triton environments
    triton = None
    tl = None
    HAS_TRITON = False


def triton_slice_available(tensor: torch.Tensor) -> bool:
    """Return True when the fused CUDA Triton 3D Fourier-slice path can run."""
    return bool(HAS_TRITON and tensor.is_cuda)


if HAS_TRITON:  # pragma: no cover - requires CUDA + Triton runtime

    @triton.jit
    def _load_chan(vol_ptr, ch_base, gx, gy, gz, D, H, W, stride_z, stride_y, mq):
        inb = mq & (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H) & (gz >= 0) & (gz < D)
        cz_i = tl.minimum(tl.maximum(gz, 0), D - 1)
        cy_i = tl.minimum(tl.maximum(gy, 0), H - 1)
        cx_i = tl.minimum(tl.maximum(gx, 0), W - 1)
        off = ch_base + cz_i * stride_z + cy_i * stride_y + cx_i
        return tl.load(vol_ptr + off, mask=inb, other=0.0).to(tl.float32)

    @triton.jit
    def _fourier_slice_3d_fwd_kernel(
        vol_ptr,  # *const T [B_vol, 2, D, H, W] contiguous
        u_ptr,  # *const float32 [N_pose, 3]
        v_ptr,  # *const float32 [N_pose, 3]
        fx_ptr,  # *const float32 [K]
        fy_ptr,  # *const float32 [K]
        out_ptr,  # *mut T [B_vol, N_pose, 2, K]
        cx,
        cy,
        cz,
        B_vol: tl.constexpr,
        N_pose,
        K,
        D,
        H,
        W,
        BLOCK: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_bp = tl.program_id(1)
        b_vol = pid_bp // N_pose
        n_pose = pid_bp % N_pose

        q_offs = pid_q * BLOCK + tl.arange(0, BLOCK)
        mq = q_offs < K

        fx = tl.load(fx_ptr + q_offs, mask=mq, other=0.0).to(tl.float32)
        fy = tl.load(fy_ptr + q_offs, mask=mq, other=0.0).to(tl.float32)

        ub = u_ptr + n_pose * 3
        vb = v_ptr + n_pose * 3
        ux = tl.load(ub + 0).to(tl.float32)
        uy = tl.load(ub + 1).to(tl.float32)
        uz = tl.load(ub + 2).to(tl.float32)
        vx = tl.load(vb + 0).to(tl.float32)
        vy = tl.load(vb + 1).to(tl.float32)
        vz = tl.load(vb + 2).to(tl.float32)

        px = cx + fx * ux + fy * vx
        py = cy + fx * uy + fy * vy
        pz = cz + fx * uz + fy * vz

        x0 = tl.floor(px)
        y0 = tl.floor(py)
        z0 = tl.floor(pz)
        tx = px - x0
        ty = py - y0
        tz = pz - z0

        ix0 = x0.to(tl.int32)
        iy0 = y0.to(tl.int32)
        iz0 = z0.to(tl.int32)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1

        vol_dhw = D * H * W
        base_re = b_vol * (2 * vol_dhw)
        base_im = base_re + vol_dhw
        stride_z = H * W
        stride_y = W

        # Accumulate real & imaginary parts using shared fractional weights
        w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
        w001 = tx * (1.0 - ty) * (1.0 - tz)
        w010 = (1.0 - tx) * ty * (1.0 - tz)
        w011 = tx * ty * (1.0 - tz)
        w100 = (1.0 - tx) * (1.0 - ty) * tz
        w101 = tx * (1.0 - ty) * tz
        w110 = (1.0 - tx) * ty * tz
        w111 = tx * ty * tz

        re_val = (
            _load_chan(vol_ptr, base_re, ix0, iy0, iz0, D, H, W, stride_z, stride_y, mq) * w000
            + _load_chan(vol_ptr, base_re, ix1, iy0, iz0, D, H, W, stride_z, stride_y, mq) * w001
            + _load_chan(vol_ptr, base_re, ix0, iy1, iz0, D, H, W, stride_z, stride_y, mq) * w010
            + _load_chan(vol_ptr, base_re, ix1, iy1, iz0, D, H, W, stride_z, stride_y, mq) * w011
            + _load_chan(vol_ptr, base_re, ix0, iy0, iz1, D, H, W, stride_z, stride_y, mq) * w100
            + _load_chan(vol_ptr, base_re, ix1, iy0, iz1, D, H, W, stride_z, stride_y, mq) * w101
            + _load_chan(vol_ptr, base_re, ix0, iy1, iz1, D, H, W, stride_z, stride_y, mq) * w110
            + _load_chan(vol_ptr, base_re, ix1, iy1, iz1, D, H, W, stride_z, stride_y, mq) * w111
        )
        im_val = (
            _load_chan(vol_ptr, base_im, ix0, iy0, iz0, D, H, W, stride_z, stride_y, mq) * w000
            + _load_chan(vol_ptr, base_im, ix1, iy0, iz0, D, H, W, stride_z, stride_y, mq) * w001
            + _load_chan(vol_ptr, base_im, ix0, iy1, iz0, D, H, W, stride_z, stride_y, mq) * w010
            + _load_chan(vol_ptr, base_im, ix1, iy1, iz0, D, H, W, stride_z, stride_y, mq) * w011
            + _load_chan(vol_ptr, base_im, ix0, iy0, iz1, D, H, W, stride_z, stride_y, mq) * w100
            + _load_chan(vol_ptr, base_im, ix1, iy0, iz1, D, H, W, stride_z, stride_y, mq) * w101
            + _load_chan(vol_ptr, base_im, ix0, iy1, iz1, D, H, W, stride_z, stride_y, mq) * w110
            + _load_chan(vol_ptr, base_im, ix1, iy1, iz1, D, H, W, stride_z, stride_y, mq) * w111
        )

        out_re_base = (b_vol * N_pose + n_pose) * (2 * K)
        out_im_base = out_re_base + K
        tl.store(out_ptr + out_re_base + q_offs, re_val.to(out_ptr.dtype.element_ty), mask=mq)
        tl.store(out_ptr + out_im_base + q_offs, im_val.to(out_ptr.dtype.element_ty), mask=mq)

    @triton.jit
    def _fourier_slice_3d_bwd_kernel(
        vol_ptr,  # *const T [B_vol, 2, D, H, W]
        go_ptr,  # *const T [B_vol, N_pose, 2, K]
        u_ptr,  # *const float32 [N_pose, 3]
        v_ptr,  # *const float32 [N_pose, 3]
        fx_ptr,  # *const float32 [K]
        fy_ptr,  # *const float32 [K]
        gu_part_ptr,  # *mut float32 [N_pose, grid_q, 3]
        gv_part_ptr,  # *mut float32 [N_pose, grid_q, 3]
        cx,
        cy,
        cz,
        B_vol: tl.constexpr,
        N_pose,
        K,
        D,
        H,
        W,
        grid_q,
        BLOCK: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        n_pose = tl.program_id(1)

        q_offs = pid_q * BLOCK + tl.arange(0, BLOCK)
        mq = q_offs < K

        fx = tl.load(fx_ptr + q_offs, mask=mq, other=0.0).to(tl.float32)
        fy = tl.load(fy_ptr + q_offs, mask=mq, other=0.0).to(tl.float32)

        ub = u_ptr + n_pose * 3
        vb = v_ptr + n_pose * 3
        ux = tl.load(ub + 0).to(tl.float32)
        uy = tl.load(ub + 1).to(tl.float32)
        uz = tl.load(ub + 2).to(tl.float32)
        vx = tl.load(vb + 0).to(tl.float32)
        vy = tl.load(vb + 1).to(tl.float32)
        vz = tl.load(vb + 2).to(tl.float32)

        px = cx + fx * ux + fy * vx
        py = cy + fx * uy + fy * vy
        pz = cz + fx * uz + fy * vz

        x0 = tl.floor(px)
        y0 = tl.floor(py)
        z0 = tl.floor(pz)
        tx = px - x0
        ty = py - y0
        tz = pz - z0

        ix0 = x0.to(tl.int32)
        iy0 = y0.to(tl.int32)
        iz0 = z0.to(tl.int32)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1

        vol_dhw = D * H * W
        stride_z = H * W
        stride_y = W

        gux = tl.zeros([BLOCK], dtype=tl.float32)
        guy = tl.zeros([BLOCK], dtype=tl.float32)
        guz = tl.zeros([BLOCK], dtype=tl.float32)
        gvx = tl.zeros([BLOCK], dtype=tl.float32)
        gvy = tl.zeros([BLOCK], dtype=tl.float32)
        gvz = tl.zeros([BLOCK], dtype=tl.float32)

        omx = 1.0 - tx
        omy = 1.0 - ty
        omz = 1.0 - tz

        for b_vol in tl.static_range(B_vol):
            go_re_base = (b_vol * N_pose + n_pose) * (2 * K)
            go_im_base = go_re_base + K
            gr = tl.load(go_ptr + go_re_base + q_offs, mask=mq, other=0.0).to(tl.float32)
            gi = tl.load(go_ptr + go_im_base + q_offs, mask=mq, other=0.0).to(tl.float32)

            base_re = b_vol * (2 * vol_dhw)
            base_im = base_re + vol_dhw

            r000 = _load_chan(vol_ptr, base_re, ix0, iy0, iz0, D, H, W, stride_z, stride_y, mq)
            r001 = _load_chan(vol_ptr, base_re, ix1, iy0, iz0, D, H, W, stride_z, stride_y, mq)
            r010 = _load_chan(vol_ptr, base_re, ix0, iy1, iz0, D, H, W, stride_z, stride_y, mq)
            r011 = _load_chan(vol_ptr, base_re, ix1, iy1, iz0, D, H, W, stride_z, stride_y, mq)
            r100 = _load_chan(vol_ptr, base_re, ix0, iy0, iz1, D, H, W, stride_z, stride_y, mq)
            r101 = _load_chan(vol_ptr, base_re, ix1, iy0, iz1, D, H, W, stride_z, stride_y, mq)
            r110 = _load_chan(vol_ptr, base_re, ix0, iy1, iz1, D, H, W, stride_z, stride_y, mq)
            r111 = _load_chan(vol_ptr, base_re, ix1, iy1, iz1, D, H, W, stride_z, stride_y, mq)

            i000 = _load_chan(vol_ptr, base_im, ix0, iy0, iz0, D, H, W, stride_z, stride_y, mq)
            i001 = _load_chan(vol_ptr, base_im, ix1, iy0, iz0, D, H, W, stride_z, stride_y, mq)
            i010 = _load_chan(vol_ptr, base_im, ix0, iy1, iz0, D, H, W, stride_z, stride_y, mq)
            i011 = _load_chan(vol_ptr, base_im, ix1, iy1, iz0, D, H, W, stride_z, stride_y, mq)
            i100 = _load_chan(vol_ptr, base_im, ix0, iy0, iz1, D, H, W, stride_z, stride_y, mq)
            i101 = _load_chan(vol_ptr, base_im, ix1, iy0, iz1, D, H, W, stride_z, stride_y, mq)
            i110 = _load_chan(vol_ptr, base_im, ix0, iy1, iz1, D, H, W, stride_z, stride_y, mq)
            i111 = _load_chan(vol_ptr, base_im, ix1, iy1, iz1, D, H, W, stride_z, stride_y, mq)

            drdx = ((r001 - r000) * omy + (r011 - r010) * ty) * omz + ((r101 - r100) * omy + (r111 - r110) * ty) * tz
            drdy = ((r010 - r000) * omx + (r011 - r001) * tx) * omz + ((r110 - r100) * omx + (r111 - r101) * tx) * tz
            drdz = ((r100 - r000) * omx + (r101 - r001) * tx) * omy + ((r110 - r010) * omx + (r111 - r011) * tx) * ty

            didx = ((i001 - i000) * omy + (i011 - i010) * ty) * omz + ((i101 - i100) * omy + (i111 - i110) * ty) * tz
            didy = ((i010 - i000) * omx + (i011 - i001) * tx) * omz + ((i110 - i100) * omx + (i111 - i101) * tx) * tz
            didz = ((i100 - i000) * omx + (i101 - i001) * tx) * omy + ((i110 - i010) * omx + (i111 - i011) * tx) * ty

            gx_acc = gr * drdx + gi * didx
            gy_acc = gr * drdy + gi * didy
            gz_acc = gr * drdz + gi * didz

            gux += gx_acc * fx
            guy += gy_acc * fx
            guz += gz_acc * fx
            gvx += gx_acc * fy
            gvy += gy_acc * fy
            gvz += gz_acc * fy

        part_off = (n_pose * grid_q + pid_q) * 3
        tl.store(gu_part_ptr + part_off + 0, tl.sum(gux, axis=0))
        tl.store(gu_part_ptr + part_off + 1, tl.sum(guy, axis=0))
        tl.store(gu_part_ptr + part_off + 2, tl.sum(guz, axis=0))
        tl.store(gv_part_ptr + part_off + 0, tl.sum(gvx, axis=0))
        tl.store(gv_part_ptr + part_off + 1, tl.sum(gvy, axis=0))
        tl.store(gv_part_ptr + part_off + 2, tl.sum(gvz, axis=0))


    class _TritonFourierSlice3DFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, vol_ri, u_xyz, v_xyz, fx, fy, cx, cy, cz):
            B_vol, _, D, H, W = vol_ri.shape
            N_pose = u_xyz.shape[0]
            K = fx.numel()
            out = torch.empty((B_vol, N_pose, 2, K), device=vol_ri.device, dtype=vol_ri.dtype)
            BLOCK = 256
            grid = (triton.cdiv(K, BLOCK), B_vol * N_pose)
            _fourier_slice_3d_fwd_kernel[grid](
                vol_ri.contiguous(),
                u_xyz.float().contiguous(),
                v_xyz.float().contiguous(),
                fx.float().contiguous(),
                fy.float().contiguous(),
                out,
                float(cx),
                float(cy),
                float(cz),
                B_vol=B_vol,
                N_pose=N_pose,
                K=K,
                D=D,
                H=H,
                W=W,
                BLOCK=BLOCK,
            )
            ctx.save_for_backward(vol_ri, u_xyz, v_xyz, fx, fy)
            ctx.center = (float(cx), float(cy), float(cz))
            return out

        @staticmethod
        def backward(ctx, grad_out):
            vol_ri, u_xyz, v_xyz, fx, fy = ctx.saved_tensors
            cx, cy, cz = ctx.center
            B_vol, _, D, H, W = vol_ri.shape
            N_pose = u_xyz.shape[0]
            K = fx.numel()
            BLOCK = 256
            grid_q = triton.cdiv(K, BLOCK)
            gu_part = torch.empty((N_pose, grid_q, 3), device=vol_ri.device, dtype=torch.float32)
            gv_part = torch.empty((N_pose, grid_q, 3), device=vol_ri.device, dtype=torch.float32)
            _fourier_slice_3d_bwd_kernel[(grid_q, N_pose)](
                vol_ri.contiguous(),
                grad_out.contiguous(),
                u_xyz.float().contiguous(),
                v_xyz.float().contiguous(),
                fx.float().contiguous(),
                fy.float().contiguous(),
                gu_part,
                gv_part,
                cx,
                cy,
                cz,
                B_vol=B_vol,
                N_pose=N_pose,
                K=K,
                D=D,
                H=H,
                W=W,
                grid_q=grid_q,
                BLOCK=BLOCK,
            )
            gu = gu_part.sum(dim=1).to(dtype=u_xyz.dtype)
            gv = gv_part.sum(dim=1).to(dtype=v_xyz.dtype)
            return None, gu, gv, None, None, None, None, None


def fourier_slice_3d_torch(
    vol_ri: torch.Tensor,
    u_xyz: torch.Tensor,
    v_xyz: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    center_xyz: Tuple[float, float, float],
) -> torch.Tensor:
    """
    Zero-volume-duplication PyTorch 3D Fourier slice returning ``(B_vol, N_pose, 2, K)``.

    Packs all ``N_pose`` poses along the depth dimension of 5D ``grid_sample`` so
    ``vol_ri`` ``(B_vol, 2, D, H, W)`` is never expanded or duplicated across poses.
    """
    B_vol, _, D, H, W = vol_ri.shape
    N_pose = u_xyz.shape[0]
    K = fx.numel()
    cx, cy, cz = center_xyz

    # p_xyz: (N_pose, K, 3) in voxel coordinates [0, W-1] x [0, H-1] x [0, D-1]
    fx_v = fx.view(1, K, 1)
    fy_v = fy.view(1, K, 1)
    u_v = u_xyz.view(N_pose, 1, 3)
    v_v = v_xyz.view(N_pose, 1, 3)
    px = cx + fx_v * u_v[..., 0:1] + fy_v * v_v[..., 0:1]
    py = cy + fx_v * u_v[..., 1:2] + fy_v * v_v[..., 1:2]
    pz = cz + fx_v * u_v[..., 2:3] + fy_v * v_v[..., 2:3]

    gx = 2.0 * px / float(max(W - 1, 1)) - 1.0
    gy = 2.0 * py / float(max(H - 1, 1)) - 1.0
    gz = 2.0 * pz / float(max(D - 1, 1)) - 1.0
    grid = torch.cat([gx, gy, gz], dim=-1)  # (N_pose, K, 3)

    # Fold (N_pose, K) as (D_out=N_pose, H_out=K, W_out=1) and expand across B_vol (stride-0 view)
    grid_5d = grid.view(1, N_pose, K, 1, 3).expand(B_vol, -1, -1, -1, -1)

    work_vol = vol_ri
    work_grid = grid_5d
    # CPU grid_sample only reliably supports float32/float64
    if not vol_ri.is_cuda and vol_ri.dtype in (torch.float16, torch.bfloat16):
        work_vol = vol_ri.float()
    if work_grid.dtype != work_vol.dtype:
        work_grid = work_grid.to(dtype=work_vol.dtype)

    sampled = F.grid_sample(
        work_vol,
        work_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B_vol, 2, N_pose, K, 1)
    return sampled.squeeze(-1).permute(0, 2, 1, 3)  # (B_vol, N_pose, 2, K)


def fourier_slice_3d(
    vol_ri: torch.Tensor,
    u_xyz: torch.Tensor,
    v_xyz: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    center_xyz: Tuple[float, float, float],
    use_triton: bool = True,
) -> torch.Tensor:
    """Dispatch 3D Fourier slice to fused Triton kernel on CUDA when available."""
    if use_triton and triton_slice_available(vol_ri) and not vol_ri.requires_grad:
        cx, cy, cz = center_xyz
        return _TritonFourierSlice3DFn.apply(vol_ri, u_xyz, v_xyz, fx, fy, cx, cy, cz)
    return fourier_slice_3d_torch(vol_ri, u_xyz, v_xyz, fx, fy, center_xyz)


def trilinear_slice_analytical_ref(
    vol_ri: torch.Tensor,
    u_xyz: torch.Tensor,
    v_xyz: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    center_xyz: Tuple[float, float, float],
) -> torch.Tensor:
    """
    Pure-PyTorch explicit 8-neighbor trilinear lookup with floor bounds check.

    Exists strictly to unit-test exact numerical equivalence between the manual
    Triton stencil equations and ``F.grid_sample(..., align_corners=True)`` on CPU.
    """
    B_vol, _, D, H, W = vol_ri.shape
    N_pose = u_xyz.shape[0]
    K = fx.numel()
    cx, cy, cz = center_xyz

    px = cx + fx.view(1, K) * u_xyz[:, 0:1] + fy.view(1, K) * v_xyz[:, 0:1]
    py = cy + fx.view(1, K) * u_xyz[:, 1:2] + fy.view(1, K) * v_xyz[:, 1:2]
    pz = cz + fx.view(1, K) * u_xyz[:, 2:3] + fy.view(1, K) * v_xyz[:, 2:3]

    x0 = torch.floor(px)
    y0 = torch.floor(py)
    z0 = torch.floor(pz)
    tx, ty, tz = px - x0, py - y0, pz - z0
    ix0, iy0, iz0 = x0.long(), y0.long(), z0.long()

    def _gather(gx: torch.Tensor, gy: torch.Tensor, gz: torch.Tensor) -> torch.Tensor:
        inb = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H) & (gz >= 0) & (gz < D)
        gx_c = gx.clamp(0, W - 1)
        gy_c = gy.clamp(0, H - 1)
        gz_c = gz.clamp(0, D - 1)
        vals = vol_ri[:, :, gz_c, gy_c, gx_c].permute(0, 2, 1, 3)  # (B_vol, N_pose, 2, K)
        return vals * inb[:, None, :].to(vals.dtype).unsqueeze(0)

    omx, omy, omz = 1.0 - tx, 1.0 - ty, 1.0 - tz
    corners = [
        (0, 0, 0, omx * omy * omz),
        (1, 0, 0, tx * omy * omz),
        (0, 1, 0, omx * ty * omz),
        (1, 1, 0, tx * ty * omz),
        (0, 0, 1, omx * omy * tz),
        (1, 0, 1, tx * omy * tz),
        (0, 1, 1, omx * ty * tz),
        (1, 1, 1, tx * ty * tz),
    ]
    acc = torch.zeros((B_vol, N_pose, 2, K), device=vol_ri.device, dtype=vol_ri.dtype)
    for dx, dy, dz, w in corners:
        acc = acc + _gather(ix0 + dx, iy0 + dy, iz0 + dz) * w[:, None, :].unsqueeze(0)
    return acc
