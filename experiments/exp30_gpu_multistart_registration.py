"""
Experiment 30: End-to-End GPU Multi-Start 6-DoF Biplanar 2D/3D Registration Benchmark.

Evaluates 64-way parallel multi-start registration on DeepFluoro subject01 at
256x256 pinhole geometry (SDD = 1020 mm, 1.0 mm pixel pitch) against ray-marched
targets (nanodrr n=500), measuring wall-clock registration time, peak working
VRAM, final 3D mTRE (mm), and <2.0 mm / <5.0 mm capture rates.

Compares:
  1. nanodrr torch (f32, n=500, B=64; micro-batched if OOM)
  2. nanodrr triton+compile (bf16, n=500, B=64)
  3. PoU-FVR Triton standalone (2x4x2@80 bf16, B=64, SE(3)-invariant warp+sum)
  4. Hybrid PoU-FVR Triton (2x2x2@64 bf16, 35 steps @ B=64) + nanodrr polish (15 steps @ B=1)
"""

import argparse
import csv
import gc
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fvr import PerspectiveSlabFVR
from fvr.cone_beam_fvr import edge_dominant_ncc
from fvr.triton_slice import HAS_TRITON


def _se3_exp(theta: torch.Tensor) -> torch.Tensor:
    """Differentiable exponential map from ``(B, 6) = [rx, ry, rz, tx, ty, tz]`` (rad, mm) to ``(B, 4, 4)``."""
    rot = theta[:, :3]
    trans = theta[:, 3:6]
    B = theta.shape[0]
    skew = torch.zeros((B, 3, 3), device=theta.device, dtype=theta.dtype)
    skew[:, 0, 1], skew[:, 1, 0] = -rot[:, 2], rot[:, 2]
    skew[:, 0, 2], skew[:, 2, 0] = rot[:, 1], -rot[:, 1]
    skew[:, 1, 2], skew[:, 2, 1] = -rot[:, 0], rot[:, 0]
    R = torch.matrix_exp(skew)
    T = torch.eye(4, device=theta.device, dtype=theta.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    return T


def _make_k_inv_pure(sdd: float, delx: float, dely: float, height: int, width: int, device: torch.device) -> torch.Tensor:
    fx = float(sdd) / float(delx)
    fy = float(sdd) / float(dely)
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    return torch.linalg.inv(K).unsqueeze(0)


def _euler_zyx_deg_to_mat(rot_deg: torch.Tensor) -> torch.Tensor:
    rad = rot_deg * (math.pi / 180.0)
    rz, ry, rx = rad[:, 2], rad[:, 1], rad[:, 0]
    cz, sz = torch.cos(rz), torch.sin(rz)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cx, sx = torch.cos(rx), torch.sin(rx)
    R = torch.zeros((rot_deg.shape[0], 3, 3), device=rot_deg.device, dtype=torch.float32)
    R[:, 0, 0] = cy * cz
    R[:, 0, 1] = cz * sx * sy - cx * sz
    R[:, 0, 2] = cx * cz * sy + sx * sz
    R[:, 1, 0] = cy * sz
    R[:, 1, 1] = cx * cz + sx * sy * sz
    R[:, 1, 2] = cx * sy * sz - cz * sx
    R[:, 2, 0] = -sy
    R[:, 2, 1] = cy * sx
    R[:, 2, 2] = cx * cy
    return R


def _make_ap_rt_inv_pure(rot_deg: torch.Tensor, xyz_mm: torch.Tensor) -> torch.Tensor:
    """Fallback camera-to-world SE(3) matching rotated AP/LAT pinhole geometry for CPU smoke tests."""
    B = rot_deg.shape[0]
    R = _euler_zyx_deg_to_mat(rot_deg)
    R_ap = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], device=rot_deg.device, dtype=torch.float32)
    T = torch.eye(4, device=rot_deg.device, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R @ R_ap
    T[:, :3, 3] = torch.einsum("bij,bj->bi", R, xyz_mm)
    return T


def _build_fiducials(vol_shape_zyx: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Build 27 centered anatomical landmark coordinates (mm) spanning the interior volume."""
    d, h, w = vol_shape_zyx[-3:]
    ext = [0.28 * float(s) for s in (w, h, d)]
    xs = torch.linspace(-ext[0], ext[0], 3, device=device)
    ys = torch.linspace(-ext[1], ext[1], 3, device=device)
    zs = torch.linspace(-ext[2], ext[2], 3, device=device)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)


def _compute_mtre(dT_rel: torch.Tensor, pts_mm: torch.Tensor) -> float:
    """Mean target registration error (mm) of relative SE(3) error ``dT_rel (4, 4)`` on landmarks ``(K, 3)``."""
    R = dT_rel[:3, :3]
    t = dT_rel[:3, 3]
    disp = (pts_mm @ R.T + t.unsqueeze(0)) - pts_mm
    return float(disp.norm(dim=-1).mean())


def _sample_multistart_theta(B: int, device: torch.device, seed: int, rot_deg_lim: float = 12.0, trans_mm_lim: float = 15.0) -> torch.Tensor:
    """Sample (B, 6) multi-start seeds in [rad, mm], keeping index 0 at the unperturbed init."""
    rng = np.random.default_rng(seed)
    theta = np.zeros((B, 6), dtype=np.float32)
    if B > 1:
        rot_lim = rot_deg_lim * (math.pi / 180.0)
        theta[1:, :3] = rng.uniform(-rot_lim, rot_lim, size=(B - 1, 3)).astype(np.float32)
        theta[1:, 3:] = rng.uniform(-trans_mm_lim, trans_mm_lim, size=(B - 1, 3)).astype(np.float32)
    return torch.from_numpy(theta).to(device=device)


def _biplanar_loss_per_cand(
    imA: torch.Tensor,
    imB: torch.Tensor,
    refA: torch.Tensor,
    refB: torch.Tensor,
) -> torch.Tensor:
    def _pair_sim(p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        p0 = p.float() - p.float().mean(dim=(-2, -1), keepdim=True)
        r0 = r.float() - r.float().mean(dim=(-2, -1), keepdim=True)
        num = (p0 * r0).sum(dim=(-2, -1))
        den = (p0.square().sum(dim=(-2, -1)) * r0.square().sum(dim=(-2, -1))).clamp(min=1e-12).sqrt()
        img_ncc = (num / den).squeeze(1)
        return 0.75 * edge_dominant_ncc(p.float(), r.float(), reduction="none") + 0.25 * img_ncc

    return 1.0 - 0.5 * (_pair_sim(imA, refA) + _pair_sim(imB, refB))


def _run_adam_multistart(
    render_pair_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    theta_init: torch.Tensor,
    refA: torch.Tensor,
    refB: torch.Tensor,
    steps: int,
    lr_rot: float = 0.02,
    lr_trans: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run independent per-candidate Adam across all B candidates simultaneously."""
    B = int(theta_init.shape[0])
    rot_p = theta_init[:, :3].clone().detach().requires_grad_(True)
    trans_p = theta_init[:, 3:6].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam(
        [{"params": [rot_p], "lr": lr_rot}, {"params": [trans_p], "lr": lr_trans}],
        betas=(0.8, 0.99),
    )
    t_max = max(1, steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(s, t_max) / t_max))
    )

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        dT = _se3_exp(torch.cat([rot_p, trans_p], dim=-1))
        imA, imB = render_pair_fn(dT)
        losses = _biplanar_loss_per_cand(imA, imB, refA, refB)
        losses.sum().backward()
        if rot_p.grad is not None:
            rot_p.grad.clamp_(-0.25, 0.25)
        if trans_p.grad is not None:
            trans_p.grad.clamp_(-2.5, 2.5)
        opt.step()
        sched.step()

    with torch.no_grad():
        theta_final = torch.cat([rot_p, trans_p], dim=-1)
        imA, imB = render_pair_fn(_se3_exp(theta_final))
        final_losses = _biplanar_loss_per_cand(imA, imB, refA, refB)
        if B == 1:
            best_idx = 0
        else:
            best_idx = int(torch.argmin(final_losses))
    return theta_final[best_idx : best_idx + 1].detach(), final_losses[best_idx : best_idx + 1].detach()


def load_deepfluoro_subject01() -> str:
    local = ROOT / "data" / "deepfluoro" / "deepfluoro" / "subject01" / "volume.nii.gz"
    if local.exists():
        return str(local)
    from nanodrr.data import download_deepfluoro

    img_path, _ = download_deepfluoro(1)
    return img_path


def run_multistart_benchmark(smoke: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and hasattr(torch, "_dynamo"):
        torch._dynamo.config.cache_size_limit = 64

    try:
        from nanodrr.camera import make_k_inv, make_rt_inv
        from nanodrr.data import Subject as NanoSubject
        from nanodrr.drr import DRR as NanoDRR

        has_nanodrr = True
    except ImportError:
        has_nanodrr = False

    from diffdrr.data import read

    vol_path = load_deepfluoro_subject01()
    d_subj_cpu = read(vol_path, orientation="AP")
    H = 128 if smoke else 256
    delx = 2.0 * (128.0 / H)
    sdd_mm = 1020.0
    B_cand = 8 if smoke else 64
    n_trials = 1 if smoke else 10
    total_steps = 4 if smoke else 50
    hybrid_global_steps = 3 if smoke else 35
    hybrid_polish_steps = 1 if smoke else 15

    if has_nanodrr:
        k_inv = make_k_inv(sdd=sdd_mm, delx=delx, dely=delx, x0=0.0, y0=0.0, height=H, width=H).to(device)
        n_subj_f32 = NanoSubject.from_filepath(vol_path).to(dtype=torch.float32, device=device)
        n_subj_bf16 = NanoSubject.from_filepath(vol_path).to(dtype=torch.bfloat16, device=device) if device.type == "cuda" else n_subj_f32
        nano_cone = NanoDRR(k_inv=k_inv, sdd=torch.tensor([sdd_mm], device=device), height=H, width=H)
        nano_comp = torch.compile(nano_cone, mode="default") if device.type == "cuda" else nano_cone
        iso = n_subj_f32.isocenter.to(device=device, dtype=torch.float32)

        def _base_pair():
            rotA = torch.tensor([[5.0, -3.0, 4.0]], device=device)
            xyz = torch.tensor([[0.0, 850.0, 0.0]], device=device)
            rtA_n = make_rt_inv(rotA, xyz, orientation="AP", isocenter=iso)
            rtA_f = make_rt_inv(rotA, xyz, orientation="AP", isocenter=torch.zeros(3, device=device))
            return rtA_n, rtA_f

    else:
        k_inv = _make_k_inv_pure(sdd_mm, delx, delx, H, H, device)
        iso = torch.zeros(3, device=device)

        def _base_pair():
            rotA = torch.tensor([[5.0, -3.0, 4.0]], device=device)
            xyz = torch.tensor([[0.0, 850.0, 0.0]], device=device)
            rtA = _make_ap_rt_inv_pure(rotA, xyz)
            return rtA, rtA

    lat = PerspectiveSlabFVR.make_pinhole_lattice(k_inv, sdd_mm=sdd_mm, height=H, width=H)

    # PoU-FVR renderers: (2x4x2@80) standalone & (2x2x2@64) fast global stage
    use_tri = device.type == "cuda" and HAS_TRITON
    fvr_hi = PerspectiveSlabFVR.from_diffdrr_subject(
        d_subj_cpu, grid_blocks=(2, 2, 2) if smoke else (2, 4, 2), block_res=48 if smoke else 80, pad_mult=2, use_triton=use_tri
    ).to(device)
    fvr_fast = PerspectiveSlabFVR.from_diffdrr_subject(
        d_subj_cpu, grid_blocks=(2, 2, 2), block_res=48 if smoke else 64, pad_mult=2, use_triton=use_tri
    ).to(device)
    if device.type == "cuda":
        fvr_hi.to_precision(torch.bfloat16)
        fvr_fast.to_precision(torch.bfloat16)

    rtA_n0, rtA_f0 = _base_pair()
    P_iso = torch.eye(4, device=device, dtype=torch.float32)
    P_iso[:3, 3] = iso
    P_neg_iso = torch.eye(4, device=device, dtype=torch.float32)
    P_neg_iso[:3, 3] = -iso
    Rz90 = torch.tensor(
        [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    rtB_f0 = Rz90 @ rtA_f0
    rtB_n0 = (P_iso @ Rz90 @ P_neg_iso) @ rtA_n0

    def _fvr_pair(fvr_mod: PerspectiveSlabFVR, initA_f: torch.Tensor, initB_f: torch.Tensor, dT: torch.Tensor):
        rtA = dT @ initA_f
        rtB = dT @ initB_f
        return fvr_mod.forward_pinhole(rtA, lat), fvr_mod.forward_pinhole(rtB, lat)

    def _nano_pair(mod, subj, dT: torch.Tensor, **kwargs):
        off = P_iso @ dT @ P_neg_iso
        return mod(subj, off @ rtA_n0, **kwargs), mod(subj, off @ rtB_n0, **kwargs)

    vol_shape = tuple(int(x) for x in d_subj_cpu.density.data.shape)
    fiducials = _build_fiducials(vol_shape, device)

    # Verify initial alignment between PoU-FVR and nanodrr across BOTH AP & LAT views
    if has_nanodrr:
        with torch.no_grad():
            bk = "torch" if device.type != "cuda" else "auto"
            refA0 = nano_cone(n_subj_f32, rtA_n0, n_samples=500, backend=bk)
            refB0 = nano_cone(n_subj_f32, rtB_n0, n_samples=500, backend=bk)
            pouA0 = fvr_hi.forward_pinhole(rtA_f0, lat)
            pouB0 = fvr_hi.forward_pinhole(rtB_f0, lat)

            def _p_ncc(x, y):
                x0, y0 = x.float().flatten() - x.float().mean(), y.float().flatten() - y.float().mean()
                return float((x0 * y0).sum() / (x0.norm() * y0.norm() + 1e-8))

            print(
                f"[Init Parity Check] PoU-FVR vs nanodrr(n=500): AP NCC={_p_ncc(pouA0, refA0):.4f}, "
                f"LAT NCC={_p_ncc(pouB0, refB0):.4f}"
            )

    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(
        f"=== Experiment 30: Biplanar Multi-Start 6-DoF Registration (H={H}, B={B_cand}, "
        f"trials={n_trials}, steps={total_steps}) | Device={gpu_label} ==="
    )

    trial_rows: List[Dict[str, object]] = []
    rng_gt = np.random.default_rng(2026)

    for trial in range(n_trials):
        rot_gt = rng_gt.uniform(-10.0, 10.0, size=(1, 3)).astype(np.float32) * (math.pi / 180.0)
        trans_gt = rng_gt.uniform(-12.0, 12.0, size=(1, 3)).astype(np.float32)
        theta_gt = torch.from_numpy(np.concatenate([rot_gt, trans_gt], axis=-1)).to(device=device)
        dT_gt = _se3_exp(theta_gt)
        init_mtre = _compute_mtre(torch.linalg.inv(dT_gt[0]), fiducials)

        rtA_f_gt = dT_gt @ rtA_f0
        rtB_f_gt = dT_gt @ rtB_f0
        rtA_n_gt = (P_iso @ dT_gt @ P_neg_iso) @ rtA_n0
        rtB_n_gt = (P_iso @ dT_gt @ P_neg_iso) @ rtB_n0

        with torch.no_grad():
            if has_nanodrr:
                xrayA = nano_cone(n_subj_f32, rtA_n_gt, n_samples=500, backend="torch" if device.type != "cuda" else "auto")
                xrayB = nano_cone(n_subj_f32, rtB_n_gt, n_samples=500, backend="torch" if device.type != "cuda" else "auto")
            else:
                xrayA = fvr_hi.forward_pinhole(rtA_f_gt, lat)
                xrayB = fvr_hi.forward_pinhole(rtB_f_gt, lat)

        theta_seeds = _sample_multistart_theta(B_cand, device=device, seed=1000 + trial)
        print(f"\n--- Trial {trial + 1}/{n_trials} | Init mTRE = {init_mtre:.2f} mm ---")

        def _record(method_name: str, run_callable: Callable[[], Tuple[torch.Tensor, torch.Tensor]]):
            if device.type == "cuda":
                torch.cuda.synchronize()
                base_bytes = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            else:
                base_bytes = 0
            t0 = time.perf_counter()
            theta_best, best_loss = run_callable()
            if device.type == "cuda":
                torch.cuda.synchronize()
                peak_mb = max(0, torch.cuda.max_memory_allocated() - base_bytes) / (1024.0 * 1024.0)
            else:
                peak_mb = 0.0
            elapsed = time.perf_counter() - t0
            dT_est = _se3_exp(theta_best)[0]
            dT_err = torch.linalg.inv(dT_gt[0]) @ dT_est
            final_mtre = _compute_mtre(dT_err, fiducials)
            succ2 = final_mtre < 2.0
            succ5 = final_mtre < 5.0
            print(
                f"  {method_name:<40} time={elapsed:6.2f}s | vram={peak_mb:6.1f} MB | "
                f"mTRE={final_mtre:5.2f} mm | <2mm={succ2} | loss={float(best_loss):.4f}"
            )
            trial_rows.append({
                "device": gpu_label,
                "trial": trial + 1,
                "H": H,
                "B": B_cand,
                "steps": total_steps,
                "method": method_name,
                "init_mtre_mm": round(init_mtre, 3),
                "final_mtre_mm": round(final_mtre, 3),
                "success_2mm": int(succ2),
                "success_5mm": int(succ5),
                "time_s": round(elapsed, 4),
                "peak_vram_mb": round(peak_mb, 2),
                "final_edge_loss": round(float(best_loss), 5),
            })
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Warm up torch.compile / Triton at full B_cand so Trial 1 avoids div16/shape recompilation
        if trial == 0:
            fvr_hi.warmup_pinhole(lat, batch_size=B_cand, backward=True)
            fvr_fast.warmup_pinhole(lat, batch_size=B_cand, backward=True)
            _run_adam_multistart(
                lambda dT: _fvr_pair(fvr_hi, rtA_f0, rtB_f0, dT), theta_seeds, xrayA, xrayB, steps=1
            )
            _run_adam_multistart(
                lambda dT: _fvr_pair(fvr_fast, rtA_f0, rtB_f0, dT), theta_seeds, xrayA, xrayB, steps=1
            )
            if has_nanodrr and device.type == "cuda":
                _run_adam_multistart(
                    lambda dT: _nano_pair(nano_comp, n_subj_bf16, dT, n_samples=500),
                    theta_seeds,
                    xrayA,
                    xrayB,
                    steps=1,
                )

        _record(
            "PoU-FVR Triton (2x4x2@80 bf16)" if device.type == "cuda" else "PoU-FVR Pinhole (f32)",
            lambda: _run_adam_multistart(
                lambda dT: _fvr_pair(fvr_hi, rtA_f0, rtB_f0, dT), theta_seeds, xrayA, xrayB, steps=total_steps
            ),
        )

        if has_nanodrr and device.type == "cuda":
            def _run_hybrid():
                theta_top, _ = _run_adam_multistart(
                    lambda dT: _fvr_pair(fvr_fast, rtA_f0, rtB_f0, dT),
                    theta_seeds,
                    xrayA,
                    xrayB,
                    steps=hybrid_global_steps,
                )
                return _run_adam_multistart(
                    lambda dT: _nano_pair(nano_comp, n_subj_bf16, dT, n_samples=500),
                    theta_top,
                    xrayA,
                    xrayB,
                    steps=hybrid_polish_steps,
                    lr_rot=0.008,
                    lr_trans=0.5,
                )

            _record("Hybrid PoU-Triton + nanodrr polish", _run_hybrid)

            _record(
                "nanodrr triton+compile (bf16 n=500)",
                lambda: _run_adam_multistart(
                    lambda dT: _nano_pair(nano_comp, n_subj_bf16, dT, n_samples=500),
                    theta_seeds,
                    xrayA,
                    xrayB,
                    steps=total_steps,
                ),
            )

            if trial < 3:  # Run slow PyTorch ray-marching baseline on first 3 trials (~1 min total) unless OOM
                try:
                    _record(
                        "nanodrr torch (f32 n=500)",
                        lambda: _run_adam_multistart(
                            lambda dT: _nano_pair(nano_cone, n_subj_f32, dT, n_samples=500, backend="torch"),
                            theta_seeds,
                            xrayA,
                            xrayB,
                            steps=total_steps,
                        ),
                    )
                except torch.OutOfMemoryError:
                    print("  nanodrr torch (f32 n=500)                OOM at B=64 during autograd")
                    trial_rows.append({
                        "device": gpu_label, "trial": trial + 1, "H": H, "B": B_cand, "steps": total_steps,
                        "method": "nanodrr torch (f32 n=500)", "init_mtre_mm": round(init_mtre, 3),
                        "final_mtre_mm": "OOM", "success_2mm": 0, "success_5mm": 0,
                        "time_s": "OOM", "peak_vram_mb": "OOM", "final_edge_loss": "OOM",
                    })
                    torch.cuda.empty_cache()

    # Build cohort summary
    summary_rows: List[Dict[str, object]] = []
    methods_seen = []
    for r in trial_rows:
        if r["method"] not in methods_seen:
            methods_seen.append(r["method"])
    print("\n=== Experiment 30 Multi-Start Registration Summary ===")
    for m_name in methods_seen:
        sub = [r for r in trial_rows if r["method"] == m_name and r["final_mtre_mm"] != "OOM"]
        if not sub:
            summary_rows.append({"method": m_name, "n_trials": 0, "mean_mtre_mm": "OOM"})
            continue
        mtres = [float(r["final_mtre_mm"]) for r in sub]
        times = [float(r["time_s"]) for r in sub]
        vrams = [float(r["peak_vram_mb"]) for r in sub]
        sr2 = 100.0 * sum(int(r["success_2mm"]) for r in sub) / len(sub)
        sr5 = 100.0 * sum(int(r["success_5mm"]) for r in sub) / len(sub)
        row_s = {
            "device": gpu_label,
            "method": m_name,
            "n_trials": len(sub),
            "mean_init_mtre_mm": round(statistics.mean(float(r["init_mtre_mm"]) for r in sub), 3),
            "mean_mtre_mm": round(statistics.mean(mtres), 3),
            "median_mtre_mm": round(statistics.median(mtres), 3),
            "success_2mm_pct": round(sr2, 1),
            "success_5mm_pct": round(sr5, 1),
            "mean_time_s": round(statistics.mean(times), 3),
            "mean_peak_vram_mb": round(statistics.mean(vrams), 1),
        }
        summary_rows.append(row_s)
        print(
            f"  {m_name:<38} n={len(sub):2d} | mean_mTRE={row_s['mean_mtre_mm']:5.2f} mm "
            f"(med={row_s['median_mtre_mm']:5.2f}) | <2mm={sr2:5.1f}% | "
            f"time={row_s['mean_time_s']:5.2f}s | VRAM={row_s['mean_peak_vram_mb']:6.1f} MB"
        )

    if smoke:
        print("\n[Smoke mode] Skipping CSV overwrite.")
        return

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "exp30_gpu_multistart_reg.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
        w.writeheader()
        w.writerows(trial_rows)
    with open(out_dir / "exp30_gpu_multistart_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved CSVs to {out_dir / 'exp30_gpu_multistart_reg.csv'} and {out_dir / 'exp30_gpu_multistart_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run fast CPU sanity check without overwriting CSVs.")
    args = parser.parse_args()
    run_multistart_benchmark(smoke=args.smoke)
