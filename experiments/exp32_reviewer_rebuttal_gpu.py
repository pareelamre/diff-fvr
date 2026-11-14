"""
Experiment 32: Unified GPU Reviewer-Rebuttal Validation Suite
=============================================================
Runs all adversarial peer-review rebuttal experiments in a single GPU pass:

  1. exp32a — Matched Multi-Start (B=64) Real Clinical DeepFluoro Cohort (subject01-subject06):
     - Stratified into Tier 1 (Bounded ~7-20 mm init) and Tier 2 (Wide-Basin ~18-30 mm init,
       wider +/-16 deg / +/-22 mm multi-start box without pinning seed[0] alone to rescue).
     - Compares 5 matched pipelines with calibrated nanodrr <-> DiffDRR coordinate parity:
         (a) DiffDRR Single-Start (f32 B=1)
         (b) nanodrr triton+compile Multi-Start (bf16 n=500, B=64)  [matched B=64 competitor]
         (c) PoU-FVR Triton Multi-Start (2x4x2@80 bf16, B=64)
         (d) Hybrid PoU-Triton + nanodrr polish (<0.5 GB VRAM)        [low-VRAM Siddon polish]
         (e) Hybrid PoU-Triton + DiffDRR polish                       [sub-mm Siddon polish]
     - Plus a dedicated DiffDRR Multi-Start (f32 B=64) timed/VRAM probe on subject01
       (with automatic micro-batch fallback on CUDA OOM) to establish exact speed & memory.
     - Records prep_s separately from reg_time_s, Peak VRAM, IQR (Q25/Q75), and paired
       Wilcoxon signed-rank p-values.

  2. exp32b — Precision Parity Ablation (float32 vs. bfloat16 PoU-FVR):
     - Evaluates forward TrueNCC vs DiffDRR and B=64 multi-start registration accuracy/latency
       across all 6 DeepFluoro subjects in both float32 and bfloat16.

  3. exp32c — Clinical Spine & Abdominal CT Cone-Beam GPU Evaluation (H=256 & H=512):
     - Evaluates B=64 cone-beam 6-DoF registration (SDD=1020 mm, Poisson-Gaussian clinical
       noise) on actual spine/abdominal CT volumes shipped in the repository:
         - Kaggle Lumbar Spine CT (100x512x512, 1.0x0.276x0.276 mm)
         - AMOS Thoracolumbar/Abdominal CT (90x768x768, 5.0x0.570x0.570 mm)
         - GE Clinical QA CT (28x512x512, 5.63x0.488x0.488 mm)
       against independent spatial ray-marched targets (nanodrr n=500, or DiffDRR in --smoke).

Outputs:
  - outputs/exp32_deepfluoro_matched_b64.csv
  - outputs/exp32_deepfluoro_summary.csv
  - outputs/exp32_precision_ablation.csv
  - outputs/exp32_spine_ct_conebeam.csv
  - outputs/exp32_spine_ct_summary.csv
"""

import argparse
import csv
import gc
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert

from fvr import PerspectiveSlabFVR
from fvr.cone_beam_fvr import edge_dominant_ncc
from fvr.dicom_loader import load_dicom_series, resample_isotropic_cube
from fvr.triton_slice import HAS_TRITON

SUBJECTS = [f"subject{i:02d}" for i in range(1, 7)]

# Proper SO(3) rotation (det=+1) mapping nanodrr OpenCV camera (+x_u, +y_v, +z_ray)
# to DiffDRR canonical AP detector camera frame (-x_col, -y_sdd, -z_row).
C_CAM_NANO_TO_DIFF = torch.tensor(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
    dtype=torch.float32,
)


def _has_nanodrr() -> bool:
    try:
        import nanodrr  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_deepfluoro_root() -> Path:
    """Locate or download DeepFluoro (eigenvivek/xvr-data) for subject01-subject06."""
    for cand in (ROOT / "data" / "deepfluoro" / "deepfluoro", ROOT / "data" / "deepfluoro"):
        if all((cand / s / "xrays").is_dir() and (cand / s / "fiducials.pt").exists() for s in SUBJECTS):
            return cand

    from huggingface_hub import snapshot_download

    target_dir = ROOT / "data" / "deepfluoro"
    target_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading DeepFluoro dataset (eigenvivek/xvr-data) for subject01-subject06...")
    snapshot_download(
        repo_id="eigenvivek/xvr-data",
        repo_type="dataset",
        allow_patterns="deepfluoro/*",
        local_dir=str(target_dir),
    )
    for cand in (target_dir / "deepfluoro", target_dir):
        if all((cand / s / "xrays").is_dir() and (cand / s / "fiducials.pt").exists() for s in SUBJECTS):
            return cand
    raise FileNotFoundError(f"Could not locate subject01-subject06 under {target_dir}")


def _ncc_2d(a: torch.Tensor, b: torch.Tensor) -> float:
    a0 = a.float().flatten()
    b0 = b.float().flatten()
    a0 = a0 - a0.mean()
    b0 = b0 - b0.mean()
    return float((a0 * b0).sum() / (a0.norm() * b0.norm() + 1e-8))


def _se3_exp(theta: torch.Tensor) -> torch.Tensor:
    """Differentiable exponential map from (B, 6) = [rx, ry, rz, tx, ty, tz] (rad, mm) to (B, 4, 4)."""
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


def _compute_mtre(pose_est: RigidTransform, pose_gt: RigidTransform, fids: torch.Tensor) -> float:
    T_err = RigidTransform(pose_est.matrix @ pose_gt.matrix.inverse())
    return float((T_err(fids) - fids).norm(dim=-1).mean().item())


def _biplanar_edge_loss(imA: torch.Tensor, imB: torch.Tensor, refA: torch.Tensor, refB: torch.Tensor) -> torch.Tensor:
    lA = edge_dominant_ncc(imA.float(), refA.float(), reduction="none")
    lB = edge_dominant_ncc(imB.float(), refB.float(), reduction="none")
    return 1.0 - 0.5 * (lA + lB)


def _sample_multistart_theta(
    B: int,
    device: torch.device,
    seed: int,
    rot_deg: float,
    trans_mm: float,
    keep_zero: bool = True,
) -> torch.Tensor:
    """Sample (B, 6) seeds in [rad, mm]. Optionally keeps seed[0]=0 for Tier 1 baseline parity."""
    rng = np.random.default_rng(seed)
    theta = np.zeros((B, 6), dtype=np.float32)
    start_idx = 1 if (keep_zero and B > 1) else 0
    if B > start_idx:
        rot_lim = rot_deg * (math.pi / 180.0)
        theta[start_idx:, :3] = rng.uniform(-rot_lim, rot_lim, size=(B - start_idx, 3)).astype(np.float32)
        theta[start_idx:, 3:] = rng.uniform(-trans_mm, trans_mm, size=(B - start_idx, 3)).astype(np.float32)
    return torch.from_numpy(theta).to(device=device)


def _optimize_multistart(
    render_pair_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    theta_init: torch.Tensor,
    xrayA: torch.Tensor,
    xrayB: torch.Tensor,
    steps: int,
    lr_rot: float = 0.015,
    lr_trans: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        losses = _biplanar_edge_loss(imA, imB, xrayA, xrayB)
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
        final_losses = _biplanar_edge_loss(imA, imB, xrayA, xrayB)
        best_idx = 0 if B == 1 else int(torch.argmin(final_losses))
    return theta_final[best_idx : best_idx + 1].detach(), final_losses[best_idx : best_idx + 1].detach()


def _optimize_multistart_microbatched(
    render_pair_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    theta_init: torch.Tensor,
    xrayA: torch.Tensor,
    xrayB: torch.Tensor,
    steps: int,
    micro_batch: int = 64,
    lr_rot: float = 0.015,
    lr_trans: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identical per-candidate Adam update supporting gradient accumulation via micro-batching."""
    B = int(theta_init.shape[0])
    if micro_batch >= B:
        return _optimize_multistart(render_pair_fn, theta_init, xrayA, xrayB, steps, lr_rot, lr_trans)

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
        for start in range(0, B, micro_batch):
            end = min(B, start + micro_batch)
            dT = _se3_exp(torch.cat([rot_p[start:end], trans_p[start:end]], dim=-1))
            imA, imB = render_pair_fn(dT)
            losses = _biplanar_edge_loss(imA, imB, xrayA, xrayB)
            losses.sum().backward()
        if rot_p.grad is not None:
            rot_p.grad.clamp_(-0.25, 0.25)
        if trans_p.grad is not None:
            trans_p.grad.clamp_(-2.5, 2.5)
        opt.step()
        sched.step()

    with torch.no_grad():
        theta_final = torch.cat([rot_p, trans_p], dim=-1)
        all_losses = []
        for start in range(0, B, micro_batch):
            end = min(B, start + micro_batch)
            imA, imB = render_pair_fn(_se3_exp(theta_final[start:end]))
            all_losses.append(_biplanar_edge_loss(imA, imB, xrayA, xrayB))
        final_losses = torch.cat(all_losses, dim=0)
        best_idx = 0 if B == 1 else int(torch.argmin(final_losses))
    return theta_final[best_idx : best_idx + 1].detach(), final_losses[best_idx : best_idx + 1].detach()


def _load_deepfluoro_view(subject, subj_dir: str, vid: str, H: int, device: torch.device):
    pt = torch.load(os.path.join(subj_dir, "xrays", f"{vid}.pt"), weights_only=False)
    intr = pt["intrinsics"]
    pose = RigidTransform(pt["pose"].to(device=device, dtype=torch.float32))
    dcm = pydicom.dcmread(os.path.join(subj_dir, "xrays", f"{vid}.dcm")).pixel_array.astype("float32")
    off = (dcm.shape[0] - intr["height"]) // 2
    crop = -dcm[off : off + intr["height"], off : off + intr["width"]]
    xray = (
        F.interpolate(
            torch.from_numpy(crop).unsqueeze(0).unsqueeze(0),
            size=(H, H),
            mode="bilinear",
            align_corners=False,
        )
        .flip(-1)
        .to(device=device, dtype=torch.float32)
    )
    delx_scaled = intr["delx"] * (intr["height"] / H)
    drr = DRR(
        subject,
        sdd=intr["sdd"],
        height=H,
        delx=delx_scaled,
        x0=intr["x0"],
        y0=intr["y0"],
    ).to(device)
    intr_scaled = {
        "sdd": float(intr["sdd"]),
        "delx": float(delx_scaled),
        "x0": float(intr["x0"]),
        "y0": float(intr["y0"]),
        "H": int(H),
    }
    return pose, xray, drr, intr_scaled


def _find_max_optical_pair(subject, subj_dir: str, H: int = 64) -> Tuple[str, str, float]:
    pts = sorted(Path(subj_dir, "xrays").glob("*.pt"))
    vids = [p.stem for p in pts]
    cpu = torch.device("cpu")
    axes = []
    for vid in vids:
        pose, _, drr, _ = _load_deepfluoro_view(subject, subj_dir, vid, H=H, device=cpu)
        with torch.no_grad():
            src, trg = drr.detector(pose, calibration=None)
            c_det = trg.squeeze(0).view(H, H, 3).mean(dim=(0, 1))
            ray = c_det - src.squeeze(0).squeeze(0)
            axes.append(ray / (ray.norm() + 1e-12))

    best_ang, best_pair = -1.0, (vids[0], vids[1])
    for i in range(len(vids)):
        for j in range(i + 1, len(vids)):
            c = torch.dot(axes[i], axes[j]).clamp(-1.0, 1.0).item()
            ang = math.degrees(math.acos(c))
            if ang > best_ang:
                best_ang = ang
                best_pair = (vids[i], vids[j])
    return best_pair[0], best_pair[1], best_ang


def _calibrate_nanodrr_to_diffdrr(
    vol_path: str,
    drr_ref,
    pose_ref: RigidTransform,
    intr: Dict[str, float],
    device: torch.device,
    min_ncc: float = 0.985,
):
    """
    Construct nanodrr Subject & DRR calibrated cleanly against DiffDRR's detector frame.
    Checks principal-point sign and sub-voxel isocenter alignment against drr_ref(pose_ref)
    so nanodrr operates on an exact apples-to-apples world coordinate frame.
    """
    from nanodrr.camera import make_k_inv
    from nanodrr.data import Subject as NanoSubject
    from nanodrr.drr import DRR as NanoDRR

    H = int(intr["H"])
    sdd = float(intr["sdd"])
    delx = float(intr["delx"])
    x0 = float(intr["x0"])
    y0 = float(intr["y0"])

    n_subj_f32 = NanoSubject.from_filepath(vol_path).to(dtype=torch.float32, device=device)
    n_subj_bf16 = (
        NanoSubject.from_filepath(vol_path).to(dtype=torch.bfloat16, device=device)
        if device.type == "cuda"
        else n_subj_f32
    )
    iso = n_subj_f32.isocenter.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        ref_img = drr_ref(pose_ref)
        # Re-orient DiffDRR world center (0,0,0) into nanodrr voxel-centered isocenter
        # and verify sign convention of (x0, y0) plus optional half-voxel shift.
        C_cam = C_CAM_NANO_TO_DIFF.to(device)
        vox_sp = getattr(n_subj_f32, "spacing", torch.zeros(3, device=device))
        if not torch.is_tensor(vox_sp):
            vox_sp = torch.tensor(vox_sp, device=device, dtype=torch.float32)
        vox_sp = vox_sp.to(device=device, dtype=torch.float32).flatten()[:3]

        best_ncc = -1.0
        best_cfg = None
        pose_mat = pose_ref.matrix[0]

        for x_sgn in (1.0, -1.0):
            for y_sgn in (1.0, -1.0):
                k_cand = make_k_inv(
                    sdd=sdd, delx=delx, dely=delx, x0=x_sgn * x0, y0=y_sgn * y0, height=H, width=H
                ).to(device)
                nano_cand = NanoDRR(k_inv=k_cand, sdd=torch.tensor([sdd], device=device), height=H, width=H)
                for iso_mul in (0.0, -0.5, 0.5):
                    iso_cand = iso + iso_mul * vox_sp
                    rt = torch.eye(4, device=device, dtype=torch.float32)
                    rt[:3, :3] = pose_mat[:3, :3] @ C_cam
                    rt[:3, 3] = pose_mat[:3, 3] + iso_cand
                    out = nano_cand(n_subj_f32, rt.unsqueeze(0), n_samples=500, backend="torch" if device.type != "cuda" else "auto")
                    val = _ncc_2d(out, ref_img)
                    if val > best_ncc:
                        best_ncc = val
                        best_cfg = (k_cand, iso_cand.clone(), x_sgn, y_sgn, iso_mul)

        k_best, iso_best, x_sgn, y_sgn, iso_mul = best_cfg
        if best_ncc < min_ncc:
            raise RuntimeError(
                f"nanodrr calibration failed against DiffDRR on {vol_path}: best NCC={best_ncc:.4f} < {min_ncc}"
            )
        nano_mod = NanoDRR(k_inv=k_best, sdd=torch.tensor([sdd], device=device), height=H, width=H)
        return n_subj_f32, n_subj_bf16, nano_mod, iso_best, best_ncc


def _diff_pose_to_nano_rt(pose_mats: torch.Tensor, iso: torch.Tensor) -> torch.Tensor:
    """Convert batched DiffDRR poses (B, 4, 4) into aligned nanodrr rt_inv (B, 4, 4)."""
    C_cam = C_CAM_NANO_TO_DIFF.to(device=pose_mats.device, dtype=pose_mats.dtype)
    rt = torch.zeros_like(pose_mats)
    rt[:, :3, :3] = pose_mats[:, :3, :3] @ C_cam.unsqueeze(0)
    rt[:, :3, 3] = pose_mats[:, :3, 3] + iso.unsqueeze(0)
    rt[:, 3, 3] = 1.0
    return rt


def _make_nano_for_view(intr: Dict[str, float], k_inv_base: torch.Tensor, device: torch.device):
    """Instantiate a NanoDRR module with identical principal point convention to view A."""
    from nanodrr.drr import DRR as NanoDRR

    return NanoDRR(k_inv=k_inv_base, sdd=torch.tensor([intr["sdd"]], device=device), height=int(intr["H"]), width=int(intr["H"]))


def _stats_iqr(vals: List[float]) -> Tuple[float, float, float, float]:
    s = sorted(vals)
    q25, q75 = np.percentile(s, [25.0, 75.0])
    return float(statistics.mean(s)), float(statistics.median(s)), float(q25), float(q75)


def _paired_wilcoxon(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 1.0
    diffs = np.asarray(x) - np.asarray(y)
    if np.allclose(diffs, 0.0):
        return 1.0
    return float(wilcoxon(x, y, zero_method="wilcox").pvalue)


def _eval_and_record(
    trial_rows: List[Dict[str, object]],
    device: torch.device,
    common: Dict[str, object],
    method_name: str,
    prep_s: float,
    pA_init_mat: torch.Tensor,
    poseA: RigidTransform,
    fids: torch.Tensor,
    run_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
        base_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        base_bytes = 0
    t0 = time.perf_counter()
    theta_best, best_loss = run_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = max(0, torch.cuda.max_memory_allocated() - base_bytes) / (1024.0 * 1024.0)
    else:
        peak_mb = 0.0
    reg_s = time.perf_counter() - t0
    pose_final = RigidTransform(_se3_exp(theta_best) @ pA_init_mat)
    final_mtre = _compute_mtre(pose_final, poseA, fids)
    s2 = int(final_mtre < 2.0)
    s5 = int(final_mtre < 5.0)
    print(
        f"    {method_name:<46} reg={reg_s:6.2f}s (prep={prep_s:4.2f}s) | "
        f"vram={peak_mb:6.1f} MB | mTRE={final_mtre:5.2f} mm | <2mm={bool(s2)}"
    )
    row = dict(common)
    row.update({
        "method": method_name,
        "prep_s": round(prep_s, 3),
        "reg_time_s": round(reg_s, 4),
        "total_time_s": round(prep_s + reg_s, 4),
        "peak_vram_mb": round(peak_mb, 2),
        "final_mtre_mm": round(final_mtre, 3),
        "success_2mm": s2,
        "success_5mm": s5,
        "final_edge_loss": round(float(best_loss), 5),
    })
    trial_rows.append(row)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_mtre


def run_exp32a_and_b(
    smoke: bool,
    device: torch.device,
    use_tri: bool,
    has_nano: bool,
    out_dir: Path,
) -> None:
    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    H = 96 if smoke else 256
    subjects = ["subject01"] if smoke else SUBJECTS
    B_cand = 8 if smoke else 64
    steps_drr = 2 if smoke else 40
    steps_pou = 2 if smoke else 50
    steps_nano = 2 if smoke else 50
    hyb_g_steps = 2 if smoke else 35
    hyb_p_steps = 2 if smoke else 12

    # Tier 1: Bounded (rot <= 0.11 rad, trans <= 12 mm; ~7-20 mm)
    # Tier 2: Wide-Basin Out-of-Basin (rot <= 0.20 rad ~11.5 deg, trans <= 18 mm; ~18-30 mm,
    #         searched with wider +/-16 deg / +/-22 mm multi-start box)
    tiers = [("Tier1_Bounded", 0.09, 10.0, 12.0, 15.0, 1 if smoke else 2)]
    if not smoke:
        tiers.append(("Tier2_WideBasin", 0.20, 18.0, 16.0, 22.0, 2))
    else:
        tiers.append(("Tier2_WideBasin", 0.20, 18.0, 16.0, 22.0, 1))

    root_df = _resolve_deepfluoro_root()
    print(f"\n=== EXP32A: Matched Multi-Start DeepFluoro Cohort | H={H}, B={B_cand} | nanodrr={has_nano} ===")

    trial_rows: List[Dict[str, object]] = []
    prec_rows: List[Dict[str, object]] = []
    warmed_fvr = False
    warmed_drr = False
    warmed_nano = False

    for sidx, sname in enumerate(subjects, start=1):
        subj_dir = str(root_df / sname)
        vol_path = os.path.join(subj_dir, "volume.nii.gz")
        fids = torch.load(os.path.join(subj_dir, "fiducials.pt"), weights_only=False).to(device=device, dtype=torch.float32)
        subject_cpu = read(vol_path, orientation="AP")
        vA, vB, opt_sep_deg = _find_max_optical_pair(subject_cpu, subj_dir)
        poseA, xrayA, drrA, intrA = _load_deepfluoro_view(subject_cpu, subj_dir, vA, H=H, device=device)
        poseB, xrayB, drrB, intrB = _load_deepfluoro_view(subject_cpu, subj_dir, vB, H=H, device=device)
        T_AB = RigidTransform(poseA.matrix.inverse() @ poseB.matrix)

        t_p0 = time.perf_counter()
        fvr_hi_f32 = PerspectiveSlabFVR.from_diffdrr_subject(
            subject_cpu,
            grid_blocks=(2, 2, 2) if smoke else (2, 4, 2),
            block_res=48 if smoke else 80,
            pad_mult=2,
            use_triton=use_tri,
        ).to(device)
        t_prep_hi = time.perf_counter() - t_p0

        t_pf0 = time.perf_counter()
        fvr_fast = PerspectiveSlabFVR.from_diffdrr_subject(
            subject_cpu,
            grid_blocks=(2, 2, 2),
            block_res=48 if smoke else 64,
            pad_mult=2,
            use_triton=use_tri,
        ).to(device)
        if device.type == "cuda":
            fvr_fast.to_precision(torch.bfloat16)
        t_prep_fast = time.perf_counter() - t_pf0

        fvr_hi_bf16 = PerspectiveSlabFVR.from_diffdrr_subject(
            subject_cpu,
            grid_blocks=(2, 2, 2) if smoke else (2, 4, 2),
            block_res=48 if smoke else 80,
            pad_mult=2,
            use_triton=use_tri,
        ).to(device)
        if device.type == "cuda":
            fvr_hi_bf16.to_precision(torch.bfloat16)

        latA_f32 = fvr_hi_f32.cache_diffdrr_geometry(drrA, H)
        latB_f32 = fvr_hi_f32.cache_diffdrr_geometry(drrB, H)
        latA_bf16 = fvr_hi_bf16.cache_diffdrr_geometry(drrA, H)
        latB_bf16 = fvr_hi_bf16.cache_diffdrr_geometry(drrB, H)
        latA_fast = fvr_fast.cache_diffdrr_geometry(drrA, H)
        latB_fast = fvr_fast.cache_diffdrr_geometry(drrB, H)

        if not warmed_fvr and device.type == "cuda":
            fvr_hi_f32.warmup_pinhole(latA_f32, batch_size=B_cand, backward=True)
            fvr_hi_bf16.warmup_pinhole(latA_bf16, batch_size=B_cand, backward=True)
            fvr_fast.warmup_pinhole(latA_fast, batch_size=B_cand, backward=True)
            warmed_fvr = True

        with torch.no_grad():
            drr_gt_a = drrA(poseA)
            pou_gt_f32 = fvr_hi_f32.forward_diffdrr(drrA, poseA, H)
            pou_gt_bf16 = fvr_hi_bf16.forward_diffdrr(drrA, poseA, H)
            ncc_pou_f32 = _ncc_2d(pou_gt_f32, drr_gt_a)
            ncc_pou_bf16 = _ncc_2d(pou_gt_bf16, drr_gt_a)

        # Calibrate nanodrr per-subject against drrA(poseA) and drrB(poseB)
        nano_prep_s = 0.0
        ncc_nano = 0.0
        nanoA_comp = nanoB_comp = None
        n_subj_f32 = n_subj_bf16 = iso_n = None
        if has_nano:
            t_n0 = time.perf_counter()
            n_subj_f32, n_subj_bf16, nanoA, iso_n, ncc_nanoA = _calibrate_nanodrr_to_diffdrr(
                vol_path, drrA, poseA, intrA, device, min_ncc=0.975 if smoke else 0.985
            )
            _, _, nanoB, _, ncc_nanoB = _calibrate_nanodrr_to_diffdrr(
                vol_path, drrB, poseB, intrB, device, min_ncc=0.975 if smoke else 0.985
            )
            nano_subj = n_subj_bf16 if device.type == "cuda" else n_subj_f32
            if device.type == "cuda":
                nanoA_comp = torch.compile(nanoA, mode="default")
                nanoB_comp = torch.compile(nanoB, mode="default")
            else:
                nanoA_comp, nanoB_comp = nanoA, nanoB
            nano_prep_s = time.perf_counter() - t_n0
            ncc_nano = 0.5 * (ncc_nanoA + ncc_nanoB)

            if not warmed_nano and device.type == "cuda":
                pA_w = _diff_pose_to_nano_rt(poseA.matrix.expand(B_cand, -1, -1).clone(), iso_n).requires_grad_(True)
                pB_w = _diff_pose_to_nano_rt(poseB.matrix.expand(B_cand, -1, -1).clone(), iso_n).requires_grad_(True)
                imAw = nanoA_comp(nano_subj, pA_w, n_samples=500, backend="auto")
                imBw = nanoB_comp(nano_subj, pB_w, n_samples=500, backend="auto")
                _biplanar_edge_loss(imAw, imBw, xrayA, xrayB).sum().backward()
                pA_w1 = _diff_pose_to_nano_rt(poseA.matrix.clone(), iso_n).requires_grad_(True)
                pB_w1 = _diff_pose_to_nano_rt(poseB.matrix.clone(), iso_n).requires_grad_(True)
                _biplanar_edge_loss(
                    nanoA_comp(nano_subj, pA_w1, n_samples=500, backend="auto"),
                    nanoB_comp(nano_subj, pB_w1, n_samples=500, backend="auto"),
                    xrayA,
                    xrayB,
                ).sum().backward()
                torch.cuda.synchronize()
                warmed_nano = True

        if not warmed_drr and device.type == "cuda":
            r_w = torch.zeros(1, 3, device=device, requires_grad=True)
            d_w = convert(r_w, torch.zeros(1, 3, device=device), parameterization="euler_angles", convention="ZYX")
            _biplanar_edge_loss(
                drrA(RigidTransform(d_w.matrix @ poseA.matrix)),
                drrB(RigidTransform(d_w.matrix @ poseB.matrix)),
                xrayA,
                xrayB,
            ).sum().backward()
            torch.cuda.synchronize()
            warmed_drr = True

        print(
            f"\n=== {sname} ({vA}+{vB}, sep={opt_sep_deg:.1f}deg) | "
            f"TrueNCC(PoU f32={ncc_pou_f32:.4f}, bf16={ncc_pou_bf16:.4f}, nano={ncc_nano:.4f}) | "
            f"prep(PoU={t_prep_hi:.2f}s, Fast={t_prep_fast:.2f}s, nano={nano_prep_s:.2f}s) ==="
        )

        # Record exp32b precision parity forward metrics
        prec_rows.append({
            "subject": sname,
            "H": H,
            "B": B_cand,
            "stage": "ForwardParity",
            "precision": "float32",
            "true_ncc": round(ncc_pou_f32, 5),
            "init_mtre_mm": 0.0,
            "final_mtre_mm": 0.0,
            "reg_time_s": 0.0,
            "peak_vram_mb": 0.0,
        })
        prec_rows.append({
            "subject": sname,
            "H": H,
            "B": B_cand,
            "stage": "ForwardParity",
            "precision": "bfloat16",
            "true_ncc": round(ncc_pou_bf16, 5),
            "init_mtre_mm": 0.0,
            "final_mtre_mm": 0.0,
            "reg_time_s": 0.0,
            "peak_vram_mb": 0.0,
        })

        for tier_name, rot_scale, trans_scale, s_rot_deg, s_trans_mm, n_inits in tiers:
            for init_idx in range(n_inits):
                rng = np.random.default_rng(3200 + sidx * 100 + (0 if tier_name == "Tier1_Bounded" else 50) + init_idx)
                dr0 = torch.tensor(rng.uniform(-rot_scale, rot_scale, size=(1, 3)), dtype=torch.float32, device=device)
                dt0 = torch.tensor(rng.uniform(-trans_scale, trans_scale, size=(1, 3)), dtype=torch.float32, device=device)
                pose_init = RigidTransform(convert(dr0, dt0, parameterization="euler_angles", convention="ZYX").matrix @ poseA.matrix)
                init_mtre = _compute_mtre(pose_init, poseA, fids)
                pA_init_mat = pose_init.matrix
                pB_init_mat = pA_init_mat @ T_AB.matrix

                # Tier 2 drops seed[0]=0 pinning to verify pure wide-basin multi-start recovery without init anchor.
                keep_zero = (tier_name == "Tier1_Bounded")
                seeds = _sample_multistart_theta(
                    B_cand, device=device, seed=4200 + sidx * 100 + init_idx + (0 if keep_zero else 50),
                    rot_deg=s_rot_deg, trans_mm=s_trans_mm, keep_zero=keep_zero,
                )

                def _make_drr_pair():
                    def _fn(dT: torch.Tensor):
                        return drrA(RigidTransform(dT @ pA_init_mat)), drrB(RigidTransform(dT @ pB_init_mat))
                    return _fn

                def _make_fvr_pair(mod: PerspectiveSlabFVR, lA, lB):
                    def _fn(dT: torch.Tensor):
                        return mod._forward_from_lattice(dT @ pA_init_mat, lA), mod._forward_from_lattice(dT @ pB_init_mat, lB)
                    return _fn

                def _make_nano_pair():
                    bk = "torch" if device.type != "cuda" else "auto"
                    nsub = n_subj_bf16 if device.type == "cuda" else n_subj_f32
                    def _fn(dT: torch.Tensor):
                        rtA = _diff_pose_to_nano_rt(dT @ pA_init_mat, iso_n)
                        rtB = _diff_pose_to_nano_rt(dT @ pB_init_mat, iso_n)
                        return nanoA_comp(nsub, rtA, n_samples=500, backend=bk), nanoB_comp(nsub, rtB, n_samples=500, backend=bk)
                    return _fn

                common = {
                    "device": gpu_label,
                    "subject": sname,
                    "tier": tier_name,
                    "pair": f"{vA}+{vB}",
                    "angle_deg": round(opt_sep_deg, 1),
                    "init_idx": init_idx + 1,
                    "H": H,
                    "B": B_cand,
                    "init_mtre_mm": round(init_mtre, 3),
                }
                print(f"  [{tier_name} | Init {init_idx + 1}/{n_inits} | init_mTRE = {init_mtre:.2f} mm | search=+/-{s_rot_deg:.0f}deg,+/-{s_trans_mm:.0f}mm]")

                _eval_and_record(
                    trial_rows, device, common, "DiffDRR Single-Start (f32 B=1)", 0.0,
                    pA_init_mat, poseA, fids,
                    lambda: _optimize_multistart(_make_drr_pair(), seeds[:1], xrayA, xrayB, steps=steps_drr),
                )

                if has_nano:
                    _eval_and_record(
                        trial_rows, device, common, f"nanodrr triton+compile (bf16 n=500, B={B_cand})", nano_prep_s,
                        pA_init_mat, poseA, fids,
                        lambda: _optimize_multistart(_make_nano_pair(), seeds, xrayA, xrayB, steps=steps_nano),
                    )

                _eval_and_record(
                    trial_rows, device, common, f"PoU-FVR Triton (2x4x2@80 bf16, B={B_cand})", t_prep_hi,
                    pA_init_mat, poseA, fids,
                    lambda: _optimize_multistart(_make_fvr_pair(fvr_hi_bf16, latA_bf16, latB_bf16), seeds, xrayA, xrayB, steps=steps_pou),
                )

                if has_nano:
                    def _run_hyb_nano():
                        th_top, _ = _optimize_multistart(
                            _make_fvr_pair(fvr_fast, latA_fast, latB_fast), seeds, xrayA, xrayB, steps=hyb_g_steps
                        )
                        return _optimize_multistart(
                            _make_nano_pair(), th_top, xrayA, xrayB, steps=hyb_p_steps, lr_rot=0.004, lr_trans=0.4
                        )

                    _eval_and_record(
                        trial_rows, device, common, "Hybrid PoU-Triton + nanodrr polish (<0.5GB)", t_prep_fast,
                        pA_init_mat, poseA, fids, _run_hyb_nano,
                    )

                def _run_hyb_drr():
                    th_top, _ = _optimize_multistart(
                        _make_fvr_pair(fvr_fast, latA_fast, latB_fast), seeds, xrayA, xrayB, steps=hyb_g_steps
                    )
                    return _optimize_multistart(
                        _make_drr_pair(), th_top, xrayA, xrayB, steps=hyb_p_steps, lr_rot=0.004, lr_trans=0.4
                    )

                _eval_and_record(
                    trial_rows, device, common, "Hybrid PoU-Triton + DiffDRR polish", t_prep_fast,
                    pA_init_mat, poseA, fids, _run_hyb_drr,
                )

                # Probe DiffDRR Multi-Start (f32 B=64) ONCE on subject01 Tier1 (micro-batched fallback on OOM)
                if sname == "subject01" and tier_name == "Tier1_Bounded" and init_idx == 0:
                    mb_used = [B_cand]
                    def _run_drr_b64_once():
                        for mb in ([B_cand, 32, 16, 8] if B_cand >= 32 else [B_cand]):
                            try:
                                if device.type == "cuda":
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()
                                    torch.cuda.reset_peak_memory_stats()
                                mb_used[0] = mb
                                return _optimize_multistart_microbatched(
                                    _make_drr_pair(), seeds, xrayA, xrayB, steps=steps_drr, micro_batch=mb
                                )
                            except torch.cuda.OutOfMemoryError:
                                gc.collect()
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                                continue
                        raise RuntimeError("DiffDRR B=64 OOM even at micro_batch=8")

                    _eval_and_record(
                        trial_rows, device, common, f"DiffDRR Multi-Start Probe (f32 B={B_cand})", 0.0,
                        pA_init_mat, poseA, fids, _run_drr_b64_once,
                    )
                    trial_rows[-1]["method"] = f"DiffDRR Multi-Start Probe (f32 B={B_cand}, mb={mb_used[0]})"

                # exp32b precision ablation: first init of each tier (12 cases across 6 subjects)
                if init_idx == 0 and tier_name == "Tier1_Bounded":
                    for prec_lbl, mod_p, lA_p, lB_p in [
                        ("float32", fvr_hi_f32, latA_f32, latB_f32),
                        ("bfloat16", fvr_hi_bf16, latA_bf16, latB_bf16),
                    ]:
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                            base_b = torch.cuda.memory_allocated()
                            torch.cuda.reset_peak_memory_stats()
                        else:
                            base_b = 0
                        tp0 = time.perf_counter()
                        th_p, _ = _optimize_multistart(
                            _make_fvr_pair(mod_p, lA_p, lB_p), seeds, xrayA, xrayB, steps=steps_pou
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                            pk_p = max(0, torch.cuda.max_memory_allocated() - base_b) / (1024.0 * 1024.0)
                        else:
                            pk_p = 0.0
                        tm_p = time.perf_counter() - tp0
                        mtre_p = _compute_mtre(RigidTransform(_se3_exp(th_p) @ pA_init_mat), poseA, fids)
                        prec_rows.append({
                            "subject": sname,
                            "H": H,
                            "B": B_cand,
                            "stage": f"Reg_{tier_name}",
                            "precision": prec_lbl,
                            "true_ncc": round(ncc_pou_f32 if prec_lbl == "float32" else ncc_pou_bf16, 5),
                            "init_mtre_mm": round(init_mtre, 3),
                            "final_mtre_mm": round(mtre_p, 3),
                            "reg_time_s": round(tm_p, 4),
                            "peak_vram_mb": round(pk_p, 2),
                        })

    # Write raw CSVs & build summary with IQR + paired Wilcoxon vs DiffDRR Single-Start and nanodrr B=64
    summary_rows: List[Dict[str, object]] = []
    seen_methods = []
    for r in trial_rows:
        if r["method"] not in seen_methods:
            seen_methods.append(str(r["method"]))

    subsets = [
        ("All_DeepFluoro", lambda r: True),
        ("Tier1_Bounded", lambda r: r["tier"] == "Tier1_Bounded"),
        ("Tier2_WideBasin", lambda r: r["tier"] == "Tier2_WideBasin"),
        ("WideAngle_gt15deg", lambda r: float(r["angle_deg"]) > 15.0),
    ]

    print("\n=== EXP32A SUMMARY (Median [IQR] & Wilcoxon p vs DiffDRR B=1 / nanodrr B=64) ===")
    for sub_lbl, pred in subsets:
        base_drr = [r for r in trial_rows if pred(r) and r["method"] == "DiffDRR Single-Start (f32 B=1)"]
        nano_name = f"nanodrr triton+compile (bf16 n=500, B={B_cand})"
        base_nano = [r for r in trial_rows if pred(r) and r["method"] == nano_name]
        drr_map = {(r["subject"], r["tier"], r["init_idx"]): float(r["final_mtre_mm"]) for r in base_drr}
        nano_map = {(r["subject"], r["tier"], r["init_idx"]): float(r["final_mtre_mm"]) for r in base_nano}

        for m in seen_methods:
            sub = [r for r in trial_rows if pred(r) and r["method"] == m]
            if not sub:
                continue
            mtres = [float(r["final_mtre_mm"]) for r in sub]
            mean_m, med_m, q25, q75 = _stats_iqr(mtres)
            s2 = 100.0 * sum(int(r["success_2mm"]) for r in sub) / len(sub)
            s5 = 100.0 * sum(int(r["success_5mm"]) for r in sub) / len(sub)
            prep_m = statistics.mean(float(r["prep_s"]) for r in sub)
            reg_m = statistics.mean(float(r["reg_time_s"]) for r in sub)
            vram_m = statistics.mean(float(r["peak_vram_mb"]) for r in sub)

            keys = [(r["subject"], r["tier"], r["init_idx"]) for r in sub]
            p_drr = _paired_wilcoxon(mtres, [drr_map[k] for k in keys if k in drr_map]) if len(drr_map) == len(sub) else 1.0
            p_nano = _paired_wilcoxon(mtres, [nano_map[k] for k in keys if k in nano_map]) if (nano_map and len(nano_map) == len(sub)) else 1.0

            print(
                f"  [{sub_lbl:<17}] {m:<46} n={len(sub):2d} | med={med_m:5.2f} [{q25:4.2f},{q75:5.2f}] mm | "
                f"<2mm={s2:5.1f}% | reg={reg_m:5.2f}s | vram={vram_m:6.1f}MB | p_drr={p_drr:.3g}, p_nano={p_nano:.3g}"
            )
            summary_rows.append({
                "device": gpu_label,
                "subset": sub_lbl,
                "method": m,
                "n_cases": len(sub),
                "mean_mtre_mm": round(mean_m, 3),
                "median_mtre_mm": round(med_m, 3),
                "iqr_q25_mm": round(q25, 3),
                "iqr_q75_mm": round(q75, 3),
                "success_2mm_pct": round(s2, 1),
                "success_5mm_pct": round(s5, 1),
                "mean_prep_s": round(prep_m, 3),
                "mean_reg_s": round(reg_m, 3),
                "mean_total_s": round(prep_m + reg_m, 3),
                "mean_peak_vram_mb": round(vram_m, 1),
                "p_wilcoxon_vs_diffdrr_b1": round(p_drr, 6),
                "p_wilcoxon_vs_nanodrr_b64": round(p_nano, 6),
            })

    if not smoke:
        for path, rows in [
            (out_dir / "exp32_deepfluoro_matched_b64.csv", trial_rows),
            (out_dir / "exp32_deepfluoro_summary.csv", summary_rows),
            (out_dir / "exp32_precision_ablation.csv", prec_rows),
        ]:
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"Saved {path}")


def _save_centered_isotropic_nii(vol_hu: np.ndarray, spacing_zyx: Tuple[float, float, float], target_res: int, out_path: Path) -> Tuple[Tuple[int, ...], float]:
    """Resample CT to centered isotropic physical mm cube and write NIfTI for DiffDRR & nanodrr."""
    vol_iso, (vx, vy, vz) = resample_isotropic_cube(vol_hu, spacing_zyx, target_res=target_res)
    img = sitk.GetImageFromArray(vol_iso.astype(np.float32))
    img.SetSpacing((float(vx), float(vy), float(vz)))
    half = 0.5 * (target_res - 1)
    img.SetOrigin((-half * float(vx), -half * float(vy), -half * float(vz)))
    sitk.WriteImage(img, str(out_path))
    return vol_iso.shape, float(vx)


def _add_poisson_gaussian_noise(img: torch.Tensor, photon_count: float = 5e4, gaussian_sigma: float = 0.015, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    arr = img.detach().float().cpu().numpy()
    mn, mx = float(arr.min()), float(arr.max())
    norm = np.clip((arr - mn) / (mx - mn + 1e-8), 0.0, 1.0)
    noisy = rng.poisson(norm * photon_count) / photon_count + rng.normal(0.0, gaussian_sigma, size=arr.shape)
    noisy = np.clip(noisy, 0.0, 1.0) * (mx - mn) + mn
    return torch.from_numpy(noisy.astype(np.float32)).to(device=img.device)


def run_exp32c_spine_ct(
    smoke: bool,
    device: torch.device,
    use_tri: bool,
    has_nano: bool,
    out_dir: Path,
) -> None:
    """Evaluate cone-beam B=64 registration on repository Spine & Abdominal CT datasets."""
    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n=== EXP32C: Clinical Spine & Abdominal CT Cone-Beam GPU Evaluation | nanodrr={has_nano} ===")

    ct_specs: List[Tuple[str, Callable[[], Tuple[np.ndarray, Tuple[float, float, float]]], List[int]]] = [
        (
            "Kaggle_Lumbar_Spine_CT",
            lambda: load_dicom_series(ROOT / "data" / "kaggle_lumbar" / "DICOM"),
            [96] if smoke else [256, 512],
        ),
        (
            "AMOS_Thoracolumbar_CT",
            lambda: (
                lambda itk: (
                    sitk.GetArrayFromImage(itk).astype(np.float32),
                    (float(itk.GetSpacing()[2]), float(itk.GetSpacing()[1]), float(itk.GetSpacing()[0])),
                )
            )(sitk.ReadImage(str(ROOT / "data" / "amos" / "amos_0001.nii.gz"))),
            [96] if smoke else [256],
        ),
    ]
    if not smoke:
        ct_specs.append((
            "GE_Clinical_CT",
            lambda: load_dicom_series(ROOT / "data" / "dicom"),
            [256],
        ))

    B_cand = 8 if smoke else 64
    sdd_mm = 1020.0
    sad_mm = 850.0
    steps_pou = 2 if smoke else 50
    steps_nano = 2 if smoke else 50
    hyb_g_steps = 2 if smoke else 35
    hyb_p_steps = 2 if smoke else 12

    rows: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for ds_idx, (ds_name, loader_fn, res_list) in enumerate(ct_specs):
            vol_hu, spacing_zyx = loader_fn()
            nii_path = tmp_path / f"{ds_name}.nii.gz"
            iso_shape, vox_mm = _save_centered_isotropic_nii(vol_hu, spacing_zyx, target_res=96 if smoke else 160, out_path=nii_path)
            subject_cpu = read(str(nii_path), orientation="AP")

            t_p0 = time.perf_counter()
            fvr_hi = PerspectiveSlabFVR.from_diffdrr_subject(
                subject_cpu,
                grid_blocks=(2, 2, 2) if smoke else (2, 4, 2),
                block_res=48 if smoke else 80,
                pad_mult=2,
                use_triton=use_tri,
            ).to(device)
            if device.type == "cuda":
                fvr_hi.to_precision(torch.bfloat16)
            prep_hi = time.perf_counter() - t_p0

            t_pf0 = time.perf_counter()
            fvr_fast = PerspectiveSlabFVR.from_diffdrr_subject(
                subject_cpu,
                grid_blocks=(2, 2, 2),
                block_res=48 if smoke else 64,
                pad_mult=2,
                use_triton=use_tri,
            ).to(device)
            if device.type == "cuda":
                fvr_fast.to_precision(torch.bfloat16)
            prep_fast = time.perf_counter() - t_pf0

            # 3D interior anatomical fiducials spanning ~60% of physical CT FOV (mm), shape (1, K, 3)
            phys_fov_mm = iso_shape[0] * vox_mm
            half_ext = 0.25 * phys_fov_mm
            ax_pts = torch.linspace(-half_ext, half_ext, 3, device=device)
            gx, gy, gz = torch.meshgrid(ax_pts, ax_pts, ax_pts, indexing="ij")
            fids = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1).unsqueeze(0)

            for H in res_list:
                # Scale detector FOV to cover physical CT FOV cleanly at magnification SDD/SAD=1.2
                det_fov_mm = max(256.0, 1.25 * phys_fov_mm)
                delx = det_fov_mm / float(H)
                drrA = DRR(subject_cpu, sdd=sdd_mm, height=H, delx=delx, x0=0.0, y0=0.0).to(device)
                drrB = DRR(subject_cpu, sdd=sdd_mm, height=H, delx=delx, x0=0.0, y0=0.0).to(device)

                # Orthogonal AP (0 deg) and LAT (90 deg) C-arm poses at SAD=850 mm
                poseA_gt = convert(
                    torch.tensor([[0.0, 0.0, 0.0]], device=device),
                    torch.tensor([[0.0, sad_mm, 0.0]], device=device),
                    parameterization="euler_angles",
                    convention="ZYX",
                )
                poseB_gt = convert(
                    torch.tensor([[math.pi / 2.0, 0.0, 0.0]], device=device),
                    torch.tensor([[0.0, sad_mm, 0.0]], device=device),
                    parameterization="euler_angles",
                    convention="ZYX",
                )
                T_AB = RigidTransform(poseA_gt.matrix.inverse() @ poseB_gt.matrix)

                latA_hi = fvr_hi.cache_diffdrr_geometry(drrA, H)
                latB_hi = fvr_hi.cache_diffdrr_geometry(drrB, H)
                latA_fast = fvr_fast.cache_diffdrr_geometry(drrA, H)
                latB_fast = fvr_fast.cache_diffdrr_geometry(drrB, H)
                if device.type == "cuda":
                    fvr_hi.warmup_pinhole(latA_hi, batch_size=B_cand, backward=True)
                    fvr_fast.warmup_pinhole(latA_fast, batch_size=B_cand, backward=True)

                intr = {"sdd": sdd_mm, "delx": delx, "x0": 0.0, "y0": 0.0, "H": float(H)}
                nano_comp = n_subj_f32 = n_subj_bf16 = iso_n = None
                if has_nano:
                    n_subj_f32, n_subj_bf16, nano_mod, iso_n, ncc_n = _calibrate_nanodrr_to_diffdrr(
                        str(nii_path), drrA, poseA_gt, intr, device, min_ncc=0.95
                    )
                    nano_comp = torch.compile(nano_mod, mode="default") if device.type == "cuda" else nano_mod
                    nsub = n_subj_bf16 if device.type == "cuda" else n_subj_f32
                    if device.type == "cuda":
                        pw = _diff_pose_to_nano_rt(poseA_gt.matrix.expand(B_cand, -1, -1).clone(), iso_n).requires_grad_(True)
                        nano_comp(nsub, pw, n_samples=500, backend="auto").sum().backward()
                        pw1 = _diff_pose_to_nano_rt(poseA_gt.matrix.clone(), iso_n).requires_grad_(True)
                        nano_comp(nsub, pw1, n_samples=500, backend="auto").sum().backward()
                        torch.cuda.synchronize()
                else:
                    ncc_n = 1.0

                n_trials = 1 if smoke else (4 if H == 512 else 6)
                print(f"\n  [{ds_name} | H={H} | vox={vox_mm:.2f}mm | TrueNCC(nano)={ncc_n:.4f} | trials={n_trials}]")

                for t_idx in range(n_trials):
                    rng = np.random.default_rng(7100 + ds_idx * 100 + H + t_idx)
                    dr0 = torch.tensor(rng.uniform(-0.17, 0.17, size=(1, 3)), dtype=torch.float32, device=device)
                    dt0 = torch.tensor(rng.uniform(-14.0, 14.0, size=(1, 3)), dtype=torch.float32, device=device)
                    pose_init = RigidTransform(convert(dr0, dt0, parameterization="euler_angles", convention="ZYX").matrix @ poseA_gt.matrix)
                    init_mtre = _compute_mtre(pose_init, poseA_gt, fids)
                    pA_init_mat = pose_init.matrix
                    pB_init_mat = pA_init_mat @ T_AB.matrix

                    # Independent spatial ray-marched targets degraded by Poisson-Gaussian fluoroscopy noise
                    with torch.no_grad():
                        if has_nano:
                            bk = "torch" if device.type != "cuda" else "auto"
                            imA_clean = nano_mod(n_subj_f32, _diff_pose_to_nano_rt(poseA_gt.matrix, iso_n), n_samples=500, backend=bk)
                            imB_clean = nano_mod(n_subj_f32, _diff_pose_to_nano_rt(poseB_gt.matrix, iso_n), n_samples=500, backend=bk)
                        else:
                            imA_clean = drrA(poseA_gt)
                            imB_clean = drrB(poseB_gt)
                        xrayA = _add_poisson_gaussian_noise(imA_clean, seed=8000 + t_idx)
                        xrayB = _add_poisson_gaussian_noise(imB_clean, seed=9000 + t_idx)

                    seeds = _sample_multistart_theta(
                        B_cand, device=device, seed=8500 + ds_idx * 100 + t_idx, rot_deg=14.0, trans_mm=18.0, keep_zero=False
                    )

                    def _make_fvr_pair(mod, lA, lB):
                        def _fn(dT):
                            return mod._forward_from_lattice(dT @ pA_init_mat, lA), mod._forward_from_lattice(dT @ pB_init_mat, lB)
                        return _fn

                    def _make_nano_pair():
                        bk = "torch" if device.type != "cuda" else "auto"
                        nsub_use = n_subj_bf16 if device.type == "cuda" else n_subj_f32
                        def _fn(dT):
                            return (
                                nano_comp(nsub_use, _diff_pose_to_nano_rt(dT @ pA_init_mat, iso_n), n_samples=500, backend=bk),
                                nano_comp(nsub_use, _diff_pose_to_nano_rt(dT @ pB_init_mat, iso_n), n_samples=500, backend=bk),
                            )
                        return _fn

                    common = {
                        "device": gpu_label,
                        "dataset": ds_name,
                        "H": H,
                        "B": B_cand,
                        "trial": t_idx + 1,
                        "init_mtre_mm": round(init_mtre, 3),
                    }
                    print(f"    Trial {t_idx + 1}/{n_trials} | init_mTRE = {init_mtre:.2f} mm")

                    if ds_name == "Kaggle_Lumbar_Spine_CT" and H <= 256:
                        _eval_and_record(
                            rows, device, common, "DiffDRR Single-Start (f32 B=1)", 0.0,
                            pA_init_mat, poseA_gt, fids,
                            lambda: _optimize_multistart(
                                lambda dT: (drrA(RigidTransform(dT @ pA_init_mat)), drrB(RigidTransform(dT @ pB_init_mat))),
                                seeds[:1], xrayA, xrayB, steps=2 if smoke else 40,
                            ),
                        )

                    _eval_and_record(
                        rows, device, common, f"PoU-FVR Triton (2x4x2@80 bf16, B={B_cand})", prep_hi,
                        pA_init_mat, poseA_gt, fids,
                        lambda: _optimize_multistart(_make_fvr_pair(fvr_hi, latA_hi, latB_hi), seeds, xrayA, xrayB, steps=steps_pou),
                    )

                    if has_nano:
                        if H <= 256:
                            _eval_and_record(
                                rows, device, common, f"nanodrr triton+compile (bf16 n=500, B={B_cand})", 0.0,
                                pA_init_mat, poseA_gt, fids,
                                lambda: _optimize_multistart(_make_nano_pair(), seeds, xrayA, xrayB, steps=steps_nano),
                            )

                        def _run_hyb_nano():
                            th, _ = _optimize_multistart(_make_fvr_pair(fvr_fast, latA_fast, latB_fast), seeds, xrayA, xrayB, steps=hyb_g_steps)
                            return _optimize_multistart(_make_nano_pair(), th, xrayA, xrayB, steps=hyb_p_steps, lr_rot=0.004, lr_trans=0.4)

                        _eval_and_record(
                            rows, device, common, "Hybrid PoU-Triton + nanodrr polish (<0.5GB)", prep_fast,
                            pA_init_mat, poseA_gt, fids, _run_hyb_nano,
                        )
                    else:
                        def _run_hyb_drr():
                            th, _ = _optimize_multistart(_make_fvr_pair(fvr_fast, latA_fast, latB_fast), seeds, xrayA, xrayB, steps=hyb_g_steps)
                            return _optimize_multistart(
                                lambda dT: (drrA(RigidTransform(dT @ pA_init_mat)), drrB(RigidTransform(dT @ pB_init_mat))),
                                th, xrayA, xrayB, steps=hyb_p_steps, lr_rot=0.004, lr_trans=0.4
                            )
                        _eval_and_record(
                            rows, device, common, "Hybrid PoU-Triton + DiffDRR polish", prep_fast,
                            pA_init_mat, poseA_gt, fids, _run_hyb_drr,
                        )

    summary = []
    seen_groups = []
    for r in rows:
        k = (r["dataset"], r["H"], r["method"])
        if k not in seen_groups:
            seen_groups.append(k)
    print("\n=== EXP32C SPINE & CT CONE-BEAM SUMMARY ===")
    for ds_name, H_val, method_name in seen_groups:
        sub = [r for r in rows if r["dataset"] == ds_name and r["H"] == H_val and r["method"] == method_name]
        mtres = [float(r["final_mtre_mm"]) for r in sub]
        mean_m, med_m, q25, q75 = _stats_iqr(mtres)
        s2 = 100.0 * sum(int(r["success_2mm"]) for r in sub) / len(sub)
        reg_m = statistics.mean(float(r["reg_time_s"]) for r in sub)
        vram_m = statistics.mean(float(r["peak_vram_mb"]) for r in sub)
        print(
            f"  [{ds_name:<22} H={H_val}] {method_name:<45} n={len(sub)} | "
            f"med={med_m:5.2f} [{q25:4.2f},{q75:5.2f}] mm | <2mm={s2:5.1f}% | reg={reg_m:5.2f}s | vram={vram_m:6.1f}MB"
        )
        summary.append({
            "device": gpu_label,
            "dataset": ds_name,
            "H": H_val,
            "method": method_name,
            "n_trials": len(sub),
            "mean_mtre_mm": round(mean_m, 3),
            "median_mtre_mm": round(med_m, 3),
            "iqr_q25_mm": round(q25, 3),
            "iqr_q75_mm": round(q75, 3),
            "success_2mm_pct": round(s2, 1),
            "mean_reg_s": round(reg_m, 3),
            "mean_peak_vram_mb": round(vram_m, 1),
        })

    if not smoke:
        for path, data in [
            (out_dir / "exp32_spine_ct_conebeam.csv", rows),
            (out_dir / "exp32_spine_ct_summary.csv", summary),
        ]:
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
            print(f"Saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Fast CPU smoke test covering all code paths")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and hasattr(torch, "_dynamo"):
        torch._dynamo.config.cache_size_limit = 64
    use_tri = device.type == "cuda" and HAS_TRITON
    has_nano = _has_nanodrr()

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    t_all = time.perf_counter()

    run_exp32a_and_b(args.smoke, device, use_tri, has_nano, out_dir)
    run_exp32c_spine_ct(args.smoke, device, use_tri, has_nano, out_dir)

    print(f"\n=== EXP32 COMPLETE in {(time.perf_counter() - t_all):.1f}s ===")


if __name__ == "__main__":
    main()
