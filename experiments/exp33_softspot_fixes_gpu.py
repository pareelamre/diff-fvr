"""
Experiment 33: Soft-Spot Fixes Validation (Isotropic 4x4x4 PoU & Top-K Hybrid Polish)
===================================================================================
Targets the exact remaining reviewer soft spots identified after exp32:

  1. Standalone PoU-FVR on wide/coarse CT cohorts (GE Clinical CT & AMOS 0001):
     - Upgrades slab geometry from anisotropic (2,4,2)@80 to isotropic (4,4,4)@64 bf16,
       quartering per-slab optical depth thickness along both AP and LAT axes.
     - Benchmarks on DeepFluoro (N=24) and Spine/Abdominal CT cohorts (exp33a / exp33c).

  2. <0.5 GB Hybrid PoU-Triton + nanodrr polish & Top-K Multi-Survivor Handoff:
     - Fixes joint View-A/View-B per-axis half-voxel isocenter parity in nanodrr
       calibration ((-0.5, 0.0, +0.5)^3 across even/odd volumes such as subject05).
     - Preserves top_k=4 lowest-loss candidates (instead of collapsing to B=1) into a
       20-step float32 spatial raycast polish (lr_rot=0.008, lr_trans=0.8) for both
       Hybrid+nanodrr (<0.5 GB) and Hybrid+DiffDRR.

Outputs:
  - outputs/exp33_deepfluoro_fixes.csv
  - outputs/exp33_deepfluoro_summary.csv
  - outputs/exp33_spine_ct_fixes.csv
  - outputs/exp33_spine_ct_summary.csv
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
import SimpleITK as sitk
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert

from fvr import PerspectiveSlabFVR
from fvr.dicom_loader import load_dicom_series
from fvr.triton_slice import HAS_TRITON

from experiments.exp32_reviewer_rebuttal_gpu import (
    C_CAM_NANO_TO_DIFF,
    SUBJECTS,
    _add_poisson_gaussian_noise,
    _biplanar_edge_loss,
    _compute_mtre,
    _diff_pose_to_nano_rt,
    _eval_and_record,
    _find_max_optical_pair,
    _has_nanodrr,
    _load_deepfluoro_view,
    _ncc_2d,
    _paired_wilcoxon,
    _resolve_deepfluoro_root,
    _sample_multistart_theta,
    _save_centered_isotropic_nii,
    _se3_exp,
    _stats_iqr,
)


def _optimize_multistart_topk(
    render_pair_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    theta_init: torch.Tensor,
    xrayA: torch.Tensor,
    xrayB: torch.Tensor,
    steps: int,
    top_k: int = 1,
    lr_rot: float = 0.015,
    lr_trans: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched Adam + cosine annealing returning top_k (K, 6) lowest-loss survivors."""
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
        k_keep = min(max(1, top_k), B)
        top_vals, top_idx = torch.topk(final_losses, k=k_keep, largest=False)
    return theta_final[top_idx].detach(), top_vals.detach()


def _calibrate_nanodrr_pair(
    vol_path: str,
    drrA,
    poseA: RigidTransform,
    intrA: Dict[str, float],
    drrB,
    poseB: RigidTransform,
    intrB: Dict[str, float],
    device: torch.device,
    min_ncc: float = 0.985,
):
    """
    Calibrate nanodrr jointly on View A + View B with per-axis {-0.5, 0, +0.5} voxel shifts.
    Fixes single-view / uniform iso_mul ambiguity across odd/even CT dimensions.
    """
    from nanodrr.camera import make_k_inv
    from nanodrr.data import Subject as NanoSubject
    from nanodrr.drr import DRR as NanoDRR

    n_subj_f32 = NanoSubject.from_filepath(vol_path).to(dtype=torch.float32, device=device)
    n_subj_bf16 = (
        NanoSubject.from_filepath(vol_path).to(dtype=torch.bfloat16, device=device)
        if device.type == "cuda"
        else n_subj_f32
    )
    iso = n_subj_f32.isocenter.to(device=device, dtype=torch.float32)
    C_cam = C_CAM_NANO_TO_DIFF.to(device)
    vox_sp = getattr(n_subj_f32, "spacing", torch.ones(3, device=device))
    if not torch.is_tensor(vox_sp):
        vox_sp = torch.tensor(vox_sp, device=device, dtype=torch.float32)
    vox_sp = vox_sp.to(device=device, dtype=torch.float32).flatten()[:3]

    H = int(intrA["H"])
    bk = "torch" if device.type != "cuda" else "auto"

    with torch.no_grad():
        refA = drrA(poseA)
        refB = drrB(poseB)
        pA = poseA.matrix[0]
        pB = poseB.matrix[0]

        best_ncc = -1.0
        best_cfg = None
        for x_sgn in (1.0, -1.0):
            for y_sgn in (1.0, -1.0):
                kA = make_k_inv(
                    sdd=float(intrA["sdd"]),
                    delx=float(intrA["delx"]),
                    dely=float(intrA["delx"]),
                    x0=x_sgn * float(intrA["x0"]),
                    y0=y_sgn * float(intrA["y0"]),
                    height=H,
                    width=H,
                ).to(device)
                kB = make_k_inv(
                    sdd=float(intrB["sdd"]),
                    delx=float(intrB["delx"]),
                    dely=float(intrB["delx"]),
                    x0=x_sgn * float(intrB["x0"]),
                    y0=y_sgn * float(intrB["y0"]),
                    height=H,
                    width=H,
                ).to(device)
                nA = NanoDRR(k_inv=kA, sdd=torch.tensor([float(intrA["sdd"])], device=device), height=H, width=H)
                nB = NanoDRR(k_inv=kB, sdd=torch.tensor([float(intrB["sdd"])], device=device), height=H, width=H)

                # Sweep orthogonal/per-axis half-voxel offsets in batch=27 for speed
                offs = torch.tensor(
                    [[ix, iy, iz] for ix in (-0.5, 0.0, 0.5) for iy in (-0.5, 0.0, 0.5) for iz in (-0.5, 0.0, 0.5)],
                    device=device,
                    dtype=torch.float32,
                ) * vox_sp.unsqueeze(0)
                iso_batch = iso.unsqueeze(0) + offs
                M = iso_batch.shape[0]
                rtA = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(M, 1, 1)
                rtB = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(M, 1, 1)
                rtA[:, :3, :3] = pA[:3, :3] @ C_cam
                rtB[:, :3, :3] = pB[:3, :3] @ C_cam
                rtA[:, :3, 3] = pA[:3, 3].unsqueeze(0) + iso_batch
                rtB[:, :3, 3] = pB[:3, 3].unsqueeze(0) + iso_batch

                outA = nA(n_subj_f32, rtA, n_samples=500, backend=bk)
                outB = nB(n_subj_f32, rtB, n_samples=500, backend=bk)
                for m in range(M):
                    val = 0.5 * (_ncc_2d(outA[m : m + 1], refA) + _ncc_2d(outB[m : m + 1], refB))
                    if val > best_ncc:
                        best_ncc = val
                        best_cfg = (kA, kB, iso_batch[m].clone())

        if best_cfg is None or best_ncc < min_ncc:
            raise RuntimeError(f"Joint nanodrr calibration failed on {vol_path}: best NCC={best_ncc:.4f}")

        kA_b, kB_b, iso_best = best_cfg
        nanoA = NanoDRR(k_inv=kA_b, sdd=torch.tensor([float(intrA["sdd"])], device=device), height=H, width=H)
        nanoB = NanoDRR(k_inv=kB_b, sdd=torch.tensor([float(intrB["sdd"])], device=device), height=H, width=H)
        return n_subj_f32, n_subj_bf16, nanoA, nanoB, iso_best, best_ncc


def run_exp33a_deepfluoro(
    smoke: bool,
    device: torch.device,
    use_tri: bool,
    has_nano: bool,
    out_dir: Path,
) -> None:
    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    H = 96 if smoke else 256
    B_cand = 8 if smoke else 64
    top_k = 2 if smoke else 4
    steps_pou = 2 if smoke else 50
    hyb_g_steps = 2 if smoke else 35
    hyb_p_steps = 2 if smoke else 20

    subjects = SUBJECTS[:1] if smoke else SUBJECTS
    tiers = [
        ("Tier1_Bounded", 0.10, 8.0, 10.0, 12.0, 1 if smoke else 2),
    ]
    if not smoke:
        tiers.append(("Tier2_WideBasin", 0.20, 18.0, 16.0, 22.0, 2))

    root_df = _resolve_deepfluoro_root()
    print(f"\n=== EXP33A: Soft-Spot Fixes DeepFluoro Cohort | H={H}, B={B_cand}, top_k={top_k} | nanodrr={has_nano} ===")

    trial_rows: List[Dict[str, object]] = []
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

        t0 = time.perf_counter()
        fvr_iso = PerspectiveSlabFVR.from_diffdrr_subject(
            subject_cpu,
            grid_blocks=(2, 2, 2) if smoke else (4, 4, 4),
            block_res=48 if smoke else 64,
            pad_mult=2,
            use_triton=use_tri,
        ).to(device)
        if device.type == "cuda":
            fvr_iso.to_precision(torch.bfloat16)
        t_prep_iso = time.perf_counter() - t0

        tf0 = time.perf_counter()
        fvr_fast = PerspectiveSlabFVR.from_diffdrr_subject(
            subject_cpu,
            grid_blocks=(2, 2, 2),
            block_res=48 if smoke else 64,
            pad_mult=2,
            use_triton=use_tri,
        ).to(device)
        if device.type == "cuda":
            fvr_fast.to_precision(torch.bfloat16)
        t_prep_fast = time.perf_counter() - tf0

        latA_iso = fvr_iso.cache_diffdrr_geometry(drrA, H)
        latB_iso = fvr_iso.cache_diffdrr_geometry(drrB, H)
        latA_fast = fvr_fast.cache_diffdrr_geometry(drrA, H)
        latB_fast = fvr_fast.cache_diffdrr_geometry(drrB, H)

        if not warmed_fvr and device.type == "cuda":
            fvr_iso.warmup_pinhole(latA_iso, batch_size=B_cand, backward=True)
            fvr_fast.warmup_pinhole(latA_fast, batch_size=B_cand, backward=True)
            warmed_fvr = True

        nanoA_comp = nanoB_comp = n_subj_f32 = iso_n = None
        nano_prep_s = 0.0
        ncc_nano = 0.0
        if has_nano:
            tn0 = time.perf_counter()
            n_subj_f32, _, nanoA, nanoB, iso_n, ncc_nano = _calibrate_nanodrr_pair(
                vol_path, drrA, poseA, intrA, drrB, poseB, intrB, device, min_ncc=0.975 if smoke else 0.985
            )
            if device.type == "cuda":
                nanoA_comp = torch.compile(nanoA, mode="default")
                nanoB_comp = torch.compile(nanoB, mode="default")
            else:
                nanoA_comp, nanoB_comp = nanoA, nanoB
            nano_prep_s = time.perf_counter() - tn0

            if not warmed_nano and device.type == "cuda":
                # Warm up both B=top_k (Hybrid f32 polish) and B=1
                for bw in (top_k, 1):
                    pwA = _diff_pose_to_nano_rt(poseA.matrix.expand(bw, -1, -1).clone(), iso_n).requires_grad_(True)
                    pwB = _diff_pose_to_nano_rt(poseB.matrix.expand(bw, -1, -1).clone(), iso_n).requires_grad_(True)
                    _biplanar_edge_loss(
                        nanoA_comp(n_subj_f32, pwA, n_samples=500, backend="auto"),
                        nanoB_comp(n_subj_f32, pwB, n_samples=500, backend="auto"),
                        xrayA,
                        xrayB,
                    ).sum().backward()
                torch.cuda.synchronize()
                warmed_nano = True

        if not warmed_drr and device.type == "cuda":
            rw = torch.zeros(top_k, 3, device=device, requires_grad=True)
            dw = convert(rw, torch.zeros(top_k, 3, device=device), parameterization="euler_angles", convention="ZYX")
            _biplanar_edge_loss(
                drrA(RigidTransform(dw.matrix @ poseA.matrix)),
                drrB(RigidTransform(dw.matrix @ poseB.matrix)),
                xrayA,
                xrayB,
            ).sum().backward()
            torch.cuda.synchronize()
            warmed_drr = True

        print(f"\n[{sname} | B_iso={fvr_iso.B} | pair={vA}+{vB} ({opt_sep_deg:.1f} deg) | JointNanoNCC={ncc_nano:.4f}]")

        for tier_idx, (tier_name, i_rot, i_trans, s_rot_deg, s_trans_mm, n_inits) in enumerate(tiers):
            for init_idx in range(n_inits):
                rng = np.random.default_rng(3300 + sidx * 100 + tier_idx * 20 + init_idx)
                dr0 = torch.tensor(rng.uniform(-i_rot, i_rot, size=(1, 3)), dtype=torch.float32, device=device)
                dt0 = torch.tensor(rng.uniform(-i_trans, i_trans, size=(1, 3)), dtype=torch.float32, device=device)
                pose_init = RigidTransform(convert(dr0, dt0, parameterization="euler_angles", convention="ZYX").matrix @ poseA.matrix)
                init_mtre = _compute_mtre(pose_init, poseA, fids)
                pA_init_mat = pose_init.matrix
                pB_init_mat = pA_init_mat @ T_AB.matrix

                keep0 = tier_name == "Tier1_Bounded"
                seeds = _sample_multistart_theta(
                    B_cand,
                    device=device,
                    seed=6300 + sidx * 100 + tier_idx * 20 + init_idx,
                    rot_deg=s_rot_deg,
                    trans_mm=s_trans_mm,
                    keep_zero=keep0,
                )

                def _make_fvr_pair(mod, lA, lB):
                    def _fn(dT):
                        return mod._forward_from_lattice(dT @ pA_init_mat, lA), mod._forward_from_lattice(dT @ pB_init_mat, lB)
                    return _fn

                def _make_drr_pair():
                    def _fn(dT):
                        return drrA(RigidTransform(dT @ pA_init_mat)), drrB(RigidTransform(dT @ pB_init_mat))
                    return _fn

                def _make_nano_f32_pair():
                    bk = "torch" if device.type != "cuda" else "auto"
                    def _fn(dT):
                        rtA = _diff_pose_to_nano_rt(dT @ pA_init_mat, iso_n)
                        rtB = _diff_pose_to_nano_rt(dT @ pB_init_mat, iso_n)
                        return nanoA_comp(n_subj_f32, rtA, n_samples=500, backend=bk), nanoB_comp(n_subj_f32, rtB, n_samples=500, backend=bk)
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
                    "top_k": top_k,
                    "init_mtre_mm": round(init_mtre, 3),
                }
                print(f"  [{tier_name} | Init {init_idx + 1}/{n_inits} | init_mTRE={init_mtre:.2f} mm]")

                _eval_and_record(
                    trial_rows, device, common, f"PoU-FVR Triton (4x4x4@64 bf16, B={B_cand})", t_prep_iso,
                    pA_init_mat, poseA, fids,
                    lambda: _optimize_multistart_topk(_make_fvr_pair(fvr_iso, latA_iso, latB_iso), seeds, xrayA, xrayB, steps=steps_pou, top_k=1),
                )

                if has_nano:
                    def _run_hyb_nano_topk():
                        th_k, _ = _optimize_multistart_topk(
                            _make_fvr_pair(fvr_fast, latA_fast, latB_fast), seeds, xrayA, xrayB, steps=hyb_g_steps, top_k=top_k
                        )
                        return _optimize_multistart_topk(
                            _make_nano_f32_pair(), th_k, xrayA, xrayB, steps=hyb_p_steps, top_k=1, lr_rot=0.008, lr_trans=0.8
                        )

                    _eval_and_record(
                        trial_rows, device, common, "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)", t_prep_fast,
                        pA_init_mat, poseA, fids, _run_hyb_nano_topk,
                    )

                def _run_hyb_drr_topk():
                    th_k, _ = _optimize_multistart_topk(
                        _make_fvr_pair(fvr_fast, latA_fast, latB_fast), seeds, xrayA, xrayB, steps=hyb_g_steps, top_k=top_k
                    )
                    return _optimize_multistart_topk(
                        _make_drr_pair(), th_k, xrayA, xrayB, steps=hyb_p_steps, top_k=1, lr_rot=0.008, lr_trans=0.8
                    )

                _eval_and_record(
                    trial_rows, device, common, "Hybrid PoU-Triton (K=4) + DiffDRR polish", t_prep_fast,
                    pA_init_mat, poseA, fids, _run_hyb_drr_topk,
                )

    summary_rows: List[Dict[str, object]] = []
    subsets = [
        ("All_DeepFluoro", trial_rows),
        ("Tier1_Bounded", [r for r in trial_rows if r["tier"] == "Tier1_Bounded"]),
        ("Tier2_WideBasin", [r for r in trial_rows if r["tier"] == "Tier2_WideBasin"]),
        ("WideAngle_gt15deg", [r for r in trial_rows if float(r["angle_deg"]) >= 15.0]),
    ]
    methods_order = []
    for r in trial_rows:
        if r["method"] not in methods_order:
            methods_order.append(r["method"])

    print("\n=== EXP33A DEEPFLUORO SOFT-SPOT FIX SUMMARY ===")
    for sub_name, sub_list in subsets:
        if not sub_list:
            continue
        for mname in methods_order:
            mrows = [r for r in sub_list if r["method"] == mname]
            if not mrows:
                continue
            mtres = [float(r["final_mtre_mm"]) for r in mrows]
            mean_m, med_m, q25, q75 = _stats_iqr(mtres)
            s2 = 100.0 * sum(int(r["success_2mm"]) for r in mrows) / len(mrows)
            s5 = 100.0 * sum(int(r["success_5mm"]) for r in mrows) / len(mrows)
            prep_m = statistics.mean(float(r["prep_s"]) for r in mrows)
            reg_m = statistics.mean(float(r["reg_time_s"]) for r in mrows)
            vram_m = statistics.mean(float(r["peak_vram_mb"]) for r in mrows)
            print(
                f"  [{sub_name:<18}] {mname:<52} n={len(mrows):2d} | med={med_m:5.3f} [{q25:5.3f},{q75:5.3f}] | "
                f"<2mm={s2:5.1f}% | <5mm={s5:5.1f}% | prep={prep_m:4.2f}s reg={reg_m:5.2f}s | vram={vram_m:6.1f}MB"
            )
            summary_rows.append({
                "device": gpu_label,
                "subset": sub_name,
                "method": mname,
                "n_cases": len(mrows),
                "mean_mtre_mm": round(mean_m, 3),
                "median_mtre_mm": round(med_m, 3),
                "iqr_q25_mm": round(q25, 3),
                "iqr_q75_mm": round(q75, 3),
                "success_2mm_pct": round(s2, 1),
                "success_5mm_pct": round(s5, 1),
                "mean_prep_s": round(prep_m, 3),
                "mean_reg_s": round(reg_m, 3),
                "mean_total_s": round(prep_m + reg_m, 3),
                "mean_vram_mb": round(vram_m, 1),
            })

    for path, rows in [
        (out_dir / "exp33_deepfluoro_fixes.csv", trial_rows),
        (out_dir / "exp33_deepfluoro_summary.csv", summary_rows),
    ]:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {path}")


def run_exp33c_spine_ct(
    smoke: bool,
    device: torch.device,
    use_tri: bool,
    has_nano: bool,
    out_dir: Path,
) -> None:
    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n=== EXP33C: Soft-Spot Fixes Clinical Spine & Abdominal CT Cone-Beam | nanodrr={has_nano} ===")

    ct_specs: List[Tuple[str, Callable[[], Tuple[np.ndarray, Tuple[float, float, float]]], List[int]]] = [
        ("Kaggle_Lumbar_Spine_CT", lambda: load_dicom_series(ROOT / "data" / "kaggle_lumbar" / "DICOM"), [96] if smoke else [256, 512]),
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
        ct_specs.append(("GE_Clinical_CT", lambda: load_dicom_series(ROOT / "data" / "dicom"), [256]))

    B_cand = 8 if smoke else 64
    top_k = 2 if smoke else 4
    sdd_mm, sad_mm = 1020.0, 850.0
    steps_pou = 2 if smoke else 50
    hyb_g_steps = 2 if smoke else 35
    hyb_p_steps = 2 if smoke else 20

    rows: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for ds_idx, (ds_name, loader_fn, res_list) in enumerate(ct_specs):
            vol_hu, spacing_zyx = loader_fn()
            nii_path = tmp_path / f"{ds_name}.nii.gz"
            iso_shape, vox_mm = _save_centered_isotropic_nii(vol_hu, spacing_zyx, target_res=96 if smoke else 160, out_path=nii_path)
            subject_cpu = read(str(nii_path), orientation="AP")

            t0 = time.perf_counter()
            fvr_iso = PerspectiveSlabFVR.from_diffdrr_subject(
                subject_cpu,
                grid_blocks=(2, 2, 2) if smoke else (4, 4, 4),
                block_res=48 if smoke else 64,
                pad_mult=2,
                use_triton=use_tri,
            ).to(device)
            if device.type == "cuda":
                fvr_iso.to_precision(torch.bfloat16)
            prep_iso = time.perf_counter() - t0

            tf0 = time.perf_counter()
            fvr_fast = PerspectiveSlabFVR.from_diffdrr_subject(
                subject_cpu,
                grid_blocks=(2, 2, 2),
                block_res=48 if smoke else 64,
                pad_mult=2,
                use_triton=use_tri,
            ).to(device)
            if device.type == "cuda":
                fvr_fast.to_precision(torch.bfloat16)
            prep_fast = time.perf_counter() - tf0

            phys_fov_mm = iso_shape[0] * vox_mm
            half_ext = 0.25 * phys_fov_mm
            ax_pts = torch.linspace(-half_ext, half_ext, 3, device=device)
            gx, gy, gz = torch.meshgrid(ax_pts, ax_pts, ax_pts, indexing="ij")
            fids = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1).unsqueeze(0)

            for H in res_list:
                det_fov_mm = max(256.0, 1.25 * phys_fov_mm)
                delx = det_fov_mm / float(H)
                drrA = DRR(subject_cpu, sdd=sdd_mm, height=H, delx=delx, x0=0.0, y0=0.0).to(device)
                drrB = DRR(subject_cpu, sdd=sdd_mm, height=H, delx=delx, x0=0.0, y0=0.0).to(device)
                poseA_gt = convert(torch.tensor([[0.0, 0.0, 0.0]], device=device), torch.tensor([[0.0, sad_mm, 0.0]], device=device), parameterization="euler_angles", convention="ZYX")
                poseB_gt = convert(torch.tensor([[math.pi / 2.0, 0.0, 0.0]], device=device), torch.tensor([[0.0, sad_mm, 0.0]], device=device), parameterization="euler_angles", convention="ZYX")
                T_AB = RigidTransform(poseA_gt.matrix.inverse() @ poseB_gt.matrix)

                latA_iso = fvr_iso.cache_diffdrr_geometry(drrA, H)
                latB_iso = fvr_iso.cache_diffdrr_geometry(drrB, H)
                latA_fast = fvr_fast.cache_diffdrr_geometry(drrA, H)
                latB_fast = fvr_fast.cache_diffdrr_geometry(drrB, H)
                if device.type == "cuda":
                    fvr_iso.warmup_pinhole(latA_iso, batch_size=B_cand, backward=True)
                    fvr_fast.warmup_pinhole(latA_fast, batch_size=B_cand, backward=True)

                intr = {"sdd": sdd_mm, "delx": delx, "x0": 0.0, "y0": 0.0, "H": float(H)}
                nanoA_comp = nanoB_comp = n_subj_f32 = iso_n = None
                if has_nano:
                    n_subj_f32, _, nanoA, nanoB, iso_n, ncc_n = _calibrate_nanodrr_pair(
                        str(nii_path), drrA, poseA_gt, intr, drrB, poseB_gt, intr, device, min_ncc=0.95
                    )
                    if device.type == "cuda":
                        nanoA_comp = torch.compile(nanoA, mode="default")
                        nanoB_comp = torch.compile(nanoB, mode="default")
                        with torch.no_grad():
                            refAw, refBw = drrA(poseA_gt).detach(), drrB(poseB_gt).detach()
                        for bw in (top_k, 1):
                            pwA = _diff_pose_to_nano_rt(poseA_gt.matrix.expand(bw, -1, -1).clone(), iso_n).requires_grad_(True)
                            pwB = _diff_pose_to_nano_rt(poseB_gt.matrix.expand(bw, -1, -1).clone(), iso_n).requires_grad_(True)
                            _biplanar_edge_loss(
                                nanoA_comp(n_subj_f32, pwA, n_samples=500, backend="auto"),
                                nanoB_comp(n_subj_f32, pwB, n_samples=500, backend="auto"),
                                refAw,
                                refBw,
                            ).sum().backward()
                        torch.cuda.synchronize()
                    else:
                        nanoA_comp, nanoB_comp = nanoA, nanoB
                else:
                    ncc_n = 1.0

                n_trials = 1 if smoke else (4 if H == 512 else 6)
                print(f"\n  [{ds_name} | H={H} | B_iso={fvr_iso.B} | vox={vox_mm:.2f}mm | JointNanoNCC={ncc_n:.4f}]")

                for t_idx in range(n_trials):
                    rng = np.random.default_rng(7100 + ds_idx * 100 + H + t_idx)
                    dr0 = torch.tensor(rng.uniform(-0.17, 0.17, size=(1, 3)), dtype=torch.float32, device=device)
                    dt0 = torch.tensor(rng.uniform(-14.0, 14.0, size=(1, 3)), dtype=torch.float32, device=device)
                    pose_init = RigidTransform(convert(dr0, dt0, parameterization="euler_angles", convention="ZYX").matrix @ poseA_gt.matrix)
                    init_mtre = _compute_mtre(pose_init, poseA_gt, fids)
                    pA_init_mat = pose_init.matrix
                    pB_init_mat = pA_init_mat @ T_AB.matrix

                    with torch.no_grad():
                        if has_nano:
                            bk = "torch" if device.type != "cuda" else "auto"
                            imA_c = nanoA_comp(n_subj_f32, _diff_pose_to_nano_rt(poseA_gt.matrix, iso_n), n_samples=500, backend=bk)
                            imB_c = nanoB_comp(n_subj_f32, _diff_pose_to_nano_rt(poseB_gt.matrix, iso_n), n_samples=500, backend=bk)
                        else:
                            imA_c, imB_c = drrA(poseA_gt), drrB(poseB_gt)
                        xrayA = _add_poisson_gaussian_noise(imA_c, seed=8000 + t_idx)
                        xrayB = _add_poisson_gaussian_noise(imB_c, seed=9000 + t_idx)

                    seeds = _sample_multistart_theta(
                        B_cand, device=device, seed=8500 + ds_idx * 100 + t_idx, rot_deg=14.0, trans_mm=18.0, keep_zero=False
                    )

                    def _make_fvr_pair(mod, lA, lB):
                        def _fn(dT):
                            return mod._forward_from_lattice(dT @ pA_init_mat, lA), mod._forward_from_lattice(dT @ pB_init_mat, lB)
                        return _fn

                    common = {
                        "device": gpu_label,
                        "dataset": ds_name,
                        "H": H,
                        "B": B_cand,
                        "top_k": top_k,
                        "trial": t_idx + 1,
                        "init_mtre_mm": round(init_mtre, 3),
                    }
                    print(f"    Trial {t_idx + 1}/{n_trials} | init_mTRE = {init_mtre:.2f} mm")

                    _eval_and_record(
                        rows, device, common, f"PoU-FVR Triton (4x4x4@64 bf16, B={B_cand})", prep_iso,
                        pA_init_mat, poseA_gt, fids,
                        lambda: _optimize_multistart_topk(_make_fvr_pair(fvr_iso, latA_iso, latB_iso), seeds, xrayA, xrayB, steps=steps_pou, top_k=1),
                    )

                    if has_nano:
                        def _make_nano_f32_pair():
                            bk = "torch" if device.type != "cuda" else "auto"
                            def _fn(dT):
                                return (
                                    nanoA_comp(n_subj_f32, _diff_pose_to_nano_rt(dT @ pA_init_mat, iso_n), n_samples=500, backend=bk),
                                    nanoB_comp(n_subj_f32, _diff_pose_to_nano_rt(dT @ pB_init_mat, iso_n), n_samples=500, backend=bk),
                                )
                            return _fn

                        def _run_hyb_nano_topk():
                            th_k, _ = _optimize_multistart_topk(_make_fvr_pair(fvr_fast, latA_fast, latB_fast), seeds, xrayA, xrayB, steps=hyb_g_steps, top_k=top_k)
                            return _optimize_multistart_topk(_make_nano_f32_pair(), th_k, xrayA, xrayB, steps=hyb_p_steps, top_k=1, lr_rot=0.008, lr_trans=0.8)

                        _eval_and_record(
                            rows, device, common, "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)", prep_fast,
                            pA_init_mat, poseA_gt, fids, _run_hyb_nano_topk,
                        )
                    else:
                        def _run_hyb_drr_topk():
                            th_k, _ = _optimize_multistart_topk(_make_fvr_pair(fvr_fast, latA_fast, latB_fast), seeds, xrayA, xrayB, steps=hyb_g_steps, top_k=top_k)
                            return _optimize_multistart_topk(
                                lambda dT: (drrA(RigidTransform(dT @ pA_init_mat)), drrB(RigidTransform(dT @ pB_init_mat))),
                                th_k, xrayA, xrayB, steps=hyb_p_steps, top_k=1, lr_rot=0.008, lr_trans=0.8,
                            )

                        _eval_and_record(
                            rows, device, common, "Hybrid PoU-Triton (K=4) + DiffDRR polish", prep_fast,
                            pA_init_mat, poseA_gt, fids, _run_hyb_drr_topk,
                        )

    summary = []
    seen_groups = []
    for r in rows:
        k = (r["dataset"], r["H"], r["method"])
        if k not in seen_groups:
            seen_groups.append(k)
    print("\n=== EXP33C SPINE & CT SOFT-SPOT FIX SUMMARY ===")
    for ds_name, H_val, method_name in seen_groups:
        sub = [r for r in rows if r["dataset"] == ds_name and r["H"] == H_val and r["method"] == method_name]
        mtres = [float(r["final_mtre_mm"]) for r in sub]
        mean_m, med_m, q25, q75 = _stats_iqr(mtres)
        s2 = 100.0 * sum(int(r["success_2mm"]) for r in sub) / len(sub)
        prep_m = statistics.mean(float(r["prep_s"]) for r in sub)
        reg_m = statistics.mean(float(r["reg_time_s"]) for r in sub)
        vram_m = statistics.mean(float(r["peak_vram_mb"]) for r in sub)
        print(
            f"  [{ds_name:<22} H={H_val}] {method_name:<52} n={len(sub)} | "
            f"med={med_m:5.3f} [{q25:5.3f},{q75:5.3f}] mm | <2mm={s2:5.1f}% | reg={reg_m:5.2f}s | vram={vram_m:6.1f}MB"
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
            "mean_prep_s": round(prep_m, 3),
            "mean_reg_s": round(reg_m, 3),
            "mean_vram_mb": round(vram_m, 1),
        })

    for path, out_rows in [
        (out_dir / "exp33_spine_ct_fixes.csv", rows),
        (out_dir / "exp33_spine_ct_summary.csv", summary),
    ]:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 33: Soft-Spot Fixes GPU Validation")
    parser.add_argument("--smoke", action="store_true", help="Fast CPU/GPU sanity smoke run.")
    parser.add_argument("--no-nanodrr", action="store_true", help="Skip nanodrr comparison even if installed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    args = parser.parse_args()

    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda")
    use_tri = bool(device.type == "cuda" and HAS_TRITON)
    has_nano = (not args.no_nanodrr) and _has_nanodrr()
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    run_exp33a_deepfluoro(args.smoke, device, use_tri, has_nano, out_dir)
    run_exp33c_spine_ct(args.smoke, device, use_tri, has_nano, out_dir)
    print(f"\nEXP33 complete in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
