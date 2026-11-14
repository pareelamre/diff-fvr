"""
Experiment 31: Multi-Subject Real Clinical Fluoroscopy GPU Registration Cohort
=============================================================================
Evaluates batched 6-DoF biplanar registration on ALL 6 real clinical cadaveric
pelvic subjects of DeepFluoro (eigenvivek/xvr-data: subject01-subject06) using
paired CT volumes, physical C-arm fluoroscopy DICOMs, calibrated DICOM intrinsics
(sdd, delx, x0, y0), and 3D anatomical fiducials.pt.

Across 3 randomized initializations per subject (18 real-fluoroscopy cases total,
spanning ~7-20 mm initial 3D anatomical mTRE), compares:
  1. DiffDRR Single-Start (Siddon f32, B=1, 40 steps, Edge-NCC)
  2. PoU-FVR Triton Multi-Start (2x4x2@80 bf16, B=64, 50 steps, Edge-NCC)
  3. Hybrid PoU-Triton (2x2x2@64 bf16, 35 steps @ B=64) + DiffDRR polish (12 steps @ B=1)

Outputs:
  - outputs/exp31_deepfluoro_gpu_cohort.csv
  - outputs/exp31_deepfluoro_gpu_summary.csv
"""

import argparse
import csv
import gc
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert

from fvr import PerspectiveSlabFVR
from fvr.cone_beam_fvr import edge_dominant_ncc
from fvr.triton_slice import HAS_TRITON

SUBJECTS = [f"subject{i:02d}" for i in range(1, 7)]


def _resolve_deepfluoro_root() -> Path:
    """Ensure DeepFluoro subject01-subject06 (CT + real DICOM xrays + fiducials.pt) are available."""
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


def _load_view(subject, subj_dir: str, vid: str, H: int, device: torch.device):
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
    drr = DRR(
        subject,
        sdd=intr["sdd"],
        height=H,
        delx=intr["delx"] * (intr["height"] / H),
        x0=intr["x0"],
        y0=intr["y0"],
    ).to(device)
    return pose, xray, drr


def _find_max_optical_pair(subject, subj_dir: str, H: int = 64) -> Tuple[str, str, float]:
    pts = sorted(Path(subj_dir, "xrays").glob("*.pt"))
    vids = [p.stem for p in pts]
    cpu = torch.device("cpu")
    axes = []
    for vid in vids:
        pose, _, drr = _load_view(subject, subj_dir, vid, H=H, device=cpu)
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


def _compute_mtre(pose_est: RigidTransform, pose_gt: RigidTransform, fids: torch.Tensor) -> float:
    T_err = RigidTransform(pose_est.matrix @ pose_gt.matrix.inverse())
    return float((T_err(fids) - fids).norm(dim=-1).mean().item())


def _ncc_2d(a: torch.Tensor, b: torch.Tensor) -> float:
    a0 = a.float().flatten()
    b0 = b.float().flatten()
    a0 = a0 - a0.mean()
    b0 = b0 - b0.mean()
    return float((a0 * b0).sum() / (a0.norm() * b0.norm() + 1e-8))


def _biplanar_edge_loss(imA: torch.Tensor, imB: torch.Tensor, refA: torch.Tensor, refB: torch.Tensor) -> torch.Tensor:
    lA = edge_dominant_ncc(imA.float(), refA.float(), reduction="none")
    lB = edge_dominant_ncc(imB.float(), refB.float(), reduction="none")
    return 1.0 - 0.5 * (lA + lB)


def _sample_multistart_theta(B: int, device: torch.device, seed: int, rot_deg: float = 8.0, trans_mm: float = 12.0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    theta = np.zeros((B, 6), dtype=np.float32)
    if B > 1:
        rot_lim = rot_deg * (math.pi / 180.0)
        theta[1:, :3] = rng.uniform(-rot_lim, rot_lim, size=(B - 1, 3)).astype(np.float32)
        theta[1:, 3:] = rng.uniform(-trans_mm, trans_mm, size=(B - 1, 3)).astype(np.float32)
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


def run_cohort(smoke: bool = False, H: int = 256, n_inits: int = 3) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    use_tri = device.type == "cuda" and HAS_TRITON

    if smoke:
        H = 128
        subjects = ["subject01"]
        n_inits = 1
        B_cand = 8
        steps_drr = 3
        steps_pou = 3
        hyb_global_steps = 2
        hyb_polish_steps = 1
    else:
        subjects = SUBJECTS
        B_cand = 64
        steps_drr = 40
        steps_pou = 50
        hyb_global_steps = 35
        hyb_polish_steps = 12

    root_df = _resolve_deepfluoro_root()
    print(
        f"=== Experiment 31: DeepFluoro Real Clinical Cohort (H={H}, B={B_cand}, "
        f"subjects={len(subjects)}, inits={n_inits}) | Device={gpu_label} | Triton={use_tri} ==="
    )

    trial_rows: List[Dict[str, object]] = []
    warmed_fvr = False
    warmed_drr = False

    for sidx, sname in enumerate(subjects, start=1):
        subj_dir = str(root_df / sname)
        fids = torch.load(os.path.join(subj_dir, "fiducials.pt"), weights_only=False).to(device=device, dtype=torch.float32)
        subject_cpu = read(os.path.join(subj_dir, "volume.nii.gz"), orientation="AP")
        vA, vB, opt_sep_deg = _find_max_optical_pair(subject_cpu, subj_dir)
        poseA, xrayA, drrA = _load_view(subject_cpu, subj_dir, vA, H=H, device=device)
        poseB, xrayB, drrB = _load_view(subject_cpu, subj_dir, vB, H=H, device=device)
        T_AB = RigidTransform(poseA.matrix.inverse() @ poseB.matrix)

        t_prep0 = time.perf_counter()
        fvr_hi = PerspectiveSlabFVR.from_diffdrr_subject(
            subject_cpu,
            grid_blocks=(2, 2, 2) if smoke else (2, 4, 2),
            block_res=48 if smoke else 80,
            pad_mult=2,
            use_triton=use_tri,
        ).to(device)
        fvr_fast = PerspectiveSlabFVR.from_diffdrr_subject(
            subject_cpu,
            grid_blocks=(2, 2, 2),
            block_res=48 if smoke else 64,
            pad_mult=2,
            use_triton=use_tri,
        ).to(device)
        if device.type == "cuda":
            fvr_hi.to_precision(torch.bfloat16)
            fvr_fast.to_precision(torch.bfloat16)
        prep_s = time.perf_counter() - t_prep0

        latA_hi = fvr_hi.cache_diffdrr_geometry(drrA, H)
        latB_hi = fvr_hi.cache_diffdrr_geometry(drrB, H)
        latA_fast = fvr_fast.cache_diffdrr_geometry(drrA, H)
        latB_fast = fvr_fast.cache_diffdrr_geometry(drrB, H)

        if not warmed_fvr and device.type == "cuda":
            # Full-batch warmup eliminates first-trial Triton/cuFFT cold-start overhead across all subjects.
            fvr_hi.warmup_pinhole(latA_hi, batch_size=B_cand, backward=True)
            fvr_fast.warmup_pinhole(latA_fast, batch_size=B_cand, backward=True)
            warmed_fvr = True

        with torch.no_grad():
            drr_gt_a = drrA(poseA)
            pou_gt_a = fvr_hi.forward_diffdrr(drrA, poseA, H)
            true_ncc = _ncc_2d(pou_gt_a, drr_gt_a)

        if not warmed_drr and device.type == "cuda":
            p_w = RigidTransform(poseA.matrix.clone())
            r_w = torch.zeros(1, 3, device=device, requires_grad=True)
            d_w = convert(r_w, torch.zeros(1, 3, device=device), parameterization="euler_angles", convention="ZYX")
            _biplanar_edge_loss(drrA(RigidTransform(d_w.matrix @ p_w.matrix)), drrB(RigidTransform(d_w.matrix @ poseB.matrix)), xrayA, xrayB).sum().backward()
            torch.cuda.synchronize()
            warmed_drr = True

        print(f"\n=== {sname} ({vA}+{vB}, sep={opt_sep_deg:.1f}deg, TrueNCC={true_ncc:.4f}, prep={prep_s:.2f}s) ===")

        for init_idx in range(n_inits):
            rng = np.random.default_rng(3100 + sidx * 10 + init_idx)
            if init_idx == 0:
                rot_scale, trans_scale = 0.055, 7.0  # bounded ~7-10 mm init
            else:
                rot_scale, trans_scale = 0.11, 12.0  # harder out-of-basin ~12-20 mm init
            dr0 = torch.tensor(rng.uniform(-rot_scale, rot_scale, size=(1, 3)), dtype=torch.float32, device=device)
            dt0 = torch.tensor(rng.uniform(-trans_scale, trans_scale, size=(1, 3)), dtype=torch.float32, device=device)
            pose_init = RigidTransform(convert(dr0, dt0, parameterization="euler_angles", convention="ZYX").matrix @ poseA.matrix)
            init_mtre = _compute_mtre(pose_init, poseA, fids)

            pA_init_mat = pose_init.matrix
            pB_init_mat = pA_init_mat @ T_AB.matrix

            def _make_drr_pair(base_a: torch.Tensor, base_b: torch.Tensor):
                def _fn(dT: torch.Tensor):
                    pA = RigidTransform(dT @ base_a)
                    pB = RigidTransform(dT @ base_b)
                    return drrA(pA), drrB(pB)

                return _fn

            def _make_fvr_pair(fvr_mod: PerspectiveSlabFVR, lA, lB, base_a: torch.Tensor, base_b: torch.Tensor):
                def _fn(dT: torch.Tensor):
                    return fvr_mod._forward_from_lattice(dT @ base_a, lA), fvr_mod._forward_from_lattice(dT @ base_b, lB)

                return _fn

            seeds = _sample_multistart_theta(B_cand, device=device, seed=4100 + sidx * 10 + init_idx)
            print(f"  [Init {init_idx + 1}/{n_inits} | init_mTRE = {init_mtre:.2f} mm]")

            def _eval_and_record(method_name: str, run_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]]):
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
                elapsed = time.perf_counter() - t0
                pose_final = RigidTransform(_se3_exp(theta_best) @ pA_init_mat)
                final_mtre = _compute_mtre(pose_final, poseA, fids)
                s2 = int(final_mtre < 2.0)
                s5 = int(final_mtre < 5.0)
                print(
                    f"    {method_name:<38} time={elapsed:6.2f}s | vram={peak_mb:6.1f} MB | "
                    f"mTRE={final_mtre:5.2f} mm | <2mm={bool(s2)}"
                )
                trial_rows.append({
                    "device": gpu_label,
                    "subject": sname,
                    "pair": f"{vA}+{vB}",
                    "angle_deg": round(opt_sep_deg, 1),
                    "true_ncc": round(true_ncc, 4),
                    "init_idx": init_idx + 1,
                    "H": H,
                    "B": B_cand,
                    "method": method_name,
                    "init_mtre_mm": round(init_mtre, 3),
                    "final_mtre_mm": round(final_mtre, 3),
                    "success_2mm": s2,
                    "success_5mm": s5,
                    "time_s": round(elapsed, 4),
                    "peak_vram_mb": round(peak_mb, 2),
                    "final_edge_loss": round(float(best_loss), 5),
                })
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            _eval_and_record(
                "DiffDRR Single-Start (f32 B=1)",
                lambda: _optimize_multistart(
                    _make_drr_pair(pA_init_mat, pB_init_mat),
                    seeds[:1],
                    xrayA,
                    xrayB,
                    steps=steps_drr,
                ),
            )
            _eval_and_record(
                "PoU-FVR Triton Multi-Start (B=64)" if device.type == "cuda" else "PoU-FVR Pinhole Multi-Start",
                lambda: _optimize_multistart(
                    _make_fvr_pair(fvr_hi, latA_hi, latB_hi, pA_init_mat, pB_init_mat),
                    seeds,
                    xrayA,
                    xrayB,
                    steps=steps_pou,
                ),
            )

            def _run_hybrid():
                theta_top, _ = _optimize_multistart(
                    _make_fvr_pair(fvr_fast, latA_fast, latB_fast, pA_init_mat, pB_init_mat),
                    seeds,
                    xrayA,
                    xrayB,
                    steps=hyb_global_steps,
                )
                return _optimize_multistart(
                    _make_drr_pair(pA_init_mat, pB_init_mat),
                    theta_top,
                    xrayA,
                    xrayB,
                    steps=hyb_polish_steps,
                    lr_rot=0.004,
                    lr_trans=0.4,
                )

            _eval_and_record("Hybrid PoU-Triton + DiffDRR polish", _run_hybrid)

    if not smoke:
        out_dir = ROOT / "outputs"
        out_dir.mkdir(exist_ok=True)
        raw_csv = out_dir / "exp31_deepfluoro_gpu_cohort.csv"
        with raw_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
            w.writeheader()
            w.writerows(trial_rows)

    summary_rows: List[Dict[str, object]] = []
    seen_methods: List[str] = []
    for r in trial_rows:
        if r["method"] not in seen_methods:
            seen_methods.append(str(r["method"]))

    print("\n=== Experiment 31 Real DeepFluoro Cohort Summary ===")
    for subset_label, pred in [("All 6 Subjects", lambda r: True), ("Wide-Angle (>15deg)", lambda r: float(r["angle_deg"]) > 15.0)]:
        for m in seen_methods:
            sub = [r for r in trial_rows if r["method"] == m and pred(r)]
            if not sub:
                continue
            mtres = [float(r["final_mtre_mm"]) for r in sub]
            s2 = 100.0 * sum(int(r["success_2mm"]) for r in sub) / len(sub)
            s5 = 100.0 * sum(int(r["success_5mm"]) for r in sub) / len(sub)
            tm = statistics.mean(float(r["time_s"]) for r in sub)
            vr = statistics.mean(float(r["peak_vram_mb"]) for r in sub)
            init_m = statistics.mean(float(r["init_mtre_mm"]) for r in sub)
            print(
                f"  [{subset_label:<19}] {m:<36} n={len(sub):2d} | mTRE={statistics.mean(mtres):5.2f} mm "
                f"(med={statistics.median(mtres):5.2f}) | <2mm={s2:5.1f}% | <5mm={s5:5.1f}% | "
                f"time={tm:5.2f}s | VRAM={vr:6.1f} MB"
            )
            summary_rows.append({
                "device": gpu_label,
                "subset": subset_label,
                "method": m,
                "n_cases": len(sub),
                "mean_init_mtre_mm": round(init_m, 3),
                "mean_mtre_mm": round(statistics.mean(mtres), 3),
                "median_mtre_mm": round(statistics.median(mtres), 3),
                "success_2mm_pct": round(s2, 1),
                "success_5mm_pct": round(s5, 1),
                "mean_time_s": round(tm, 3),
                "mean_peak_vram_mb": round(vr, 1),
            })

    if smoke:
        print("\n[Smoke test] Skipped outputs/exp31_*.csv write.")
    else:
        sum_csv = out_dir / "exp31_deepfluoro_gpu_summary.csv"
        with sum_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSaved {raw_csv} and {sum_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Fast CPU smoke test on subject01")
    ap.add_argument("--res", type=int, default=256, help="Detector resolution H (default: 256)")
    ap.add_argument("--inits", type=int, default=3, help="Random initializations per subject (default: 3)")
    args = ap.parse_args()
    run_cohort(smoke=args.smoke, H=args.res, n_inits=args.inits)
