"""
Experiment 27: Geometry-Matched Speedup & Clinical Cone-Beam Robustness Benchmark.

Resolves three critical methodological gaps:
1. Part A (Geometry-Matched Speed & Gradient Latency, DeepFluoro subject01 at 128x128 & 256x256):
   - Matched Parallel-vs-Parallel (SDD = 1e6 mm orthographic):
     Parallel Diff-FVR vs. DiffDRR Siddon (SDD=1e6) vs. DiffDRR Trilinear (SDD=1e6).
   - Matched Cone-Beam-vs-Cone-Beam (SDD = 1020 mm clinical C-arm):
     Cone-Beam PoU-FVR (2x2x2) vs. DiffDRR Siddon (SDD=1020) vs. DiffDRR Trilinear (SDD=1020).
   Reports 3D FFT prep time, single-view forward ms, forward+backward ms, and amortized
   total optimization time across 10, 40, and 100 iterations.

2. Part B (Cone-Beam & Clinical Robustness):
   - Cohort 1 (Compact-to-Standard C-Arm Mag Stress on GE Clinical DICOM CT):
     Standalone PerspectiveSlabFVR (2x2x2) vs. calibrated Orthographic Diff-FVR across
     SAD/SDD = 600/900 (1.50x), 750/1050 (1.40x), 850/1200 (1.41x), 1000/1400 (1.40x).
   - Cohort 2 (Real Clinical Fluoroscopy DeepFluoro subject01-06, SDD=1020 mm):
     Standalone PerspectiveSlabFVR (2x2x2) vs. DiffDRR-Only vs. PoU-FVR -> DiffDRR Hybrid
     with unit-decoupled gradient clipping and exact optical-ray separation reporting.
"""

import csv
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d

from fvr import RobustDiffFVREngine, PerspectiveSlabFVR
from fvr.cone_beam_fvr import edge_dominant_ncc
from fvr.dicom_loader import load_dicom_series, resample_isotropic_cube, hu_to_attenuation
from fvr.registration_3d2d import compute_mtre_6dof
from experiments.exp9_perspective_conebeam_stress import project_conebeam_3d, rotate_volume_3d
from experiments.exp9b_perspective_isocenter_calibrated import calibrate_to_isocenter
from experiments.exp26_deepfluoro_multisubject import (
    SUBJECTS,
    DeepFluoroDiffFVR,
    grad_ncc_2d,
    load_view,
    compute_mtre,
)


def _norm01(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def _ncc_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p0 = p - p.mean()
    q0 = q - q.mean()
    return 1.0 - (p0 * q0).sum() / (p0.norm() * q0.norm() + 1e-8)


def _find_max_optical_pair(subject, subj_dir: str, H: int = 128):
    """Find available view pair maximizing true optical-axis angular baseline (src -> det center)."""
    pts = sorted(Path(subj_dir, "xrays").glob("*.pt"))
    vids = [p.stem for p in pts]
    axes = []
    for vid in vids:
        pose, _, drr = load_view(subject, subj_dir, vid, H=H)
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


def _benchmark_forward_backward(fn_render, params, repeats=5, warmup=1):
    for _ in range(warmup):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        out = fn_render()
        out.sum().backward()

    fwd_times, fb_times = [], []
    for _ in range(repeats):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        t0 = time.perf_counter()
        out = fn_render()
        t1 = time.perf_counter()
        out.sum().backward()
        t2 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000.0)
        fb_times.append((t2 - t0) * 1000.0)
    return float(np.median(fwd_times)), float(np.median(fb_times))


def run_part_a_speed_benchmark(out_csv: Path):
    print("=== PART A: Geometry-Matched Speed & Amortized Cost Benchmark ===")
    subj_dir = str(ROOT / "data" / "deepfluoro" / "deepfluoro" / "subject01")
    subject = read(os.path.join(subj_dir, "volume.nii.gz"), orientation="AP")
    pose_t, _, drr_ref = load_view(subject, subj_dir, "003", H=128)
    pt = torch.load(os.path.join(subj_dir, "xrays", "003.pt"), weights_only=False)
    intr = pt["intrinsics"]
    fov_mm = float(intr["delx"]) * int(intr["height"])
    sdd_cone = float(intr["sdd"])

    t0 = time.perf_counter()
    fvr_par = PerspectiveSlabFVR.from_diffdrr_subject(
        subject, grid_blocks=(1, 1, 1), block_res=128, pad_mult=2
    )
    prep_par_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    fvr_cone = PerspectiveSlabFVR.from_diffdrr_subject(
        subject, grid_blocks=(2, 2, 2), block_res=80, pad_mult=2
    )
    prep_cone_s = time.perf_counter() - t0

    print(
        f"FFT Prep Times: Parallel(1x1x1,R=128)={prep_par_s:.2f}s | Cone-Beam(2x2x2,R=80)={prep_cone_s:.2f}s"
    )

    rows = []
    # Centered orthographic pose straddling CT isocenter (src=+5e5, det=-5e5 along AP Y-axis)
    pose_par = convert(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 5.0e5, 0.0]]),
        parameterization="euler_angles",
        convention="ZYX",
    )

    for res in (128, 256):
        delx_cone = fov_mm / float(res)
        delx_par = 200.0 / float(res)  # matched patient-plane FOV at isocenter
        drr_par_sid = DRR(subject, sdd=1.0e6, height=res, delx=delx_par, renderer="siddon")
        drr_par_tri = DRR(subject, sdd=1.0e6, height=res, delx=delx_par, renderer="trilinear")
        drr_con_sid = DRR(subject, sdd=sdd_cone, height=res, delx=delx_cone, x0=intr["x0"], y0=intr["y0"], renderer="siddon")
        drr_con_tri = DRR(subject, sdd=sdd_cone, height=res, delx=delx_cone, x0=intr["x0"], y0=intr["y0"], renderer="trilinear")

        def make_pose_fn(renderer_fn, base_pose):
            dr = torch.zeros((1, 3), requires_grad=True)
            dt = torch.zeros((1, 3), requires_grad=True)

            def call():
                delta = convert(dr, dt, parameterization="euler_angles", convention="ZYX")
                p = RigidTransform(delta.matrix @ base_pose.matrix)
                return renderer_fn(p)

            return [dr, dt], call

        configs = [
            ("Parallel (SDD=1e6 mm)", "Parallel Diff-FVR (1x1x1)", prep_par_s, pose_par, lambda p: fvr_par.forward_diffdrr(drr_par_sid, p, res)),
            ("Parallel (SDD=1e6 mm)", "DiffDRR Siddon (SDD=1e6)", 0.0, pose_par, lambda p: drr_par_sid(p)),
            ("Parallel (SDD=1e6 mm)", "DiffDRR Trilinear (SDD=1e6)", 0.0, pose_par, lambda p: drr_par_tri(p)),
            (f"Cone-Beam (SDD={sdd_cone:.0f} mm)", "Cone-Beam PoU-FVR (2x2x2)", prep_cone_s, pose_t, lambda p: fvr_cone.forward_diffdrr(drr_con_sid, p, res)),
            (f"Cone-Beam (SDD={sdd_cone:.0f} mm)", "DiffDRR Siddon (Cone)", 0.0, pose_t, lambda p: drr_con_sid(p)),
            (f"Cone-Beam (SDD={sdd_cone:.0f} mm)", "DiffDRR Trilinear (Cone)", 0.0, pose_t, lambda p: drr_con_tri(p)),
        ]

        for geom_lbl, method_lbl, prep_s, bp, fn in configs:
            params, call = make_pose_fn(fn, bp)
            reps = 5 if res == 128 else 3
            fwd_ms, fb_ms = _benchmark_forward_backward(call, params, repeats=reps, warmup=1)
            step_2view_s = 2.0 * fb_ms / 1000.0
            tot_10 = prep_s + 10.0 * step_2view_s
            tot_40 = prep_s + 40.0 * step_2view_s
            tot_100 = prep_s + 100.0 * step_2view_s
            print(
                f"  [H={res}] {geom_lbl:25s} | {method_lbl:26s} | prep={prep_s:4.2f}s | "
                f"fwd={fwd_ms:7.2f}ms | fwd+bwd={fb_ms:7.2f}ms | 40-step={tot_40:5.2f}s | 100-step={tot_100:5.2f}s"
            )
            rows.append({
                "detector_res": res,
                "geometry": geom_lbl,
                "method": method_lbl,
                "fft_prep_s": round(prep_s, 3),
                "fwd_ms": round(fwd_ms, 2),
                "fwd_bwd_ms": round(fb_ms, 2),
                "total_10steps_2view_s": round(tot_10, 2),
                "total_40steps_2view_s": round(tot_40, 2),
                "total_100steps_2view_s": round(tot_100, 2),
            })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}\n")


def run_part_b_robustness(out_csv: Path):
    print("=== PART B1: Compact-to-Standard C-Arm Mag Stress (GE Clinical DICOM) ===")
    vol_hu, spacing = load_dicom_series(ROOT / "data" / "dicom")
    vol_cubic, gs = resample_isotropic_cube(vol_hu, spacing, 96)
    vox = float(gs[0])
    vol_mu = hu_to_attenuation(vol_cubic)
    fov = 96 * vox

    cb_fvr = PerspectiveSlabFVR.from_volume(
        vol_mu, vox_mm=vox, grid_blocks=(2, 2, 2), block_res=64, pad_mult=2, center_mode="grid_center"
    )
    ortho_engine = RobustDiffFVREngine(vol_mu, pad_factor=2)

    carm_configs = [
        ("Compact Mobile C-Arm", 600.0, 900.0),
        ("Mini Surgical C-Arm", 750.0, 1050.0),
        ("Standard Fluoroscopy", 850.0, 1200.0),
        ("Fixed Interventional", 1000.0, 1400.0),
    ]
    rows = []

    for label, sad, sdd in carm_configs:
        mag = sdd / sad
        eff_vox = vox / mag
        np.random.seed(909)
        ortho_raw_mtres, ortho_cal_mtres, ortho_times = [], [], []
        pou_mtres, pou_times = [], []
        true_nccs = []

        for _trial in range(5):
            true_pose = (
                float(np.random.uniform(-4, 4)),
                float(np.random.uniform(-4, 4)),
                float(np.random.uniform(-4, 4)),
                float(np.random.uniform(-5, 5)),
                float(np.random.uniform(-4, 4)),
                float(np.random.uniform(-4, 4)),
            )
            rot_vol = rotate_volume_3d(vol_mu, *true_pose)
            t_ap = torch.from_numpy(project_conebeam_3d(rot_vol, sad=sad, sdd=sdd, det_pixels=96, view="ap").astype("float32"))
            t_lt = torch.from_numpy(project_conebeam_3d(rot_vol, sad=sad, sdd=sdd, det_pixels=96, view="lateral").astype("float32"))
            t_ap_n, t_lt_n = _norm01(t_ap), _norm01(t_lt)

            with torch.no_grad():
                p0 = cb_fvr.forward_carm_view(*[torch.tensor(v) for v in true_pose], sad, sdd, 96, fov, "ap")
                true_nccs.append(1.0 - float(_ncc_loss(p0, t_ap)))

            # Method 1a: Uncalibrated Orthographic Diff-FVR (native cone mismatch)
            res_raw = ortho_engine.register_biplanar_robust(t_ap.numpy(), t_lt.numpy(), handle_metal=False, max_iter=60)
            est_raw = res_raw["final_pose"]
            ortho_raw_mtres.append(compute_mtre_6dof(true_pose, tuple(est_raw), vol_mu.shape, spacing=(vox,) * 3))

            # Method 1b: Isocenter-Calibrated Orthographic Diff-FVR
            t_ap_cal = calibrate_to_isocenter(t_ap.numpy(), sad, sdd, 96)
            t_lt_cal = calibrate_to_isocenter(t_lt.numpy(), sad, sdd, 96)
            t0 = time.perf_counter()
            res_cal = ortho_engine.register_biplanar_robust(t_ap_cal, t_lt_cal, handle_metal=False, max_iter=60)
            ortho_times.append(time.perf_counter() - t0)
            est_cal = res_cal["final_pose"]
            ortho_cal_mtres.append(compute_mtre_6dof(true_pose, tuple(est_cal), vol_mu.shape, spacing=(vox,) * 3))

            # Method 2: Standalone True Cone-Beam PerspectiveSlabFVR (2x2x2)
            r = torch.zeros(3, requires_grad=True)
            t = torch.zeros(3, requires_grad=True)
            opt = torch.optim.Adam([{"params": [r], "lr": 0.65}, {"params": [t], "lr": 0.65}])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=90, eta_min=0.01)
            t0 = time.perf_counter()
            for _ in range(90):
                opt.zero_grad()
                p_ap = cb_fvr.forward_carm_view(r[0], r[1], r[2], t[0], t[1], t[2], sad, sdd, 96, fov, "ap")
                p_lt = cb_fvr.forward_carm_view(r[0], r[1], r[2], t[0], t[1], t[2], sad, sdd, 96, fov, "lateral")
                loss = 0.5 * (_ncc_loss(p_ap, t_ap_n) + _ncc_loss(p_lt, t_lt_n))
                loss.backward()
                torch.nn.utils.clip_grad_norm_([r], 2.5)
                torch.nn.utils.clip_grad_norm_([t], 2.5)
                opt.step()
                sched.step()
            pou_times.append(time.perf_counter() - t0)
            est_p = (r[0].item(), r[1].item(), r[2].item(), t[0].item(), t[1].item(), t[2].item())
            pou_mtres.append(compute_mtre_6dof(true_pose, est_p, vol_mu.shape, spacing=(vox,) * 3))

        print(
            f"  {label:22s} ({sad:.0f}/{sdd:.0f}mm Mag={mag:.2f}x): TrueNCC={np.mean(true_nccs):.4f} | "
            f"Uncal-Ortho={np.mean(ortho_raw_mtres):.2f}mm | Cal-Ortho={np.mean(ortho_cal_mtres):.2f}mm ({np.mean(ortho_times):.2f}s) | "
            f"Cone PoU-FVR={np.mean(pou_mtres):.2f}mm ({np.mean(pou_times):.2f}s)"
        )
        rows.append({
            "cohort": f"GE C-Arm {label} ({sad:.0f}/{sdd:.0f}mm)",
            "separation_or_mag": f"{mag:.2f}x",
            "true_pose_ncc": round(float(np.mean(true_nccs)), 4),
            "uncal_ortho_or_drr_mtre_mm": round(float(np.mean(ortho_raw_mtres)), 3),
            "cal_ortho_or_drr_time_s": round(float(np.mean(ortho_times)), 2),
            "cal_ortho_mtre_mm": round(float(np.mean(ortho_cal_mtres)), 3),
            "pou_conefvr_mtre_mm": round(float(np.mean(pou_mtres)), 3),
            "pou_conefvr_time_s": round(float(np.mean(pou_times)), 2),
            "hybrid_mtre_mm": "",
            "hybrid_time_s": "",
        })

    print("\n=== PART B2: Real Clinical Fluoroscopy (DeepFluoro subject01-06) ===")
    H = 128

    for sidx, sname in enumerate(SUBJECTS, start=1):
        subj_dir = str(ROOT / "data" / "deepfluoro" / "deepfluoro" / sname)
        fids = torch.load(os.path.join(subj_dir, "fiducials.pt"), weights_only=False)
        subject = read(os.path.join(subj_dir, "volume.nii.gz"), orientation="AP")
        vA, vB, opt_sep_deg = _find_max_optical_pair(subject, subj_dir, H=H)
        poseA, xrayA, drrA = load_view(subject, subj_dir, vA, H=H)
        poseB, xrayB, drrB = load_view(subject, subj_dir, vB, H=H)
        _, xrayA_64, drrA_64 = load_view(subject, subj_dir, vA, H=64)
        _, xrayB_64, drrB_64 = load_view(subject, subj_dir, vB, H=64)
        # Exact rigid biplanar relative transform: P_B = P_A @ T_AB
        T_AB = RigidTransform(poseA.matrix.inverse() @ poseB.matrix)

        fvr_cone = PerspectiveSlabFVR.from_diffdrr_subject(subject, grid_blocks=(2, 2, 2), block_res=64, pad_mult=2)

        with torch.no_grad():
            drr_a = drrA(poseA)
            pou_a = fvr_cone.forward_diffdrr(drrA, poseA, H)
            true_ncc_drr = 1.0 - float(_ncc_loss(pou_a, drr_a))

        rng = np.random.default_rng(2727 + sidx)
        dr0 = torch.tensor(rng.uniform(-0.055, 0.055, size=(1, 3)), dtype=torch.float32)
        dt0 = torch.tensor(rng.uniform(-7.0, 7.0, size=(1, 3)), dtype=torch.float32)
        pose_init = RigidTransform(convert(dr0, dt0, parameterization="euler_angles", convention="ZYX").matrix @ poseA.matrix)

        def build_pose(r_par, t_par, base=pose_init):
            delta = convert(r_par, t_par, parameterization="euler_angles", convention="ZYX")
            return RigidTransform(delta.matrix @ base.matrix)

        def loss_edge(pA, pB):
            return 1.0 - 0.5 * (edge_dominant_ncc(pA, xrayA) + edge_dominant_ncc(pB, xrayB))

        init_mtre = compute_mtre(pose_init, poseA, fids)

        # 1. DiffDRR-Only Siddon (25 steps, single-start, rigid T_AB + edge_dominant_ncc)
        r_d = torch.zeros(1, 3, requires_grad=True)
        t_d = torch.zeros(1, 3, requires_grad=True)
        opt_d = torch.optim.Adam([{"params": [r_d], "lr": 0.004}, {"params": [t_d], "lr": 1.0}])
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=25, eta_min=0.0002)
        t0 = time.perf_counter()
        for _ in range(25):
            opt_d.zero_grad()
            pa = build_pose(r_d, t_d)
            pb = RigidTransform(pa.matrix @ T_AB.matrix)
            l = loss_edge(drrA(pa), drrB(pb))
            l.backward()
            torch.nn.utils.clip_grad_norm_([r_d], 0.25)
            torch.nn.utils.clip_grad_norm_([t_d], 2.0)
            opt_d.step()
            sched_d.step()
        drr_time = time.perf_counter() - t0
        drr_mtre = compute_mtre(build_pose(r_d.detach(), t_d.detach()), poseA, fids)

        # 2. Single-Start Cone-Beam PoU-FVR (50 steps, rigid T_AB + edge_dominant_ncc)
        r_s = torch.zeros(1, 3, requires_grad=True)
        t_s = torch.zeros(1, 3, requires_grad=True)
        opt_s = torch.optim.Adam([{"params": [r_s], "lr": 0.005}, {"params": [t_s], "lr": 1.0}])
        sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=50, eta_min=0.0002)
        t0 = time.perf_counter()
        for _ in range(50):
            opt_s.zero_grad()
            pa = build_pose(r_s, t_s)
            pb = RigidTransform(pa.matrix @ T_AB.matrix)
            l = loss_edge(fvr_cone.forward_diffdrr(drrA, pa, H), fvr_cone.forward_diffdrr(drrB, pb, H))
            l.backward()
            torch.nn.utils.clip_grad_norm_([r_s], 0.25)
            torch.nn.utils.clip_grad_norm_([t_s], 2.0)
            opt_s.step()
            sched_s.step()
        single_pou_mtre = compute_mtre(build_pose(r_s.detach(), t_s.detach()), poseA, fids)

        # 3. Standalone Multi-Start Edge PoU-FVR (H=64 screen 24 -> Top-4 -> H=128 polish, ZERO DiffDRR)
        t0 = time.perf_counter()
        pose_ms, _ = fvr_cone.register_diffdrr_multistart(
            drrA, drrB, xrayA, xrayB, pose_init, T_AB,
            drrA_coarse=drrA_64, drrB_coarse=drrB_64,
            xrayA_coarse=xrayA_64, xrayB_coarse=xrayB_64,
            n_candidates=24, rot_rad=0.055, trans_mm=8.0,
            top_k=4, coarse_steps=15, fine_steps=45,
            fine_drr_polish_steps=0, seed=100 + sidx,
        )
        pou_time = time.perf_counter() - t0
        pou_mtre = compute_mtre(pose_ms, poseA, fids)

        # 4. Hybrid: Multi-Start PoU-FVR + 8-step DiffDRR Edge-NCC micro-polish
        r_h = torch.zeros(1, 3, requires_grad=True)
        t_h = torch.zeros(1, 3, requires_grad=True)
        opt_h2 = torch.optim.Adam([{"params": [r_h], "lr": 0.002}, {"params": [t_h], "lr": 0.4}])
        t1 = time.perf_counter()
        for _ in range(8):
            opt_h2.zero_grad()
            pa = build_pose(r_h, t_h, base=pose_ms)
            pb = RigidTransform(pa.matrix @ T_AB.matrix)
            l = loss_edge(drrA(pa), drrB(pb))
            l.backward()
            opt_h2.step()
        hyb_time = pou_time + (time.perf_counter() - t1)
        hyb_mtre = compute_mtre(build_pose(r_h.detach(), t_h.detach(), base=pose_ms), poseA, fids)

        print(
            f"  {sname} ({vA}+{vB}, opt_sep={opt_sep_deg:4.1f}deg, Init={init_mtre:5.2f}mm, TrueNCC={true_ncc_drr:.4f}): "
            f"DiffDRR={drr_mtre:.2f}mm ({drr_time:.1f}s) | SinglePoU={single_pou_mtre:.2f}mm | "
            f"MultiStart PoU-FVR={pou_mtre:.2f}mm ({pou_time:.2f}s) | "
            f"Hybrid={hyb_mtre:.2f}mm ({hyb_time:.2f}s)"
        )
        rows.append({
            "cohort": f"DeepFluoro {sname} ({vA}+{vB})",
            "separation_or_mag": f"{opt_sep_deg:.1f} deg",
            "true_pose_ncc": round(true_ncc_drr, 4),
            "uncal_ortho_or_drr_mtre_mm": round(drr_mtre, 3),
            "cal_ortho_or_drr_time_s": round(drr_time, 2),
            "cal_ortho_mtre_mm": round(single_pou_mtre, 3),
            "pou_conefvr_mtre_mm": round(pou_mtre, 3),
            "pou_conefvr_time_s": round(pou_time, 2),
            "hybrid_mtre_mm": round(hyb_mtre, 3),
            "hybrid_time_s": round(hyb_time, 2),
        })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-b-only", action="store_true")
    args = ap.parse_args()
    out_dir = ROOT / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.part_b_only:
        run_part_a_speed_benchmark(out_dir / "table27_geometry_matched_benchmark.csv")
    run_part_b_robustness(out_dir / "table28_conebeam_clinical_robustness.csv")
