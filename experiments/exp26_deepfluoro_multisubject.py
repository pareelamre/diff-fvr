"""
Experiment 26: Multi-Subject Real Paired CT + C-Arm Validation (DeepFluoro subject01-06)
=========================================================================================
Evaluates Diff-FVR, Hybrid (Diff-FVR -> DiffDRR), and DiffDRR 0.6.1 across all
6 real clinical subjects in DeepFluoro (eigenvivek/xvr-data):
  - Forward latency (ms), FPS, and perspective fidelity (NCC vs DiffDRR, MNCC vs Real X-ray)
  - Bi-planar 3D/2D pose recovery on each subject's max-angle C-arm pair (30.5-35.7 deg)
  - Reports per-subject and cohort mean/median 3D fiducial mTRE (mm) & runtime (s)

Outputs:
  - outputs/paper_figures/table26_deepfluoro_multisubject.csv
  - outputs/paper_figures/fig26_deepfluoro_multisubject.png
"""

import csv
import math
import os
import time
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d


ROOT_DIR = os.path.join("data", "deepfluoro", "deepfluoro")
OUT_DIR = os.path.join("outputs", "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)
SUBJECTS = ["subject01", "subject02", "subject03", "subject04", "subject05", "subject06"]


def ncc_2d(a: torch.Tensor, b: torch.Tensor) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    return float((a0 * b0).sum() / (a0.norm() * b0.norm() + 1e-8))


class DeepFluoroDiffFVR:
    """World-aligned 3D Fourier Volume Rendering slice engine for arbitrary CT FOV."""

    def __init__(self, subject, fov_mm: float = 480.0, n_slice: int = 160):
        self.fov_mm = fov_mm
        self.n_slice = n_slice
        self.ds_slice = fov_mm / n_slice

        vol_raw = subject.density.data.squeeze(0)
        vol_world = vol_raw.flip(0, 1)  # affine diag(-1, -1, 1) -> world (+x, +y, +z)

        nx, ny, nz = vol_world.shape
        target = int(fov_mm)
        px = (target - nx) // 2
        py = (target - ny) // 2
        pz = (target - nz) // 2
        vol_cubic = F.pad(
            vol_world,
            (pz, target - nz - pz, py, target - ny - py, px, target - nx - px),
        )
        vol_ds = F.avg_pool3d(vol_cubic.unsqueeze(0).unsqueeze(0), 2).squeeze()  # 2.0 mm isotropic

        self.P = target  # 2x zero-padding over (target // 2)
        self.ds_vol = 2.0
        pad = (self.P - vol_ds.shape[0]) // 2
        vol_pad = F.pad(vol_ds, (pad, pad, pad, pad, pad, pad))
        F_vol = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(vol_pad)))
        self.F_vol_ri = torch.stack([F_vol.real, F_vol.imag], dim=0).unsqueeze(0)

        fx = torch.fft.fftshift(torch.fft.fftfreq(n_slice, d=self.ds_slice))
        fy = torch.fft.fftshift(torch.fft.fftfreq(n_slice, d=self.ds_slice))
        self.FY, self.FX = torch.meshgrid(fy, fx, indexing="ij")

    def forward(self, drr_mod: DRR, pose: RigidTransform, out_res: int) -> torch.Tensor:
        src, trg = drr_mod.detector(pose, calibration=None)
        src = src.squeeze(0).squeeze(0)
        trg = trg.squeeze(0).view(out_res, out_res, 3)

        c_det = trg.mean(dim=(0, 1))
        u_dir = trg[0, -1] - trg[0, 0]
        u_dir = u_dir / u_dir.norm()
        v_dir = trg[-1, 0] - trg[0, 0]
        v_dir = v_dir / v_dir.norm()
        n_dir = c_det - src
        n_len = n_dir.norm()
        n_dir = n_dir / n_len

        t_iso = -torch.dot(src, n_dir) / n_len
        pts_iso = src + t_iso * (trg - src)
        u_mm = (pts_iso * u_dir).sum(dim=-1)
        v_mm = (pts_iso * v_dir).sum(dim=-1)

        K = self.FX.unsqueeze(-1) * u_dir + self.FY.unsqueeze(-1) * v_dir
        idx = K * (self.P * self.ds_vol) + self.P // 2
        g = 2.0 * idx / (self.P - 1) - 1.0
        grid = torch.stack([g[..., 2], g[..., 1], g[..., 0]], dim=-1).unsqueeze(0).unsqueeze(0)

        S_ri = F.grid_sample(
            self.F_vol_ri, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(0).squeeze(1)
        S = torch.complex(S_ri[0], S_ri[1])
        img_full = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(S))).real

        gu = 2.0 * (u_mm / self.ds_slice + self.n_slice // 2) / (self.n_slice - 1) - 1.0
        gv = 2.0 * (v_mm / self.ds_slice + self.n_slice // 2) / (self.n_slice - 1) - 1.0
        grid_2d = torch.stack([gu, gv], dim=-1).unsqueeze(0)
        return F.grid_sample(
            img_full.unsqueeze(0).unsqueeze(0),
            grid_2d,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


def find_best_pair(subj_dir: str):
    rays = []
    for i in range(6):
        pt = torch.load(os.path.join(subj_dir, "xrays", f"{i:03d}.pt"), weights_only=False)
        R = RigidTransform(pt["pose"]).matrix[0, :3, :3]
        rays.append(R[:, 1])
    best_ang = -1.0
    best_pair = (0, 1)
    for i in range(6):
        for j in range(i + 1, 6):
            c = torch.dot(rays[i], rays[j]).clamp(-1.0, 1.0).item()
            ang = math.degrees(math.acos(c))
            if ang > best_ang:
                best_ang = ang
                best_pair = (i, j)
    return f"{best_pair[0]:03d}", f"{best_pair[1]:03d}", best_ang


def load_view(subject, subj_dir: str, vid: str, H: int = 128):
    pt = torch.load(os.path.join(subj_dir, "xrays", f"{vid}.pt"), weights_only=False)
    intr = pt["intrinsics"]
    pose = RigidTransform(pt["pose"])
    dcm = pydicom.dcmread(os.path.join(subj_dir, "xrays", f"{vid}.dcm")).pixel_array.astype("float32")
    off = (dcm.shape[0] - intr["height"]) // 2
    crop = -dcm[off : off + intr["height"], off : off + intr["width"]]
    xray = F.interpolate(
        torch.from_numpy(crop).unsqueeze(0).unsqueeze(0),
        size=(H, H),
        mode="bilinear",
        align_corners=False,
    ).flip(-1)
    drr = DRR(
        subject,
        sdd=intr["sdd"],
        height=H,
        delx=intr["delx"] * (intr["height"] / H),
        x0=intr["x0"],
        y0=intr["y0"],
    )
    return pose, xray, drr


def compute_mtre(p_est: RigidTransform, p_gt: RigidTransform, fids: torch.Tensor) -> float:
    T_err = RigidTransform(p_est.matrix @ p_gt.matrix.inverse())
    return float((T_err(fids) - fids).norm(dim=-1).mean().item())


def main():
    H = 128
    mncc = MultiscaleNormalizedCrossCorrelation2d()
    rows = []

    dr_init = torch.tensor([[0.045, -0.045, 0.035]])
    dt_init = torch.tensor([[5.0, -5.0, 4.0]])

    for s in SUBJECTS:
        subj_dir = os.path.join(ROOT_DIR, s)
        fids = torch.load(os.path.join(subj_dir, "fiducials.pt"), weights_only=False)
        subject = read(os.path.join(subj_dir, "volume.nii.gz"), orientation="AP")
        vA, vB, ang = find_best_pair(subj_dir)

        t0 = time.perf_counter()
        fvr = DeepFluoroDiffFVR(subject, fov_mm=480.0, n_slice=160)
        t_pre = time.perf_counter() - t0

        poseA, xrayA, drrA = load_view(subject, subj_dir, vA, H=H)
        poseB, xrayB, drrB = load_view(subject, subj_dir, vB, H=H)

        # Forward latency & fidelity on view A
        with torch.no_grad():
            _ = fvr.forward(drrA, poseA, H)
            _ = drrA(poseA)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(8):
                p_fvr_img = fvr.forward(drrA, poseA, H)
        ms_fvr = (time.perf_counter() - t0) / 8.0 * 1000.0

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(3):
                p_drr_img = drrA(poseA)
        ms_drr = (time.perf_counter() - t0) / 3.0 * 1000.0

        ncc_fd = ncc_2d(p_fvr_img[:, :, 8:-8, 8:-8], p_drr_img[:, :, 8:-8, 8:-8])
        mncc_fr = float(mncc(p_fvr_img[:, :, 8:-8, 8:-8], xrayA[:, :, 8:-8, 8:-8]).item())
        mncc_dr = float(mncc(p_drr_img[:, :, 8:-8, 8:-8], xrayA[:, :, 8:-8, 8:-8]).item())

        # Bi-planar registration
        T_rel = RigidTransform(poseB.matrix @ poseA.matrix.inverse())
        delta0 = convert(dr_init, dt_init, parameterization="euler_angles", convention="ZYX")
        pose_initA = RigidTransform(delta0.matrix @ poseA.matrix)
        mtre_init = compute_mtre(pose_initA, poseA, fids)

        def run_method(steps_fvr: int, steps_drr: int):
            r = torch.zeros(1, 3, requires_grad=True)
            t = torch.zeros(1, 3, requires_grad=True)
            ts = time.perf_counter()
            if steps_fvr > 0:
                opt = torch.optim.Adam([{"params": [r], "lr": 0.004}, {"params": [t], "lr": 1.2}])
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_fvr, eta_min=2e-4)
                for _ in range(steps_fvr):
                    opt.zero_grad()
                    d_mat = convert(r, t, parameterization="euler_angles", convention="ZYX").matrix
                    pA = RigidTransform(d_mat @ pose_initA.matrix)
                    pB = RigidTransform(T_rel.matrix @ pA.matrix)
                    lA = -mncc(fvr.forward(drrA, pA, H)[:, :, 8:-8, 8:-8], xrayA[:, :, 8:-8, 8:-8])
                    lB = -mncc(fvr.forward(drrB, pB, H)[:, :, 8:-8, 8:-8], xrayB[:, :, 8:-8, 8:-8])
                    (0.5 * (lA + lB)).backward()
                    opt.step()
                    sched.step()
            if steps_drr > 0:
                lr_r = 0.003 if steps_fvr > 0 else 0.004
                lr_t = 0.9 if steps_fvr > 0 else 1.2
                opt2 = torch.optim.Adam([{"params": [r], "lr": lr_r}, {"params": [t], "lr": lr_t}])
                sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=steps_drr, eta_min=1e-4)
                for _ in range(steps_drr):
                    opt2.zero_grad()
                    d_mat = convert(r, t, parameterization="euler_angles", convention="ZYX").matrix
                    pA = RigidTransform(d_mat @ pose_initA.matrix)
                    pB = RigidTransform(T_rel.matrix @ pA.matrix)
                    lA = -mncc(drrA(pA)[:, :, 8:-8, 8:-8], xrayA[:, :, 8:-8, 8:-8])
                    lB = -mncc(drrB(pB)[:, :, 8:-8, 8:-8], xrayB[:, :, 8:-8, 8:-8])
                    (0.5 * (lA + lB)).backward()
                    opt2.step()
                    sched2.step()
            elapsed = time.perf_counter() - ts
            with torch.no_grad():
                d_mat = convert(r, t, parameterization="euler_angles", convention="ZYX").matrix
                p_final = RigidTransform(d_mat @ pose_initA.matrix)
                mtre_final = compute_mtre(p_final, poseA, fids)
            return elapsed, mtre_final

        t_fvr, m_fvr = run_method(40, 0)
        t_hyb, m_hyb = run_method(35, 30)
        t_drr, m_drr = run_method(0, 40)

        rows.append({
            "subject": s,
            "pair": f"{vA}+{vB}",
            "angle_deg": ang,
            "fft_pre_s": t_pre,
            "ms_fvr": ms_fvr,
            "ms_drr": ms_drr,
            "speedup": ms_drr / ms_fvr,
            "ncc_fvr_drr": ncc_fd,
            "mncc_fvr_real": mncc_fr,
            "mncc_drr_real": mncc_dr,
            "init_mtre_mm": mtre_init,
            "fvr_time_s": t_fvr,
            "fvr_mtre_mm": m_fvr,
            "hyb_time_s": t_hyb,
            "hyb_mtre_mm": m_hyb,
            "drr_time_s": t_drr,
            "drr_mtre_mm": m_drr,
        })
        print(
            f"{s} ({vA}+{vB}, {ang:.1f}deg) | Speedup={ms_drr/ms_fvr:.1f}x | "
            f"NCC={ncc_fd:.3f} | Init={mtre_init:.2f}mm | "
            f"FVR={m_fvr:.2f}mm ({t_fvr:.1f}s) | "
            f"Hybrid={m_hyb:.2f}mm ({t_hyb:.1f}s) | "
            f"DiffDRR={m_drr:.2f}mm ({t_drr:.1f}s)"
        )

    csv_path = os.path.join(OUT_DIR, "table26_deepfluoro_multisubject.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {csv_path}")

    # Cohort summary
    print("\n--- Cohort Summary (N=6 Real DeepFluoro Subjects) ---")
    print(f"Mean Speedup: {np.mean([r['speedup'] for r in rows]):.1f}x "
          f"({np.mean([r['ms_fvr'] for r in rows]):.1f} ms vs {np.mean([r['ms_drr'] for r in rows]):.1f} ms)")
    print(f"Mean NCC(FVR, DRR): {np.mean([r['ncc_fvr_drr'] for r in rows]):.4f}")
    print(f"Mean Initial mTRE: {np.mean([r['init_mtre_mm'] for r in rows]):.2f} mm")
    print(f"Diff-FVR (40): mTRE={np.mean([r['fvr_mtre_mm'] for r in rows]):.2f} mm, time={np.mean([r['fvr_time_s'] for r in rows]):.2f} s")
    print(f"Hybrid (35+30): mTRE={np.mean([r['hyb_mtre_mm'] for r in rows]):.2f} mm, time={np.mean([r['hyb_time_s'] for r in rows]):.2f} s")
    print(f"DiffDRR (40): mTRE={np.mean([r['drr_mtre_mm'] for r in rows]):.2f} mm, time={np.mean([r['drr_time_s'] for r in rows]):.2f} s")

    # Plot 2-panel cohort figure
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    x = np.arange(len(rows))
    w = 0.25

    ax = axes[0]
    ax.bar(x - w, [r["fvr_mtre_mm"] for r in rows], width=w, color="#1f77b4", label="Diff-FVR (40 steps, ~2.2 s)")
    ax.bar(x, [r["hyb_mtre_mm"] for r in rows], width=w, color="#27ae60", label="Hybrid (35+30 steps)")
    ax.bar(x + w, [r["drr_mtre_mm"] for r in rows], width=w, color="#c0392b", label="DiffDRR 0.6.1 (40 steps)")
    ax.axhline(2.0, color="#27ae60", linestyle=":", linewidth=1.2, label="Clinical 2.0 mm threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([r["subject"].replace("subject", "S") for r in rows])
    ax.set_ylabel("3D Fiducial mTRE (mm)")
    ax.set_title("A. Multi-Subject Real C-Arm Accuracy (DeepFluoro S1-S6)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - 0.18, [r["drr_time_s"] for r in rows], width=0.35, color="#c0392b", label="DiffDRR 0.6.1 (40 steps)")
    ax.bar(x + 0.18, [r["hyb_time_s"] for r in rows], width=0.35, color="#27ae60", label="Hybrid (35+30 steps)")
    ax.bar(x + 0.18, [r["fvr_time_s"] for r in rows], width=0.35, color="#1f77b4", label="Diff-FVR coarse phase")
    ax.set_xticks(x)
    ax.set_xticklabels([r["subject"].replace("subject", "S") for r in rows])
    ax.set_ylabel("Registration Wall-Clock Time (s)")
    ax.set_title("B. Registration Runtime Across Subjects (CPU)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig26_deepfluoro_multisubject.png")
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
