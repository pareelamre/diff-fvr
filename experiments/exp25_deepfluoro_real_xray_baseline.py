"""
Experiment 25: Real Paired CT + C-Arm Fluoroscopy Validation & DiffDRR Head-to-Head
==================================================================================
Addresses three key validation pillars on real clinical data (DeepFluoro subject01):
  1. Real Paired CT + C-Arm Fluoroscopy Validation:
     Evaluates on real pelvic CT (398x197x341, 1.0 mm) and 10 real C-arm DICOMs
     (000..009) with ground-truth SE(3) poses and 14 3D anatomical fiducials.
  2. Direct Head-to-Head Baseline Comparison:
     Compares Diff-FVR against DiffDRR 0.6.1 (Siddon cone-beam raycasting) on
     the exact same CT, real X-rays, MNCC objective, and hardware:
       - Single-DRR forward latency (ms) & throughput (FPS) across all 10 views
       - Registration runtime (s) and 3D fiducial mTRE (mm) on bi-planar pairs
  3. Cone-Beam Perspective Parallax & Hybrid Coarse-to-Fine:
     Quantifies the parallel+isocenter-magnification vs. cone-beam perspective
     gap (NCC across all 10 views) and benchmarks Diff-FVR -> DiffDRR Hybrid.

Outputs:
  - outputs/paper_figures/table25_deepfluoro_real_baseline.csv
  - outputs/paper_figures/fig25_deepfluoro_real_baseline.png
"""

import csv
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


DATA_DIR = os.path.join("data", "deepfluoro", "deepfluoro", "subject01")
OUT_DIR = os.path.join("outputs", "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)


def ncc_2d(a: torch.Tensor, b: torch.Tensor) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    return float((a0 * b0).sum() / (a0.norm() * b0.norm() + 1e-8))


class DeepFluoroDiffFVR:
    """World-aligned 3D Fourier Volume Rendering slice engine with isocenter magnification."""

    def __init__(self, subject, fov_mm: float = 400.0, n_slice: int = 160):
        self.fov_mm = fov_mm
        self.n_slice = n_slice
        self.ds_slice = fov_mm / n_slice

        # subject.density.data has shape (1, X, Y, Z) with affine diag(-1, -1, 1)
        vol_raw = subject.density.data.squeeze(0)
        vol_world = vol_raw.flip(0, 1)  # align voxel axes with world (+x, +y, +z)

        nx, ny, nz = vol_world.shape
        px = (400 - nx) // 2
        py = (400 - ny) // 2
        pz = (400 - nz) // 2
        vol_400 = F.pad(vol_world, (pz, 400 - nz - pz, py, 400 - ny - py, px, 400 - nx - px))
        vol_200 = F.avg_pool3d(vol_400.unsqueeze(0).unsqueeze(0), 2).squeeze()  # 2.0 mm isotropic

        self.P = 400
        self.ds_vol = 2.0
        pad = (self.P - 200) // 2
        vol_pad = F.pad(vol_200, (pad, pad, pad, pad, pad, pad))
        F_vol = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(vol_pad)))
        self.F_vol_ri = torch.stack([F_vol.real, F_vol.imag], dim=0).unsqueeze(0)  # (1, 2, P, P, P)

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

        # Isocenter plane intersection & magnification scaling
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


def load_view(subject, vid: str, H: int = 128):
    pt = torch.load(os.path.join(DATA_DIR, "xrays", f"{vid}.pt"), weights_only=False)
    intr = pt["intrinsics"]
    pose = RigidTransform(pt["pose"])
    dcm = pydicom.dcmread(os.path.join(DATA_DIR, "xrays", f"{vid}.dcm")).pixel_array.astype("float32")
    off = (dcm.shape[0] - intr["height"]) // 2
    crop = -dcm[off : off + intr["height"], off : off + intr["height"]]
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
    print("Loading DeepFluoro subject01 CT & fiducials...")
    fids = torch.load(os.path.join(DATA_DIR, "fiducials.pt"), weights_only=False)
    subject = read(os.path.join(DATA_DIR, "volume.nii.gz"), orientation="AP")

    t_pre0 = time.perf_counter()
    fvr = DeepFluoroDiffFVR(subject, fov_mm=400.0, n_slice=160)
    t_pre = time.perf_counter() - t_pre0
    print(f"Diff-FVR 3D FFT precomputation: {t_pre:.2f} s")

    mncc = MultiscaleNormalizedCrossCorrelation2d()

    # 1. Forward-model fidelity & latency across all 10 real C-arm views
    views = {}
    fwd_rows = []
    print("\n--- Part 1: 10-View Forward Fidelity & Single-DRR Latency (H=128) ---")
    for i in range(10):
        vid = f"{i:03d}"
        pose, xray, drr = load_view(subject, vid, H=H)
        views[vid] = (pose, xray, drr)

        # Warmup
        with torch.no_grad():
            _ = fvr.forward(drr, pose, H)
            _ = drr(pose)

        # Time Diff-FVR (mean of 10)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                p_fvr = fvr.forward(drr, pose, H)
        ms_fvr = (time.perf_counter() - t0) / 10.0 * 1000.0

        # Time DiffDRR (mean of 3)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(3):
                p_drr = drr(pose)
        ms_drr = (time.perf_counter() - t0) / 3.0 * 1000.0

        ncc_fvr_drr = ncc_2d(p_fvr[:, :, 8:-8, 8:-8], p_drr[:, :, 8:-8, 8:-8])
        mncc_fvr_real = float(mncc(p_fvr[:, :, 8:-8, 8:-8], xray[:, :, 8:-8, 8:-8]).item())
        mncc_drr_real = float(mncc(p_drr[:, :, 8:-8, 8:-8], xray[:, :, 8:-8, 8:-8]).item())
        speedup = ms_drr / ms_fvr
        fwd_rows.append({
            "view": vid,
            "ms_fvr": ms_fvr,
            "ms_drr": ms_drr,
            "speedup": speedup,
            "ncc_fvr_vs_drr": ncc_fvr_drr,
            "mncc_fvr_vs_real": mncc_fvr_real,
            "mncc_drr_vs_real": mncc_drr_real,
        })
        print(
            f"View {vid}: FVR={ms_fvr:5.2f} ms ({1000/ms_fvr:5.1f} FPS) | "
            f"DiffDRR={ms_drr:7.2f} ms ({1000/ms_drr:4.2f} FPS) | "
            f"Speedup={speedup:6.1f}x | NCC(FVR,DRR)={ncc_fvr_drr:.4f} | "
            f"MNCC(FVR,Real)={mncc_fvr_real:.4f} vs DRR={mncc_drr_real:.4f}"
        )

    # 2. Bi-planar SE(3) Registration Benchmark on 4 diverse real C-arm pairs
    pairs = [("000", "002"), ("001", "003"), ("004", "007"), ("006", "009")]
    perturbations = [
        (torch.tensor([[0.05, -0.05, 0.04]]), torch.tensor([[5.0, -6.0, 4.0]])),
        (torch.tensor([[-0.04, 0.05, -0.05]]), torch.tensor([[-6.0, 5.0, -5.0]])),
        (torch.tensor([[0.05, 0.04, -0.04]]), torch.tensor([[4.0, 6.0, -5.0]])),
        (torch.tensor([[-0.05, -0.04, 0.05]]), torch.tensor([[-5.0, -5.0, 6.0]])),
    ]

    reg_rows = []
    print("\n--- Part 2: Bi-Planar Real C-Arm Registration (Diff-FVR vs Hybrid vs DiffDRR) ---")
    for (vA, vB), (dr, dt) in zip(pairs, perturbations):
        poseA, xrayA, drrA = views[vA]
        poseB, xrayB, drrB = views[vB]
        T_rel = RigidTransform(poseB.matrix @ poseA.matrix.inverse())
        delta_init = convert(dr, dt, parameterization="euler_angles", convention="ZYX")
        pose_initA = RigidTransform(delta_init.matrix @ poseA.matrix)
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

        t_fvr, m_fvr = run_method(35, 0)
        t_hyb, m_hyb = run_method(30, 15)
        t_drr, m_drr = run_method(0, 25)

        reg_rows.append({
            "pair": f"{vA}+{vB}",
            "init_mtre_mm": mtre_init,
            "fvr_time_s": t_fvr, "fvr_mtre_mm": m_fvr,
            "hyb_time_s": t_hyb, "hyb_mtre_mm": m_hyb,
            "drr_time_s": t_drr, "drr_mtre_mm": m_drr,
        })
        print(
            f"Pair {vA}+{vB} (init={mtre_init:.2f} mm) | "
            f"Diff-FVR: {t_fvr:5.2f}s, {m_fvr:5.2f} mm | "
            f"Hybrid: {t_hyb:5.2f}s, {m_hyb:5.2f} mm | "
            f"DiffDRR: {t_drr:6.2f}s, {m_drr:5.2f} mm"
        )

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "table25_deepfluoro_real_baseline.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "id", "metric_a", "metric_b", "metric_c", "metric_d", "metric_e", "metric_f", "metric_g"])
        for r in fwd_rows:
            w.writerow([
                "forward_10views", r["view"],
                f"{r['ms_fvr']:.3f}", f"{r['ms_drr']:.3f}", f"{r['speedup']:.2f}",
                f"{r['ncc_fvr_vs_drr']:.4f}", f"{r['mncc_fvr_vs_real']:.4f}",
                f"{r['mncc_drr_vs_real']:.4f}", ""
            ])
        for r in reg_rows:
            w.writerow([
                "reg_biplanar", r["pair"],
                f"{r['init_mtre_mm']:.3f}",
                f"{r['fvr_time_s']:.3f}", f"{r['fvr_mtre_mm']:.3f}",
                f"{r['hyb_time_s']:.3f}", f"{r['hyb_mtre_mm']:.3f}",
                f"{r['drr_time_s']:.3f}", f"{r['drr_mtre_mm']:.3f}",
            ])
    print(f"\nSaved {csv_path}")

    # Plot 3-panel publication figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=180)

    # Panel A: Single-DRR Latency (log scale) across 10 views
    ax = axes[0]
    x = np.arange(len(fwd_rows))
    ms_f = [r["ms_fvr"] for r in fwd_rows]
    ms_d = [r["ms_drr"] for r in fwd_rows]
    ax.bar(x - 0.18, ms_d, width=0.35, color="#c0392b", label="DiffDRR 0.6.1 (Siddon)")
    ax.bar(x + 0.18, ms_f, width=0.35, color="#1f77b4", label="Diff-FVR (Fourier Slice)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r["view"] for r in fwd_rows], fontsize=8)
    ax.set_xlabel("DeepFluoro subject01 C-Arm View")
    ax.set_ylabel("Single-DRR Latency (ms, log scale)")
    ax.set_title(f"A. Forward Speed ({np.mean(ms_d)/np.mean(ms_f):.0f}x Speedup)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel B: Perspective vs Fourier Alignment across 10 views
    ax = axes[1]
    ncc_fd = [r["ncc_fvr_vs_drr"] for r in fwd_rows]
    mncc_fr = [r["mncc_fvr_vs_real"] for r in fwd_rows]
    mncc_dr = [r["mncc_drr_vs_real"] for r in fwd_rows]
    ax.plot(x, ncc_fd, "o-", color="#2c3e50", label="NCC(Diff-FVR, DiffDRR)")
    ax.plot(x, mncc_dr, "s--", color="#c0392b", label="MNCC(DiffDRR, Real X-ray)")
    ax.plot(x, mncc_fr, "^-", color="#1f77b4", label="MNCC(Diff-FVR, Real X-ray)")
    ax.set_xticks(x)
    ax.set_xticklabels([r["view"] for r in fwd_rows], fontsize=8)
    ax.set_ylim(0.75, 1.01)
    ax.set_xlabel("DeepFluoro subject01 C-Arm View")
    ax.set_ylabel("Similarity Score")
    ax.set_title("B. Real C-Arm & Perspective Fidelity")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(True, alpha=0.3)

    # Panel C: Runtime vs Final 3D Fiducial mTRE (Bi-planar Real C-Arm)
    ax = axes[2]
    labels = ["Initial", "Diff-FVR\n(35 steps)", "Hybrid\n(30+15)", "DiffDRR\n(25 steps)"]
    mean_mtre = [
        np.mean([r["init_mtre_mm"] for r in reg_rows]),
        np.mean([r["fvr_mtre_mm"] for r in reg_rows]),
        np.mean([r["hyb_mtre_mm"] for r in reg_rows]),
        np.mean([r["drr_mtre_mm"] for r in reg_rows]),
    ]
    mean_time = [
        0.0,
        np.mean([r["fvr_time_s"] for r in reg_rows]),
        np.mean([r["hyb_time_s"] for r in reg_rows]),
        np.mean([r["drr_time_s"] for r in reg_rows]),
    ]
    colors = ["#7f8c8d", "#1f77b4", "#27ae60", "#c0392b"]
    bars = ax.bar(labels, mean_mtre, color=colors, width=0.55)
    for bar, tm in zip(bars, mean_time):
        h = bar.get_height()
        txt = f"{h:.2f} mm" if tm == 0 else f"{h:.2f} mm\n({tm:.1f} s)"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.25, txt, ha="center", va="bottom", fontsize=8)
    ax.axhline(2.0, color="#27ae60", linestyle=":", linewidth=1.2, label="Clinical 2.0 mm threshold")
    ax.set_ylim(0, max(mean_mtre) * 1.28)
    ax.set_ylabel("3D Fiducial mTRE (mm)")
    ax.set_title("C. Real C-Arm Bi-Planar Registration")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig25_deepfluoro_real_baseline.png")
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
