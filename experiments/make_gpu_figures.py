"""Generate publication-grade NVIDIA A100 figures (fig29 & fig30) from verified exp29/30/31 CSVs."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG_OUT = OUT / "paper_figures"
TEX_FIG = Path(
    r"C:\Users\paree\.gemini\antigravity\brain\db84ca2a-35bb-4b49-a099-1ebdc0eb4333\scratch\tex_build\figures"
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.28, linewidth=0.7)


def make_fig29() -> Path:
    bench = _read_csv(OUT / "exp29_gpu_benchmark.csv")
    ablat = _read_csv(OUT / "exp29_pou_ablation.csv")

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.35), dpi=300)
    Hs = [128, 200, 256, 512]
    x = np.arange(len(Hs))

    # --- Panel (a): B=64 Forward Throughput (FPS) ---
    ax = axes[0]
    _style_ax(ax)
    fwd_series = [
        ("PoU-FVR Triton+CUDAGraph (bf16 fwd)", "PoU-FVR Triton+Graph (bf16)", "#0B6E4F", "o-", 2.2),
        ("PoU-FVR Triton (2x2x2 bf16 fwd)", "PoU-FVR Triton eager (bf16)", "#1B9E77", "s--", 1.8),
        ("TorchDiffFVR Zero-IFFT NCC", "Diff-FVR Zero-IFFT NCC (ortho)", "#2B59C3", "^-", 1.9),
        ("nanodrr triton+compile (bf16 n=200 fwd)", "nanodrr triton+comp (n=200)", "#D95F02", "d--", 1.7),
        ("nanodrr triton+compile (bf16 n=500 fwd)", "nanodrr triton+comp (n=500)", "#E7298A", "v-", 1.8),
        ("nanodrr torch (n=500 fwd)", "nanodrr torch (n=500, mb=16@512)", "#666666", "x:", 1.6),
    ]
    for m_key, label, color, fmt, lw in fwd_series:
        vals = []
        for h in Hs:
            match = [
                r for r in bench
                if int(r["H"]) == h and int(r["B"]) == 64 and r["method"].startswith(m_key)
            ]
            vals.append(float(match[0]["fps"]))
        ax.plot(x, vals, fmt, color=color, label=label, linewidth=lw, markersize=5.5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}$^2$" for h in Hs])
    ax.set_xlabel("Detector Resolution ($H \\times H$, $B=64$)")
    ax.set_ylabel("Throughput (FPS, log scale)")
    ax.set_title("(a) A100 Batched Forward Throughput ($B=64$)")
    ax.set_ylim(100, 1.8e5)
    ax.annotate("26,506 FPS\n(185x PyTorch / 23x Triton n=500)",
                xy=(3, 26506), xytext=(1.55, 65000),
                fontsize=7.6, fontweight="bold", color="#0B6E4F",
                arrowprops=dict(arrowstyle="->", color="#0B6E4F", lw=1.0))
    ax.legend(loc="lower left", frameon=True, framealpha=0.92)

    # --- Panel (b): Batched Fwd+Bwd Latency Scaling ---
    ax = axes[1]
    _style_ax(ax)
    bwd_series = [
        ("PoU-FVR Triton (2x2x2 bf16 fwd+bwd)", 64, "PoU-FVR Triton (B=64)", "#0B6E4F", "o-", 2.2),
        ("nanodrr triton (bf16 n=500 fwd+bwd)", 64, "nanodrr Triton n=500 (B=64)", "#E7298A", "v-", 1.9),
        ("PoU-FVR Triton (2x2x2 bf16 fwd+bwd)", 16, "PoU-FVR Triton (B=16)", "#1B9E77", "s--", 1.7),
        ("nanodrr triton (bf16 n=500 fwd+bwd)", 16, "nanodrr Triton n=500 (B=16)", "#D95F02", "d--", 1.7),
    ]
    for m_key, b_val, label, color, fmt, lw in bwd_series:
        vals = [
            float([r for r in bench if int(r["H"]) == h and int(r["B"]) == b_val and r["method"] == m_key][0]["wall_ms"])
            for h in Hs
        ]
        ax.plot(x, vals, fmt, color=color, label=label, linewidth=lw, markersize=5.5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}$^2$" for h in Hs])
    ax.set_xlabel("Detector Resolution ($H \\times H$)")
    ax.set_ylabel("Fwd+Bwd Wall Latency (ms, log scale)")
    ax.set_title("(b) Differentiable Step Latency (Fwd+Bwd)")
    ax.set_ylim(2.5, 250)
    ax.annotate("6.91x faster\n(21.8 vs 151.0 ms)",
                xy=(3, 21.843), xytext=(1.55, 42.0),
                fontsize=7.8, fontweight="bold", color="#0B6E4F",
                arrowprops=dict(arrowstyle="->", color="#0B6E4F", lw=1.0))
    ax.annotate("5.49x\n(7.1 vs 38.8 ms)",
                xy=(2, 7.074), xytext=(0.95, 3.05),
                fontsize=7.6, color="#0B6E4F",
                arrowprops=dict(arrowstyle="->", color="#0B6E4F", lw=0.9))
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    # --- Panel (c): PoU-FVR Block / Resolution Pareto ---
    ax = axes[2]
    _style_ax(ax)
    ablat_sorted = sorted(ablat, key=lambda r: float(r["working_mem_mb"]))
    vrams = [float(r["working_mem_mb"]) for r in ablat_sorted]
    nccs = [float(r["ncc_vs_raymarch"]) for r in ablat_sorted]
    enccs = [float(r["edge_ncc_vs_raymarch"]) for r in ablat_sorted]
    walls = [float(r["wall_ms"]) for r in ablat_sorted]
    labels = [f"{r['grid_blocks'].strip('()').replace(', ', 'x')}@{r['block_res']}" for r in ablat_sorted]

    ax.plot(vrams, nccs, "o-", color="#2B59C3", linewidth=1.8, markersize=5.2, label="Intensity NCC")
    ax.plot(vrams, enccs, "s-", color="#0B6E4F", linewidth=1.8, markersize=5.2, label="Edge-NCC")

    label_offsets = {
        "1x1x1@64": (1.0, -0.025),
        "2x2x2@64": (1.2, -0.045),
        "2x4x2@80": (-6.5, 0.045),
        "2x4x2@96": (-6.8, 0.045),
        "4x4x2@80": (-6.0, 0.045),
    }
    for v_mb, e_val, lbl in zip(vrams, enccs, labels):
        if lbl in label_offsets:
            dx, dy = label_offsets[lbl]
            ax.annotate(lbl, xy=(v_mb, e_val),
                        xytext=(v_mb + dx, e_val + dy),
                        fontsize=7.3, color="#1a1a1a",
                        arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6))

    ax.text(
        0.52, 0.80,
        f"Flat latency across all\n11 configs: {min(walls):.2f}–{max(walls):.2f} ms",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=7.8, bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef7f3", edgecolor="#0B6E4F", lw=0.8),
    )
    ax.set_xlabel("PoU Spectral Cache Memory (MB, bf16)")
    ax.set_ylabel("Fidelity vs. Raymarching ($256^2$)")
    ax.set_title("(c) PoU-FVR Block/Res Pareto Ablation")
    ax.set_ylim(0.35, 1.02)
    ax.set_xlim(-1.0, 33.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.94)

    fig.tight_layout(w_pad=2.2)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    TEX_FIG.mkdir(parents=True, exist_ok=True)
    out_path = FIG_OUT / "fig29_a100_gpu_benchmark.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(out_path, TEX_FIG / out_path.name)
    return out_path


def make_fig30() -> Path:
    exp30 = _read_csv(OUT / "exp30_gpu_multistart_reg.csv")
    exp31 = _read_csv(OUT / "exp31_deepfluoro_gpu_cohort.csv")

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.45), dpi=300)

    # --- Panel (a): exp30 B=64 Multi-Start Registration (10 DeepFluoro trials) ---
    ax = axes[0]
    _style_ax(ax)
    m30_keys = [
        "PoU-FVR Triton (2x4x2@80 bf16)",
        "Hybrid PoU-Triton + nanodrr polish",
        "nanodrr triton+compile (bf16 n=500)",
    ]
    m30_labels = [
        "PoU-FVR Triton\n(2.16 s | 1.35 GB)",
        "Hybrid PoU+nanodrr\n(1.34 s | 0.46 GB)",
        "nanodrr Triton\n(6.04 s | 0.36 GB)",
    ]
    m30_colors = ["#0B6E4F", "#2B59C3", "#D95F02"]
    data30 = [
        [float(r["final_mtre_mm"]) for r in exp30 if r["method"] == k]
        for k in m30_keys
    ]
    init30 = [float(r["init_mtre_mm"]) for r in exp30 if r["method"] == m30_keys[0]]

    bp = ax.boxplot(data30, positions=[0, 1, 2], widths=0.45, patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], m30_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.32)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.4)
    for med in bp["medians"]:
        med.set_color("#111111")
        med.set_linewidth(1.8)

    rng = np.random.default_rng(7)
    for i, (vals, col) in enumerate(zip(data30, m30_colors)):
        jitter = rng.uniform(-0.11, 0.11, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=col, s=26, zorder=4, edgecolors="white", linewidths=0.5)
        ax.text(i, max(vals) + 0.14, f"mean {np.mean(vals) + 1e-6:.2f} mm", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=col)

    ax.axhline(2.0, color="#b22222", linestyle="--", linewidth=1.1, label="2.0 mm clinical target")
    ax.text(2.85, 1.98, "nanodrr torch f32:\nOOM @ B=64", ha="center", va="top",
            fontsize=7.7, color="#8b0000", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdecea", edgecolor="#b22222", lw=0.8))
    ax.set_xlim(-0.55, 3.35)
    ax.set_ylim(0.25, 2.35)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(m30_labels, fontsize=7.9)
    ax.set_ylabel("Final 6-DoF mTRE (mm)")
    ax.set_title(f"(a) Biplanar $B=64$ Registration (init {np.mean(init30):.1f} mm)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    # --- Panel (b): exp31 Per-Subject DeepFluoro Clinical Cohort (n=18) ---
    ax = axes[1]
    _style_ax(ax)
    subjects = [f"subject0{i}" for i in range(1, 7)]
    m31_keys = [
        ("DiffDRR Single-Start (f32 B=1)", "DiffDRR Single-Start (50% <2mm)", "#C0392B"),
        ("PoU-FVR Triton Multi-Start (B=64)", "PoU-FVR Triton B=64 (100% <5mm)", "#2B59C3"),
        ("Hybrid PoU-Triton + DiffDRR polish", "Hybrid PoU+DiffDRR (100% <2mm)", "#0B6E4F"),
    ]
    sx = np.arange(len(subjects))
    width = 0.25
    for idx, (m_key, label, col) in enumerate(m31_keys):
        means = [
            float(np.mean([float(r["final_mtre_mm"]) for r in exp31 if r["subject"] == s and r["method"] == m_key]))
            for s in subjects
        ]
        maxs = [
            float(np.max([float(r["final_mtre_mm"]) for r in exp31 if r["subject"] == s and r["method"] == m_key]))
            for s in subjects
        ]
        bars = ax.bar(sx + (idx - 1) * width, means, width=width, color=col, alpha=0.85, label=label, edgecolor="white")
        for bx, mx in zip(bars, maxs):
            if mx > 5.0:
                ax.text(bx.get_x() + bx.get_width() / 2.0, bx.get_height() + 0.35, f"max\n{mx:.0f}",
                        ha="center", va="bottom", fontsize=6.6, color="#922B21")

    ax.axhline(2.0, color="#b22222", linestyle="--", linewidth=1.1)
    ax.set_xticks(sx)
    ax.set_xticklabels([f"S0{i}\n({[r['angle_deg'] for r in exp31 if r['subject']==f'subject0{i}'][0]}°)" for i in range(1, 7)], fontsize=7.8)
    ax.set_ylabel("Mean mTRE across 3 Inits (mm)")
    ax.set_title("(b) Real Clinical Cohort (6 DeepFluoro Patients)")
    ax.set_ylim(0, 17.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)

    # --- Panel (c): Cohort Time vs Accuracy Tradeoff ---
    ax = axes[2]
    _style_ax(ax)
    cohort_pts = [
        ("DiffDRR Single-Start (f32 B=1)", "DiffDRR Single-Start\n(7.22 mm | 50% <2mm | 5.49 GB)", "#C0392B", "^", (3.26, 7.22), (2.25, 6.2)),
        ("PoU-FVR Triton Multi-Start (B=64)", "PoU-FVR Triton B=64\n(1.79 mm | 100% <5mm | 1.35 GB)", "#2B59C3", "s", (2.11, 1.79), (2.25, 3.15)),
        ("Hybrid PoU-Triton + DiffDRR polish", "Hybrid PoU+DiffDRR\n(1.07 mm | 100% <2mm | 1.82 s)", "#0B6E4F", "o", (1.82, 1.07), (2.25, 0.35)),
    ]
    for m_key, lbl, col, mk, _, txt_xy in cohort_pts:
        rows = [r for r in exp31 if r["method"] == m_key]
        ts = [float(r["time_s"]) for r in rows]
        es = [float(r["final_mtre_mm"]) for r in rows]
        ax.scatter(ts, es, color=col, alpha=0.28, s=28, edgecolors="none")
        mx, my = float(np.mean(ts)), float(np.mean(es))
        ax.scatter([mx], [my], color=col, marker=mk, s=110, edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(lbl, xy=(mx, my), xytext=txt_xy, fontsize=7.6, fontweight="bold", color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9))

    ax.axhline(2.0, color="#b22222", linestyle="--", linewidth=1.1, label="2.0 mm clinical threshold")
    ax.set_xlabel("Mean Registration Wall Time on A100 (s)")
    ax.set_ylabel("Final 6-DoF mTRE (mm)")
    ax.set_title("(c) DeepFluoro Time–Accuracy Pareto ($n=18$)")
    ax.set_xlim(1.35, 4.55)
    ax.set_ylim(-0.65, 10.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    fig.tight_layout(w_pad=2.2)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    TEX_FIG.mkdir(parents=True, exist_ok=True)
    out_path = FIG_OUT / "fig30_gpu_multistart_cohort.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(out_path, TEX_FIG / out_path.name)
    return out_path


def main() -> None:
    p29 = make_fig29()
    p30 = make_fig30()
    print(f"Generated {p29} and {p30} (and copied to {TEX_FIG})")


if __name__ == "__main__":
    main()
