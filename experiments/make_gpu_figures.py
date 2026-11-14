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
import pandas as pd


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


def make_fig31() -> Path:
    """Generate publication figure from verified exp32 matched B=64, precision, & spine CT CSVs."""
    df_trials = _read_csv(OUT / "exp32_deepfluoro_matched_b64.csv")
    df_sum = _read_csv(OUT / "exp32_deepfluoro_summary.csv")
    prec = _read_csv(OUT / "exp32_precision_ablation.csv")
    spine = _read_csv(OUT / "exp32_spine_ct_summary.csv")

    plt.rcParams.update({
        "font.size": 9.2,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.2,
        "legend.fontsize": 7.6,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.75), dpi=300)

    # --- Panel (a): Matched B=64 DeepFluoro mTRE Boxplots + Speed/VRAM Labels (n=24) ---
    ax = axes[0]
    _style_ax(ax)
    methods_a = [
        ("DiffDRR Single-Start (f32 B=1)", "DiffDRR\nB=1 (f32)", "#C0392B"),
        ("nanodrr triton+compile (bf16 n=500, B=64)", "nanodrr\nB=64 (bf16)", "#E7298A"),
        ("PoU-FVR Triton (2x4x2@80 bf16, B=64)", "PoU-FVR\nB=64 (bf16)", "#2B59C3"),
        ("Hybrid PoU-Triton + nanodrr polish (<0.5GB)", "Hybrid+\nnanodrr", "#D95F02"),
        ("Hybrid PoU-Triton + DiffDRR polish", "Hybrid+\nDiffDRR", "#0B6E4F"),
    ]
    data_a = [
        [float(r["final_mtre_mm"]) for r in df_trials if r["method"] == m_key]
        for m_key, _, _ in methods_a
    ]
    bp = ax.boxplot(
        data_a,
        positions=np.arange(len(methods_a)),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#111111", linewidth=1.6),
        whiskerprops=dict(linewidth=1.1, color="#444444"),
        capprops=dict(linewidth=1.1, color="#444444"),
    )
    rng = np.random.default_rng(7)
    for idx, (patch, vals, (m_key, _, col)) in enumerate(zip(bp["boxes"], data_a, methods_a)):
        patch.set_facecolor(col)
        patch.set_alpha(0.32)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.3)
        jit = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(np.full(len(vals), idx) + jit, np.clip(vals, 0.0, 29.0), color=col, s=16, alpha=0.82, zorder=4)
        srow = [r for r in df_sum if r["subset"] == "All_DeepFluoro" and r["method"] == m_key][0]
        med_v = float(srow["median_mtre_mm"])
        reg_v = float(srow["mean_reg_s"])
        vram_v = float(srow["mean_peak_vram_mb"]) / 1024.0
        lbl_ys = [19.8, 11.8, 4.5, 6.8, 3.8]
        ax.text(
            idx, lbl_ys[idx], f"med {med_v:.2f}\n{reg_v:.2f}s | {vram_v:.2f}G",
            ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=col, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=col, alpha=0.92, lw=0.6),
        )

    ax.axhline(2.0, color="#b22222", linestyle="--", linewidth=1.0, label="2.0 mm threshold")
    ax.set_xticks(np.arange(len(methods_a)))
    ax.set_xticklabels([x[1] for x in methods_a], fontsize=7.6)
    ax.set_ylabel("Final 6-DoF mTRE (mm) — Median & IQR")
    ax.set_title("(a) Matched $B=64$ DeepFluoro ($n=24$, S01–S06)")
    ax.set_ylim(-0.6, 31.2)
    ax.text(
        0.98, 0.96,
        "DiffDRR B=64 probe (S01):\n184.1 s | 21.6 GB | OOM@64 (mb=4)\nWilcoxon Hybrid+DiffDRR:\np=1.5e-4 vs B=1 | p=2.3e-4 vs nano",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color="#333333",
        bbox=dict(boxstyle="round,pad=0.28", fc="#f8f9fa", ec="#bbbbbb", lw=0.7),
    )

    # --- Panel (b): Stratified Tier 1 vs Tier 2 Wide-Basin + f32/bf16 Inset ---
    ax = axes[1]
    _style_ax(ax)
    m_b = [
        ("DiffDRR Single-Start (f32 B=1)", "DiffDRR B=1", "#C0392B"),
        ("nanodrr triton+compile (bf16 n=500, B=64)", "nanodrr B=64", "#E7298A"),
        ("PoU-FVR Triton (2x4x2@80 bf16, B=64)", "PoU-FVR B=64", "#2B59C3"),
        ("Hybrid PoU-Triton + nanodrr polish (<0.5GB)", "Hybrid+nano", "#D95F02"),
        ("Hybrid PoU-Triton + DiffDRR polish", "Hybrid+DiffDRR", "#0B6E4F"),
    ]
    bx = np.arange(len(m_b))
    bw = 0.36
    t1_s2 = [float([r for r in df_sum if r["subset"] == "Tier1_Bounded" and r["method"] == k][0]["success_2mm_pct"]) for k, _, _ in m_b]
    t2_s2 = [float([r for r in df_sum if r["subset"] == "Tier2_WideBasin" and r["method"] == k][0]["success_2mm_pct"]) for k, _, _ in m_b]
    t2_s5 = [float([r for r in df_sum if r["subset"] == "Tier2_WideBasin" and r["method"] == k][0]["success_5mm_pct"]) for k, _, _ in m_b]

    bars1 = ax.bar(bx - bw / 2, t1_s2, width=bw, color="#34495E", alpha=0.88, label="Tier 1 Bounded (<2 mm)")
    bars2 = ax.bar(bx + bw / 2, t2_s2, width=bw, color="#1B9E77", alpha=0.88, label="Tier 2 Wide-Basin (<2 mm)")
    ax.plot(bx + bw / 2, t2_s5, "D--", color="#0B6E4F", markersize=4.5, lw=1.2, label="Tier 2 Gross Lock (<5 mm)")

    for bar, v in zip(bars1, t1_s2):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.7, color="#2C3E50", fontweight="bold")
    for bar, v in zip(bars2, t2_s2):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.7, color="#0B6E4F", fontweight="bold")

    ax.set_xticks(bx)
    ax.set_xticklabels([x[1] for x in m_b], rotation=18, ha="right", fontsize=7.6)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("(b) Tier 1 Bounded vs. Tier 2 Wide-Basin (Unanchored)")
    ax.set_ylim(0, 128)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=6.9)

    # Small inset comparing f32 vs bf16 mTRE parity placed cleanly above PoU-FVR/Hybrid+nano bars
    f32_vals = [float(r["final_mtre_mm"]) for r in prec if r["stage"] == "Reg_Tier1_Bounded" and r["precision"] == "float32"]
    bf16_vals = [float(r["final_mtre_mm"]) for r in prec if r["stage"] == "Reg_Tier1_Bounded" and r["precision"] == "bfloat16"]
    max_diff = max(abs(a - b) for a, b in zip(f32_vals, bf16_vals))
    ax.text(
        0.64, 0.96,
        f"f32 vs bf16 Parity (S01–S06):\nmean {np.mean(f32_vals):.3f} vs {np.mean(bf16_vals):.3f} mm\nmax |Δ| = {max_diff:.3f} mm",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.7, color="#1A3A6E",
        bbox=dict(boxstyle="round,pad=0.24", fc="#EEF4FF", ec="#2B59C3", lw=0.7),
    )

    # --- Panel (c): Spine & Abdominal CT Cone-Beam GPU Evaluation (exp32c) ---
    ax = axes[2]
    _style_ax(ax)
    cohorts_c = [
        ("Kaggle_Lumbar_Spine_CT", 256, "Kaggle\nLumbar 256²"),
        ("Kaggle_Lumbar_Spine_CT", 512, "Kaggle\nLumbar 512²"),
        ("AMOS_Thoracolumbar_CT", 256, "AMOS 0001\nTh-Lumbar"),
        ("GE_Clinical_CT", 256, "GE Clinical\nQA CT"),
    ]
    m_c = [
        ("PoU-FVR Triton (2x4x2@80 bf16, B=64)", "PoU-FVR B=64", "#2B59C3"),
        ("Hybrid PoU-Triton + nanodrr polish (<0.5GB)", "Hybrid+nano (<0.5G)", "#D95F02"),
        ("nanodrr triton+compile (bf16 n=500, B=64)", "nanodrr B=64", "#E7298A"),
    ]
    cx = np.arange(len(cohorts_c))
    cw = 0.25
    for midx, (m_key, lbl, col) in enumerate(m_c):
        vals, s2s = [], []
        for ds, hval, _ in cohorts_c:
            match = [r for r in spine if r["dataset"] == ds and int(r["H"]) == hval and r["method"] == m_key]
            vals.append(float(match[0]["median_mtre_mm"]) if match else np.nan)
            s2s.append(float(match[0]["success_2mm_pct"]) if match else np.nan)
        offs = (midx - 1) * cw
        bars = ax.bar(cx + offs, np.nan_to_num(vals, nan=0.0), width=cw, color=col, alpha=0.86, label=lbl)
        for bx_i, v, s2 in zip(bars, vals, s2s):
            if np.isnan(v):
                ax.text(bx_i.get_x() + bx_i.get_width() / 2, 0.2, "N/A", ha="center", va="bottom", fontsize=6.4, color="#666666")
            else:
                ax.text(bx_i.get_x() + bx_i.get_width() / 2, v + 0.18, f"{v:.2f}\n({s2:.0f}%)",
                        ha="center", va="bottom", fontsize=6.1, fontweight="bold", color=col)

    ax.axhline(2.0, color="#b22222", linestyle="--", linewidth=1.0)
    ax.set_xticks(cx)
    ax.set_xticklabels([c[2] for c in cohorts_c], fontsize=7.6)
    ax.set_ylabel("Median 6-DoF mTRE (mm) & <2 mm Rate")
    ax.set_title("(c) Clinical Spine & Abdominal CT Cone-Beam ($B=64$)")
    ax.set_ylim(0, 12.2)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.2)
    ax.text(
        0.52, 0.96,
        "Kaggle 256² DiffDRR B=1:\nmed 4.83 mm (16.7% <2mm)\nvs PoU-FVR: 0.88 mm (100%)",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.7, color="#922B21",
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFF5F5", ec="#C0392B", lw=0.6),
    )

    fig.tight_layout(w_pad=2.0)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    TEX_FIG.mkdir(parents=True, exist_ok=True)
    out_path = FIG_OUT / "fig31_rebuttal_ablation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(out_path, TEX_FIG / out_path.name)
    return out_path


def make_fig32() -> Path:
    """Publication Fig 32: Exp33 soft-spot fixes (4x4x4 PoU slabs & K=4 <0.5GB Hybrid polish)."""
    df32 = pd.read_csv(OUT / "exp32_deepfluoro_summary.csv")
    df33 = pd.read_csv(OUT / "exp33_deepfluoro_summary.csv")
    ct32 = pd.read_csv(OUT / "exp32_spine_ct_summary.csv")
    ct33 = pd.read_csv(OUT / "exp33_spine_ct_summary.csv")

    def _row(df, col, val, method):
        sub = df[(df[col] == val) & (df["method"] == method)]
        return sub.iloc[0]

    def _ct_row(df, ds, H, method):
        sub = df[(df["dataset"] == ds) & (df["H"] == H) & (df["method"] == method)]
        return sub.iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), dpi=300)
    plt.subplots_adjust(wspace=0.28, top=0.83, bottom=0.22, left=0.055, right=0.985)

    # Panel (a): DeepFluoro N=24 before/after fixes
    ax = axes[0]
    pairs = [
        ("Standalone\nPoU-FVR", _row(df32, "subset", "All_DeepFluoro", "PoU-FVR Triton (2x4x2@80 bf16, B=64)"), _row(df33, "subset", "All_DeepFluoro", "PoU-FVR Triton (4x4x4@64 bf16, B=64)")),
        ("Hybrid+nano\n(<0.5 GB)", _row(df32, "subset", "All_DeepFluoro", "Hybrid PoU-Triton + nanodrr polish (<0.5GB)"), _row(df33, "subset", "All_DeepFluoro", "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)")),
        ("Hybrid+\nDiffDRR", _row(df32, "subset", "All_DeepFluoro", "Hybrid PoU-Triton + DiffDRR polish"), _row(df33, "subset", "All_DeepFluoro", "Hybrid PoU-Triton (K=4) + DiffDRR polish")),
    ]
    x = np.arange(len(pairs))
    w = 0.34
    v32 = [float(p[1]["median_mtre_mm"]) for p in pairs]
    v33 = [float(p[2]["median_mtre_mm"]) for p in pairs]
    s32 = [float(p[1]["success_2mm_pct"]) for p in pairs]
    s33 = [float(p[2]["success_2mm_pct"]) for p in pairs]
    ax.bar(x - w / 2, v32, width=w, color="#9ca3af", edgecolor="#374151", linewidth=0.8, label="exp32 (2x4x2@80 / K=1)")
    ax.bar(x + w / 2, v33, width=w, color="#0f766e", edgecolor="#134e4a", linewidth=0.8, label="exp33 (4x4x4@64 / K=4)")
    for i in range(len(pairs)):
        ax.text(x[i] - w / 2, v32[i] + 0.06, f"{v32[i]:.2f}\n({s32[i]:.0f}%)", ha="center", va="bottom", fontsize=7.5, color="#374151")
        ax.text(x[i] + w / 2, v33[i] + 0.06, f"{v33[i]:.2f}\n({s33[i]:.0f}%)", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0f766e")
    ax.axhline(2.0, color="#b91c1c", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=8.5)
    ax.set_ylabel("Median 6-DoF mTRE (mm) & <2 mm Rate", fontsize=9)
    ax.set_ylim(0, 3.15)
    ax.set_title("(a) DeepFluoro (N=24): Soft-Spot Fixes", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92)

    # Panel (b): Standalone PoU-FVR (2x4x2@80 vs 4x4x4@64) across Clinical CTs
    ax = axes[1]
    cohorts = [
        ("Kaggle\n256²", "Kaggle_Lumbar_Spine_CT", 256),
        ("Kaggle\n512²", "Kaggle_Lumbar_Spine_CT", 512),
        ("AMOS\n256²", "AMOS_Thoracolumbar_CT", 256),
        ("GE CT\n256²", "GE_Clinical_CT", 256),
    ]
    xc = np.arange(len(cohorts))
    c32 = [_ct_row(ct32, ds, H, "PoU-FVR Triton (2x4x2@80 bf16, B=64)") for _, ds, H in cohorts]
    c33 = [_ct_row(ct33, ds, H, "PoU-FVR Triton (4x4x4@64 bf16, B=64)") for _, ds, H in cohorts]
    m32 = [float(r["median_mtre_mm"]) for r in c32]
    m33 = [float(r["median_mtre_mm"]) for r in c33]
    p32 = [float(r["success_2mm_pct"]) for r in c32]
    p33 = [float(r["success_2mm_pct"]) for r in c33]
    ax.bar(xc - w / 2, m32, width=w, color="#f59e0b", edgecolor="#92400e", linewidth=0.8, label="Standalone PoU 2x4x2@80")
    ax.bar(xc + w / 2, m33, width=w, color="#2563eb", edgecolor="#1e3a8a", linewidth=0.8, label="Standalone PoU 4x4x4@64")
    for i in range(len(cohorts)):
        ax.text(xc[i] - w / 2, m32[i] + 0.18, f"{m32[i]:.2f}\n({p32[i]:.0f}%)", ha="center", va="bottom", fontsize=7.2, color="#92400e")
        ax.text(xc[i] + w / 2, m33[i] + 0.18, f"{m33[i]:.2f}\n({p33[i]:.0f}%)", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1e3a8a")
    ax.axhline(2.0, color="#b91c1c", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(xc)
    ax.set_xticklabels([c[0] for c in cohorts], fontsize=8.5)
    ax.set_ylabel("Median mTRE (mm) & <2 mm Rate", fontsize=9)
    ax.set_ylim(0, 11.6)
    ax.set_title("(b) Standalone PoU-FVR: Isotropic 4x4x4 Slabs", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)

    # Panel (c): <0.5 GB Hybrid+nanodrr (K=1 vs K=4 f32 polish) across all cohorts
    ax = axes[2]
    all_c = [
        ("DeepFluoro\n(N=24, 464MB)", float(_row(df32, "subset", "All_DeepFluoro", "Hybrid PoU-Triton + nanodrr polish (<0.5GB)")["median_mtre_mm"]),
         float(_row(df33, "subset", "All_DeepFluoro", "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)")["median_mtre_mm"]),
         float(_row(df33, "subset", "All_DeepFluoro", "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)")["success_2mm_pct"])),
        ("Kaggle 256²\n(464MB)", float(_ct_row(ct32, "Kaggle_Lumbar_Spine_CT", 256, "Hybrid PoU-Triton + nanodrr polish (<0.5GB)")["median_mtre_mm"]),
         float(_ct_row(ct33, "Kaggle_Lumbar_Spine_CT", 256, "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)")["median_mtre_mm"]),
         float(_ct_row(ct33, "Kaggle_Lumbar_Spine_CT", 256, "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)")["success_2mm_pct"])),
        ("Kaggle 512²\n(908MB)", float(_ct_row(ct32, "Kaggle_Lumbar_Spine_CT", 512, "Hybrid PoU-Triton + nanodrr polish (<0.5GB)")["median_mtre_mm"]),
         float(_ct_row(ct33, "Kaggle_Lumbar_Spine_CT", 512, "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)")["median_mtre_mm"]),
         float(_ct_row(ct33, "Kaggle_Lumbar_Spine_CT", 512, "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)")["success_2mm_pct"])),
        ("AMOS 256²\n(465MB)", float(_ct_row(ct32, "AMOS_Thoracolumbar_CT", 256, "Hybrid PoU-Triton + nanodrr polish (<0.5GB)")["median_mtre_mm"]),
         float(_ct_row(ct33, "AMOS_Thoracolumbar_CT", 256, "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)")["median_mtre_mm"]),
         float(_ct_row(ct33, "AMOS_Thoracolumbar_CT", 256, "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)")["success_2mm_pct"])),
        ("GE CT 256²\n(464MB)", float(_ct_row(ct32, "GE_Clinical_CT", 256, "Hybrid PoU-Triton + nanodrr polish (<0.5GB)")["median_mtre_mm"]),
         float(_ct_row(ct33, "GE_Clinical_CT", 256, "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)")["median_mtre_mm"]),
         float(_ct_row(ct33, "GE_Clinical_CT", 256, "Hybrid PoU-Triton (K=4) + nanodrr f32 polish (<0.5GB)")["success_2mm_pct"])),
    ]
    xh = np.arange(len(all_c))
    h32 = [c[1] for c in all_c]
    h33 = [c[2] for c in all_c]
    s_h = [c[3] for c in all_c]
    ax.bar(xh - w / 2, h32, width=w, color="#9ca3af", edgecolor="#374151", linewidth=0.8, label="Hybrid+nano K=1 (exp32)")
    ax.bar(xh + w / 2, h33, width=w, color="#7c3aed", edgecolor="#4c1d95", linewidth=0.8, label="Hybrid+nano K=4 f32 (exp33)")
    for i in range(len(all_c)):
        ax.text(xh[i] - w / 2, h32[i] + 0.08, f"{h32[i]:.2f}", ha="center", va="bottom", fontsize=7.2, color="#374151")
        ax.text(xh[i] + w / 2, h33[i] + 0.08, f"{h33[i]:.2f}\n({s_h[i]:.0f}%)", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color="#4c1d95")
    ax.axhline(2.0, color="#b91c1c", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(xh)
    ax.set_xticklabels([c[0] for c in all_c], fontsize=7.6)
    ax.set_ylabel("Median mTRE (mm) & <2 mm Rate", fontsize=9)
    ax.set_ylim(0, 6.0)
    ax.set_title("(c) Low-VRAM Hybrid+nanodrr: K=4 f32 Polish", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92)

    for a in axes:
        _style_ax(a)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    TEX_FIG.mkdir(parents=True, exist_ok=True)
    out_path = FIG_OUT / "fig32_softspot_fixes.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(out_path, TEX_FIG / out_path.name)
    return out_path


def main() -> None:
    p29 = make_fig29()
    p30 = make_fig30()
    p31 = make_fig31()
    p32 = make_fig32()
    print(f"Generated {p29}, {p30}, {p31}, and {p32} (and copied to {TEX_FIG})")


if __name__ == "__main__":
    main()
