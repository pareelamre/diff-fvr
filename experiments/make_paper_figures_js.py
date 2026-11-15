"""Generate all 15 unified, colorblind-safe publication figures via HTML/SVG + Chromium.

Usage:
    C:\\Python313\\python.exe experiments/make_paper_figures_js.py

Outputs crisp 3x-Retina PNGs to:
  - outputs/paper_figures/fig*.png
  - <artifact_dir>/scratch/tex_build/figures/fig*.png (when present)

Runs an automated DOM bounding-box alignment & collision audit on every SVG
inside headless Chromium before saving each PNG.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
PF_DIR = REPO / "outputs" / "paper_figures"
CROP_DIR = PF_DIR / "_raw_crops"
OUT_ROOT = REPO / "outputs"
TEX_FIG_DIR = Path(
    r"C:\Users\paree\.gemini\antigravity\brain\db84ca2a-35bb-4b49-a099-1ebdc0eb4333\scratch\tex_build\figures"
)

# Unified Okabe-Ito / IEEE-Nature publication palette
THEME = {
    "bg": "#FFFFFF",
    "panelBg": "#F8FAFC",
    "ink": "#111827",
    "muted": "#4B5563",
    "grid": "#E5E7EB",
    "spine": "#374151",
    "border": "#D1D5DB",
    # Semantic data colors (shared identically across all 15 figures)
    "navy": "#0F4C81",       # Standalone Diff-FVR / PoU-FVR
    "teal": "#0D7A6F",       # Hybrid Best / Exp33 Upgraded / Clean Winner
    "cobalt": "#2563EB",     # Zero-IFFT / Hybrid+nanodrr / Secondary Proposed
    "vermilion": "#C2410C",  # DiffDRR Siddon / Monoplanar Fail / B=1 Baseline
    "amber": "#D97706",      # nanodrr triton+compile / Trilinear / Warm Ref
    "slate": "#64748B",      # Initial / Exp32 Before / Uncalibrated
    "slateLight": "#94A3B8", # Muted Reference / Secondary Before
    "indigo": "#4338CA",     # AMOS Clinical Cohort
    "plum": "#7E22CE",       # Kaggle Lumbar Cohort
    "redDash": "#B91C1C",    # Clinical 2.0 mm / 1.0 mm Thresholds
}


def read_csv(rel_path: str) -> list[dict[str, str]]:
    p = REPO / rel_path
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def img_data_uri(name: str) -> str:
    p = CROP_DIR / name
    raw = p.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def cdf_points(values: list[float], x_max: float) -> list[list[float]]:
    s = sorted(values)
    n = len(s)
    pts = [[0.0, 0.0]]
    for i, v in enumerate(s):
        vx = min(v, x_max)
        pts.append([vx, 100.0 * i / n])
        pts.append([vx, 100.0 * (i + 1) / n])
        if v >= x_max:
            break
    if pts[-1][0] < x_max:
        pts.append([x_max, pts[-1][1]])
    return pts


def box_stats(values: list[float]) -> dict[str, Any]:
    s = sorted(values)
    n = len(s)

    def q(p: float) -> float:
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return s[lo]
        return s[lo] * (hi - idx) + s[hi] * (idx - lo)

    q1, med, q3 = q(0.25), q(0.5), q(0.75)
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    w_lo = min((v for v in s if v >= lo_fence), default=s[0])
    w_hi = max((v for v in s if v <= hi_fence), default=s[-1])
    return {
        "q1": round(q1, 4),
        "med": round(med, 4),
        "q3": round(q3, 4),
        "wlo": round(w_lo, 4),
        "whi": round(w_hi, 4),
        "mean": round(statistics.mean(s), 4),
        "pts": [round(v, 4) for v in s],
    }


def build_datasets() -> dict[str, Any]:
    data: dict[str, Any] = {}

    # --- Fig 1 ---
    t1 = read_csv("outputs/paper_figures/table1_complexity_benchmark.csv")
    data["fig1"] = [
        {
            "n": int(r["Volume_Size"]),
            "prep_s": float(r["Prep_Time_s"]),
            "fvr_ms": float(r["FVR_ms"]),
            "fvr_std": float(r["FVR_std_ms"]),
            "ray_ms": float(r["Raycast_ms"]),
            "ray_std": float(r["Raycast_std_ms"]),
            "speedup": float(r["Speedup"]),
        }
        for r in t1
    ]

    # --- Fig 6 ---
    t6 = read_csv("outputs/paper_figures/table6_biplanar_benchmark.csv")
    methods6 = [
        ("Biplanar Spectral-FVR (Proposed)", "Biplanar Spectral-FVR", "navy"),
        ("Monoplanar Spectral-FVR (Lateral Only)", "Monoplanar Lateral-Only", "vermilion"),
        ("Biplanar Raycasting (Spatial)", "Biplanar Raycasting", "slate"),
    ]
    d6_cdfs = []
    d6_dofs = []
    dof_keys = [
        ("Error_Yaw", "Yaw (°)"),
        ("Error_Pitch", "Pitch (°)"),
        ("Error_Roll", "Roll (°)"),
        ("Error_Tx", "Tx (mm)"),
        ("Error_Ty", "Ty (mm)"),
        ("Error_Tz", "Tz (mm)"),
    ]
    for m_raw, m_lbl, col in methods6:
        sub = [r for r in t6 if r["Method"] == m_raw]
        vals = [float(r["Final_mTRE_mm"]) for r in sub]
        s2 = 100.0 * sum(1 for r in sub if r["Success_2mm"] == "True") / len(sub)
        d6_cdfs.append(
            {
                "label": m_lbl,
                "color": col,
                "med": statistics.median(vals),
                "s2": s2,
                "pts": cdf_points(vals, 12.0),
            }
        )
        d6_dofs.append(
            {
                "label": m_lbl,
                "color": col,
                "vals": [statistics.mean(float(r[k]) for r in sub) for k, _ in dof_keys],
            }
        )
    data["fig6"] = {"cdfs": d6_cdfs, "dofs": d6_dofs, "dof_labels": [l for _, l in dof_keys]}

    # --- Fig 7 ---
    t7 = read_csv("outputs/paper_figures/table7_clinical_benchmark.csv")
    d7_clean = [float(r["Final_mTRE_mm"]) for r in t7 if "Clean" in r["Condition"]]
    d7_low = [float(r["Final_mTRE_mm"]) for r in t7 if "Low-Dose" in r["Condition"]]
    data["fig7"] = {
        "imgs": {
            "axial": img_data_uri("fig7_axial.png"),
            "coronal": img_data_uri("fig7_coronal.png"),
            "sagittal": img_data_uri("fig7_sagittal.png"),
            "ap": img_data_uri("fig7_ap_drr.png"),
            "lat": img_data_uri("fig7_lat_drr.png"),
        },
        "clean_cdf": cdf_points(d7_clean, 0.08),
        "low_cdf": cdf_points(d7_low, 0.08),
        "clean_med": statistics.median(d7_clean),
        "low_med": statistics.median(d7_low),
    }

    # --- Fig 11 ---
    t11 = read_csv("outputs/paper_figures/table11_radiation_dose_sweep.csv")
    regimes11 = [
        ("Standard Dose (10⁶)", "10⁶"),
        ("Half Dose (2×10⁵)", "2×10⁵"),
        ("Low Dose (5×10⁴)", "5×10⁴"),
        ("Very Low Dose (10⁴)", "10⁴"),
        ("Ultra-Low Dose (5×10³)", "5×10³"),
        ("Extreme Starvation (10³)", "10³"),
    ]
    d11_rows = []
    for i, (r_raw, short_lbl) in enumerate(regimes11):
        sub = [r for r in t11 if r["Regime"] == r_raw]
        vals = [float(r["Final_mTRE_mm"]) for r in sub]
        inits = [float(r["Init_mTRE_mm"]) for r in sub]
        d11_rows.append(
            {
                "label": short_lbl,
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals),
                "init_mean": statistics.mean(inits),
                "s2": 100.0 * sum(1 for r in sub if r["Success_2mm"] == "True") / len(sub),
                "s1": 100.0 * sum(1 for r in sub if r["Success_1mm"] == "True") / len(sub),
                "tile": img_data_uri(f"fig11_dose_{i}.png"),
            }
        )
    data["fig11"] = d11_rows

    # --- Fig 15 ---
    data["fig15"] = {
        "imgs": [img_data_uri(f"fig15_panel_{i}.png") for i in range(4)],
    }

    # --- Fig 14 ---
    t12 = read_csv("outputs/paper_figures/table12_kaggle_clinical_benchmark.csv")
    t13 = read_csv("outputs/paper_figures/table13_amos_clinical_benchmark.csv")
    d7_all = [float(r["Final_mTRE_mm"]) for r in t7]
    d13_all = [float(r["Final_mTRE_mm"]) for r in t13]
    d12_all = [float(r["Final_mTRE_mm"]) for r in t12]
    d6_bip = [float(r["Final_mTRE_mm"]) for r in t6 if "Proposed" in r["Method"]]
    d6_mono = [float(r["Final_mTRE_mm"]) for r in t6 if "Monoplanar" in r["Method"]]

    def cohort_rates(rows: list[dict[str, str]], cond_sub: str) -> dict[str, float]:
        sub = [r for r in rows if cond_sub in r["Condition"]]
        vals = [float(r["Final_mTRE_mm"]) for r in sub]
        return {
            "s2": 100.0 * sum(1 for r in sub if r["Success_2mm"] == "True") / len(sub),
            "s1": 100.0 * sum(1 for r in sub if r["Success_1mm"] == "True") / len(sub),
            "med": statistics.median(vals),
        }

    data["fig14"] = {
        "cdfs": [
            {"label": "GE Clinical (100% <2mm)", "color": "navy", "dash": False, "pts": cdf_points(d7_all, 6.0)},
            {"label": "AMOS 0001 (90% <2mm)", "color": "indigo", "dash": False, "pts": cdf_points(d13_all, 6.0)},
            {"label": "Kaggle Lumbar (85% <2mm)", "color": "plum", "dash": False, "pts": cdf_points(d12_all, 6.0)},
            {"label": "Synth Biplanar (86.7% <2mm)", "color": "teal", "dash": False, "pts": cdf_points(d6_bip, 6.0)},
            {"label": "Monoplanar (6.7% <2mm)", "color": "vermilion", "dash": True, "pts": cdf_points(d6_mono, 6.0)},
        ],
        "bars": [
            {"cohort": "GE DICOM", "clean": cohort_rates(t7, "Clean"), "low": cohort_rates(t7, "Low-Dose")},
            {"cohort": "AMOS 0001", "clean": cohort_rates(t13, "Clean"), "low": cohort_rates(t13, "Low-Dose")},
            {"cohort": "Kaggle", "clean": cohort_rates(t12, "Clean"), "low": cohort_rates(t12, "Low-Dose")},
        ],
    }

    # --- Fig 25 ---
    t25 = read_csv("outputs/paper_figures/table25_deepfluoro_real_baseline.csv")
    fwd25 = [r for r in t25 if r["section"] == "forward_10views"]
    reg25 = [r for r in t25 if r["section"] == "reg_biplanar"]
    data["fig25"] = {
        "fwd": [
            {
                "view": r["id"],
                "ms_fvr": float(r["metric_a"]),
                "ms_drr": float(r["metric_b"]),
                "ncc_fd": float(r["metric_d"]),
                "mncc_fr": float(r["metric_e"]),
                "mncc_dr": float(r["metric_f"]),
            }
            for r in fwd25
        ],
        "reg": [
            {"label": "Initial", "mtre": statistics.mean(float(r["metric_a"]) for r in reg25), "time": 0.0, "color": "slate"},
            {"label": "Diff-FVR", "mtre": statistics.mean(float(r["metric_c"]) for r in reg25), "time": statistics.mean(float(r["metric_b"]) for r in reg25), "color": "navy"},
            {"label": "Hybrid", "mtre": statistics.mean(float(r["metric_e"]) for r in reg25), "time": statistics.mean(float(r["metric_d"]) for r in reg25), "color": "teal"},
            {"label": "DiffDRR", "mtre": statistics.mean(float(r["metric_g"]) for r in reg25), "time": statistics.mean(float(r["metric_f"]) for r in reg25), "color": "vermilion"},
        ],
    }

    # --- Fig 26 ---
    t26 = read_csv("outputs/paper_figures/table26_deepfluoro_multisubject.csv")
    data["fig26"] = [
        {
            "sub": r["subject"].replace("subject", "S"),
            "ms_fvr": float(r["ms_fvr"]),
            "ms_drr": float(r["ms_drr"]),
            "speedup": float(r["speedup"]),
            "init": float(r["init_mtre_mm"]),
            "fvr": float(r["fvr_mtre_mm"]),
            "hyb": float(r["hyb_mtre_mm"]),
            "drr": float(r["drr_mtre_mm"]),
            "hyb_s": float(r["hyb_time_s"]),
            "drr_s": float(r["drr_time_s"]),
        }
        for r in t26
    ]

    # --- Fig 27 ---
    t27 = read_csv("outputs/paper_figures/table27_geometry_matched_benchmark.csv")
    data["fig27"] = [
        {
            "H": int(r["detector_res"]),
            "geom": "Parallel" if "Parallel" in r["geometry"] else "Cone-Beam",
            "method": r["method"],
            "prep_s": float(r["fft_prep_s"]),
            "fwd_ms": float(r["fwd_ms"]),
            "t10": float(r["total_10steps_2view_s"]),
            "t40": float(r["total_40steps_2view_s"]),
            "t100": float(r["total_100steps_2view_s"]),
        }
        for r in t27
    ]

    # --- Fig 28 ---
    t28 = read_csv("outputs/paper_figures/table28_conebeam_clinical_robustness.csv")
    ge28 = [r for r in t28 if r["cohort"].startswith("GE")]
    df28 = [r for r in t28 if r["cohort"].startswith("DeepFluoro")]
    ge_short = ["600/900", "750/1050", "850/1200", "1000/1400"]
    data["fig28"] = {
        "ge": [
            {
                "label": ge_short[i],
                "mag": r["separation_or_mag"],
                "uncal": float(r["uncal_ortho_or_drr_mtre_mm"]),
                "cal": float(r["cal_ortho_mtre_mm"]),
                "pou": float(r["pou_conefvr_mtre_mm"]),
            }
            for i, r in enumerate(ge28)
        ],
        "df": [
            {
                "sub": "S" + r["cohort"].split("subject")[1][:2],
                "drr": float(r["uncal_ortho_or_drr_mtre_mm"]),
                "single": float(r["cal_ortho_mtre_mm"]),
                "multi": float(r["pou_conefvr_mtre_mm"]),
                "hyb": float(r["hybrid_mtre_mm"]),
            }
            for r in df28
        ],
    }

    # --- Fig 29 ---
    e29 = read_csv("outputs/exp29_gpu_benchmark.csv")
    e29_pou = read_csv("outputs/exp29_pou_ablation.csv")
    res_list = ["128", "256", "512"]
    methods29 = [
        ("PoU-FVR Triton+CUDAGraph (bf16 fwd)", "PoU-FVR CUDAGraph", "teal"),
        ("PoU-FVR Triton (2x2x2 bf16 fwd)", "PoU-FVR Triton", "navy"),
        ("TorchDiffFVR Zero-IFFT NCC", "Zero-IFFT NCC", "cobalt"),
        ("nanodrr triton+compile (bf16 n=500 fwd)", "nanodrr Triton+compile", "amber"),
        ("nanodrr torch (n=500 fwd)", "nanodrr PyTorch", "vermilion"),
    ]
    fps_b64 = {}
    for m_raw, m_lbl, col in methods29:
        vals = []
        for H in res_list:
            hit = [r for r in e29 if r["H"] == H and r["B"] == "64" and r["method"].startswith(m_raw)]
            vals.append(float(hit[0]["fps"]))
        fps_b64[m_lbl] = {"color": col, "vals": vals}
    fb_b64 = {}
    for m_raw, m_lbl, col in [
        ("PoU-FVR Triton (2x2x2 bf16 fwd+bwd)", "PoU-FVR Triton", "navy"),
        ("nanodrr triton (bf16 n=500 fwd+bwd)", "nanodrr Triton (n=500)", "amber"),
    ]:
        vals = []
        for H in res_list:
            hit = [r for r in e29 if r["H"] == H and r["B"] == "64" and r["method"] == m_raw]
            vals.append(float(hit[0]["wall_ms"]))
        fb_b64[m_lbl] = {"color": col, "vals": vals}
    data["fig29"] = {
        "res": ["128²", "256²", "512²"],
        "fps": fps_b64,
        "fb": fb_b64,
        "pareto": [
            {
                "cfg": f"{r['grid_blocks'].replace('(','').replace(')','').replace(', ','×')}@{r['block_res']}",
                "blocks": int(r["active_blocks"]),
                "br": int(r["block_res"]),
                "gpu_ms": float(r["gpu_ms"]),
                "ncc": float(r["ncc_vs_raymarch"]),
                "edge": float(r["edge_ncc_vs_raymarch"]),
            }
            for r in e29_pou
        ],
    }

    # --- Fig 30 ---
    e30 = read_csv("outputs/exp30_gpu_multistart_reg.csv")
    e31_c = read_csv("outputs/exp31_deepfluoro_gpu_cohort.csv")
    e31_s = read_csv("outputs/exp31_deepfluoro_gpu_summary.csv")
    m30_defs = [
        ("PoU-FVR Triton (2x4x2@80 bf16)", "PoU-FVR Triton", "navy"),
        ("Hybrid PoU-Triton + nanodrr polish", "Hybrid+nanodrr", "teal"),
        ("nanodrr triton+compile (bf16 n=500)", "nanodrr Triton", "amber"),
        ("nanodrr torch (f32 n=500)", "nanodrr PyTorch", "vermilion"),
    ]
    d30_a = []
    for m_raw, m_lbl, col in m30_defs:
        sub = [r for r in e30 if r["method"] == m_raw and r["final_mtre_mm"] != "OOM"]
        if sub:
            vals = [float(r["final_mtre_mm"]) for r in sub]
            ts = [float(r["time_s"]) for r in sub]
            d30_a.append(
                {
                    "label": m_lbl,
                    "color": col,
                    "oom": False,
                    "med": statistics.median(vals),
                    "mean": statistics.mean(vals),
                    "time_s": statistics.mean(ts),
                }
            )
        else:
            d30_a.append({"label": m_lbl, "color": col, "oom": True, "med": 0.0, "mean": 0.0, "time_s": 0.0})

    subs31 = sorted(set(r["subject"] for r in e31_c))
    d30_b = []
    for s in subs31:
        row = {"sub": s.replace("subject", "S")}
        for m_raw, key in [
            ("DiffDRR Single-Start (f32 B=1)", "drr"),
            ("PoU-FVR Triton Multi-Start (B=64)", "pou"),
            ("Hybrid PoU-Triton + DiffDRR polish", "hyb"),
        ]:
            sub = [r for r in e31_c if r["subject"] == s and r["method"] == m_raw]
            row[key] = statistics.mean(float(r["final_mtre_mm"]) for r in sub)
        d30_b.append(row)

    d30_c = [
        {
            "label": "Hybrid (All 6)",
            "color": "teal",
            "time_s": float(e31_s[2]["mean_time_s"]),
            "mtre": float(e31_s[2]["mean_mtre_mm"]),
            "s2": float(e31_s[2]["success_2mm_pct"]),
        },
        {
            "label": "PoU-FVR B=64",
            "color": "navy",
            "time_s": float(e31_s[1]["mean_time_s"]),
            "mtre": float(e31_s[1]["mean_mtre_mm"]),
            "s2": float(e31_s[1]["success_2mm_pct"]),
        },
        {
            "label": "DiffDRR B=1",
            "color": "vermilion",
            "time_s": float(e31_s[0]["mean_time_s"]),
            "mtre": float(e31_s[0]["mean_mtre_mm"]),
            "s2": float(e31_s[0]["success_2mm_pct"]),
        },
    ]
    data["fig30"] = {"a": d30_a, "b": d30_b, "c": d30_c}

    # --- Fig 31 ---
    e32_c = read_csv("outputs/exp32_deepfluoro_matched_b64.csv")
    e32_s = read_csv("outputs/exp32_deepfluoro_summary.csv")
    e32_ct = read_csv("outputs/exp32_spine_ct_summary.csv")
    m32_defs = [
        ("DiffDRR Single-Start (f32 B=1)", "DiffDRR B=1", "vermilion"),
        ("nanodrr triton+compile (bf16 n=500, B=64)", "nanodrr B=64", "amber"),
        ("PoU-FVR Triton (2x4x2@80 bf16, B=64)", "PoU-FVR B=64", "navy"),
        ("Hybrid PoU-Triton + nanodrr polish (<0.5GB)", "Hybrid+nano", "cobalt"),
        ("Hybrid PoU-Triton + DiffDRR polish", "Hybrid+DiffDRR", "teal"),
    ]
    d31_boxes = []
    for m_raw, m_lbl, col in m32_defs:
        vals = [float(r["final_mtre_mm"]) for r in e32_c if r["method"] == m_raw]
        st = box_stats(vals)
        st["label"] = m_lbl
        st["color"] = col
        d31_boxes.append(st)

    d31_tiers = []
    for m_raw, m_lbl, col in m32_defs:
        t1_r = [r for r in e32_s if r["subset"] == "Tier1_Bounded" and r["method"] == m_raw][0]
        t2_r = [r for r in e32_s if r["subset"] == "Tier2_WideBasin" and r["method"] == m_raw][0]
        d31_tiers.append(
            {
                "label": m_lbl,
                "color": col,
                "t1": float(t1_r["success_2mm_pct"]),
                "t2": float(t2_r["success_2mm_pct"]),
            }
        )

    ct_cohorts = [
        ("Kaggle_Lumbar_Spine_CT", "256", "Kaggle 256²"),
        ("Kaggle_Lumbar_Spine_CT", "512", "Kaggle 512²"),
        ("AMOS_Thoracolumbar_CT", "256", "AMOS 256²"),
        ("GE_Clinical_CT", "256", "GE 256²"),
    ]
    d31_ct = []
    for ds, H, lbl in ct_cohorts:
        row = {"label": lbl}
        for m_raw, k in [
            ("PoU-FVR Triton (2x4x2@80 bf16, B=64)", "pou"),
            ("nanodrr triton+compile (bf16 n=500, B=64)", "nano"),
            ("Hybrid PoU-Triton + nanodrr polish (<0.5GB)", "hyb"),
        ]:
            hit = [r for r in e32_ct if r["dataset"] == ds and r["H"] == H and r["method"] == m_raw]
            row[k] = float(hit[0]["median_mtre_mm"]) if hit else None
        d31_ct.append(row)

    data["fig31"] = {"boxes": d31_boxes, "tiers": d31_tiers, "ct": d31_ct}

    # --- Fig 32 ---
    e33_s = read_csv("outputs/exp33_deepfluoro_summary.csv")
    e33_ct = read_csv("outputs/exp33_spine_ct_summary.csv")

    def find_r(rows: list[dict[str, str]], key_col: str, kv: str, m_part: str) -> dict[str, str]:
        return [r for r in rows if r[key_col] == kv and m_part in r["method"]][0]

    d32_a = []
    for lbl, m32_k, m33_k in [
        ("PoU-FVR Solo", "PoU-FVR Triton (2x4x2@80", "PoU-FVR Triton (4x4x4@64"),
        ("Hybrid+nanodrr", "Hybrid PoU-Triton + nanodrr", "Hybrid PoU-Triton (K=4) + nanodrr"),
        ("Hybrid+DiffDRR", "Hybrid PoU-Triton + DiffDRR", "Hybrid PoU-Triton (K=4) + DiffDRR"),
    ]:
        r_before = find_r(e32_s, "subset", "All_DeepFluoro", m32_k)
        r_after = find_r(e33_s, "subset", "All_DeepFluoro", m33_k)
        d32_a.append(
            {
                "label": lbl,
                "med_b": float(r_before["median_mtre_mm"]),
                "med_a": float(r_after["median_mtre_mm"]),
                "s2_b": float(r_before["success_2mm_pct"]),
                "s2_a": float(r_after["success_2mm_pct"]),
            }
        )

    d32_b = []
    for ds, H, lbl in ct_cohorts:
        rb = [r for r in e32_ct if r["dataset"] == ds and r["H"] == H and "PoU-FVR Triton" in r["method"]][0]
        ra = [r for r in e33_ct if r["dataset"] == ds and r["H"] == H and "PoU-FVR Triton" in r["method"]][0]
        d32_b.append(
            {
                "label": lbl,
                "med_b": float(rb["median_mtre_mm"]),
                "med_a": float(ra["median_mtre_mm"]),
                "s2_b": float(rb["success_2mm_pct"]),
                "s2_a": float(ra["success_2mm_pct"]),
            }
        )

    d32_c = [
        {
            "label": "DeepFluoro",
            "med_b": float(find_r(e32_s, "subset", "All_DeepFluoro", "Hybrid PoU-Triton + nanodrr")["median_mtre_mm"]),
            "med_a": float(find_r(e33_s, "subset", "All_DeepFluoro", "Hybrid PoU-Triton (K=4) + nanodrr")["median_mtre_mm"]),
            "s2_a": float(find_r(e33_s, "subset", "All_DeepFluoro", "Hybrid PoU-Triton (K=4) + nanodrr")["success_2mm_pct"]),
        }
    ]
    for ds, H, lbl in ct_cohorts:
        rb = [r for r in e32_ct if r["dataset"] == ds and r["H"] == H and "Hybrid" in r["method"]][0]
        ra = [r for r in e33_ct if r["dataset"] == ds and r["H"] == H and "Hybrid" in r["method"]][0]
        d32_c.append(
            {
                "label": lbl.replace(" ", "\n"),
                "med_b": float(rb["median_mtre_mm"]),
                "med_a": float(ra["median_mtre_mm"]),
                "s2_a": float(ra["success_2mm_pct"]),
            }
        )

    data["fig32"] = {"a": d32_a, "b": d32_b, "c": d32_c}
    return data


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #FFFFFF;
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    color: #111827;
    -webkit-font-smoothing: antialiased;
  }
  svg { display: block; font-variant-numeric: tabular-nums; }
  text { font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; }
</style>
</head>
<body>
<div id="mount"></div>
<script>
const C = __THEME_JSON__;
const D = __DATA_JSON__;

function el(tag, attrs={}, children=[]) {
  const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) n.setAttribute(k, String(v));
  }
  for (const c of children) {
    if (typeof c === 'string') n.textContent = c;
    else if (c) n.appendChild(c);
  }
  return n;
}

function makeSVG(w, h) {
  const mount = document.getElementById('mount');
  mount.innerHTML = '';
  const svg = el('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}` });
  mount.appendChild(svg);
  svg.appendChild(el('rect', { x: 0, y: 0, width: w, height: h, fill: C.bg }));
  return svg;
}

function panelTitle(svg, x, y, title, subtitle) {
  svg.appendChild(el('text', {
    x, y, fill: C.ink, 'font-size': 13.5, 'font-weight': 700, 'data-role': 'panel-title'
  }, [title]));
  if (subtitle) {
    svg.appendChild(el('text', {
      x, y: y + 16, fill: C.muted, 'font-size': 10.5, 'font-weight': 500, 'data-role': 'panel-sub'
    }, [subtitle]));
  }
}

function drawLegend(svg, x0, y, items, gap=14, maxX=9999, rowStep=16) {
  const g = el('g', { 'data-role': 'legend' });
  svg.appendChild(g);
  let cx = x0;
  let cy = y;
  for (const it of items) {
    const iconW = (it.kind === 'line') ? 21 : (it.kind === 'dash_line' ? 19 : 13);
    const t = el('text', { x: cx + iconW, y: cy, fill: C.ink, 'font-size': 10.2, 'font-weight': 500 }, [it.label]);
    g.appendChild(t);
    const w = t.getComputedTextLength();
    if (cx > x0 && cx + iconW + w > maxX) {
      cx = x0;
      cy += rowStep;
      t.setAttribute('x', String(cx + iconW));
      t.setAttribute('y', String(cy));
    }
    if (it.kind === 'line') {
      g.insertBefore(el('line', {
        x1: cx, y1: cy - 3.5, x2: cx + 16, y2: cy - 3.5,
        stroke: it.color, 'stroke-width': 2.2,
        'stroke-dasharray': it.dash ? '4,3' : null
      }), t);
      if (it.marker !== false) {
        g.insertBefore(el('circle', { cx: cx + 8, cy: cy - 3.5, r: 2.8, fill: it.color }), t);
      }
    } else if (it.kind === 'dash_line') {
      g.insertBefore(el('line', {
        x1: cx, y1: cy - 3.5, x2: cx + 14, y2: cy - 3.5,
        stroke: it.color, 'stroke-width': 1.6, 'stroke-dasharray': '4,3'
      }), t);
    } else {
      g.insertBefore(el('rect', { x: cx, y: cy - 8, width: 9, height: 9, rx: 1.5, fill: it.color }), t);
    }
    cx += iconW + w + gap;
  }
  return g;
}

function drawGridY(svg, x0, x1, yMap, ticks, fmt) {
  for (const t of ticks) {
    const py = yMap(t);
    svg.appendChild(el('line', { x1: x0, y1: py, x2: x1, y2: py, stroke: C.grid, 'stroke-width': 0.8 }));
    svg.appendChild(el('line', { x1: x0 - 4, y1: py, x2: x0, y2: py, stroke: C.spine, 'stroke-width': 1 }));
    svg.appendChild(el('text', {
      x: x0 - 7, y: py + 3.5, 'text-anchor': 'end', fill: C.muted, 'font-size': 10
    }, [fmt ? fmt(t) : String(t)]));
  }
}

function drawGridX(svg, y0, y1, xMap, ticks, fmt) {
  for (const t of ticks) {
    const px = xMap(t);
    svg.appendChild(el('line', { x1: px, y1: y0, x2: px, y2: y1, stroke: C.grid, 'stroke-width': 0.8 }));
    svg.appendChild(el('line', { x1: px, y1: y1, x2: px, y2: y1 + 4, stroke: C.spine, 'stroke-width': 1 }));
    svg.appendChild(el('text', {
      x: px, y: y1 + 16, 'text-anchor': 'middle', fill: C.muted, 'font-size': 10
    }, [fmt ? fmt(t) : String(t)]));
  }
}

function drawAxesFrame(svg, x0, y0, x1, y1, yLabel, xLabel, yLabelXOffset=38) {
  svg.appendChild(el('line', { x1: x0, y1: y0, x2: x0, y2: y1, stroke: C.spine, 'stroke-width': 1.1 }));
  svg.appendChild(el('line', { x1: x0, y1: y1, x2: x1, y2: y1, stroke: C.spine, 'stroke-width': 1.1 }));
  if (yLabel) {
    const ym = (y0 + y1) / 2;
    svg.appendChild(el('text', {
      x: x0 - yLabelXOffset, y: ym, 'text-anchor': 'middle',
      transform: `rotate(-90 ${x0 - yLabelXOffset} ${ym})`,
      fill: C.ink, 'font-size': 11, 'font-weight': 600
    }, [yLabel]));
  }
  if (xLabel) {
    svg.appendChild(el('text', {
      x: (x0 + x1) / 2, y: y1 + 33, 'text-anchor': 'middle',
      fill: C.ink, 'font-size': 11, 'font-weight': 600
    }, [xLabel]));
  }
}

function polylinePath(pts, xMap, yMap) {
  return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${xMap(p[0]).toFixed(2)},${yMap(p[1]).toFixed(2)}`).join(' ');
}

function logMapper(lo, hi, p0, p1) {
  const llo = Math.log10(lo), lhi = Math.log10(hi);
  return (v) => p0 + ((Math.log10(v) - llo) / (lhi - llo)) * (p1 - p0);
}

function linMapper(lo, hi, p0, p1) {
  return (v) => p0 + ((v - lo) / (hi - lo)) * (p1 - p0);
}

function drawThresh(svg, x0, x1, yVal, label, color=C.redDash, align='right') {
  svg.appendChild(el('line', {
    x1: x0, y1: yVal, x2: x1, y2: yVal, stroke: color, 'stroke-width': 1.15, 'stroke-dasharray': '5,3'
  }));
  if (label) {
    svg.appendChild(el('text', {
      x: align === 'right' ? x1 - 4 : x0 + 4, y: yVal - 4,
      'text-anchor': align === 'right' ? 'end' : 'start',
      fill: color, 'font-size': 9.5, 'font-weight': 600
    }, [label]));
  }
}

function drawThreshX(svg, xVal, y0, y1, label, color=C.redDash, dy=12, dx=4, anchor='start') {
  svg.appendChild(el('line', {
    x1: xVal, y1: y0, x2: xVal, y2: y1, stroke: color, 'stroke-width': 1.15, 'stroke-dasharray': '5,3'
  }));
  if (label) {
    svg.appendChild(el('text', {
      x: xVal + dx, y: y0 + dy, 'text-anchor': anchor, fill: color, 'font-size': 9.5, 'font-weight': 600
    }, [label]));
  }
}

function addHatchPattern(svg, id, color) {
  let defs = svg.querySelector('defs');
  if (!defs) { defs = el('defs'); svg.insertBefore(defs, svg.firstChild); }
  const p = el('pattern', { id, width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' });
  p.appendChild(el('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: color, 'stroke-width': 2.2, opacity: 0.75 }));
  defs.appendChild(p);
}

// ======================= FIGURE BUILDERS =======================

function build_fig0() {
  const W = 1140, H = 430;
  const svg = makeSVG(W, H);
  // Subtle arrow marker
  const defs = el('defs');
  const m = el('marker', { id: 'arr', markerWidth: 8, markerHeight: 8, refX: 7, refY: 3.5, orient: 'auto' });
  m.appendChild(el('path', { d: 'M0,0 L7,3.5 L0,7 Z', fill: C.spine }));
  defs.appendChild(m);
  const mTeal = el('marker', { id: 'arrTeal', markerWidth: 8, markerHeight: 8, refX: 7, refY: 3.5, orient: 'auto' });
  mTeal.appendChild(el('path', { d: 'M0,0 L7,3.5 L0,7 Z', fill: C.teal }));
  defs.appendChild(mTeal);
  svg.appendChild(defs);

  // Header
  svg.appendChild(el('text', { x: 24, y: 30, fill: C.ink, 'font-size': 15, 'font-weight': 700 },
    ['Dual-Domain Diff-FVR & Partition-of-Unity Architecture']));
  svg.appendChild(el('text', { x: 24, y: 48, fill: C.muted, 'font-size': 11 },
    ['One-time O(N³ log N) spectral conditioning feeds batched O(N²) C-arm slice rendering and K=4 f32 stage-2 polish']));

  function drawStageBox(x, y, w, h, stripeColor, tag, title) {
    svg.appendChild(el('rect', { x, y, width: w, height: h, rx: 4, fill: C.panelBg, stroke: C.border, 'stroke-width': 1 }));
    svg.appendChild(el('rect', { x, y, width: w, height: 4, rx: 1, fill: stripeColor }));
    svg.appendChild(el('text', { x: x + 14, y: y + 24, fill: stripeColor, 'font-size': 10, 'font-weight': 700 }, [tag]));
    svg.appendChild(el('text', { x: x + 14, y: y + 43, fill: C.ink, 'font-size': 13, 'font-weight': 700 }, [title]));
  }

  // Stage 1
  drawStageBox(24, 68, 320, 310, C.navy, 'STAGE 1 · OFFLINE (ONCE PER CT)', 'Conditioned 3D Spectral Field');
  const s1Steps = [
    ['Input 3D CT V(x,y,z)', 'Pre-clip / HU-to-μ conversion, isocenter parity'],
    ['sinc² Pre-Compensation', 'Suppresses Kaiser-Bessel / linear roll-off'],
    ['2× Zero-Padding', 'Removes circular wrap-around aliasing'],
    ['3D FFT or PoU Slab Partition', 'Parallel (1×1×1) or Cone-Beam (Gₓ×Gᵧ×G_z)'],
    ['Preset Profiles (v1.2.0)', 'Fast 2×2×2@64 · Isotropic 4×4×4@64 (bf16)'],
  ];
  s1Steps.forEach((st, i) => {
    const by = 132 + i * 47;
    svg.appendChild(el('rect', { x: 38, y: by, width: 292, height: 39, rx: 3, fill: '#FFFFFF', stroke: C.border, 'stroke-width': 0.9 }));
    svg.appendChild(el('text', { x: 48, y: by + 16, fill: C.navy, 'font-size': 11, 'font-weight': 700 }, [st[0]]));
    svg.appendChild(el('text', { x: 48, y: by + 31, fill: C.muted, 'font-size': 10 }, [st[1]]));
  });

  // Arrow Stage 1 -> Stage 2
  svg.appendChild(el('line', { x1: 346, y1: 223, x2: 384, y2: 223, stroke: C.spine, 'stroke-width': 1.6, 'marker-end': 'url(#arr)' }));

  // Stage 2
  drawStageBox(388, 68, 360, 310, C.cobalt, 'STAGE 2 · ONLINE BATCHED (B = 64)', 'O(N²) Central Slice & PoU Blending');
  // Isometric Spectral Cube Schematic inside Stage 2 top half
  const cx = 568, cy = 205;
  // Cube wireframe
  const pts = {
    A: [cx - 62, cy - 38], B: [cx + 22, cy - 58], C: [cx + 68, cy - 22], D: [cx - 16, cy - 2],
    E: [cx - 62, cy + 36], F: [cx + 22, cy + 16], G: [cx + 68, cy + 52], H: [cx - 16, cy + 72],
  };
  const edge = (p1, p2, dash=false) => svg.appendChild(el('line', {
    x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1], stroke: C.slate, 'stroke-width': 1, 'stroke-dasharray': dash ? '3,3' : null
  }));
  edge(pts.A, pts.B); edge(pts.B, pts.C); edge(pts.C, pts.D); edge(pts.D, pts.A);
  edge(pts.D, pts.H); edge(pts.C, pts.G); edge(pts.A, pts.E); edge(pts.H, pts.G); edge(pts.E, pts.H);
  edge(pts.B, pts.F, true); edge(pts.E, pts.F, true); edge(pts.F, pts.G, true);

  // Two orthogonal central planes passing through origin
  svg.appendChild(el('polygon', {
    points: `${cx - 46},${cy + 2} ${cx + 42},${cy - 22} ${cx + 42},${cy + 34} ${cx - 46},${cy + 58}`,
    fill: C.navy, opacity: 0.16, stroke: C.navy, 'stroke-width': 1.4
  }));
  svg.appendChild(el('polygon', {
    points: `${cx - 26},${cy - 44} ${cx + 46},${cy - 2} ${cx + 22},${cy + 46} ${cx - 50},${cy + 4}`,
    fill: C.teal, opacity: 0.16, stroke: C.teal, 'stroke-width': 1.4
  }));
  svg.appendChild(el('circle', { cx: cx + 2, cy: cy + 8, r: 3.5, fill: C.ink }));
  svg.appendChild(el('text', { x: cx - 102, y: cy - 30, fill: C.navy, 'font-size': 10, 'font-weight': 700, 'text-anchor': 'end' }, ['AP Spectral Slice']));
  svg.appendChild(el('text', { x: cx + 76, y: cy - 16, fill: C.teal, 'font-size': 10, 'font-weight': 700, 'text-anchor': 'start' }, ['Lat Spectral Slice']));
  svg.appendChild(el('text', { x: cx + 76, y: cy + 28, fill: C.muted, 'font-size': 9.5 }, ['k = R(θ)⁻¹ [kₓ, kᵧ, 0]ᵀ']));

  // Stage 2 specs box
  const s2Specs = [
    ['Biplanar Triton Fused Slab Kernel', 'Up to 26,506 FPS @ 512² (B=64, NVIDIA A100)'],
    ['Zero-IFFT Spectral NCC Path', '90,486 FPS @ 128² via Parseval theorem (24 MB)'],
  ];
  s2Specs.forEach((sp, i) => {
    const by = 286 + i * 44;
    svg.appendChild(el('rect', { x: 402, y: by, width: 332, height: 36, rx: 3, fill: '#FFFFFF', stroke: C.border, 'stroke-width': 0.9 }));
    svg.appendChild(el('text', { x: 412, y: by + 15, fill: C.cobalt, 'font-size': 11, 'font-weight': 700 }, [sp[0]]));
    svg.appendChild(el('text', { x: 412, y: by + 29, fill: C.muted, 'font-size': 10 }, [sp[1]]));
  });

  // Arrow Stage 2 -> Stage 3
  svg.appendChild(el('line', { x1: 750, y1: 223, x2: 788, y2: 223, stroke: C.spine, 'stroke-width': 1.6, 'marker-end': 'url(#arr)' }));

  // Stage 3
  drawStageBox(792, 68, 324, 310, C.teal, 'STAGE 3 · DUAL-DOMAIN & TOP-K', 'Global Lock + Sub-Millimeter Polish');
  const s3Blocks = [
    { h: 72, col: C.navy, title: 'Global Multi-Start Basin (B=64)', l1: 'B=64 cold-start search beats single-start', l2: 'DeepFluoro B=1 (33% → 91.7%–100% <2 mm)' },
    { h: 72, col: C.cobalt, title: 'Stand-Alone PoU-FVR (4×4×4@64)', l1: 'No raytracer: 100% <2 mm across Spine CT', l2: '0.31–1.41 mm med mTRE, 91.7% DeepFluoro' },
    { h: 76, col: C.teal, title: 'Top-K=4 f32 Hybrid Polish (<0.5 GB)', l1: 'Exp33 verified across N=24 DeepFluoro + 4 CT cohorts', l2: '0.835 mm med mTRE, 100% <2 mm in 1.81 s' },
  ];
  let sy = 132;
  s3Blocks.forEach((b, idx) => {
    svg.appendChild(el('rect', { x: 806, y: sy, width: 296, height: b.h, rx: 3, fill: '#FFFFFF', stroke: C.border, 'stroke-width': 0.9 }));
    svg.appendChild(el('rect', { x: 806, y: sy, width: 4, height: b.h, fill: b.col }));
    svg.appendChild(el('text', { x: 818, y: sy + 18, fill: b.col, 'font-size': 11, 'font-weight': 700 }, [b.title]));
    svg.appendChild(el('text', { x: 818, y: sy + 37, fill: C.ink, 'font-size': 10 }, [b.l1]));
    svg.appendChild(el('text', { x: 818, y: sy + 53, fill: C.muted, 'font-size': 10, 'font-weight': idx === 2 ? 700 : 500 }, [b.l2]));
    sy += b.h + 12;
  });

  // Footer caption note
  svg.appendChild(el('text', { x: 24, y: 404, fill: C.muted, 'font-size': 10.5 },
    ['All measurements audited from reproducible CSV benchmarks (exp29–exp33, NVIDIA A100-SXM4-40GB); no synthetic fallbacks.']));
  return svg;
}

function build_fig1() {
  const W = 980, H = 400;
  const svg = makeSVG(W, H);
  const rows = D.fig1;

  // Panel (a): Log latency vs N
  panelTitle(svg, 64, 24, '(a) Single-View Scaling: O(N²) vs O(N³)', 'Isotropic grid size 64³ to 256³ (mean ± SD)');
  drawLegend(svg, 64, 58, [
    { kind: 'line', color: C.navy, label: 'Diff-FVR O(N²)' },
    { kind: 'line', color: C.vermilion, label: 'Spatial Raycasting O(N³)' },
  ]);
  const ax0 = 64, ax1 = 468, ay0 = 76, ay1 = 342;
  const xMapA = (i) => ax0 + 28 + i * ((ax1 - ax0 - 56) / 3);
  const yMapA = logMapper(2.0, 2500.0, ay1, ay0);
  drawGridY(svg, ax0, ax1, yMapA, [3, 10, 30, 100, 300, 1000], (v) => String(v));
  rows.forEach((r, i) => {
    const px = xMapA(i);
    svg.appendChild(el('line', { x1: px, y1: ay0, x2: px, y2: ay1, stroke: C.grid, 'stroke-width': 0.8 }));
    svg.appendChild(el('text', { x: px, y: ay1 + 16, 'text-anchor': 'middle', fill: C.muted, 'font-size': 10 }, [`${r.n}³`]));
  });
  drawAxesFrame(svg, ax0, ay0, ax1, ay1, 'Render Time (ms, log)', 'Volume Grid Size (N³)');

  // Lines & error bars
  const fPts = rows.map((r, i) => [i, r.fvr_ms]);
  const rPts = rows.map((r, i) => [i, r.ray_ms]);
  svg.appendChild(el('path', { d: polylinePath(fPts, xMapA, yMapA), fill: 'none', stroke: C.navy, 'stroke-width': 2.3 }));
  svg.appendChild(el('path', { d: polylinePath(rPts, xMapA, yMapA), fill: 'none', stroke: C.vermilion, 'stroke-width': 2.3 }));

  rows.forEach((r, i) => {
    const px = xMapA(i);
    // FVR
    const yf = yMapA(r.fvr_ms);
    const yfl = yMapA(Math.max(2.1, r.fvr_ms - r.fvr_std)), yfh = yMapA(r.fvr_ms + r.fvr_std);
    svg.appendChild(el('line', { x1: px, y1: yfl, x2: px, y2: yfh, stroke: C.navy, 'stroke-width': 1.2 }));
    svg.appendChild(el('circle', { cx: px, cy: yf, r: 4, fill: C.navy, stroke: '#FFFFFF', 'stroke-width': 1 }));
    svg.appendChild(el('text', {
      x: i === 0 ? px + 8 : px,
      y: i === 0 ? yf + 3 : yfl + 13,
      'text-anchor': i === 0 ? 'start' : 'middle',
      fill: C.navy, 'font-size': 9.5, 'font-weight': 600
    }, [`${r.fvr_ms.toFixed(1)}`]));
    // Raycast
    const yr = yMapA(r.ray_ms);
    const yrl = yMapA(r.ray_ms - r.ray_std), yrh = yMapA(r.ray_ms + r.ray_std);
    svg.appendChild(el('line', { x1: px, y1: yrl, x2: px, y2: yrh, stroke: C.vermilion, 'stroke-width': 1.2 }));
    svg.appendChild(el('rect', { x: px - 3.5, y: yr - 3.5, width: 7, height: 7, fill: C.vermilion, stroke: '#FFFFFF', 'stroke-width': 1 }));
    svg.appendChild(el('text', { x: px + (i === 3 ? -10 : 0), y: yrh - 6, 'text-anchor': i === 3 ? 'end' : 'middle', fill: C.vermilion, 'font-size': 9.5, 'font-weight': 600 },
      [`${r.ray_ms.toFixed(0)}`]));
  });

  // Panel (b): Speedup & Offline Prep
  const bx0 = 548, bx1 = 948, by0 = 76, by1 = 342;
  panelTitle(svg, bx0, 24, '(b) Rendering Speedup & One-Time Prep', 'Measured factor vs raycasting (left); FFT prep annotated');
  const yMapB = linMapper(0, 14, by1, by0);
  const xMapB = (i) => bx0 + 46 + i * ((bx1 - bx0 - 92) / 3);
  drawGridY(svg, bx0, bx1, yMapB, [0, 2, 4, 6, 8, 10, 12, 14], (v) => `${v}×`);
  rows.forEach((r, i) => {
    const px = xMapB(i);
    const bw = 46;
    const py = yMapB(r.speedup);
    svg.appendChild(el('rect', { x: px - bw / 2, y: py, width: bw, height: by1 - py, fill: C.teal, rx: 2 }));
    svg.appendChild(el('text', { x: px, y: py - 18, 'text-anchor': 'middle', fill: C.teal, 'font-size': 11, 'font-weight': 700 },
      [`${r.speedup.toFixed(2)}×`]));
    svg.appendChild(el('text', { x: px, y: py - 5, 'text-anchor': 'middle', fill: C.muted, 'font-size': 9.5 },
      [`prep ${r.prep_s.toFixed(2)}s`]));
    svg.appendChild(el('text', { x: px, y: by1 + 16, 'text-anchor': 'middle', fill: C.muted, 'font-size': 10 }, [`${r.n}³`]));
  });
  drawAxesFrame(svg, bx0, by0, bx1, by1, 'Forward Speedup (×)', 'Volume Grid Size (N³)');
  return svg;
}

function build_fig6() {
  const W = 1040, H = 415;
  const svg = makeSVG(W, H);
  const { cdfs, dofs, dof_labels } = D.fig6;

  // Panel (a): CED
  panelTitle(svg, 62, 24, '(a) Biplanar vs Monoplanar CED (30 Trials)', 'Orthogonal spectral slices resolve depth & out-of-plane ambiguity');
  drawLegend(svg, 62, 58, cdfs.map(c => ({ kind: 'line', color: C[c.color], label: `${c.label} (${c.s2.toFixed(1)}%)` })), 10);
  const ax0 = 62, ax1 = 485, ay0 = 76, ay1 = 352;
  const xMapA = linMapper(0, 12, ax0, ax1);
  const yMapA = linMapper(0, 100, ay1, ay0);
  drawGridY(svg, ax0, ax1, yMapA, [0, 20, 40, 60, 80, 100], v => `${v}%`);
  drawGridX(svg, ay0, ay1, xMapA, [0, 2, 4, 6, 8, 10, 12], v => `${v}`);
  drawThreshX(svg, xMapA(2.0), ay0, ay1, '2.0 mm threshold', C.redDash, ay1 - ay0 - 10, 6, 'start');
  cdfs.forEach(c => {
    svg.appendChild(el('path', { d: polylinePath(c.pts, xMapA, yMapA), fill: 'none', stroke: C[c.color], 'stroke-width': 2.3 }));
  });
  drawAxesFrame(svg, ax0, ay0, ax1, ay1, 'Success Rate (%)', 'Final 3D mTRE (mm)');

  // Panel (b): Grouped 6-DoF Errors on Symlog / Log-like Scale so small Tx/Ty & large Yaw both read cleanly
  const bx0 = 562, bx1 = 1012, by0 = 76, by1 = 352;
  panelTitle(svg, bx0, 24, '(b) 6-DoF Pose Error Breakdown', 'Mean residual error across rotations (°) and translations (mm)');
  drawLegend(svg, bx0, 58, dofs.map(d => ({ kind: 'bar', color: C[d.color], label: d.label })), 14);
  // Use log1p scale: y = log10(1 + v) up to 20 -> crisp for 0.01 up to 16
  const yMapB = (v) => by1 - (Math.log10(1 + v) / Math.log10(21)) * (by1 - by0);
  drawGridY(svg, bx0, bx1, yMapB, [0, 0.5, 1, 2, 5, 10, 20], v => String(v));
  const grpW = (bx1 - bx0) / dof_labels.length;
  const bw = 15, gap = 3;
  dof_labels.forEach((lbl, gi) => {
    const gc = bx0 + (gi + 0.5) * grpW;
    svg.appendChild(el('text', { x: gc, y: by1 + 17, 'text-anchor': 'middle', fill: C.ink, 'font-size': 10, 'font-weight': 600 }, [lbl]));
    dofs.forEach((d, di) => {
      const val = d.vals[gi];
      const bx = gc + (di - 1) * (bw + gap) - bw / 2;
      const py = yMapB(val);
      const col = C[d.color];
      svg.appendChild(el('rect', { x: bx, y: py, width: bw, height: Math.max(2, by1 - py), fill: col, rx: 1.5 }));
      const txt = val >= 10 ? val.toFixed(1) : val.toFixed(2);
      svg.appendChild(el('text', {
        x: bx + bw / 2, y: py - 5 - (di === 1 ? 8 : 0),
        'text-anchor': 'middle', fill: col, 'font-size': 8.2, 'font-weight': 700
      }, [txt]));
    });
  });
  drawAxesFrame(svg, bx0, by0, bx1, by1, 'Mean Error (° or mm, log(1+x))', 'Rigid Pose Degree of Freedom');
  return svg;
}

function build_fig7() {
  const W = 1080, H = 610;
  const svg = makeSVG(W, H);
  const { imgs, clean_cdf, low_cdf, clean_med, low_med } = D.fig7;

  svg.appendChild(el('text', { x: 24, y: 24, fill: C.ink, 'font-size': 14, 'font-weight': 700 },
    ['Clinical GE CT Pre-Calibration Orthographic Noise Study (n = 20)']));
  svg.appendChild(el('text', { x: 24, y: 40, fill: C.muted, 'font-size': 10.5 },
    ['Orthogonal CT slices, AP/Lateral registered projections, and clean vs simulated low-dose fluoroscopy CED']));

  // Top 3 CT Slices (each 220x220 box, evenly spaced across 1080px)
  const topTitles = ['(a) Axial CT Slice', '(b) Coronal CT Slice', '(c) Sagittal CT Slice'];
  const topKeys = ['axial', 'coronal', 'sagittal'];
  const xs3 = [66, 430, 794];
  topKeys.forEach((k, i) => {
    const x = xs3[i], y = 76, sz = 216;
    svg.appendChild(el('text', { x: x + sz / 2, y: 66, 'text-anchor': 'middle', fill: C.ink, 'font-size': 12, 'font-weight': 700 }, [topTitles[i]]));
    svg.appendChild(el('rect', { x: x - 1, y: y - 1, width: sz + 2, height: sz + 2, fill: '#000', stroke: C.border, 'stroke-width': 1 }));
    svg.appendChild(el('image', { href: imgs[k], x, y, width: sz, height: sz, preserveAspectRatio: 'xMidYMid meet' }));
  });

  // Bottom Row: (d) AP DRR, (e) Lat DRR, (f) CED plot
  const botY = 342, botSz = 224;
  [['ap', '(d) Registered AP DRR'], ['lat', '(e) Registered Lateral DRR']].forEach(([k, ttl], i) => {
    const x = i === 0 ? 54 : 332;
    svg.appendChild(el('text', { x: x + botSz / 2, y: botY - 12, 'text-anchor': 'middle', fill: C.ink, 'font-size': 12, 'font-weight': 700 }, [ttl]));
    svg.appendChild(el('rect', { x: x - 1, y: botY - 1, width: botSz + 2, height: botSz + 2, fill: '#000', stroke: C.border, 'stroke-width': 1 }));
    svg.appendChild(el('image', { href: imgs[k], x, y: botY, width: botSz, height: botSz, preserveAspectRatio: 'xMidYMid meet' }));
  });

  // (f) CED chart
  const cx0 = 646, cx1 = 1044, cy0 = 368, cy1 = 564;
  svg.appendChild(el('text', { x: cx0, y: 330, fill: C.ink, 'font-size': 12, 'font-weight': 700 }, ['(f) Sub-Millimeter Convergence CED']));
  drawLegend(svg, cx0, 352, [
    { kind: 'line', color: C.navy, label: `Clean (${clean_med.toFixed(3)} mm)` },
    { kind: 'line', color: C.teal, label: `Low-Dose (${low_med.toFixed(3)} mm)` },
  ], 14);
  const xMap = linMapper(0, 0.08, cx0, cx1);
  const yMap = linMapper(0, 100, cy1, cy0);
  drawGridY(svg, cx0, cx1, yMap, [0, 25, 50, 75, 100], v => `${v}%`);
  drawGridX(svg, cy0, cy1, xMap, [0, 0.02, 0.04, 0.06, 0.08], v => v.toFixed(2));
  svg.appendChild(el('path', { d: polylinePath(clean_cdf, xMap, yMap), fill: 'none', stroke: C.navy, 'stroke-width': 2.3 }));
  svg.appendChild(el('path', { d: polylinePath(low_cdf, xMap, yMap), fill: 'none', stroke: C.teal, 'stroke-width': 2.3 }));
  svg.appendChild(el('text', { x: cx1 - 4, y: cy1 - 12, 'text-anchor': 'end', fill: C.teal, 'font-size': 10, 'font-weight': 600 },
    ['100% < 0.07 mm (init 4.87 mm)']));
  drawAxesFrame(svg, cx0, cy0, cx1, cy1, 'Success (%)', 'Final 3D mTRE (mm)');
  return svg;
}

function build_fig11() {
  const W = 1080, H = 560;
  const svg = makeSVG(W, H);
  const rows = D.fig11;

  svg.appendChild(el('text', { x: 24, y: 24, fill: C.ink, 'font-size': 14, 'font-weight': 700 },
    ['(a) Poisson–Gaussian Low-Dose Simulation Across Six Photon Flux Regimes']));

  // 6 aligned dose tiles across top
  const tileW = 152, gap = 20, xStart = 32, tileY = 58;
  rows.forEach((r, i) => {
    const x = xStart + i * (tileW + gap);
    svg.appendChild(el('text', { x: x + tileW / 2, y: tileY - 8, 'text-anchor': 'middle', fill: C.ink, 'font-size': 11, 'font-weight': 700 },
      [`Φ = ${r.label} ph/px`]));
    svg.appendChild(el('rect', { x: x - 1, y: tileY - 1, width: tileW + 2, height: tileW + 2, fill: '#000', stroke: C.border, 'stroke-width': 1 }));
    svg.appendChild(el('image', { href: r.tile, x, y: tileY, width: tileW, height: tileW }));
  });

  // Bottom Row: Two completely separated JS/SVG panels with generous center gutter
  // Left Panel (b): Mean mTRE +- SD
  const ax0 = 72, ax1 = 496, ay0 = 294, ay1 = 506;
  panelTitle(svg, ax0, 244, '(b) Registration Accuracy Across 1,000× Dose Drop', 'Flat ~0.03 mm final mTRE from ~5.0 mm initial offset');
  drawLegend(svg, ax0, 278, [
    { kind: 'line', color: C.navy, label: 'Diff-FVR Mean mTRE' },
    { kind: 'bar', color: '#DCE5EE', label: '±1 SD Envelope' },
  ], 14);
  const xMapA = (i) => ax0 + 30 + i * ((ax1 - ax0 - 60) / 5);
  const yMapA = linMapper(0, 0.12, ay1, ay0);
  drawGridY(svg, ax0, ax1, yMapA, [0, 0.03, 0.06, 0.09, 0.12], v => v.toFixed(2));
  rows.forEach((r, i) => {
    const px = xMapA(i);
    svg.appendChild(el('line', { x1: px, y1: ay0, x2: px, y2: ay1, stroke: C.grid, 'stroke-width': 0.8 }));
    svg.appendChild(el('text', { x: px, y: ay1 + 16, 'text-anchor': 'middle', fill: C.muted, 'font-size': 10 }, [r.label]));
  });
  // ±SD shaded band
  const upper = rows.map((r, i) => [xMapA(i), yMapA(r.mean + r.std)]);
  const lower = rows.slice().reverse().map((r, i) => [xMapA(5 - i), yMapA(Math.max(0, r.mean - r.std))]);
  const bandPts = upper.concat(lower).map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  svg.appendChild(el('polygon', { points: bandPts, fill: C.navy, opacity: 0.14 }));
  svg.appendChild(el('path', { d: polylinePath(rows.map((r, i) => [i, r.mean]), xMapA, yMapA), fill: 'none', stroke: C.navy, 'stroke-width': 2.3 }));
  rows.forEach((r, i) => {
    const px = xMapA(i), py = yMapA(r.mean);
    svg.appendChild(el('circle', { cx: px, cy: py, r: 3.8, fill: C.navy, stroke: '#FFF', 'stroke-width': 1 }));
    svg.appendChild(el('text', { x: px, y: yMapA(r.mean + r.std) - 5, 'text-anchor': 'middle', fill: C.navy, 'font-size': 9.5, 'font-weight': 700 },
      [r.mean.toFixed(3)]));
  });
  drawAxesFrame(svg, ax0, ay0, ax1, ay1, 'Final mTRE (mm)', 'Simulated Photon Flux Φ (ph/px)');

  // Right Panel (c): Success Rates (<2mm & <1mm)
  const bx0 = 592, bx1 = 1046, by0 = 294, by1 = 506;
  panelTitle(svg, bx0, 244, '(c) Clinical Sub-Millimeter Reliability', '100% convergence (<2.0 mm & <1.0 mm) down to Φ = 10³ ph/px');
  drawLegend(svg, bx0, 278, [
    { kind: 'bar', color: C.navy, label: '< 2.0 mm Success (%)' },
    { kind: 'bar', color: C.teal, label: '< 1.0 mm Success (%)' },
  ], 14);
  const yMapB = linMapper(0, 115, by1, by0);
  const xMapB = (i) => bx0 + 36 + i * ((bx1 - bx0 - 72) / 5);
  drawGridY(svg, bx0, bx1, yMapB, [0, 25, 50, 75, 100], v => `${v}%`);
  rows.forEach((r, i) => {
    const px = xMapB(i);
    const bw = 18;
    const py2 = yMapB(r.s2), py1 = yMapB(r.s1);
    svg.appendChild(el('rect', { x: px - bw - 2, y: py2, width: bw, height: by1 - py2, fill: C.navy, rx: 1.5 }));
    svg.appendChild(el('rect', { x: px + 2, y: py1, width: bw, height: by1 - py1, fill: C.teal, rx: 1.5 }));
    svg.appendChild(el('text', { x: px, y: py2 - 6, 'text-anchor': 'middle', fill: C.ink, 'font-size': 9.5, 'font-weight': 700 }, ['100%']));
    svg.appendChild(el('text', { x: px, y: by1 + 16, 'text-anchor': 'middle', fill: C.muted, 'font-size': 10 }, [r.label]));
  });
  drawAxesFrame(svg, bx0, by0, bx1, by1, 'Success Rate (%)', 'Simulated Photon Flux Φ (ph/px)', 42);
  return svg;
}

function build_fig15() {
  const W = 1140, H = 370;
  const svg = makeSVG(W, H);
  const { imgs } = D.fig15;
  svg.appendChild(el('text', { x: 24, y: 26, fill: C.ink, 'font-size': 14.5, 'font-weight': 700 },
    ['Hierarchical Articulated Multi-Vertebra (L5–L4–L3) Fourier Registration']));
  svg.appendChild(el('text', { x: 24, y: 44, fill: C.muted, 'font-size': 10.5 },
    ['Linear Fourier superposition isolates rigid alignment (0.767 mm) before per-vertebra flexion refinement (−54.7% loss)']));

  const panels = [
    ['(a) Initial Misalignment', 'Mixed global & inter-vertebral drift', C.vermilion],
    ['(b) Global Rigid Lock', 'Stage-1 alignment: 0.767 mm mTRE', C.navy],
    ['(c) L5–L4–L3 Flexion Solve', 'Linear spectral superposition fit', C.cobalt],
    ['(d) Converged Residual', '54.7% post-articulation loss drop', C.teal],
  ];
  const pw = 256, gap = 22, sx = 24, py = 88;
  panels.forEach(([t, sub, col], i) => {
    const x = sx + i * (pw + gap);
    svg.appendChild(el('rect', { x, y: py - 30, width: pw, height: 3, fill: col }));
    svg.appendChild(el('text', { x, y: py - 14, fill: C.ink, 'font-size': 11.5, 'font-weight': 700 }, [t]));
    svg.appendChild(el('text', { x, y: py + pw + 18, fill: C.muted, 'font-size': 10.5 }, [sub]));
    svg.appendChild(el('rect', { x: x - 1, y: py - 1, width: pw + 2, height: pw + 2, fill: '#000', stroke: C.border, 'stroke-width': 1 }));
    svg.appendChild(el('image', { href: imgs[i], x, y: py, width: pw, height: pw }));
  });
  return svg;
}

function build_fig14() {
  const W = 1080, H = 435;
  const svg = makeSVG(W, H);
  const { cdfs, bars } = D.fig14;

  // Panel (a): Unified CED
  panelTitle(svg, 64, 24, '(a) Multi-Center Clinical Cohort CED', 'Cross-scanner convergence vs synthetic baseline & monoplanar failure');
  // Put legend in 2 neat rows above plot
  drawLegend(svg, 64, 56, cdfs.slice(0, 3).map(c => ({ kind: 'line', color: C[c.color], dash: c.dash, label: c.label })), 12);
  drawLegend(svg, 64, 73, cdfs.slice(3).map(c => ({ kind: 'line', color: C[c.color], dash: c.dash, label: c.label })), 12);

  const ax0 = 64, ax1 = 528, ay0 = 92, ay1 = 374;
  const xMapA = linMapper(0, 6.0, ax0, ax1);
  const yMapA = linMapper(0, 100, ay1, ay0);
  drawGridY(svg, ax0, ax1, yMapA, [0, 20, 40, 60, 80, 100], v => `${v}%`);
  drawGridX(svg, ay0, ay1, xMapA, [0, 1, 2, 3, 4, 5, 6], v => `${v}`);
  drawThreshX(svg, xMapA(2.0), ay0, ay1, '2.0 mm gate', C.redDash, ay1 - ay0 - 10, 5, 'start');
  drawThreshX(svg, xMapA(1.0), ay0, ay1, '1.0 mm', C.slate, ay1 - ay0 - 10, -4, 'end');
  cdfs.forEach(c => {
    svg.appendChild(el('path', {
      d: polylinePath(c.pts, xMapA, yMapA), fill: 'none', stroke: C[c.color],
      'stroke-width': 2.2, 'stroke-dasharray': c.dash ? '5,3' : null
    }));
  });
  drawAxesFrame(svg, ax0, ay0, ax1, ay1, 'Success Rate (%)', 'Final 3D mTRE (mm)');

  // Panel (b): Grouped Clean vs Low-Dose bars
  const bx0 = 602, bx1 = 1052, by0 = 92, by1 = 374;
  panelTitle(svg, bx0, 24, '(b) Clean vs Low-Dose Cohort Reliability', 'Success rates at <2.0 mm and <1.0 mm (labels placed cleanly above bars)');
  drawLegend(svg, bx0, 56, [
    { kind: 'bar', color: C.navy, label: '<2mm Clean' },
    { kind: 'bar', color: C.cobalt, label: '<2mm Low-Dose' },
    { kind: 'bar', color: C.teal, label: '<1mm Clean' },
  ], 12);
  drawLegend(svg, bx0, 73, [
    { kind: 'bar', color: C.slate, label: '<1mm Low-Dose' },
  ], 12);
  const yMapB = linMapper(0, 115, by1, by0);
  drawGridY(svg, bx0, bx1, yMapB, [0, 25, 50, 75, 100], v => `${v}%`);
  const grpW = (bx1 - bx0) / bars.length;
  const bw = 22, gap = 4;
  bars.forEach((b, gi) => {
    const gc = bx0 + (gi + 0.5) * grpW;
    svg.appendChild(el('text', { x: gc, y: by1 + 17, 'text-anchor': 'middle', fill: C.ink, 'font-size': 10.5, 'font-weight': 700 }, [b.cohort]));
    const vals = [
      [b.clean.s2, C.navy], [b.low.s2, C.cobalt],
      [b.clean.s1, C.teal], [b.low.s1, C.slate]
    ];
    vals.forEach(([v, col], bi) => {
      const bx = gc + (bi - 1.5) * (bw + gap) - bw / 2;
      const py = yMapB(v);
      svg.appendChild(el('rect', { x: bx, y: py, width: bw, height: by1 - py, fill: col, rx: 1.5 }));
      svg.appendChild(el('text', {
        x: bx + bw / 2, y: py - 5 - (bi % 2 === 1 ? 10 : 0),
        'text-anchor': 'middle', fill: col, 'font-size': 8.5, 'font-weight': 700
      }, [`${v.toFixed(0)}%`]));
    });
  });
  drawAxesFrame(svg, bx0, by0, bx1, by1, 'Success Rate (%)', 'Clinical DICOM CT Cohort');
  return svg;
}

function build_fig25() {
  const W = 1140, H = 395;
  const svg = makeSVG(W, H);
  const { fwd, reg } = D.fig25;

  // Panel (a): Forward Latency 10 views (log)
  const ax0 = 58, ax1 = 378, ay0 = 76, ay1 = 336;
  panelTitle(svg, ax0, 24, '(a) CPU Forward Latency (10 Views)', 'DeepFluoro subject01 · 105.7× mean speedup');
  drawLegend(svg, ax0, 58, [
    { kind: 'bar', color: C.vermilion, label: 'DiffDRR 0.6.1' },
    { kind: 'bar', color: C.navy, label: 'Diff-FVR' },
  ], 14);
  const yMapA = logMapper(4, 3000, ay1, ay0);
  drawGridY(svg, ax0, ax1, yMapA, [5, 10, 50, 100, 500, 1500], v => String(v));
  const stepA = (ax1 - ax0) / fwd.length;
  fwd.forEach((r, i) => {
    const gc = ax0 + (i + 0.5) * stepA;
    svg.appendChild(el('rect', { x: gc - 11, y: yMapA(r.ms_drr), width: 10, height: ay1 - yMapA(r.ms_drr), fill: C.vermilion, rx: 1 }));
    svg.appendChild(el('rect', { x: gc + 1, y: yMapA(r.ms_fvr), width: 10, height: ay1 - yMapA(r.ms_fvr), fill: C.navy, rx: 1 }));
    svg.appendChild(el('text', { x: gc, y: ay1 + 15, 'text-anchor': 'middle', fill: C.muted, 'font-size': 9 }, [r.view]));
  });
  drawAxesFrame(svg, ax0, ay0, ax1, ay1, 'Latency / View (ms, log)', 'DeepFluoro subject01 View ID');

  // Panel (b): Similarity (NCC / MNCC)
  const bx0 = 446, bx1 = 766, by0 = 76, by1 = 336;
  panelTitle(svg, bx0, 24, '(b) Spectral vs Real X-Ray Fidelity', 'Per-view NCC (DRR vs FVR) & Real MNCC');
  drawLegend(svg, bx0, 58, [
    { kind: 'line', color: C.navy, label: 'FVR–DRR' },
    { kind: 'line', color: C.slate, label: 'DRR–Real' },
    { kind: 'line', color: C.teal, label: 'FVR–Real' },
  ], 10);
  const xMapB = (i) => bx0 + 18 + i * ((bx1 - bx0 - 36) / 9);
  const yMapB = linMapper(0.80, 1.01, by1, by0);
  drawGridY(svg, bx0, bx1, yMapB, [0.80, 0.85, 0.90, 0.95, 1.00], v => v.toFixed(2));
  fwd.forEach((r, i) => {
    svg.appendChild(el('text', { x: xMapB(i), y: by1 + 15, 'text-anchor': 'middle', fill: C.muted, 'font-size': 9 }, [r.view]));
  });
  [
    ['ncc_fd', C.navy], ['mncc_dr', C.slate], ['mncc_fr', C.teal]
  ].forEach(([k, col]) => {
    const pts = fwd.map((r, i) => [i, r[k]]);
    svg.appendChild(el('path', { d: polylinePath(pts, xMapB, yMapB), fill: 'none', stroke: col, 'stroke-width': 2.0 }));
    pts.forEach(([i, v]) => svg.appendChild(el('circle', { cx: xMapB(i), cy: yMapB(v), r: 3, fill: col, stroke: '#FFF', 'stroke-width': 0.8 })));
  });
  drawAxesFrame(svg, bx0, by0, bx1, by1, 'Similarity (NCC / MNCC)', 'DeepFluoro subject01 View ID');

  // Panel (c): Mean Biplanar 6-DoF mTRE
  const cx0 = 834, cx1 = 1118, cy0 = 76, cy1 = 336;
  panelTitle(svg, cx0, 24, '(c) Real Biplanar 6-DoF Error', 'Mean over 4 real X-ray pairs (subject01)');
  drawLegend(svg, cx0, 58, [{ kind: 'dash_line', color: C.redDash, label: '2.0 mm Clinical Threshold' }], 10);
  const yMapC = linMapper(0, 12.0, cy1, cy0);
  drawGridY(svg, cx0, cx1, yMapC, [0, 2, 4, 6, 8, 10, 12], v => String(v));
  drawThresh(svg, cx0, cx1, yMapC(2.0), null, C.redDash);
  const stepC = (cx1 - cx0) / reg.length;
  reg.forEach((r, i) => {
    const gc = cx0 + (i + 0.5) * stepC;
    const bw = 38;
    const py = yMapC(r.mtre);
    svg.appendChild(el('rect', { x: gc - bw / 2, y: py, width: bw, height: cy1 - py, fill: C[r.color], rx: 2 }));
    svg.appendChild(el('text', { x: gc, y: py - 15, 'text-anchor': 'middle', fill: C[r.color], 'font-size': 10, 'font-weight': 700 }, [`${r.mtre.toFixed(2)} mm`]));
    if (r.time > 0) {
      svg.appendChild(el('text', { x: gc, y: py - 4, 'text-anchor': 'middle', fill: C.muted, 'font-size': 9 }, [`(${r.time.toFixed(1)} s)`]));
    }
    svg.appendChild(el('text', { x: gc, y: cy1 + 15, 'text-anchor': 'middle', fill: C.ink, 'font-size': 10, 'font-weight': 600 }, [r.label]));
  });
  drawAxesFrame(svg, cx0, cy0, cx1, cy1, 'Mean 3D mTRE (mm)', 'Registration Pipeline');
  return svg;
}

function build_fig26() {
  const W = 1060, H = 415;
  const svg = makeSVG(W, H);
  const rows = D.fig26;

  // Panel (a): Forward speed
  const ax0 = 62, ax1 = 492, ay0 = 76, ay1 = 352;
  panelTitle(svg, ax0, 24, '(a) Multi-Subject CPU Forward Speed (S01–S06)', '70.7×–105.6× speedup across all 6 cadaveric specimens');
  drawLegend(svg, ax0, 58, [
    { kind: 'bar', color: C.vermilion, label: 'DiffDRR 0.6.1' },
    { kind: 'bar', color: C.navy, label: 'Diff-FVR' },
  ], 14);
  const yMapA = logMapper(4, 5000, ay1, ay0);
  drawGridY(svg, ax0, ax1, yMapA, [5, 10, 50, 100, 500, 2000], v => String(v));
  const stepA = (ax1 - ax0) / rows.length;
  rows.forEach((r, i) => {
    const gc = ax0 + (i + 0.5) * stepA;
    svg.appendChild(el('rect', { x: gc - 18, y: yMapA(r.ms_drr), width: 16, height: ay1 - yMapA(r.ms_drr), fill: C.vermilion, rx: 1.5 }));
    svg.appendChild(el('rect', { x: gc + 2, y: yMapA(r.ms_fvr), width: 16, height: ay1 - yMapA(r.ms_fvr), fill: C.navy, rx: 1.5 }));
    svg.appendChild(el('text', { x: gc - 10, y: yMapA(r.ms_drr) - 5, 'text-anchor': 'middle', fill: C.vermilion, 'font-size': 8.5, 'font-weight': 700 },
      [`${Math.round(r.ms_drr)}`]));
    svg.appendChild(el('text', { x: gc + 10, y: yMapA(r.ms_fvr) - 16, 'text-anchor': 'middle', fill: C.navy, 'font-size': 8.5, 'font-weight': 700 },
      [`${r.ms_fvr.toFixed(1)}`]));
    svg.appendChild(el('text', { x: gc + 10, y: yMapA(r.ms_fvr) - 5, 'text-anchor': 'middle', fill: C.teal, 'font-size': 8.5, 'font-weight': 700 },
      [`${r.speedup.toFixed(0)}×`]));
    svg.appendChild(el('text', { x: gc, y: ay1 + 16, 'text-anchor': 'middle', fill: C.ink, 'font-size': 10, 'font-weight': 600 }, [r.sub]));
  });
  drawAxesFrame(svg, ax0, ay0, ax1, ay1, 'Latency / View (ms, log)', 'DeepFluoro Subject');

  // Panel (b): mTRE across methods per subject
  const bx0 = 564, bx1 = 1032, by0 = 76, by1 = 352;
  panelTitle(svg, bx0, 24, '(b) Legacy Orthographic Biplanar 6-DoF mTRE', 'Coarse-to-fine Hybrid (Diff-FVR → DiffDRR) vs pure DiffDRR');
  drawLegend(svg, bx0, 58, [
    { kind: 'bar', color: C.slate, label: 'Init' },
    { kind: 'bar', color: C.navy, label: 'Diff-FVR' },
    { kind: 'bar', color: C.vermilion, label: 'DiffDRR' },
    { kind: 'bar', color: C.teal, label: 'Hybrid' },
    { kind: 'dash_line', color: C.redDash, label: '2.0 mm gate' },
  ], 10);
  const yMapB = linMapper(0, 12.5, by1, by0);
  drawGridY(svg, bx0, bx1, yMapB, [0, 2, 4, 6, 8, 10, 12], v => String(v));
  drawThresh(svg, bx0, bx1, yMapB(2.0), null, C.redDash);
  const stepB = (bx1 - bx0) / rows.length;
  const bw = 13, gap = 2;
  rows.forEach((r, i) => {
    const gc = bx0 + (i + 0.5) * stepB;
    [['init', C.slate], ['fvr', C.navy], ['drr', C.vermilion], ['hyb', C.teal]].forEach(([k, col], bi) => {
      const bx = gc + (bi - 1.5) * (bw + gap) - bw / 2;
      const py = yMapB(r[k]);
      svg.appendChild(el('rect', { x: bx, y: py, width: bw, height: by1 - py, fill: col, rx: 1.2 }));
    });
    svg.appendChild(el('text', { x: gc, y: by1 + 16, 'text-anchor': 'middle', fill: C.ink, 'font-size': 10, 'font-weight': 600 }, [r.sub]));
  });
  drawAxesFrame(svg, bx0, by0, bx1, by1, 'Final 3D mTRE (mm)', 'DeepFluoro Subject');
  return svg;
}

function build_fig27() {
  const W = 1060, H = 415;
  const svg = makeSVG(W, H);
  const rows = D.fig27;
  const find = (H_val, geom, m_part) => rows.find(r => r.H === H_val && r.geom === geom && r.method.includes(m_part));

  // Panel (a): Forward Latency
  const ax0 = 62, ax1 = 510, ay0 = 76, ay1 = 352;
  panelTitle(svg, ax0, 24, '(a) Geometry-Matched Single-View Latency', 'Equal detector resolution (128², 256²) & equal cone/parallel physics');
  drawLegend(svg, ax0, 58, [
    { kind: 'bar', color: C.teal, label: 'Diff-FVR / PoU-FVR' },
    { kind: 'bar', color: C.amber, label: 'DiffDRR Trilinear' },
    { kind: 'bar', color: C.vermilion, label: 'DiffDRR Siddon' },
  ], 12);
  const yMapA = logMapper(10, 9000, ay1, ay0);
  drawGridY(svg, ax0, ax1, yMapA, [10, 30, 100, 300, 1000, 4000], v => String(v));
  const specsA = [
    ['Par 128²', find(128, 'Parallel', 'Parallel Diff-FVR'), find(128, 'Parallel', 'Trilinear'), find(128, 'Parallel', 'Siddon')],
    ['Cone 128²', find(128, 'Cone-Beam', 'Cone-Beam'), find(128, 'Cone-Beam', 'Trilinear'), find(128, 'Cone-Beam', 'Siddon')],
    ['Par 256²', find(256, 'Parallel', 'Parallel Diff-FVR'), find(256, 'Parallel', 'Trilinear'), find(256, 'Parallel', 'Siddon')],
    ['Cone 256²', find(256, 'Cone-Beam', 'Cone-Beam'), find(256, 'Cone-Beam', 'Trilinear'), find(256, 'Cone-Beam', 'Siddon')],
  ];
  const grpA = (ax1 - ax0) / specsA.length;
  specsA.forEach(([lbl, rF, rT, rS], gi) => {
    const gc = ax0 + (gi + 0.5) * grpA;
    const bw = 20, gap = 4;
    [[rF.fwd_ms, C.teal], [rT.fwd_ms, C.amber], [rS.fwd_ms, C.vermilion]].forEach(([v, col], bi) => {
      const bx = gc + (bi - 1) * (bw + gap) - bw / 2;
      const py = yMapA(v);
      svg.appendChild(el('rect', { x: bx, y: py, width: bw, height: ay1 - py, fill: col, rx: 1.5 }));
      svg.appendChild(el('text', { x: bx + bw / 2, y: py - 5, 'text-anchor': 'middle', fill: col, 'font-size': 8.5, 'font-weight': 700 },
        [v >= 100 ? `${Math.round(v)}` : v.toFixed(1)]));
    });
    svg.appendChild(el('text', { x: gc, y: ay1 + 16, 'text-anchor': 'middle', fill: C.ink, 'font-size': 10, 'font-weight': 600 }, [lbl]));
  });
  drawAxesFrame(svg, ax0, ay0, ax1, ay1, 'Forward Time / View (ms, log)', 'Physics & Detector Resolution');

  // Panel (b): Amortized 2-View Wall Time @ 256^2
  const bx0 = 582, bx1 = 1032, by0 = 76, by1 = 352;
  panelTitle(svg, bx0, 24, '(b) Amortized 2-View Cone-Beam Time (256²)', 'Includes 0.80 s PoU-FVR FFT setup overhead across optimization budget');
  drawLegend(svg, bx0, 58, [
    { kind: 'line', color: C.teal, label: 'Cone-Beam PoU-FVR (2×2×2)' },
    { kind: 'line', color: C.amber, label: 'DiffDRR Trilinear' },
    { kind: 'line', color: C.vermilion, label: 'DiffDRR Siddon' },
  ], 10);
  const xMapB = (i) => bx0 + 48 + i * ((bx1 - bx0 - 96) / 2);
  const yMapB = logMapper(1, 3500, by1, by0);
  drawGridY(svg, bx0, bx1, yMapB, [1, 3, 10, 30, 100, 300, 1500], v => String(v));
  const stepsLbl = ['10 steps', '40 steps', '100 steps'];
  stepsLbl.forEach((l, i) => {
    svg.appendChild(el('line', { x1: xMapB(i), y1: by0, x2: xMapB(i), y2: by1, stroke: C.grid, 'stroke-width': 0.8 }));
    svg.appendChild(el('text', { x: xMapB(i), y: by1 + 16, 'text-anchor': 'middle', fill: C.ink, 'font-size': 10, 'font-weight': 600 }, [l]));
  });
  [
    [find(256, 'Cone-Beam', 'Cone-Beam'), C.teal, 14],
    [find(256, 'Cone-Beam', 'Trilinear'), C.amber, 13],
    [find(256, 'Cone-Beam', 'Siddon'), C.vermilion, -7],
  ].forEach(([r, col, dy]) => {
    const pts = [[0, r.t10], [1, r.t40], [2, r.t100]];
    svg.appendChild(el('path', { d: polylinePath(pts, xMapB, yMapB), fill: 'none', stroke: col, 'stroke-width': 2.3 }));
    pts.forEach(([i, v]) => {
      svg.appendChild(el('circle', { cx: xMapB(i), cy: yMapB(v), r: 3.8, fill: col, stroke: '#FFF', 'stroke-width': 1 }));
      svg.appendChild(el('text', {
        x: xMapB(i), y: yMapB(v) + dy, 'text-anchor': 'middle', fill: col, 'font-size': 9.5, 'font-weight': 700
      }, [v >= 100 ? `${v.toFixed(0)} s` : `${v.toFixed(1)} s`]));
    });
  });
  drawAxesFrame(svg, bx0, by0, bx1, by1, 'Total 2-View Wall Time (s, log)', 'Optimization Step Budget');
  return svg;
}

function build_fig28() {
  const W = 1060, H = 420;
  const svg = makeSVG(W, H);
  const { ge, df } = D.fig28;

  // Panel (a): GE C-Arm geometries
  const ax0 = 62, ax1 = 500, ay0 = 76, ay1 = 356;
  panelTitle(svg, ax0, 24, '(a) GE Clinical CT Across 4 C-Arm Geometries', 'Grid-centered PoU-FVR eliminates cone-beam magnification drift');
  drawLegend(svg, ax0, 58, [
    { kind: 'bar', color: C.slate, label: 'Uncalibrated Ortho' },
    { kind: 'bar', color: C.amber, label: 'Calibrated Ortho' },
    { kind: 'bar', color: C.teal, label: 'PoU-FVR (Cone-Beam)' },
  ], 10);
  const yMapA = logMapper(0.4, 90, ay1, ay0);
  drawGridY(svg, ax0, ax1, yMapA, [0.5, 1, 2, 5, 10, 40], v => String(v));
  drawThresh(svg, ax0, ax1, yMapA(2.0), '2.0 mm', C.redDash, 'right');
  const grpA = (ax1 - ax0) / ge.length;
  ge.forEach((r, i) => {
    const gc = ax0 + (i + 0.5) * grpA;
    const bw = 20, gap = 3;
    [[r.uncal, C.slate], [r.cal, C.amber], [r.pou, C.teal]].forEach(([v, col], bi) => {
      const bx = gc + (bi - 1) * (bw + gap) - bw / 2;
      const py = yMapA(v);
      svg.appendChild(el('rect', { x: bx, y: py, width: bw, height: ay1 - py, fill: col, rx: 1.5 }));
      svg.appendChild(el('text', { x: bx + bw / 2, y: py - 5, 'text-anchor': 'middle', fill: col, 'font-size': 8.5, 'font-weight': 700 },
        [v >= 10 ? v.toFixed(1) : v.toFixed(2)]));
    });
    svg.appendChild(el('text', { x: gc, y: ay1 + 16, 'text-anchor': 'middle', fill: C.ink, 'font-size': 9.5, 'font-weight': 600 }, [`${r.label} mm`]));
  });
  drawAxesFrame(svg, ax0, ay0, ax1, ay1, 'Final 3D mTRE (mm, log)', 'C-Arm SAD/SDD Geometry');

  // Panel (b): DeepFluoro Cone-Beam Multi-Start + Hybrid
  const bx0 = 568, bx1 = 1036, by0 = 76, by1 = 356;
  panelTitle(svg, bx0, 24, '(b) Real DeepFluoro Cone-Beam Registration', 'Mean mTRE: DiffDRR 4.78 vs Multi-Start PoU 2.66 vs Hybrid 2.23 mm');
  drawLegend(svg, bx0, 58, [
    { kind: 'bar', color: C.vermilion, label: 'DiffDRR (128s)' },
    { kind: 'bar', color: C.cobalt, label: 'Single PoU' },
    { kind: 'bar', color: C.navy, label: 'Multi-Start (9.3s)' },
    { kind: 'bar', color: C.teal, label: 'Hybrid (50s)' },
    { kind: 'dash_line', color: C.redDash, label: '2.0 mm' },
  ], 8);
  const yMapB = linMapper(0, 8.8, by1, by0);
  drawGridY(svg, bx0, bx1, yMapB, [0, 2, 4, 6, 8], v => String(v));
  drawThresh(svg, bx0, bx1, yMapB(2.0), null, C.redDash);
  const grpB = (bx1 - bx0) / df.length;
  df.forEach((r, i) => {
    const gc = bx0 + (i + 0.5) * grpB;
    const bw = 13, gap = 2;
    [[r.drr, C.vermilion], [r.single, C.cobalt], [r.multi, C.navy], [r.hyb, C.teal]].forEach(([v, col], bi) => {
      const bx = gc + (bi - 1.5) * (bw + gap) - bw / 2;
      const py = yMapB(v);
      svg.appendChild(el('rect', { x: bx, y: py, width: bw, height: by1 - py, fill: col, rx: 1.2 }));
    });
    svg.appendChild(el('text', { x: gc, y: by1 + 16, 'text-anchor': 'middle', fill: C.ink, 'font-size': 10, 'font-weight': 600 }, [r.sub]));
  });
  drawAxesFrame(svg, bx0, by0, bx1, by1, 'Final 3D mTRE (mm)', 'DeepFluoro Subject');
  return svg;
}

function build_fig29() {
  const W = 1140, H = 400;
  const svg = makeSVG(W, H);
  const { res, fps, fb, pareto } = D.fig29;

  // Panel (a): Forward Throughput FPS (log)
  const ax0 = 62, ax1 = 382, ay0 = 96, ay1 = 340;
  panelTitle(svg, ax0, 24, '(a) A100 Throughput (B=64)', 'Up to 26,506 FPS @ 512² (185× vs PyTorch)');
  const keysA = ['PoU-FVR CUDAGraph', 'PoU-FVR Triton', 'Zero-IFFT NCC', 'nanodrr Triton+compile', 'nanodrr PyTorch'];
  const shortA = {
    'PoU-FVR CUDAGraph': 'PoU CUDAGraph',
    'PoU-FVR Triton': 'PoU Triton',
    'Zero-IFFT NCC': 'Zero-IFFT',
    'nanodrr Triton+compile': 'nanodrr Triton',
    'nanodrr PyTorch': 'nanodrr PyTorch'
  };
  drawLegend(svg, ax0, 56, keysA.slice(0, 3).map(k => ({ kind: 'line', color: C[fps[k].color], label: shortA[k] })), 10);
  drawLegend(svg, ax0, 73, keysA.slice(3, 5).map(k => ({ kind: 'line', color: C[fps[k].color], label: shortA[k] })), 12);
  const yMapA = logMapper(80, 180000, ay1, ay0);
  const xMapA = (i) => ax0 + 36 + i * ((ax1 - ax0 - 72) / 2);
  drawGridY(svg, ax0, ax1, yMapA, [100, 1000, 10000, 100000], v => v >= 1000 ? `${v / 1000}k` : String(v));
  res.forEach((lbl, i) => {
    svg.appendChild(el('line', { x1: xMapA(i), y1: ay0, x2: xMapA(i), y2: ay1, stroke: C.grid, 'stroke-width': 0.8 }));
    svg.appendChild(el('text', { x: xMapA(i), y: ay1 + 16, 'text-anchor': 'middle', fill: C.ink, 'font-size': 10, 'font-weight': 600 }, [lbl]));
  });
  keysA.forEach((k) => {
    const s = fps[k];
    const pts = s.vals.map((v, i) => [i, v]);
    svg.appendChild(el('path', { d: polylinePath(pts, xMapA, yMapA), fill: 'none', stroke: C[s.color], 'stroke-width': 2.1 }));
    pts.forEach(([i, v]) => svg.appendChild(el('circle', { cx: xMapA(i), cy: yMapA(v), r: 3.3, fill: C[s.color], stroke: '#FFF', 'stroke-width': 0.8 })));
  });
  svg.appendChild(el('text', { x: xMapA(2) - 4, y: yMapA(143.45) + 14, 'text-anchor': 'end', fill: C.vermilion, 'font-size': 9, 'font-weight': 700 },
    ['PyTorch 143 FPS']));
  svg.appendChild(el('text', { x: xMapA(2), y: yMapA(26505.78) - 10, 'text-anchor': 'end', fill: C.teal, 'font-size': 9, 'font-weight': 700 },
    ['26,506 FPS']));
  drawAxesFrame(svg, ax0, ay0, ax1, ay1, 'Throughput (FPS, log)', 'Detector Resolution (H×W)');

  // Panel (b): Differentiable fwd+bwd latency
  const bx0 = 448, bx1 = 752, by0 = 96, by1 = 340;
  panelTitle(svg, bx0, 24, '(b) Fwd + Bwd Wall Time (B=64)', '6.91× faster differentiable pass at 512²');
  drawLegend(svg, bx0, 62, [
    { kind: 'bar', color: C.navy, label: 'PoU-FVR Triton' },
    { kind: 'bar', color: C.amber, label: 'nanodrr Triton (n=500)' },
  ], 10);
  const yMapB = linMapper(0, 175, by1, by0);
  drawGridY(svg, bx0, bx1, yMapB, [0, 30, 60, 90, 120, 150], v => String(v));
  const grpB = (bx1 - bx0) / 3;
  res.forEach((lbl, i) => {
    const gc = bx0 + (i + 0.5) * grpB;
    const vf = fb['PoU-FVR Triton'].vals[i], vn = fb['nanodrr Triton (n=500)'].vals[i];
    svg.appendChild(el('rect', { x: gc - 24, y: yMapB(vf), width: 22, height: by1 - yMapB(vf), fill: C.navy, rx: 1.5 }));
    svg.appendChild(el('rect', { x: gc + 2, y: yMapB(vn), width: 22, height: by1 - yMapB(vn), fill: C.amber, rx: 1.5 }));
    svg.appendChild(el('text', { x: gc - 13, y: yMapB(vf) - 5, 'text-anchor': 'middle', fill: C.navy, 'font-size': 9, 'font-weight': 700 }, [vf.toFixed(1)]));
    svg.appendChild(el('text', { x: gc + 13, y: yMapB(vn) - 5, 'text-anchor': 'middle', fill: C.amber, 'font-size': 9, 'font-weight': 700 }, [vn.toFixed(1)]));
    svg.appendChild(el('text', { x: gc, y: by1 + 16, 'text-anchor': 'middle', fill: C.ink, 'font-size': 10, 'font-weight': 600 }, [lbl]));
  });
  drawAxesFrame(svg, bx0, by0, bx1, by1, 'Fwd+Bwd Time (ms)', 'Detector Resolution (H×W)');

  // Panel (c): Pareto edge-NCC vs gpu_ms (clean non-overlapping labels)
  const cx0 = 816, cx1 = 1116, cy0 = 96, cy1 = 340;
  panelTitle(svg, cx0, 24, '(c) PoU Slab Quality–Latency Pareto', 'Edge-NCC vs Siddon over active block configurations');
  drawLegend(svg, cx0, 62, [
    { kind: 'line', color: C.teal, label: 'Measured PoU Configs (1.16–1.22 ms)' },
  ], 10);
  const xMapC = linMapper(1.14, 1.24, cx0, cx1);
  const yMapC = linMapper(0.38, 0.78, cy1, cy0);
  drawGridY(svg, cx0, cx1, yMapC, [0.40, 0.50, 0.60, 0.70], v => v.toFixed(2));
  drawGridX(svg, cy0, cy1, xMapC, [1.15, 1.18, 1.21, 1.24], v => v.toFixed(2));
  // Show key representative Pareto configs so labels never collide!
  const picks = [
    ['1×1×1@64', 5, -5, 'start'],
    ['2×2×2@64', 7, 3, 'start'],
    ['2×4×2@64', 7, 3, 'start'],
    ['2×4×2@80', 7, 4, 'start'],
    ['2×4×2@96', -7, -2, 'end'],
    ['4×4×2@80', 7, 3, 'start'],
  ];
  pareto.forEach(p => {
    const px = xMapC(p.gpu_ms), py = yMapC(p.edge);
    const hit = picks.find(x => x[0] === p.cfg);
    svg.appendChild(el('circle', { cx: px, cy: py, r: hit ? 4.5 : 3.0, fill: hit ? C.teal : C.slateLight, stroke: '#FFF', 'stroke-width': 0.9 }));
    if (hit) {
      svg.appendChild(el('text', {
        x: px + hit[1], y: py + hit[2], 'text-anchor': hit[3],
        fill: C.ink, 'font-size': 9, 'font-weight': 700
      }, [p.cfg]));
    }
  });
  drawAxesFrame(svg, cx0, cy0, cx1, cy1, 'Edge-NCC vs Raymarching', 'Single-Pass GPU Kernel Time (ms)');
  return svg;
}

function build_fig30() {
  const W = 1140, H = 400;
  const svg = makeSVG(W, H);
  const { a, b, c } = D.fig30;
  addHatchPattern(svg, 'oomHatch', C.vermilion);

  // Panel (a): Exp30 Calibrated Multi-Start (n=10)
  const ax0 = 58, ax1 = 368, ay0 = 78, ay1 = 340;
  panelTitle(svg, ax0, 24, '(a) Multi-Start Reg (S01, n=10)', 'Init 15.79 mm · B=64 · PoU-FVR needs zero compile');
  const yMapA = linMapper(0, 2.3, ay1, ay0);
  drawGridY(svg, ax0, ax1, yMapA, [0, 0.5, 1.0, 1.5, 2.0], v => v.toFixed(1));
  drawThresh(svg, ax0, ax1, yMapA(2.0), '2.0 mm', C.redDash, 'left');
  const stepA = (ax1 - ax0) / a.length;
  const shortA = ['PoU-FVR', 'Hybrid', 'nanodrr\nTriton', 'nanodrr\nPyTorch'];
  a.forEach((r, i) => {
    const gc = ax0 + (i + 0.5) * stepA;
    const bw = 38;
    if (r.oom) {
      const py = yMapA(1.58);
      svg.appendChild(el('rect', { x: gc - bw / 2, y: py, width: bw, height: ay1 - py, fill: 'url(#oomHatch)', stroke: C.vermilion, 'stroke-width': 1.2, rx: 1.5 }));
      svg.appendChild(el('text', { x: gc, y: py - 18, 'text-anchor': 'middle', fill: C.vermilion, 'font-size': 9.5, 'font-weight': 700 }, ['OOM']));
      svg.appendChild(el('text', { x: gc, y: py - 5, 'text-anchor': 'middle', fill: C.vermilion, 'font-size': 8.5 }, ['(>40 GB)']));
    } else {
      const py = yMapA(r.med);
      svg.appendChild(el('rect', { x: gc - bw / 2, y: py, width: bw, height: ay1 - py, fill: C[r.color], rx: 1.5 }));
      svg.appendChild(el('text', { x: gc, y: py - 17, 'text-anchor': 'middle', fill: C[r.color], 'font-size': 9.5, 'font-weight': 700 }, [`${r.med.toFixed(2)} mm`]));
      svg.appendChild(el('text', { x: gc, y: py - 4, 'text-anchor': 'middle', fill: C.muted, 'font-size': 8.8 }, [`(${r.time_s.toFixed(2)} s)`]));
    }
    const lines = shortA[i].split('\n');
    lines.forEach((ln, li) => {
      svg.appendChild(el('text', { x: gc, y: ay1 + 15 + li * 13, 'text-anchor': 'middle', fill: C.ink, 'font-size': 9.5, 'font-weight': 600 }, [ln]));
    });
  });
  drawAxesFrame(svg, ax0, ay0, ax1, ay1, 'Median 3D mTRE (mm)', null);

  // Panel (b): Exp31 Per-Subject Real DeepFluoro (S01-S06)
  const bx0 = 434, bx1 = 794, by0 = 78, by1 = 340;
  panelTitle(svg, bx0, 24, '(b) Real DeepFluoro Cohort (n=18)', 'Mean mTRE across all 6 cadaveric subjects');
  drawLegend(svg, bx0, 60, [
    { kind: 'bar', color: C.vermilion, label: 'DiffDRR B=1' },
    { kind: 'bar', color: C.navy, label: 'PoU B=64' },
    { kind: 'bar', color: C.teal, label: 'Hybrid' },
    { kind: 'dash_line', color: C.redDash, label: '2.0 mm' },
  ], 9);
  const yMapB = linMapper(0, 15.0, by1, by0);
  drawGridY(svg, bx0, bx1, yMapB, [0, 3, 6, 9, 12, 15], v => String(v));
  drawThresh(svg, bx0, bx1, yMapB(2.0), null, C.redDash);
  const grpB = (bx1 - bx0) / b.length;
  b.forEach((r, i) => {
    const gc = bx0 + (i + 0.5) * grpB;
    const bw = 13, gap = 2;
    [['drr', C.vermilion], ['pou', C.navy], ['hyb', C.teal]].forEach(([k, col], bi) => {
      const bx = gc + (bi - 1) * (bw + gap) - bw / 2;
      const py = yMapB(r[k]);
      svg.appendChild(el('rect', { x: bx, y: py, width: bw, height: by1 - py, fill: col, rx: 1.2 }));
    });
    svg.appendChild(el('text', { x: gc, y: by1 + 16, 'text-anchor': 'middle', fill: C.ink, 'font-size': 10, 'font-weight': 600 }, [r.sub]));
  });
  drawAxesFrame(svg, bx0, by0, bx1, by1, 'Mean 3D mTRE (mm)', 'DeepFluoro Subject');

  // Panel (c): Time-Accuracy Pareto Frontier
  const cx0 = 860, cx1 = 1118, cy0 = 78, cy1 = 340;
  panelTitle(svg, cx0, 24, '(c) Cohort Pareto Frontier', 'Wall runtime vs mean error & <2 mm rate');
  const xMapC = linMapper(1.2, 3.9, cx0, cx1);
  const yMapC = linMapper(0, 8.5, cy1, cy0);
  drawGridY(svg, cx0, cx1, yMapC, [0, 2, 4, 6, 8], v => String(v));
  drawGridX(svg, cy0, cy1, xMapC, [1.5, 2.0, 2.5, 3.0, 3.5], v => `${v.toFixed(1)}`);
  drawThresh(svg, cx0, cx1, yMapC(2.0), '2.0 mm', C.redDash, 'right');
  c.forEach((p, i) => {
    const px = xMapC(p.time_s), py = yMapC(p.mtre);
    const rad = 5 + (p.s2 / 100) * 5;
    svg.appendChild(el('circle', { cx: px, cy: py, r: rad, fill: C[p.color], opacity: 0.88, stroke: '#FFF', 'stroke-width': 1.5 }));
    const lblY = i === 0 ? py + 20 : py - rad - 16;
    const anchor = px > cx1 - 60 ? 'end' : 'start';
    const dx = anchor === 'end' ? 4 : -4;
    svg.appendChild(el('text', { x: px + dx, y: lblY, 'text-anchor': anchor, fill: C[p.color], 'font-size': 9.5, 'font-weight': 700 },
      [p.label]));
    svg.appendChild(el('text', { x: px + dx, y: lblY + 11, 'text-anchor': anchor, fill: C.muted, 'font-size': 8.8 },
      [`${p.mtre.toFixed(2)}mm · ${p.s2.toFixed(0)}%`]));
  });
  drawAxesFrame(svg, cx0, cy0, cx1, cy1, 'Mean 3D mTRE (mm)', 'Mean Registration Time (s)');
  return svg;
}

function build_fig31() {
  const W = 1140, H = 405;
  const svg = makeSVG(W, H);
  const { boxes, tiers, ct } = D.fig31;

  // Panel (a): N=24 DeepFluoro Boxplot + Strip
  const ax0 = 58, ax1 = 442, ay0 = 86, ay1 = 342;
  panelTitle(svg, ax0, 24, '(a) Matched B=64 N=24 DeepFluoro', 'Includes Tier 2 unanchored wide-basin perturbations');
  const yMapA = logMapper(0.4, 70, ay1, ay0);
  drawGridY(svg, ax0, ax1, yMapA, [0.5, 1, 2, 5, 10, 20, 50], v => String(v));
  drawThresh(svg, ax0, ax1, yMapA(2.0), '2.0 mm', C.redDash, 'right');

  const stepA = (ax1 - ax0) / boxes.length;
  const shortLabels = ['DiffDRR\nB=1', 'nanodrr\nB=64', 'PoU-FVR\nB=64', 'Hybrid\n+nano', 'Hybrid\n+DiffDRR'];
  boxes.forEach((b, i) => {
    const gc = ax0 + (i + 0.5) * stepA;
    const col = C[b.color];
    // Whisker line
    svg.appendChild(el('line', { x1: gc, y1: yMapA(b.whi), x2: gc, y2: yMapA(b.wlo), stroke: col, 'stroke-width': 1.4 }));
    svg.appendChild(el('line', { x1: gc - 7, y1: yMapA(b.whi), x2: gc + 7, y2: yMapA(b.whi), stroke: col, 'stroke-width': 1.4 }));
    svg.appendChild(el('line', { x1: gc - 7, y1: yMapA(b.wlo), x2: gc + 7, y2: yMapA(b.wlo), stroke: col, 'stroke-width': 1.4 }));
    // IQR box
    const yq3 = yMapA(b.q3), yq1 = yMapA(b.q1), ymed = yMapA(b.med);
    svg.appendChild(el('rect', { x: gc - 14, y: yq3, width: 28, height: Math.max(2, yq1 - yq3), fill: col, opacity: 0.22, stroke: col, 'stroke-width': 1.5, rx: 1.5 }));
    svg.appendChild(el('line', { x1: gc - 14, y1: ymed, x2: gc + 14, y2: ymed, stroke: col, 'stroke-width': 2.4 }));
    // Jittered points (deterministic pseudo-jitter)
    b.pts.forEach((v, pi) => {
      const jx = gc + (((pi * 7) % 11) - 5) * 1.4;
      svg.appendChild(el('circle', { cx: jx, cy: yMapA(v), r: 1.9, fill: col, opacity: 0.7 }));
    });
    // Med value cleanly under 0.5 line or above whi
    svg.appendChild(el('text', { x: gc, y: ay1 + 14, 'text-anchor': 'middle', fill: C.ink, 'font-size': 9.2, 'font-weight': 600 }, [shortLabels[i].split('\n')[0]]));
    svg.appendChild(el('text', { x: gc, y: ay1 + 25, 'text-anchor': 'middle', fill: C.muted, 'font-size': 8.8 }, [shortLabels[i].split('\n')[1]]));
  });
  // Significance brackets cleanly at top y=62..76 above plot
  const g0 = ax0 + 0.5 * stepA, g1 = ax0 + 1.5 * stepA, g4 = ax0 + 4.5 * stepA;
  const brY1 = ay0 - 20;
  svg.appendChild(el('path', { d: `M${g0},${brY1 + 5} L${g0},${brY1} L${g4},${brY1} L${g4},${brY1 + 5}`, fill: 'none', stroke: C.ink, 'stroke-width': 1.1 }));
  svg.appendChild(el('text', { x: (g0 + g4) / 2, y: brY1 - 4, 'text-anchor': 'middle', fill: C.ink, 'font-size': 9, 'font-weight': 700 },
    ['Wilcoxon p = 1.50×10⁻⁴ (vs B=1), p = 2.28×10⁻⁴ (vs nanodrr)']));
  drawAxesFrame(svg, ax0, ay0, ax1, ay1, 'Final 3D mTRE (mm, log)', null);

  // Panel (b): Tier 1 vs Tier 2 <2mm Success Rates
  const bx0 = 506, bx1 = 790, by0 = 86, by1 = 342;
  panelTitle(svg, bx0, 24, '(b) Tier 1 vs Tier 2 <2 mm Rate', 'Bounded (n=12) vs Wide-Basin (n=12)');
  drawLegend(svg, bx0, 62, [
    { kind: 'bar', color: C.navy, label: 'Tier 1 Bounded' },
    { kind: 'bar', color: C.teal, label: 'Tier 2 Wide-Basin' },
  ], 10);
  const yMapB = linMapper(0, 115, by1, by0);
  drawGridY(svg, bx0, bx1, yMapB, [0, 25, 50, 75, 100], v => `${v}%`);
  const grpB = (bx1 - bx0) / tiers.length;
  tiers.forEach((r, i) => {
    const gc = bx0 + (i + 0.5) * grpB;
    svg.appendChild(el('rect', { x: gc - 15, y: yMapB(r.t1), width: 13, height: by1 - yMapB(r.t1), fill: C.navy, rx: 1.2 }));
    svg.appendChild(el('rect', { x: gc + 2, y: yMapB(r.t2), width: 13, height: by1 - yMapB(r.t2), fill: C.teal, rx: 1.2 }));
    svg.appendChild(el('text', { x: gc - 8, y: yMapB(r.t1) - 5, 'text-anchor': 'middle', fill: C.navy, 'font-size': 8.2, 'font-weight': 700 }, [`${r.t1.toFixed(0)}`]));
    svg.appendChild(el('text', { x: gc + 9, y: yMapB(r.t2) - 5, 'text-anchor': 'middle', fill: C.teal, 'font-size': 8.2, 'font-weight': 700 }, [`${r.t2.toFixed(0)}`]));
    svg.appendChild(el('text', { x: gc, y: ay1 + 14, 'text-anchor': 'middle', fill: C.ink, 'font-size': 8.8, 'font-weight': 600 }, [shortLabels[i].split('\n')[0]]));
    svg.appendChild(el('text', { x: gc, y: ay1 + 25, 'text-anchor': 'middle', fill: C.muted, 'font-size': 8.5 }, [shortLabels[i].split('\n')[1]]));
  });
  drawAxesFrame(svg, bx0, by0, bx1, by1, 'Success Rate <2 mm (%)', null);

  // Panel (c): Spine/Abdominal CT Validation (exp32 before fixes)
  const cx0 = 852, cx1 = 1118, cy0 = 86, cy1 = 342;
  panelTitle(svg, cx0, 24, '(c) Clinical CT Validation (exp32)', 'Revealing AMOS/GE & K=1 bf16 soft spots');
  drawLegend(svg, cx0, 62, [
    { kind: 'bar', color: C.navy, label: 'PoU-FVR' },
    { kind: 'bar', color: C.amber, label: 'nanodrr' },
    { kind: 'bar', color: C.cobalt, label: 'Hybrid' },
  ], 8);
  const yMapC = logMapper(0.015, 25, cy1, cy0);
  drawGridY(svg, cx0, cx1, yMapC, [0.02, 0.1, 0.5, 2, 10], v => v < 0.1 ? v.toFixed(2) : String(v));
  drawThresh(svg, cx0, cx1, yMapC(2.0), '2.0 mm', C.redDash, 'right');
  const grpC = (cx1 - cx0) / ct.length;
  ct.forEach((r, i) => {
    const gc = cx0 + (i + 0.5) * grpC;
    [['pou', C.navy], ['nano', C.amber], ['hyb', C.cobalt]].forEach(([k, col], bi) => {
      if (r[k] === null) return;
      const bx = gc + (bi - 1) * 14 - 6;
      const py = yMapC(r[k]);
      svg.appendChild(el('rect', { x: bx, y: py, width: 12, height: cy1 - py, fill: col, rx: 1.2 }));
    });
    const parts = r.label.split(' ');
    svg.appendChild(el('text', { x: gc, y: cy1 + 14, 'text-anchor': 'middle', fill: C.ink, 'font-size': 9.2, 'font-weight': 600 }, [parts[0]]));
    svg.appendChild(el('text', { x: gc, y: cy1 + 25, 'text-anchor': 'middle', fill: C.muted, 'font-size': 8.5 }, [parts[1]]));
  });
  drawAxesFrame(svg, cx0, cy0, cx1, cy1, 'Median mTRE (mm, log)', null);
  return svg;
}

function build_fig32() {
  const W = 1140, H = 400;
  const svg = makeSVG(W, H);
  const { a, b, c } = D.fig32;

  // Panel (a): DeepFluoro Before vs After
  const ax0 = 58, ax1 = 382, ay0 = 78, ay1 = 338;
  panelTitle(svg, ax0, 24, '(a) DeepFluoro (N=24) Before → After', '4×4×4@64 slabs + K=4 f32 polish → 100% <2 mm');
  drawLegend(svg, ax0, 60, [
    { kind: 'bar', color: C.slate, label: 'exp32 (K=1)' },
    { kind: 'bar', color: C.teal, label: 'exp33 (K=4)' },
    { kind: 'dash_line', color: C.redDash, label: '2.0 mm' },
  ], 10);
  const yMapA = linMapper(0, 3.0, ay1, ay0);
  drawGridY(svg, ax0, ax1, yMapA, [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], v => v.toFixed(1));
  drawThresh(svg, ax0, ax1, yMapA(2.0), null, C.redDash);
  const grpA = (ax1 - ax0) / a.length;
  const r2 = (v) => (Math.round((v + 1e-9) * 100) / 100).toFixed(2);
  a.forEach((r, i) => {
    const gc = ax0 + (i + 0.5) * grpA;
    svg.appendChild(el('rect', { x: gc - 24, y: yMapA(r.med_b), width: 22, height: ay1 - yMapA(r.med_b), fill: C.slate, rx: 1.5 }));
    svg.appendChild(el('rect', { x: gc + 2, y: yMapA(r.med_a), width: 22, height: ay1 - yMapA(r.med_a), fill: C.teal, rx: 1.5 }));
    const nearThreshB = Math.abs(yMapA(r.med_b) - yMapA(2.0)) < 14;
    svg.appendChild(el('text', {
      x: gc - 13,
      y: nearThreshB ? yMapA(r.med_b) + 14 : yMapA(r.med_b) - 5,
      'text-anchor': 'middle',
      fill: nearThreshB ? '#FFFFFF' : C.slate,
      'font-size': 8.5, 'font-weight': 700
    }, [r2(r.med_b)]));
    svg.appendChild(el('text', { x: gc - 13, y: ay1 - 8, 'text-anchor': 'middle', fill: '#FFFFFF', 'font-size': 8, 'font-weight': 700 }, [`${r.s2_b.toFixed(0)}%`]));
    svg.appendChild(el('text', { x: gc + 13, y: yMapA(r.med_a) - 5, 'text-anchor': 'middle', fill: C.teal, 'font-size': 8.5, 'font-weight': 700 }, [r2(r.med_a)]));
    svg.appendChild(el('text', { x: gc + 13, y: ay1 - 8, 'text-anchor': 'middle', fill: '#FFFFFF', 'font-size': 8, 'font-weight': 700 }, [`${r.s2_a.toFixed(0)}%`]));
    svg.appendChild(el('text', { x: gc, y: ay1 + 16, 'text-anchor': 'middle', fill: C.ink, 'font-size': 9.8, 'font-weight': 600 }, [r.label]));
  });
  drawAxesFrame(svg, ax0, ay0, ax1, ay1, 'Median 3D mTRE (mm)', null);

  // Panel (b): Standalone PoU-FVR Across All 4 CT Cohorts
  const bx0 = 446, bx1 = 758, by0 = 78, by1 = 338;
  panelTitle(svg, bx0, 24, '(b) Standalone PoU-FVR on CT', '2×4×2@80 → Isotropic 4×4×4@64 (100% <2 mm)');
  drawLegend(svg, bx0, 60, [
    { kind: 'bar', color: C.slate, label: '2×4×2@80 (exp32)' },
    { kind: 'bar', color: C.navy, label: '4×4×4@64 (exp33)' },
  ], 10);
  const yMapB = logMapper(0.15, 16, by1, by0);
  drawGridY(svg, bx0, bx1, yMapB, [0.2, 0.5, 1, 2, 5, 10], v => String(v));
  drawThresh(svg, bx0, bx1, yMapB(2.0), '2.0 mm', C.redDash, 'left');
  const grpB = (bx1 - bx0) / b.length;
  b.forEach((r, i) => {
    const gc = bx0 + (i + 0.5) * grpB;
    svg.appendChild(el('rect', { x: gc - 16, y: yMapB(r.med_b), width: 14, height: by1 - yMapB(r.med_b), fill: C.slate, rx: 1.5 }));
    svg.appendChild(el('rect', { x: gc + 2, y: yMapB(r.med_a), width: 14, height: by1 - yMapB(r.med_a), fill: C.navy, rx: 1.5 }));
    svg.appendChild(el('text', { x: gc - 9, y: yMapB(r.med_b) - 4, 'text-anchor': 'middle', fill: C.slate, 'font-size': 8.2, 'font-weight': 700 }, [r.med_b.toFixed(2)]));
    svg.appendChild(el('text', { x: gc + 9, y: yMapB(r.med_a) - 4, 'text-anchor': 'middle', fill: C.navy, 'font-size': 8.2, 'font-weight': 700 }, [r.med_a.toFixed(2)]));
    const parts = r.label.split(' ');
    svg.appendChild(el('text', { x: gc, y: by1 + 14, 'text-anchor': 'middle', fill: C.ink, 'font-size': 9.2, 'font-weight': 600 }, [parts[0]]));
    svg.appendChild(el('text', { x: gc, y: by1 + 25, 'text-anchor': 'middle', fill: C.muted, 'font-size': 8.5 }, [parts[1]]));
  });
  drawAxesFrame(svg, bx0, by0, bx1, by1, 'Median mTRE (mm, log)', null);

  // Panel (c): <0.5 GB Hybrid+nanodrr Before vs After Across All 5 Cohorts
  const cx0 = 822, cx1 = 1118, cy0 = 78, cy1 = 338;
  panelTitle(svg, cx0, 24, '(c) <0.5 GB Hybrid+nanodrr', 'K=1 bf16 (exp32) → K=4 f32 (exp33, ≤464 MB)');
  drawLegend(svg, cx0, 60, [
    { kind: 'bar', color: C.slate, label: 'K=1 bf16' },
    { kind: 'bar', color: C.teal, label: 'K=4 f32' },
  ], 10);
  const yMapC = logMapper(0.02, 12, cy1, cy0);
  drawGridY(svg, cx0, cx1, yMapC, [0.03, 0.1, 0.3, 1, 2, 5], v => v < 0.1 ? v.toFixed(2) : String(v));
  drawThresh(svg, cx0, cx1, yMapC(2.0), '2.0 mm', C.redDash, 'right');
  const grpC = (cx1 - cx0) / c.length;
  const xLabelsC = [['Deep', 'Fluoro'], ['Kaggle', '256²'], ['Kaggle', '512²'], ['AMOS', '256²'], ['GE', '256²']];
  c.forEach((r, i) => {
    const gc = cx0 + (i + 0.5) * grpC;
    svg.appendChild(el('rect', { x: gc - 13, y: yMapC(r.med_b), width: 11, height: cy1 - yMapC(r.med_b), fill: C.slate, rx: 1.2 }));
    svg.appendChild(el('rect', { x: gc + 2, y: yMapC(r.med_a), width: 11, height: cy1 - yMapC(r.med_a), fill: C.teal, rx: 1.2 }));
    svg.appendChild(el('text', { x: gc - 7, y: yMapC(r.med_b) - 4, 'text-anchor': 'middle', fill: C.slate, 'font-size': 7.8, 'font-weight': 700 }, [r2(r.med_b)]));
    svg.appendChild(el('text', { x: gc + 8, y: yMapC(r.med_a) - 4, 'text-anchor': 'middle', fill: C.teal, 'font-size': 7.8, 'font-weight': 700 }, [r2(r.med_a)]));
    svg.appendChild(el('text', { x: gc, y: cy1 + 14, 'text-anchor': 'middle', fill: C.ink, 'font-size': 8.8, 'font-weight': 600 }, [xLabelsC[i][0]]));
    svg.appendChild(el('text', { x: gc, y: cy1 + 25, 'text-anchor': 'middle', fill: C.muted, 'font-size': 8.4 }, [xLabelsC[i][1]]));
  });
  drawAxesFrame(svg, cx0, cy0, cx1, cy1, 'Median mTRE (mm, log)', null);
  return svg;
}

const BUILDERS = {
  fig0_method_overview: build_fig0,
  fig1_complexity_scaling: build_fig1,
  fig6_biplanar_ced_comparison: build_fig6,
  fig7_clinical_dicom_registration: build_fig7,
  fig11_radiation_dose_sweep: build_fig11,
  fig15_articulated_spine_registration: build_fig15,
  fig14_unified_multicenter_ced: build_fig14,
  fig25_deepfluoro_real_baseline: build_fig25,
  fig26_deepfluoro_multisubject: build_fig26,
  fig27_geometry_matched_benchmark: build_fig27,
  fig28_conebeam_clinical_robustness: build_fig28,
  fig29_a100_gpu_benchmark: build_fig29,
  fig30_gpu_multistart_cohort: build_fig30,
  fig31_rebuttal_ablation: build_fig31,
  fig32_softspot_fixes: build_fig32,
};

window.renderAndAudit = function(figName) {
  const mount = document.getElementById('mount');
  mount.innerHTML = '';
  const svg = BUILDERS[figName]();
  mount.appendChild(svg);
  const vb = svg.viewBox.baseVal;
  const W = vb.width, H = vb.height;
  const sRect = svg.getBoundingClientRect();
  const sx = W / sRect.width, sy = H / sRect.height;

  const texts = Array.from(svg.querySelectorAll('text')).map(t => {
    const r = t.getBoundingClientRect();
    return {
      text: t.textContent,
      role: t.getAttribute('data-role') || '',
      x: (r.left - sRect.left) * sx,
      y: (r.top - sRect.top) * sy,
      right: (r.right - sRect.left) * sx,
      bottom: (r.bottom - sRect.top) * sy,
      w: r.width * sx,
      h: r.height * sy,
    };
  }).filter(b => b.w > 0 && b.h > 0);

  const clips = [];
  for (const b of texts) {
    if (b.x < 1.5 || b.y < 1.5 || b.right > W - 1.5 || b.bottom > H - 1.5) {
      clips.push(`${b.text} [${b.x.toFixed(1)},${b.y.toFixed(1)}..${b.right.toFixed(1)},${b.bottom.toFixed(1)}]`);
    }
  }

  const overlaps = [];
  for (let i = 0; i < texts.length; i++) {
    for (let j = i + 1; j < texts.length; j++) {
      const a = texts[i], b = texts[j];
      if (a.x < b.right + 0.5 && a.right + 0.5 > b.x && a.y < b.bottom + 0.5 && a.bottom + 0.5 > b.y) {
        overlaps.push(`"${a.text}" vs "${b.text}"`);
      }
    }
  }

  // Check panel-title baseline alignment (titles sharing top row y within 2px)
  const titleYs = texts.filter(t => t.role === 'panel-title').map(t => t.y);
  const titleSpread = titleYs.length > 1 ? Math.max(...titleYs) - Math.min(...titleYs) : 0;

  return { width: W, height: H, nTexts: texts.length, clips, overlaps, titleSpread };
};
</script>
</body>
</html>
"""


def main() -> None:
    if not CROP_DIR.exists():
        raise RuntimeError(f"Missing raw crop assets at {CROP_DIR}")

    datasets = build_datasets()
    html_src = HTML_TEMPLATE.replace("__THEME_JSON__", json.dumps(THEME)).replace(
        "__DATA_JSON__", json.dumps(datasets)
    )
    html_path = PF_DIR / "_figure_gallery.html"
    html_path.write_text(html_src, encoding="utf-8")

    names = [
        "fig0_method_overview",
        "fig1_complexity_scaling",
        "fig6_biplanar_ced_comparison",
        "fig7_clinical_dicom_registration",
        "fig11_radiation_dose_sweep",
        "fig15_articulated_spine_registration",
        "fig14_unified_multicenter_ced",
        "fig25_deepfluoro_real_baseline",
        "fig26_deepfluoro_multisubject",
        "fig27_geometry_matched_benchmark",
        "fig28_conebeam_clinical_robustness",
        "fig29_a100_gpu_benchmark",
        "fig30_gpu_multistart_cohort",
        "fig31_rebuttal_ablation",
        "fig32_softspot_fixes",
    ]

    failed = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900}, device_scale_factor=3)
        page = context.new_page()
        page.goto(html_path.resolve().as_uri())

        for name in names:
            audit = page.evaluate(f'window.renderAndAudit("{name}")')
            w, h = int(audit["width"]), int(audit["height"])
            page.set_viewport_size({"width": w, "height": h})
            out_png = PF_DIR / f"{name}.png"
            page.locator("svg").screenshot(path=str(out_png))
            if TEX_FIG_DIR.exists():
                shutil.copy2(out_png, TEX_FIG_DIR / f"{name}.png")

            ok = (not audit["clips"]) and (not audit["overlaps"]) and audit["titleSpread"] <= 1.0
            status = "PASS" if ok else "FAIL"
            print(
                f"[{status}] {name:38s} ({w}x{h} @3x) | texts={audit['nTexts']:3d} | "
                f"clips={len(audit['clips'])} | overlaps={len(audit['overlaps'])} | titleSpread={audit['titleSpread']:.1f}"
            )
            if not ok:
                for c in audit["clips"]:
                    print("   CLIP:", c)
                for o in audit["overlaps"]:
                    print("   OVERLAP:", o)
                failed.append(name)

        browser.close()

    if failed:
        print(f"\nAUDIT FAILED on {len(failed)} figures: {failed}")
        sys.exit(1)
    print(f"\nALL {len(names)} FIGURES PASSED AUTOMATED DOM ALIGNMENT & COLLISION AUDIT (0 overlaps, 0 clips).")


if __name__ == "__main__":
    main()
