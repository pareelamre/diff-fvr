"""
Experiment 29: CUDA GPU Triton / torch.compile Benchmark & PoU-FVR Pareto Ablation.

Benchmarks Diff-FVR (Parallel + Zero-IFFT Spectral-NCC) and Cone-Beam PoU-FVR
(PyTorch, Triton fused 3D slice, bfloat16, torch.compile / CUDA Graphs) head-to-head
against nanodrr (PyTorch + Triton + compile + bfloat16) and DiffDRR on DeepFluoro subject01.
Also evaluates the grid_blocks x block_res Pareto frontier for IEEE TMI / MICCAI.
"""

import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fvr import PerspectiveSlabFVR, TorchDiffFVREngine
from fvr.triton_slice import fourier_slice_3d, fourier_slice_3d_torch, HAS_TRITON


def _mark_cudagraph_step():
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()


def _eval_nograd(fn: Callable):
    with torch.no_grad():
        return fn()


def _cuda_time_ms(fn: Callable, warmup: int = 12, runs: int = 50) -> Tuple[float, float, float]:
    """Return (median_wall_ms, median_gpu_ms, peak_working_set_mb) on CUDA or CPU."""
    is_cuda = torch.cuda.is_available()
    for _ in range(warmup):
        if is_cuda:
            _mark_cudagraph_step()
        fn()
    if is_cuda:
        torch.cuda.synchronize()
        base_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        base_bytes = 0

    wall_ts: List[float] = []
    gpu_ts: List[float] = []
    for _ in range(runs):
        if is_cuda:
            torch.cuda.synchronize()
            _mark_cudagraph_step()
            t0 = time.perf_counter()
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            wall_ts.append((time.perf_counter() - t0) * 1000.0)
            gpu_ts.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            fn()
            wall_ts.append((time.perf_counter() - t0) * 1000.0)
            gpu_ts.append(wall_ts[-1])

    peak_extra_mb = (
        max(0, torch.cuda.max_memory_allocated() - base_bytes) / (1024.0 * 1024.0) if is_cuda else 0.0
    )
    return float(statistics.median(wall_ts)), float(statistics.median(gpu_ts)), float(peak_extra_mb)


def _capture_cuda_graph_fn(fn: Callable, rt_static: torch.Tensor) -> Callable:
    """Capture a fixed-shape CUDA Graph around ``fn(rt_static)`` for zero Python launch overhead."""
    for _ in range(3):
        fn(rt_static)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    out_buf = [None]
    with torch.cuda.graph(g):
        out_buf[0] = fn(rt_static)
    torch.cuda.synchronize()

    def _replay():
        g.replay()
        return out_buf[0]

    return _replay


def _ncc(a: torch.Tensor, b: torch.Tensor) -> float:
    a0 = a.float().flatten()
    b0 = b.float().flatten()
    a0 = a0 - a0.mean()
    b0 = b0 - b0.mean()
    return float((a0 * b0).sum() / (a0.norm() * b0.norm() + 1e-8))


def verify_triton_parity(device: torch.device) -> None:
    """Verify CUDA Triton 3D Fourier-slice kernel matches F.grid_sample forward & backward."""
    if not (device.type == "cuda" and HAS_TRITON):
        print("[Skip] Triton CUDA kernel parity skipped (no CUDA/Triton runtime).")
        return

    torch.manual_seed(0)
    B_vol, D, H, W, N_pose, P = 4, 32, 32, 32, 4, 24
    vol = torch.randn(B_vol, 2, D, H, W, device=device, dtype=torch.float32)
    coords = torch.linspace(-0.45, 0.45, P, device=device)
    fy, fx = torch.meshgrid(coords, coords, indexing="ij")
    fx_f, fy_f = fx.reshape(-1), fy.reshape(-1)
    u0 = torch.randn(N_pose, 3, device=device) * 10.0
    v0 = torch.randn(N_pose, 3, device=device) * 10.0
    c = (16.0, 16.0, 16.0)

    u_ref = u0.clone().requires_grad_(True)
    v_ref = v0.clone().requires_grad_(True)
    out_ref = fourier_slice_3d_torch(vol, u_ref, v_ref, fx_f, fy_f, c)
    w = torch.randn_like(out_ref)
    (out_ref * w).sum().backward()

    u_tri = u0.clone().requires_grad_(True)
    v_tri = v0.clone().requires_grad_(True)
    out_tri = fourier_slice_3d(vol, u_tri, v_tri, fx_f, fy_f, c, use_triton=True)
    (out_tri * w).sum().backward()

    fwd_err = float((out_tri - out_ref).abs().max())
    gu_err = float((u_tri.grad - u_ref.grad).abs().max())
    gv_err = float((v_tri.grad - v_ref.grad).abs().max())
    print(f"[Triton CUDA Parity] max_fwd_err={fwd_err:.2e}, max_gu_err={gu_err:.2e}, max_gv_err={gv_err:.2e}")
    assert fwd_err < 1e-4, f"Triton forward mismatch: {fwd_err}"
    assert gu_err < 1e-3 and gv_err < 1e-3, f"Triton gradient mismatch: {gu_err}, {gv_err}"


def load_deepfluoro_subject01() -> str:
    local = ROOT / "data" / "deepfluoro" / "deepfluoro" / "subject01" / "volume.nii.gz"
    if local.exists():
        return str(local)
    from nanodrr.data import download_deepfluoro

    img_path, _ = download_deepfluoro(1)
    return img_path


def run_benchmark() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"=== Experiment 29 GPU Benchmark | Device={device} ({gpu_label}) | Triton={HAS_TRITON} ===")
    verify_triton_parity(device)

    from diffdrr.data import read
    from diffdrr.drr import DRR as DiffDRR
    from diffdrr.pose import convert
    from nanodrr.camera import make_k_inv, make_rt_inv
    from nanodrr.data import Subject as NanoSubject
    from nanodrr.drr import DRR as NanoDRR

    vol_path = load_deepfluoro_subject01()
    d_subj_cpu = read(vol_path, orientation="AP")
    d_subj = d_subj_cpu.to(device)
    n_subj_f32 = NanoSubject.from_filepath(vol_path).to(dtype=torch.float32, device=device)
    n_subj_bf16 = NanoSubject.from_filepath(vol_path).to(dtype=torch.bfloat16, device=device) if device.type == "cuda" else n_subj_f32

    t0 = time.perf_counter()
    fvr_pou_f32 = PerspectiveSlabFVR.from_diffdrr_subject(
        d_subj_cpu, grid_blocks=(2, 2, 2), block_res=64, pad_mult=2, use_triton=False
    ).to(device)
    t_prep_pou = time.perf_counter() - t0

    fvr_pou_tri = PerspectiveSlabFVR.from_diffdrr_subject(
        d_subj_cpu, grid_blocks=(2, 2, 2), block_res=64, pad_mult=2, use_triton=True
    ).to(device)
    if device.type == "cuda":
        fvr_pou_tri.to_precision(torch.bfloat16)

    fvr_pou_comp = PerspectiveSlabFVR.from_diffdrr_subject(
        d_subj_cpu, grid_blocks=(2, 2, 2), block_res=64, pad_mult=2, use_triton=False
    ).to(device)
    if device.type == "cuda":
        fvr_pou_comp.to_precision(torch.bfloat16).compile_renderer(mode="reduce-overhead")

    vol_128 = torch.nn.functional.interpolate(
        d_subj_cpu.density.data.unsqueeze(0), size=(128, 128, 128), mode="trilinear", align_corners=False
    ).squeeze().numpy()
    fvr_torch = TorchDiffFVREngine(vol_128, pad_factor=2, device=device, use_triton=True)
    if device.type == "cuda":
        fvr_torch.to_precision(torch.bfloat16)
    fvr_torch_comp = TorchDiffFVREngine(vol_128, pad_factor=2, device=device, use_triton=False)
    if device.type == "cuda":
        fvr_torch_comp.to_precision(torch.bfloat16).compile_renderer(mode="reduce-overhead")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    rows: List[Dict[str, object]] = []

    warmup = 15 if device.type == "cuda" else 1
    runs = 60 if device.type == "cuda" else 3

    for H in [128, 200, 256, 512]:
        delx = 2.0 * (128.0 / H)
        k_inv = make_k_inv(sdd=1020.0, delx=delx, dely=delx, x0=0.0, y0=0.0, height=H, width=H).to(device)
        lat = PerspectiveSlabFVR.make_pinhole_lattice(k_inv, sdd_mm=1020.0, height=H, width=H)

        for B in [1, 16, 64]:
            if device.type != "cuda" and ((H == 512 and B > 1) or B == 64):
                continue
            rot = torch.tensor([[5.0, -3.0, 4.0]], device=device).expand(B, 3) + torch.randn(B, 3, device=device) * 2.0
            xyz = torch.tensor([[0.0, 850.0, 0.0]], device=device).expand(B, 3)
            rt_inv = make_rt_inv(rot, xyz, orientation="AP", isocenter=n_subj_f32.isocenter)
            nano_cone = NanoDRR(k_inv=k_inv, sdd=torch.tensor([1020.0], device=device), height=H, width=H)
            nano_comp = torch.compile(nano_cone, mode="reduce-overhead") if device.type == "cuda" else nano_cone

            if B == 1:
                with torch.no_grad():
                    ref_nano = nano_cone(n_subj_f32, rt_inv, n_samples=500, backend="torch" if device.type != "cuda" else "auto")
                    ref_pou = fvr_pou_f32.forward_pinhole(rt_inv, lat)
                    ref_tri = fvr_pou_tri.forward_pinhole(rt_inv, lat)
                print(
                    f"\n[H={H}x{H}] PoU vs nanodrr(n=500) NCC: fp32={_ncc(ref_pou, ref_nano):.4f}, "
                    f"triton+bf16={_ncc(ref_tri, ref_nano):.4f}"
                )

            specs: List[Tuple[str, Callable, bool]] = []
            if B == 1 and H <= 256:
                drr_diff = DiffDRR(d_subj, sdd=1020.0, height=H, delx=delx).to(device)
                p_diff = convert(
                    torch.zeros((1, 3), device=device),
                    torch.tensor([[0.0, 850.0, 0.0]], device=device),
                    parameterization="euler_angles",
                    convention="ZYX",
                )
                specs.append(("DiffDRR (f32 fwd)", lambda m=drr_diff, p=p_diff: m(p), False))

            specs.append(("nanodrr torch (n=500 fwd)", lambda m=nano_cone, s=n_subj_f32, p=rt_inv: m(s, p, n_samples=500, backend="torch"), False))
            if device.type == "cuda":
                specs.append(("nanodrr triton (bf16 n=500 fwd)", lambda m=nano_cone, s=n_subj_bf16, p=rt_inv: m(s, p, n_samples=500), False))
                specs.append(("nanodrr triton+compile (bf16 n=500 fwd)", lambda m=nano_comp, s=n_subj_bf16, p=rt_inv: m(s, p, n_samples=500), False))
                specs.append(("nanodrr triton+compile (bf16 n=200 fwd)", lambda m=nano_comp, s=n_subj_bf16, p=rt_inv: m(s, p, n_samples=200), False))

            specs.append(("PoU-FVR Pinhole (2x2x2 f32 fwd)", lambda m=fvr_pou_f32, p=rt_inv, l=lat: m.forward_pinhole(p, l), False))
            if device.type == "cuda" and HAS_TRITON:
                specs.append(("PoU-FVR Triton (2x2x2 bf16 fwd)", lambda m=fvr_pou_tri, p=rt_inv, l=lat: m.forward_pinhole(p, l), False))
                try:
                    cg_fn = _capture_cuda_graph_fn(lambda p, m=fvr_pou_tri, l=lat: m.forward_pinhole(p, l), rt_inv)
                    specs.append(("PoU-FVR Triton+CUDAGraph (bf16 fwd)", cg_fn, False))
                except Exception as exc:
                    print(f"  [Warn] CUDAGraph capture skipped for H={H}, B={B}: {exc}")
            if device.type == "cuda":
                specs.append(("PoU-FVR compile (2x2x2 bf16 fwd)", lambda m=fvr_pou_comp, p=rt_inv, l=lat: m.forward_pinhole(p, l), False))

            poses_b = torch.randn(B, 6, device=device)
            f_targ = fvr_torch.prepare_spectral_target(torch.randn(1, 1, H, H, device=device))
            specs.append(("TorchDiffFVR Zero-IFFT NCC", lambda m=fvr_torch_comp, p=poses_b, ft=f_targ: m.forward_spectral_ncc(p, ft), False))

            if B == 1:
                def _nano_fb(m=nano_cone, s=n_subj_bf16, p=rt_inv):
                    rp = p.detach().clone().requires_grad_(True)
                    m(s, rp, n_samples=500).sum().backward()

                def _pou_fb(m=fvr_pou_tri if device.type == "cuda" else fvr_pou_f32, p=rt_inv, l=lat):
                    rp = p.detach().clone().requires_grad_(True)
                    m.forward_pinhole(rp, l).sum().backward()

                if device.type == "cuda":
                    specs.append(("nanodrr triton (bf16 n=500 fwd+bwd)", _nano_fb, True))
                specs.append(("PoU-FVR Pinhole (2x2x2 fwd+bwd)", _pou_fb, True))

            print(f"--- H={H}x{H}, Batch B={B} ---")
            for name, fn, is_fb in specs:
                run_fn = fn if is_fb else (lambda f=fn: _eval_nograd(f))
                wall_ms, gpu_ms, peak_mb = _cuda_time_ms(run_fn, warmup=warmup, runs=runs)
                fps = (B * 1000.0) / wall_ms if wall_ms > 0 else 0.0
                print(
                    f"  {name:<42} wall={wall_ms:8.3f} ms | gpu={gpu_ms:8.3f} ms | "
                    f"{fps:10.1f} FPS | work_mem={peak_mb:6.1f} MB"
                )
                rows.append({
                    "device": gpu_label,
                    "H": H,
                    "B": B,
                    "method": name,
                    "wall_ms": round(wall_ms, 4),
                    "gpu_ms": round(gpu_ms, 4),
                    "fps": round(fps, 2),
                    "working_mem_mb": round(peak_mb, 2),
                    "prep_s": round(t_prep_pou, 3) if "PoU-FVR" in name else 0.0,
                })

    print("\n=== PoU-FVR Block / Resolution Pareto Ablation (H=256x256, sdd=1020mm) ===")
    H = 256
    delx = 2.0 * (128.0 / H)
    k_inv = make_k_inv(sdd=1020.0, delx=delx, dely=delx, x0=0.0, y0=0.0, height=H, width=H).to(device)
    lat = PerspectiveSlabFVR.make_pinhole_lattice(k_inv, sdd_mm=1020.0, height=H, width=H)
    rt_inv = make_rt_inv(
        torch.tensor([[5.0, -3.0, 4.0]], device=device),
        torch.tensor([[0.0, 850.0, 0.0]], device=device),
        orientation="AP",
        isocenter=n_subj_f32.isocenter,
    )
    nano_ref = NanoDRR(k_inv=k_inv, sdd=torch.tensor([1020.0], device=device), height=H, width=H)
    with torch.no_grad():
        trg_ref = nano_ref(n_subj_f32, rt_inv, n_samples=500, backend="torch" if device.type != "cuda" else "auto")

    ablation_rows: List[Dict[str, object]] = []
    for gb in [(1, 1, 1), (2, 2, 2), (2, 4, 2)]:
        for bres in [48, 64, 80]:
            if gb == (1, 1, 1) and bres == 48:
                continue
            t_p0 = time.perf_counter()
            m = PerspectiveSlabFVR.from_diffdrr_subject(
                d_subj_cpu, grid_blocks=gb, block_res=bres, pad_mult=2, use_triton=(device.type == "cuda")
            ).to(device)
            t_p = time.perf_counter() - t_p0
            if device.type == "cuda":
                m.to_precision(torch.bfloat16)
            with torch.no_grad():
                pred = m.forward_pinhole(rt_inv, lat)
            ncc_val = _ncc(pred, trg_ref)
            w_ms, g_ms, p_mb = _cuda_time_ms(
                lambda mod=m: _eval_nograd(lambda: mod.forward_pinhole(rt_inv, lat)), warmup=warmup, runs=runs
            )
            print(
                f"  blocks={gb}, block_res={bres:2d}, active_B={m.B:2d}: prep={t_p:.2f}s | "
                f"gpu={g_ms:.3f} ms | wall={w_ms:.3f} ms | NCC={ncc_val:.4f} | work_mem={p_mb:.1f} MB"
            )
            ablation_rows.append({
                "device": gpu_label,
                "grid_blocks": str(gb),
                "block_res": bres,
                "active_blocks": m.B,
                "prep_s": round(t_p, 3),
                "wall_ms": round(w_ms, 4),
                "gpu_ms": round(g_ms, 4),
                "ncc_vs_raymarch": round(ncc_val, 5),
                "working_mem_mb": round(p_mb, 2),
            })

    with open(out_dir / "exp29_gpu_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(out_dir / "exp29_pou_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        w.writeheader()
        w.writerows(ablation_rows)
    print(f"\nSaved CSVs to {out_dir / 'exp29_gpu_benchmark.csv'} and {out_dir / 'exp29_pou_ablation.csv'}")


if __name__ == "__main__":
    run_benchmark()
