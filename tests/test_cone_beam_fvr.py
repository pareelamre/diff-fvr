import sys
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fvr.cone_beam_fvr import PerspectiveSlabFVR, _make_pou_1d, edge_dominant_ncc
from fvr.triton_slice import fourier_slice_3d_torch, trilinear_slice_analytical_ref


def _ncc_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a0 = a - a.mean()
    b0 = b - b.mean()
    return 1.0 - (a0 * b0).sum() / (a0.norm() * b0.norm() + 1e-8)


class TestConeBeamFVR(unittest.TestCase):
    def test_pou_windows_sum_to_unity(self):
        for length, k in [(64, 2), (96, 3), (128, 4)]:
            ws, _ = _make_pou_1d(length, k, overlap=0.25)
            total = torch.stack(ws, dim=0).sum(dim=0)
            self.assertTrue(torch.allclose(total, torch.ones(length), atol=1e-5))

    def test_perspective_differential_magnification_front_vs_back(self):
        """Front-vs-back off-isocenter spheres must project to distinct detector radii under cone-beam."""
        n = 64
        vox_mm = 2.0
        vol = np.zeros((n, n, n), dtype=np.float32)  # (Z, Y, X)
        zz, yy, xx = np.meshgrid(
            (np.arange(n) - (n - 1) / 2.0) * vox_mm,
            (np.arange(n) - (n - 1) / 2.0) * vox_mm,
            (np.arange(n) - (n - 1) / 2.0) * vox_mm,
            indexing="ij",
        )
        sphere_front = ((zz - 16.0) ** 2 + (yy + 32.0) ** 2 + (xx - 24.0) ** 2) < 9.0 ** 2
        sphere_back = ((zz + 16.0) ** 2 + (yy - 32.0) ** 2 + (xx - 24.0) ** 2) < 9.0 ** 2
        vol[sphere_front] = 1.0
        vol[sphere_back] = 1.0

        fvr_cone = PerspectiveSlabFVR.from_volume(vol, vox_mm=vox_mm, grid_blocks=(2, 4, 2), block_res=48)
        z0 = torch.tensor(0.0)
        img = fvr_cone.forward_carm_view(z0, z0, z0, z0, z0, z0, sad_mm=300.0, sdd_mm=600.0, det_px=96, det_fov_mm=n * vox_mm, view="ap")

        u_coords = torch.linspace(-0.5, 0.5, 96) * (n * vox_mm)
        top_half = img[48:, :]
        bot_half = img[:48, :]
        x_front = float((top_half.sum(dim=0) * u_coords).sum() / top_half.sum())
        x_back = float((bot_half.sum(dim=0) * u_coords).sum() / bot_half.sum())
        self.assertGreater(x_front, x_back + 7.0, f"Expected front sphere > back sphere by >7 mm, got {x_front:.2f} vs {x_back:.2f}")

    def test_cone_beam_gradients_and_symmetry(self):
        n = 48
        vol = np.zeros((n, n, n), dtype=np.float32)
        zz, yy, xx = np.ogrid[:n, :n, :n]
        vol[((zz - 24) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2) < 12 ** 2] = 0.05

        cb = PerspectiveSlabFVR.from_volume(vol, vox_mm=2.0, grid_blocks=(2, 2, 2), block_res=40)
        target = cb.forward_carm_view(
            torch.tensor(3.0), torch.tensor(-2.0), torch.tensor(1.5),
            torch.tensor(1.0), torch.tensor(-1.0), torch.tensor(0.5),
            sad_mm=600.0, sdd_mm=900.0, det_px=48, det_fov_mm=96.0, view="ap"
        ).detach()

        params = [torch.tensor(0.0, requires_grad=True) for _ in range(6)]
        pred = cb.forward_carm_view(*params, sad_mm=600.0, sdd_mm=900.0, det_px=48, det_fov_mm=96.0, view="ap")
        loss = _ncc_loss(pred, target)
        loss.backward()
        for p in params:
            self.assertIsNotNone(p.grad)
            self.assertTrue(bool(torch.isfinite(p.grad)))
        self.assertTrue(any(abs(float(p.grad)) > 1e-4 for p in params))

    def test_grid_center_vs_pixel_center_origin(self):
        vol = np.ones((32, 32, 32), dtype=np.float32)
        fvr_px = PerspectiveSlabFVR.from_volume(vol, vox_mm=2.5, grid_blocks=(2, 2, 2), block_res=32, center_mode="pixel_center")
        fvr_gc = PerspectiveSlabFVR.from_volume(vol, vox_mm=2.5, grid_blocks=(2, 2, 2), block_res=32, center_mode="grid_center")
        delta = fvr_px.centers - fvr_gc.centers
        self.assertTrue(torch.allclose(delta, torch.full_like(delta, 1.25), atol=1e-4))
        with self.assertRaises(ValueError):
            PerspectiveSlabFVR.from_volume(vol, vox_mm=1.0, center_mode="invalid")

    def test_edge_dominant_ncc_vignetting_invariance(self):
        H = 96
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, H), indexing="ij")
        img = torch.exp(-((xx - 0.1) ** 2 + (yy + 0.05) ** 2) / 0.04)
        img = img + 0.5 * ((xx ** 2 + yy ** 2) < 0.25).float()
        ref = img[None, None]
        vignette = ref.clone() + 0.8 * xx[None, None]
        vignette[:, :, :6, :] = -10.0
        vignette[:, :, -6:, :] = -10.0
        shifted = torch.roll(ref, shifts=(4, 4), dims=(-2, -1))
        self.assertGreater(float(edge_dominant_ncc(ref, ref)), 0.999)
        self.assertGreater(float(edge_dominant_ncc(ref, vignette)), 0.98)
        self.assertLess(float(edge_dominant_ncc(shifted, vignette)), 0.50)

    def test_batched_projection_pinhole_and_half_precision_parity(self):
        """Batched PoU-FVR, forward_pinhole, and bfloat16 storage match single-pose forward & gradients."""
        torch.manual_seed(7)
        n = 32
        vol = np.zeros((n, n, n), dtype=np.float32)
        zz, yy, xx = np.ogrid[:n, :n, :n]
        vol[((zz - 14) ** 2 + (yy - 17) ** 2 + (xx - 16) ** 2) < 9 ** 2] = 0.08
        fvr = PerspectiveSlabFVR.from_volume(vol, vox_mm=2.0, grid_blocks=(2, 2, 2), block_res=32)

        H, W = 48, 48
        k_inv = torch.tensor(
            [[[1.0 / 300.0, 0.0, -W / (2.0 * 300.0)], [0.0, 1.0 / 300.0, -H / (2.0 * 300.0)], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )
        lattice = fvr.make_pinhole_lattice(k_inv, sdd_mm=600.0, height=H, width=W)

        mats = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(4, 1, 1)
        mats[:, 2, 3] = -300.0
        mats[:, 0, 3] = torch.tensor([0.0, 4.0, -3.0, 6.0])
        mats[:, 1, 3] = torch.tensor([0.0, -2.0, 5.0, 1.0])
        mats.requires_grad_(True)

        batched = fvr.forward_pinhole(mats, lattice)  # (4, 1, H, W)
        self.assertEqual(batched.shape, (4, 1, H, W))
        for b in range(4):
            single = fvr.forward_pinhole(mats[b : b + 1].detach(), lattice)
            self.assertTrue(torch.allclose(batched[b : b + 1], single, atol=1e-4))
        batched.sum().backward()
        self.assertIsNotNone(mats.grad)
        self.assertTrue(bool(torch.all(torch.isfinite(mats.grad))))
        self.assertGreater(float(mats.grad.abs().sum()), 1e-4)

        # Verify safe CPU execution after casting Fourier blocks to bfloat16
        fvr_bf16 = PerspectiveSlabFVR.from_volume(vol, vox_mm=2.0, grid_blocks=(2, 2, 2), block_res=32).to_precision(
            torch.bfloat16
        )
        m_half = mats.detach().clone().requires_grad_(True)
        out_half = fvr_bf16.forward_pinhole(m_half, lattice)
        out_half.sum().backward()
        self.assertEqual(out_half.dtype, torch.float32)
        self.assertTrue(torch.allclose(out_half, batched.detach(), atol=0.15, rtol=0.05))
        self.assertGreater(float(m_half.grad.abs().sum()), 1e-4)

    def test_trilinear_slice_analytical_parity(self):
        """Triton-matching explicit 8-neighbor stencil matches grid_sample forward & autograd gradients."""
        torch.manual_seed(11)
        B_vol, D, H, W, N_pose, P = 3, 24, 24, 24, 4, 20
        vol = torch.randn(B_vol, 2, D, H, W, dtype=torch.float32)
        coords = torch.linspace(-0.45, 0.45, P)
        fy, fx = torch.meshgrid(coords, coords, indexing="ij")
        fx_flat, fy_flat = fx.reshape(-1), fy.reshape(-1)
        u0 = torch.randn(N_pose, 3) * 8.0
        v0 = torch.randn(N_pose, 3) * 8.0
        c = (11.5, 11.5, 11.5)

        u_a = u0.clone().requires_grad_(True)
        v_a = v0.clone().requires_grad_(True)
        out_torch = fourier_slice_3d_torch(vol, u_a, v_a, fx_flat, fy_flat, c)
        w_out = torch.randn_like(out_torch)
        (out_torch * w_out).sum().backward()

        u_b = u0.clone().requires_grad_(True)
        v_b = v0.clone().requires_grad_(True)
        out_ref = trilinear_slice_analytical_ref(vol, u_b, v_b, fx_flat, fy_flat, c)
        (out_ref * w_out).sum().backward()

        self.assertTrue(torch.allclose(out_torch, out_ref, atol=1e-5))
        self.assertTrue(torch.allclose(u_a.grad, u_b.grad, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(v_a.grad, v_b.grad, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
