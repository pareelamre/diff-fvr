"""
Integration and Autograd Test Suite for TorchDiffFVREngine.
Author: Pareel Amre, TU Dresden
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import torch
import numpy as np
from src.fvr.torch_engine import TorchDiffFVREngine
from src.fvr.dicom_loader import create_spine_phantom, hu_to_attenuation


class TestTorchDiffFVREngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a compact synthetic spine phantom
        vol_hu, _ = create_spine_phantom(size=64)
        vol_mu = hu_to_attenuation(vol_hu)
        cls.vol_tensor = torch.from_numpy(vol_mu).float()
        cls.engine = TorchDiffFVREngine(cls.vol_tensor, spacing=(1.5, 1.5, 1.5), pad_factor=2.0)

    def test_forward_batch_synthesis(self):
        """Test batched DRR synthesis for AP and Lateral projections."""
        # Batch of 4 poses
        poses = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, -3.0, 1.5, 4.0, -2.0, 1.0],
        ], dtype=torch.float32)

        drrs_ap = self.engine(poses, view="ap", out_res=64)
        self.assertEqual(drrs_ap.shape, (4, 1, 64, 64))
        self.assertTrue(torch.all(torch.isfinite(drrs_ap)))
        self.assertTrue(torch.all(drrs_ap >= 0.0))

        drrs_lat = self.engine(poses, view="lateral", out_res=64)
        self.assertEqual(drrs_lat.shape, (4, 1, 64, 64))
        self.assertTrue(torch.all(torch.isfinite(drrs_lat)))

    def test_autograd_pose_gradients(self):
        """Test end-to-end PyTorch autograd backpropagation through Fourier volume sampling."""
        pose = torch.tensor([[3.0, -2.0, 1.0, 5.0, -3.0, 2.0]], dtype=torch.float32, requires_grad=True)
        drr = self.engine(pose, view="ap", out_res=64)

        loss = torch.sum(drr ** 2)
        loss.backward()

        self.assertIsNotNone(pose.grad)
        self.assertEqual(pose.grad.shape, (1, 6))
        self.assertTrue(torch.all(torch.isfinite(pose.grad)))
        self.assertFalse(torch.all(pose.grad == 0.0), "Gradients should be non-zero")

    def test_spectral_ncc_adam_optimization(self):
        """Test end-to-end gradient-based optimization using Adam on PyTorch tensors."""
        true_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        with torch.no_grad():
            target_drr = self.engine(true_pose, view="ap", out_res=64)

        # Perturbed initial pose
        est_pose = torch.tensor([[2.0, -1.5, 0.0, 3.0, -2.0, 0.0]], dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([est_pose], lr=0.15)

        initial_loss = None
        final_loss = None

        for step in range(12):
            optimizer.zero_grad()
            pred_drr = self.engine(est_pose, view="ap", out_res=64)
            loss = self.engine.spectral_ncc_loss(pred_drr, target_drr, high_pass=True)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        self.assertLess(final_loss, initial_loss, "Adam optimization should decrease Spectral-NCC loss")


if __name__ == "__main__":
    unittest.main()
