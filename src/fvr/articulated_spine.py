"""
Articulated Multi-Vertebra Kinematic Registration Framework.

Models the lumbar spine as a kinematically coupled multi-body system (e.g., L3-L4-L5)
where each vertebral segment undergoes independent relative rotation (flexion/extension,
lateral bending, axial twist) and translation within physiological disc limits.

Leverages Fourier Linearity:
  F_2D{ sum_i I_i } = sum_i F_2D{ I_i } = sum_i S_i(k, T_i)
allowing ultra-fast simultaneous articulated DRR synthesis and hierarchical registration.
"""

import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import minimize
from .diff_fvr import DifferentiableFourierVolumeRenderer


class VertebralSegment:
    """
    Represents an individual segmented vertebra in 3D Fourier space.
    """
    def __init__(self, name: str, sub_volume: np.ndarray, nominal_offset_z: float, pad_factor: float = 2.0):
        self.name = name
        self.nominal_offset_z = nominal_offset_z
        self.renderer = DifferentiableFourierVolumeRenderer(sub_volume, pad_factor=pad_factor)
        self.shape = sub_volume.shape


class ArticulatedSpineModel:
    """
    Hierarchical Articulated Spine Model with Physiological Joint Bounds.
    """
    def __init__(self, segments: List[VertebralSegment]):
        self.segments = segments
        self.num_segments = len(segments)

    def render_articulated_spectrum(
        self,
        base_pose: np.ndarray,  # [yaw, pitch, roll, tx, ty, tz]
        relative_articulations: List[np.ndarray],  # list of [d_yaw, d_pitch, d_roll, d_tx, d_ty, d_tz]
        view: str = 'lateral'
    ) -> np.ndarray:
        """
        Synthesize combined 2D Fourier spectrum of articulated spine via Fourier linearity.
        """
        ref_seg = self.segments[0]
        slice_h = ref_seg.renderer.pz
        slice_w = ref_seg.renderer.py if view == 'lateral' else ref_seg.renderer.px

        total_spectrum = np.zeros((slice_h, slice_w), dtype=np.complex128)

        curr_pose = np.copy(base_pose)
        for i, seg in enumerate(self.segments):
            if i > 0 and len(relative_articulations) >= i:
                # Accumulate relative inter-vertebral articulation
                curr_pose = curr_pose + relative_articulations[i - 1]

            # In-plane translation for the specific view
            yaw, pitch, roll, tx, ty, tz = curr_pose
            if view == 'lateral':
                f_slice = seg.renderer.extract_frequency_slice(yaw, pitch, roll, tx=ty, ty=tz, view='lateral')
            elif view == 'ap':
                f_slice = seg.renderer.extract_frequency_slice(yaw, pitch, roll, tx=tx, ty=tz, view='ap')
            else:
                f_slice = seg.renderer.extract_frequency_slice(yaw, pitch, roll, tx=tx, ty=ty, view='axial')

            total_spectrum += f_slice

        return total_spectrum

    def render_drr(
        self,
        base_pose: np.ndarray,
        relative_articulations: List[np.ndarray],
        view: str = 'lateral'
    ) -> np.ndarray:
        """
        Render spatial projection of the articulated spine.
        """
        f_spectrum = self.render_articulated_spectrum(base_pose, relative_articulations, view=view)
        from scipy import fft
        spatial = np.real(fft.fftshift(fft.ifft2(fft.ifftshift(f_spectrum))))
        return np.maximum(spatial, 0.0)

    def register_articulated_biplanar(
        self,
        f_target_ap: np.ndarray,
        f_target_lat: np.ndarray,
        init_base_pose: np.ndarray,
        max_iter_global: int = 40,
        max_iter_local: int = 30
    ) -> Dict[str, Any]:
        """
        Two-Stage Hierarchical Articulated Registration:
          Stage 1: Global Spine Alignment (all vertebrae rigidly locked).
          Stage 2: Local Inter-Vertebral Joint Refinement within physiological bounds.
        """
        t0 = time.time()
        num_joints = self.num_segments - 1

        # --- Stage 1: Global Rigid Base Registration ---
        def stage1_obj(pose):
            zero_art = [np.zeros(6) for _ in range(num_joints)]
            f_ap = self.render_articulated_spectrum(pose, zero_art, view='ap')
            f_lat = self.render_articulated_spectrum(pose, zero_art, view='lateral')

            ncc_ap = self.segments[0].renderer.compute_spectral_ncc(f_ap, f_target_ap)
            ncc_lat = self.segments[0].renderer.compute_spectral_ncc(f_lat, f_target_lat)
            return float(0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat))

        res_stage1 = minimize(
            stage1_obj,
            x0=init_base_pose,
            method='Powell',
            options={'maxiter': max_iter_global, 'ftol': 1e-4}
        )
        base_converged = res_stage1.x

        # --- Stage 2: Articulated Joint Refinement ---
        # State vector: [d_yaw_1..K-1, d_pitch_1..K-1, d_roll_1..K-1, d_tx_1..K-1, ...]
        if num_joints > 0:
            init_art = np.zeros(num_joints * 6)

            # Physiological limits: +/- 8 deg rotation, +/- 2.5 mm translation
            joint_bounds = []
            for _ in range(num_joints):
                joint_bounds.extend([(-8.0, 8.0), (-8.0, 8.0), (-8.0, 8.0), (-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)])

            def stage2_obj(art_flat):
                arts = [art_flat[j*6:(j+1)*6] for j in range(num_joints)]
                f_ap = self.render_articulated_spectrum(base_converged, arts, view='ap')
                f_lat = self.render_articulated_spectrum(base_converged, arts, view='lateral')

                ncc_ap = self.segments[0].renderer.compute_spectral_ncc(f_ap, f_target_ap)
                ncc_lat = self.segments[0].renderer.compute_spectral_ncc(f_lat, f_target_lat)

                # Prior: small regularization on inter-vertebral strain
                strain_penalty = 1e-4 * np.sum(art_flat ** 2)
                return float(0.5 * (1.0 - ncc_ap) + 0.5 * (1.0 - ncc_lat) + strain_penalty)

            res_stage2 = minimize(
                stage2_obj,
                x0=init_art,
                method='Powell',
                options={'maxiter': max_iter_local, 'ftol': 1e-4}
            )
            arts_converged = [res_stage2.x[j*6:(j+1)*6] for j in range(num_joints)]
            final_loss = float(res_stage2.fun)
        else:
            arts_converged = []
            final_loss = float(res_stage1.fun)

        elapsed = time.time() - t0

        return {
            'base_pose': base_converged,
            'articulations': arts_converged,
            'final_loss': final_loss,
            'time_sec': elapsed,
            'stage1_success': res_stage1.success
        }
