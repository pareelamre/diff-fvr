"""
Diff-FVR: Differentiable Fourier Volume Rendering for 3D-2D Registration.
Author: Pareel Amre, TU Dresden
"""

from .fvr_engine import FourierVolumeRenderer
from .diff_fvr import DifferentiableFourierVolumeRenderer
from .cone_beam_fvr import PerspectiveSlabFVR
from .core_engine import RobustDiffFVREngine, detect_radio_opaque_metal
from .gradients import SpectralGradientEngine
from .articulated_spine import ArticulatedSpineModel, VertebralSegment
from .raycasting_drr import raycast_drr
from .rotation_2d import rotate_2d_fft
from .dicom_loader import load_dicom_series, create_spine_phantom, hu_to_attenuation
from .registration_3d2d import compute_mtre_6dof
from .cli import main_render, main_register

try:
    from .torch_engine import TorchDiffFVREngine
    _has_torch = True
except ImportError:
    TorchDiffFVREngine = None
    _has_torch = False

__version__ = "1.0.0"

__all__ = [
    "FourierVolumeRenderer",
    "DifferentiableFourierVolumeRenderer",
    "PerspectiveSlabFVR",
    "RobustDiffFVREngine",
    "detect_radio_opaque_metal",
    "SpectralGradientEngine",
    "ArticulatedSpineModel",
    "VertebralSegment",
    "TorchDiffFVREngine",
    "raycast_drr",
    "rotate_2d_fft",
    "load_dicom_series",
    "create_spine_phantom",
    "hu_to_attenuation",
    "compute_mtre_6dof",
    "main_render",
    "main_register"
]
