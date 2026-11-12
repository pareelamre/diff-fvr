import os
import glob
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pydicom
import SimpleITK as sitk
from scipy.ndimage import affine_transform


def resample_isotropic_cube(volume_hu, spacing, target_res):
    """Center a CT in a cubic grid without cropping its physical field of view.

    Return HU data and (Z, Y, X) spacing in mm. Sampling coordinates preserve
    physical distances exactly; padding outside the acquired CT is air.
    """
    volume_hu = np.asarray(volume_hu)
    spacing = np.asarray(spacing, dtype=float)
    if volume_hu.ndim != 3 or min(volume_hu.shape) < 2 or not np.all(np.isfinite(volume_hu)):
        raise ValueError('CT must be a finite 3D array with at least two voxels per axis.')
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('spacing must contain three finite positive values in mm (Z, Y, X).')
    if not isinstance(target_res, (int, np.integer)) or target_res < 2:
        raise ValueError('target_res must be an integer of at least two.')
    voxel_mm = float(np.max(np.array(volume_hu.shape) * spacing) / target_res)
    ratios = voxel_mm / spacing
    offset = (np.array(volume_hu.shape) - 1) / 2 - ratios * (target_res - 1) / 2
    resampled = affine_transform(volume_hu.astype(np.float32), np.diag(ratios), offset=offset,
                                output_shape=(target_res,) * 3, order=1, cval=-1000.)
    return resampled, (voxel_mm,) * 3


def load_dicom_series(directory: str | Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load a DICOM series from a folder using SimpleITK with a fallback to pydicom.
    Returns:
        volume: 3D numpy array (Z, Y, X) of CT intensity in Hounsfield Units (HU).
        spacing: (spacing_z, spacing_y, spacing_x) in mm.
    """
    directory = str(directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Try SimpleITK first
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(directory)
        if series_ids:
            dicom_names = reader.GetGDCMSeriesFileNames(directory, series_ids[0])
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            volume = sitk.GetArrayFromImage(image)  # Shape: (Z, Y, X)
            spacing = image.GetSpacing()  # (dx, dy, dz)
            spacing_zyx = (spacing[2], spacing[1], spacing[0])
            return volume.astype(np.float32), spacing_zyx
    except Exception as e:
        print(f"[dicom_loader] SimpleITK series load failed ({e}), trying fallback...")

    # Fallback: scan for *.dcm or numbered files with pydicom
    files = glob.glob(os.path.join(directory, "*.dcm"))
    if not files:
        # Check files without extension
        all_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files = [f for f in all_files if not f.endswith(('.rar', '.zip', '.txt', '.py', '.m'))]

    if not files:
        raise ValueError(f"No DICOM files found in {directory}")

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise ValueError(f"No readable DICOM slices found in {directory}")

    # Sort slices by ImagePositionPatient[2] (Z) or SliceLocation or InstanceNumber
    def get_slice_z(ds):
        if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
            return float(ds.ImagePositionPatient[2])
        if hasattr(ds, "SliceLocation"):
            return float(ds.SliceLocation)
        return float(getattr(ds, "InstanceNumber", 0))

    slices.sort(key=get_slice_z)

    # Pixel spacing
    dz = 1.0
    if len(slices) > 1 and hasattr(slices[0], "ImagePositionPatient") and hasattr(slices[1], "ImagePositionPatient"):
        dz = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
    elif hasattr(slices[0], "SliceThickness"):
        dz = float(slices[0].SliceThickness)

    dx, dy = 1.0, 1.0
    if hasattr(slices[0], "PixelSpacing"):
        dy, dx = float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1])

    # Convert pixel arrays to Hounsfield Units
    vol_list = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        slope = float(getattr(s, "RescaleSlope", 1.0))
        arr = arr * slope + intercept
        vol_list.append(arr)

    volume = np.stack(vol_list, axis=0)
    return volume.astype(np.float32), (dz, dy, dx)


def create_spine_phantom(size: int = 128) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Generate a 3D volumetric phantom resembling human vertebrae (spine) in CT.
    Returns:
        volume: (size, size, size) in Hounsfield Units (air: -1000, soft tissue: 40, bone: 700-1200)
        spacing: (1.0, 1.0, 1.0) mm
    """
    # Coordinate grids: z: (size, 1, 1), y: (1, size, 1), x: (1, 1, size)
    z = np.arange(size, dtype=np.float32)[:, None, None]
    y = np.arange(size, dtype=np.float32)[None, :, None]
    x = np.arange(size, dtype=np.float32)[None, None, :]
    center = size / 2.0
    
    # Base background: Air (-1000 HU)
    vol = np.full((size, size, size), -1000.0, dtype=np.float32)

    # Torso body (cylinder soft tissue, ~40 HU)
    torso_mask = ((x - center) / (0.38 * size))**2 + ((y - center) / (0.32 * size))**2 <= 1.0
    vol[np.broadcast_to(torso_mask, vol.shape)] = 40.0

    # Repeating vertebrae along Z
    num_vertebrae = 5
    vertebra_period = size / num_vertebrae

    for i in range(num_vertebrae):
        vz = (i + 0.5) * vertebra_period
        v_height = vertebra_period * 0.72

        z_mask = np.abs(z - vz) <= (v_height / 2.0)

        # Vertebral body (cylinder in anterior spine: slightly shifted in -y)
        vy_body = center - 0.06 * size
        vx_body = center
        r_body = 0.12 * size
        body_dist_sq = (x - vx_body)**2 + (y - vy_body)**2

        # Outer cortical bone (dense, ~1100 HU)
        body_mask = (body_dist_sq <= r_body**2) & z_mask
        vol[body_mask] = 1100.0

        # Cancellous/trabecular bone interior (~350 HU)
        inner_body_mask = (body_dist_sq <= (r_body * 0.75)**2) & z_mask
        vol[inner_body_mask] = 400.0

        # Spinal canal (foramen - fluid/tissue, ~15 HU)
        vy_canal = center + 0.08 * size
        r_canal = 0.055 * size
        canal_mask = (((x - vx_body)**2 + (y - vy_canal)**2) <= r_canal**2) & z_mask
        vol[canal_mask] = 15.0

        # Posterior arch & spinous process (bone, ~900 HU)
        spinous_mask = (np.abs(x - center) <= 0.03 * size) & \
                       (y >= vy_canal) & (y <= (center + 0.22 * size)) & z_mask
        vol[spinous_mask] = 900.0

        # Transverse processes (left and right lateral wings)
        transverse_mask = (np.abs(y - vy_canal) <= 0.04 * size) & \
                          (np.abs(x - center) <= 0.24 * size) & \
                          (np.abs(x - center) >= 0.07 * size) & \
                          (np.abs(z - vz) <= (v_height * 0.35))
        vol[transverse_mask] = 850.0

    return vol, (1.0, 1.0, 1.0)


def hu_to_attenuation(volume_hu: np.ndarray) -> np.ndarray:
    """
    Convert CT numbers (HU) to linear X-ray attenuation coefficients (mu >= 0).
    Air (-1000 HU) -> 0.0
    Water (0 HU) -> ~0.02 mm^-1 (or normalized ~0.2)
    Bone (+1000 HU) -> ~0.04 mm^-1
    """
    # Linear conversion: mu = max(0, (HU + 1000) / 1000)
    mu = np.maximum(0.0, (volume_hu + 1000.0) / 1000.0)
    return mu.astype(np.float32)
