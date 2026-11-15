import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pydicom
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
    """Load one regular, parallel CT stack in its local voxel-axis frame.

    Returns HU and Z,Y,X spacing. Patient origin/direction are not returned;
    this API must not be used to transform patient-coordinate landmarks.
    Nonuniform spacing, inter-slice shear and mixed series fail closed rather
    than silently fabricating a regular physical grid. Reconstruct those data
    with a geometry-aware converter before using this API.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    slices = []
    paths = sorted(directory.iterdir())
    if len(paths) > 10000:
        raise ValueError('CT directory exceeds the 10000-file safety limit.')
    for path in paths:
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            continue
        if getattr(ds, 'Modality', None) == 'CT':
            if not 0 < int(ds.Rows) * int(ds.Columns) <= 4096**2:
                raise ValueError('CT matrix exceeds the supported size.')
            slices.append(ds)
    if len(slices) < 2:
        raise ValueError('At least two single-frame CT slices are required.')
    if len({str(s.SeriesInstanceUID) for s in slices}) != 1:
        raise ValueError('Mixed DICOM series; select one CT series explicitly.')
    frames = {str(getattr(s, 'FrameOfReferenceUID', '')) for s in slices}
    if len(frames) != 1 or '' in frames:
        raise ValueError('Missing or mixed CT coordinate frame identifiers.')
    if any(int(getattr(s, 'NumberOfFrames', 1)) != 1 for s in slices):
        raise ValueError('Enhanced multi-frame CT requires a geometry-aware converter.')
    if any(int(getattr(s, 'SamplesPerPixel', 0)) != 1 for s in slices):
        raise ValueError('Only single-sample CT pixels are supported.')
    if any(getattr(s, 'PhotometricInterpretation', '') != 'MONOCHROME2' for s in slices):
        raise ValueError('Only MONOCHROME2 CT pixels are supported.')
    try:
        orientations = np.asarray([s.ImageOrientationPatient for s in slices], dtype=float)
        positions = np.asarray([s.ImagePositionPatient for s in slices], dtype=float)
        spacings = np.asarray([s.PixelSpacing for s in slices], dtype=float)
    except AttributeError as exc:
        raise ValueError('CT requires orientation, position and pixel spacing metadata.') from exc
    if (orientations.shape != (len(slices), 6) or positions.shape != (len(slices), 3)
            or spacings.shape != (len(slices), 2)
            or not all(np.all(np.isfinite(a)) for a in (orientations, positions, spacings))
            or np.any(spacings <= 0)):
        raise ValueError('Invalid CT geometry metadata.')
    row, col = orientations[0, :3], orientations[0, 3:]
    if (not np.allclose(orientations, orientations[0], atol=1e-5, rtol=0)
            or not np.allclose([np.linalg.norm(row), np.linalg.norm(col), row @ col], [1, 1, 0], atol=1e-4)):
        raise ValueError('Changing or invalid CT slice orientation; reconstruct geometry first.')
    if not np.allclose(spacings, spacings[0], atol=1e-5, rtol=0):
        raise ValueError('Changing CT pixel spacing; reconstruct geometry first.')
    normal = np.cross(row, col)
    normal /= np.linalg.norm(normal)
    order = np.argsort(positions @ normal)
    positions = positions[order]
    slices = [slices[i] for i in order]
    delta = np.diff(positions, axis=0)
    dz = delta @ normal
    if np.any(dz <= 0) or not np.allclose(dz, dz[0], atol=1e-3, rtol=1e-4):
        raise ValueError('Duplicate or nonuniform CT slice positions; reconstruct geometry first.')
    if not np.allclose(delta, dz[:, None] * normal, atol=1e-3, rtol=0):
        raise ValueError('CT gantry tilt/shear is unsupported; reconstruct geometry first.')
    if len({(int(s.Rows), int(s.Columns)) for s in slices}) != 1:
        raise ValueError('Changing CT matrix dimensions.')
    if len(slices) * int(slices[0].Rows) * int(slices[0].Columns) > 512**3:
        raise ValueError('CT exceeds the supported 512-cubed voxel budget.')
    volume = np.empty((len(slices), int(slices[0].Rows), int(slices[0].Columns)), dtype=np.float32)
    for i, header in enumerate(slices):
        ds = pydicom.dcmread(header.filename)
        arr = ds.pixel_array
        if arr.shape != volume.shape[1:]:
            raise ValueError('CT decoded pixel dimensions differ from metadata.')
        volume[i] = arr.astype(np.float32) * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
    if not np.all(np.isfinite(volume)):
        raise ValueError('Non-finite CT intensity values.')
    return volume, (float(dz[0]), float(spacings[0, 0]), float(spacings[0, 1]))

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
