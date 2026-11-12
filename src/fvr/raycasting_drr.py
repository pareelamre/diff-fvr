import numpy as np
from scipy.ndimage import affine_transform
from typing import Tuple, Optional


def raycast_view(
    volume: np.ndarray,
    view: str = 'lateral',
    rotation_deg: float = 0.0,
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Direct spatial-domain ray-summation (line-integral) DRR generator
    with identical geometry and coordinate axes to FourierVolumeRenderer.
    
    Coordinates: volume shape (Z, Y, X)
      Z = Inferior -> Superior
      Y = Anterior -> Posterior
      X = Right -> Left
      
    Views:
      - 'ap': Ray along Y (Anterior-Posterior). Projection plane (X, Z).
              rotation_deg rotates around patient Z axis.
      - 'lateral': Ray along X (Right-Left). Projection plane (Y, Z).
                   rotation_deg rotates around patient Z axis.
      - 'axial': Ray along Z. Projection plane (X, Y).
    """
    vol = volume.astype(np.float64)
    nz, ny, nx = vol.shape
    center = np.array([nz / 2.0, ny / 2.0, nx / 2.0])

    theta = np.radians(rotation_deg)

    # Rotation around Z axis (patient superior-inferior axis)
    # Affine transform coordinates order: [z, y, x]
    # Rotate y and x:
    # y' = y*cos(theta) - x*sin(theta)
    # x' = y*sin(theta) + x*cos(theta)
    rz = np.array([
        [1.0, 0.0,           0.0          ],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta),  np.cos(theta)]
    ])

    offset = center - rz @ center
    rotated_vol = affine_transform(vol, matrix=rz, offset=offset, order=1, mode='constant', cval=0.0)

    view = view.lower()
    if view == 'ap':
        # Sum along Y (axis 1) -> Image plane (Z, X)
        proj = np.sum(rotated_vol, axis=1)  # shape (nz, nx)
    elif view == 'lateral':
        # Sum along X (axis 2) -> Image plane (Z, Y)
        proj = np.sum(rotated_vol, axis=2)  # shape (nz, ny)
    elif view == 'axial':
        # Sum along Z (axis 0) -> Image plane (Y, X)
        proj = np.sum(rotated_vol, axis=0)  # shape (ny, nx)
    else:
        raise ValueError(f"Unknown view: {view}")

    if output_size is not None and (proj.shape[0] != output_size[0] or proj.shape[1] != output_size[1]):
        from scipy.ndimage import zoom
        zy = output_size[0] / proj.shape[0]
        zx = output_size[1] / proj.shape[1]
        proj = zoom(proj, (zy, zx), order=1)

    return proj.astype(np.float64)


def raycast_drr(
    volume: np.ndarray,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Legacy wrapper for raycast_view('lateral', rotation_deg=yaw_deg).
    """
    return raycast_view(volume, view='lateral', rotation_deg=yaw_deg, output_size=output_size)
