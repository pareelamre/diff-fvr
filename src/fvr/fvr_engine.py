import numpy as np
from scipy import fft
from scipy.ndimage import map_coordinates
from typing import Tuple, Optional


class FourierVolumeRenderer:
    """
    Fourier Volume Rendering (FVR) based on the Fourier Projection-Slice Theorem
    (Malzbender, ACM TOG 1993).
    
    Coordinate System (Medical Standard):
      volume shape: (Z, Y, X) where:
        Z = Inferior -> Superior (Slice axis)
        Y = Anterior -> Posterior
        X = Right -> Left
        
    Radiological Projections:
      - 'ap' (Coronal): Ray along Y (Anterior-Posterior). Projection plane (X, Z).
      - 'lateral' (Sagittal): Ray along X (Right-Left). Projection plane (Y, Z).
      - 'axial' (Transverse): Ray along Z (Inferior-Superior). Projection plane (X, Y).
    """

    def __init__(self, volume: np.ndarray, pad_factor: float = 2.0, apply_premultiplication: bool = True):
        self.orig_shape = volume.shape
        self.pad_factor = pad_factor
        self.apply_premultiplication = apply_premultiplication

        # Precompute 3D FFT
        self.f_3d, self.padded_shape = self._preprocess_and_fft(volume)

    def _preprocess_and_fft(self, volume: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        vol = volume.astype(np.float64)
        nz, ny, nx = vol.shape

        pz = int(round(nz * self.pad_factor))
        py = int(round(ny * self.pad_factor))
        px = int(round(nx * self.pad_factor))

        if self.apply_premultiplication:
            def sinc2_correction(n_orig, n_pad):
                coords = (np.arange(n_orig) - n_orig / 2.0) / n_pad
                s = np.sinc(coords)
                denom = np.maximum(s ** 2, 0.05)
                return 1.0 / denom

            cz = sinc2_correction(nz, pz)[:, None, None]
            cy = sinc2_correction(ny, py)[None, :, None]
            cx = sinc2_correction(nx, px)[None, None, :]
            vol = vol * (cz * cy * cx)

        # 3D Zero-Padding
        padded = np.zeros((pz, py, px), dtype=np.float64)
        sz = (pz - nz) // 2
        sy = (py - ny) // 2
        sx = (px - nx) // 2
        padded[sz:sz + nz, sy:sy + ny, sx:sx + nx] = vol

        # 3D FFT with zero frequency centered
        f_3d = fft.fftshift(fft.fftn(fft.ifftshift(padded)))
        return f_3d, (pz, py, px)

    def render_view(
        self,
        view: str = 'lateral',
        rotation_deg: float = 0.0,
        output_size: Optional[Tuple[int, int]] = None,
        interp_order: int = 1,
        crop_to_roi: bool = True
    ) -> np.ndarray:
        """
        Render a standard radiological projection view:
          - view='ap': Anterior-Posterior projection (Coronal plane).
          - view='lateral': Lateral projection (Sagittal plane).
          - view='axial': Top-down slice sum (Transverse plane).
          
        Scale-Correction:
          Because the volume was padded by pad_factor (e.g. 2x), the 2D frequency slice
          is sampled across the full padded dimension (P_h x P_w). Inverse 2D FFT
          yields an unpadded image of size (P_h x P_w). When crop_to_roi is True,
          we crop the central (N_h x N_w) region, ensuring the reconstructed object
          has 1:1 identical physical size and pixel scale as spatial raycasting!
        """
        view = view.lower()
        theta = np.radians(rotation_deg)
        pz, py, px = self.padded_shape
        cz, cy, cx = pz / 2.0, py / 2.0, px / 2.0

        nz, ny, nx = self.orig_shape

        if view == 'ap':
            # Ray along Y. Image plane has horizontal X, vertical Z.
            target_h, target_w = nz, nx
            slice_h, slice_w = pz, px
            u_vec = np.array([0.0, -np.sin(theta), np.cos(theta)])
            v_vec = np.array([1.0, 0.0, 0.0])
        elif view == 'lateral':
            # Ray along X. Image plane has horizontal Y, vertical Z.
            target_h, target_w = nz, ny
            slice_h, slice_w = pz, py
            u_vec = np.array([0.0, np.cos(theta), np.sin(theta)])
            v_vec = np.array([1.0, 0.0, 0.0])
        elif view == 'axial':
            # Ray along Z. Image plane has horizontal X, vertical Y.
            target_h, target_w = ny, nx
            slice_h, slice_w = py, px
            u_vec = np.array([0.0, 0.0, 1.0])
            v_vec = np.array([0.0, 1.0, 0.0])
        else:
            raise ValueError(f"Unknown view: {view}. Choose 'ap', 'lateral', or 'axial'.")

        # Sample full padded 2D frequency grid (step 1.0 in frequency bins)
        u = np.linspace(-slice_w / 2.0, slice_w / 2.0, slice_w, endpoint=False)
        v = np.linspace(-slice_h / 2.0, slice_h / 2.0, slice_h, endpoint=False)
        uu, vv = np.meshgrid(u, v)

        kwz = uu * u_vec[0] + vv * v_vec[0] + cz
        kwy = uu * u_vec[1] + vv * v_vec[1] + cy
        kwx = uu * u_vec[2] + vv * v_vec[2] + cx

        coords = np.array([kwz.ravel(), kwy.ravel(), kwx.ravel()])

        real_slice = map_coordinates(self.f_3d.real, coords, order=interp_order, mode='constant', cval=0.0).reshape((slice_h, slice_w))
        imag_slice = map_coordinates(self.f_3d.imag, coords, order=interp_order, mode='constant', cval=0.0).reshape((slice_h, slice_w))
        f_slice = real_slice + 1j * imag_slice

        # 2D IFFT on the full padded slice
        spatial_proj = fft.fftshift(fft.ifft2(fft.ifftshift(f_slice)))
        drr = np.real(spatial_proj)

        # Scale normalization (Fourier slice theorem line-integral scaling)
        scale_factor = (pz * py * px) / (slice_h * slice_w)
        drr = np.maximum(drr * scale_factor, 0.0)

        if crop_to_roi:
            # Crop the central region (target_h x target_w) corresponding to the unpadded volume FOV
            sh = (slice_h - target_h) // 2
            sw = (slice_w - target_w) // 2
            drr = drr[sh:sh + target_h, sw:sw + target_w]

        if output_size is not None and (drr.shape[0] != output_size[0] or drr.shape[1] != output_size[1]):
            from scipy.ndimage import zoom
            zy = output_size[0] / drr.shape[0]
            zx = output_size[1] / drr.shape[1]
            drr = zoom(drr, (zy, zx), order=1)

        return drr
