import numpy as np
from scipy import fft


def rotate_2d_fft(image: np.ndarray, theta_rad: float, pad: bool = True) -> np.ndarray:
    """
    Rotate a 2D image (grayscale or RGB) by theta radians using Fourier shear decomposition:
    R(theta) = ShearX(-tan(theta/2)) * ShearY(sin(theta)) * ShearX(-tan(theta/2))

    Each 1D shear is performed in the frequency domain using the Fourier Shift Theorem:
    f(x - delta) <=> F(u) * exp(-2*pi*i*u*delta/N)
    
    This reproduces Pareel's original MATLAB algorithm from:
    `image_rotation_by_fft_translation.m`.
    """
    img = image.astype(np.float64)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    
    h_orig, w_orig, c = img.shape

    if pad:
        # Zero-pad image to 2x size to prevent cyclic wrap-around at corners
        h_pad, w_pad = 2 * h_orig, 2 * w_orig
        padded = np.zeros((h_pad, w_pad, c), dtype=np.float64)
        y_start = (h_pad - h_orig) // 2
        x_start = (w_pad - w_orig) // 2
        padded[y_start:y_start + h_orig, x_start:x_start + w_orig, :] = img
        img = padded

    h, w, c = img.shape
    a = np.tan(theta_rad / 2.0)
    b = -np.sin(theta_rad)

    # Frequency indices matching standard DFT ordering
    # fftfreq returns [0, 1, ..., N/2-1, -N/2, ..., -1] / N
    fx = fft.fftfreq(w)  # length w
    fy = fft.fftfreq(h)  # length h

    # 1. First Shear along X: horizontal rows shifted by offset = (y - y_center) * a
    # Phase shift: exp(-2*pi*i * fx * (y - y_center) * a)
    y_coords = np.arange(h) - h / 2.0
    phase_x = np.exp(-2j * np.pi * np.outer(y_coords * a, fx))  # shape: (h, w)

    ix = np.zeros_like(img, dtype=np.float64)
    for ch in range(c):
        f_rows = fft.fft(img[:, :, ch], axis=1)
        ix[:, :, ch] = np.real(fft.ifft(f_rows * phase_x, axis=1))

    # 2. Second Shear along Y: vertical columns shifted by offset = -(x - x_center) * b
    # Phase shift: exp(2*pi*i * fy * (x - x_center) * b)
    x_coords = np.arange(w) - w / 2.0
    phase_y = np.exp(2j * np.pi * np.outer(fy, x_coords * b))  # shape: (h, w)

    iy = np.zeros_like(ix, dtype=np.float64)
    for ch in range(c):
        f_cols = fft.fft(ix[:, :, ch], axis=0)
        iy[:, :, ch] = np.real(fft.ifft(f_cols * phase_y, axis=0))

    # 3. Third Shear along X
    i_rot = np.zeros_like(iy, dtype=np.float64)
    for ch in range(c):
        f_rows = fft.fft(iy[:, :, ch], axis=1)
        i_rot[:, :, ch] = np.real(fft.ifft(f_rows * phase_x, axis=1))

    i_rot = np.clip(i_rot, 0.0, 1.0)
    if image.ndim == 2:
        return i_rot[:, :, 0]
    return i_rot
