import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from fvr import rotate_2d_fft


def main():
    img_path = Path(__file__).resolve().parent / "sunflowers.jpg"
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    if not img_path.exists():
        print(f"Error: {img_path} not found.")
        return

    print("=" * 60)
    print("  2D Image Rotation via Fourier Shearing (3-Shear FFT)")
    print("  Reproducing Pareel's original image_rotation_by_fft_translation.m")
    print("=" * 60)

    # Load image and convert to float [0, 1]
    img = np.array(Image.open(img_path)).astype(np.float64) / 255.0
    print(f"Loaded image: {img.shape}")

    # Rotate by 60 degrees (pi / 3), matching Pareel's MATLAB code
    theta_deg = 60.0
    theta_rad = np.radians(theta_deg)
    print(f"Rotating image by {theta_deg} degrees...")

    rotated_padded = rotate_2d_fft(img, theta_rad, pad=True)
    rotated_unpadded = rotate_2d_fft(img, theta_rad, pad=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original (sunflowers.jpg)", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(rotated_padded)
    axes[1].set_title(f"Fourier Rotated ({theta_deg}°)\nWith 2x Zero Padding", fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(rotated_unpadded)
    axes[2].set_title(f"Fourier Rotated ({theta_deg}°)\nWithout Padding (Cyclic Wrap)", fontsize=12)
    axes[2].axis('off')

    out_file = out_dir / "2d_rotation_comparison.png"
    fig.savefig(out_file, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Result saved to: {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
