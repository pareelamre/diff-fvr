# Diff-FVR-Spectral physical-coordinate contract

The tensor parallel renderer uses a separate geometry from the perspective
PoU-FVR projector. A parallel view cannot independently observe translation
along its ray direction. Two independent views can constrain all three
translations; pose identifiability also depends on anatomy and view separation.

`TorchDiffFVREngine` accepts volume arrays in `(D,H,W)=(z,y,x)` order and spacing
`(sz,sy,sx)` in millimetres. The volume's central voxel, at indices
`(D//2,H//2,W//2)`, defines the object origin. Padding preserves each axis's
parity and that origin. Patient-coordinate orientation, origin and landmark
transforms remain the caller's responsibility; this tensor API does not infer
them from a NIfTI affine.

Each pose is `[yaw_Z,pitch_Y,roll_X,tx,ty,tz]`, with Euler angles in degrees
and translations in millimetres. It is an active object-to-scanner transform:
`x_scanner = Rz(yaw) Ry(pitch) Rx(roll) x_object + t`. These angle units differ
from the radian-valued perturbations used by the acquired-image experiment.

| Canonical view | Detector u | Detector v | Ray direction | In-plane translation |
|---|---|---|---|---|
| `ap` | +x | +y | +z | tx,ty |
| `lateral` | -z | +y | +x | -tz,ty |

These are mathematical view names, not an automatic anatomical AP/lateral
orientation for arbitrary clinical volumes. Image columns follow u and rows
follow v. Pixel coordinates are centered at `out_res//2`. The square detector pitch is the explicit `detector_spacing` when supplied.
Otherwise `detector_pitch(out_res)` chooses `2*r/(out_res-1)`, where
`r = sqrt(sum(((n//2+0.5)*s)**2))` encloses the full voxel box about the origin.
The pitch is fixed across poses at that resolution. It is an explicit sampling
choice, not detector calibration inferred from a medical image.

An inverse FFT is periodic with period `out_res * pitch`. To prevent anatomy
folding across the detector boundary, the renderer rejects poses whose projected
full voxel-box extent exceeds `(out_res-1)*pitch/2` in either axis. This is a
conservative whole-support renderer, not a finite-detector crop: it also counts
empty border voxels. Translations may require larger explicit pitch or a larger
output at fixed pitch. Padding the 3D spectrum does not enlarge this detector
field of view. The automatically chosen field encloses every rotation at zero
translation; translations are checked separately.

Detector frequencies use centered FFT bins in cycles/mm. The object-frequency
plane is `k_object=R^T(k_u u+k_v v)`; frequencies map to volume FFT indices using
the padded voxel count times physical spacing on each axis. Translation applies
`exp(-2*pi*i*(k_u*t_u+k_v*t_v))`. Centered spatial FFT input and centered image
output retain the stated voxel origin. Target spectra apply `ifftshift` before
`fft2` to use the same origin, on both odd and even grids. The scale factor
`sx*sy*sz / detector_spacing**2` gives line-integral units: for an axis-aligned
view at matched pitch, a spatial sum is multiplied by the ray-axis voxel length.

One periodic guard bin on each Fourier-volume face supports interpolation at
reflected Nyquist endpoints. Samples farther outside this stored volume vanish.
Trilinear interpolation of the discrete Fourier volume causes spatial roll-off
and ghosting away from exact grid-node samples, as well as finite-frequency
support error. The normalization gives physical line-integral dimensions, not
exact off-grid amplitudes. Padding reduces interpolation error at cubic storage
and FFT cost. Factors down to 1 are accepted without an accuracy guarantee.

The independent oblique test uses a shifted anisotropic Gaussian, spacing
(1.1,0.9,0.7) mm, angles (30,17,9) degrees and fractional translation
(0.37,-0.23,0.45) mm. A closed-form Gaussian integral supplies each view's
reference, without using the renderer. Relative L2 errors are 9.6385% AP and
9.5030% lateral at padding 2, versus 2.3800% and 2.4847% at padding 4. The
predeclared padding-4 acceptance is 3% per view and improvement over padding 2.
These are fixture-specific measurements, not a universal error bound. No
unconditional sinc-squared precompensation is applied: it would bias exact
on-grid line sums that already pass. Other volumes, extents and padding factors
require their own fidelity assessment. This is not an exact cone-beam or
detector-physics model.

The no-IFFT loss correlates complex spectral fields. Parseval equality applies
to the corresponding complex inverse transforms with identical weighting and
normalization; it does not make this loss equal to correlation after taking
the real part and clipping negative image values. The displayed-image forward
path retains that real/ReLU operation. High-pass cutoff is in frequency-bin
units and its DC bin is exactly zero. Trilinear interpolation and ReLU are
piecewise differentiable; at grid-node and zero-image kinks, autograd selects
one-sided derivatives. Central differences need not agree there. Finite-
difference checks therefore use two oblique poses away from those kinks, with
one check of the complex loss and one of the displayed-image loss.

Reduced-precision storage uses float32 sampling coordinates in the Torch
fallback on CPU and CUDA, upcasting stored values for sampling. Float16 casting
is rejected when spectrum values exceed its finite range. Bfloat16 still
quantizes Fourier amplitudes and is not asserted numerically equivalent to
float32.

The geometry is identified by `geometry_version = "DFVR-PARALLEL-2"`.
The correction changes the former lateral view, phase scale, pixel centering,
spacing behavior and preparation buffer. Old pose outputs and timing CSVs must
retain their original source identity. In particular, the historical exp29
objective timings do not certify corrected physical pose estimation.

CPU regressions compare canonical projections against independently summed
voxels, physical integer shifts, a known rotation, even-grid Nyquist reflection,
anisotropic Gaussian analytic integrals and uniform changes of length units.
They also compare the complex Parseval objective/gradient and finite differences
for all six parameters on an asymmetric fixture. The analytic Gaussian tolerance
is 6% relative L2, and the finite-difference tolerance is 2% relative L2 on this
fixture; these are test acceptance bounds, not population error estimates.

Call-site search found only tests and the historical exp24/exp29 parallel
benchmark scripts. Those scripts inherit the new sampling convention and cannot
reproduce the old CSVs with current source; their outputs need new provenance
and a fresh physical validation before use. Exp29 explicitly requests Triton.

The corrected parallel engine defaults to `use_triton=False`. CPU checks compare
its actual physical sampling bins, anisotropic axes and guard center with the
explicit eight-neighbor stencil for values and gradients, including samples
outside the guard. They do not execute GPU kernels. Explicit Triton opt-in
remains available explicitly. A17November2026 L4 qualification now compares
corrected CUDA Torch/Triton forward images and all six pose gradients on a25-cube
asymmetric anisotropic phantom, including oblique views and fractional shifts.
Float32 relative-L2 discrepancies are4.49e-7(image) and1.88e-7(gradient);
float16 gives1.87e-4/4.28e-5 andbfloat16 gives1.45e-3/1.15e-4.
All declared fixture tolerances pass. This is scoped numerical qualification,
not broad size/shape performance certification; the conservative Torch default
is retained. Six preselected VerSe CTs additionally pass the independent spatial
parallel-target registration study using float32 Torch. Its limitations and
exact protocols/source hashes are in outputs/journal_j34.

The separate acquired-image36-worker resource batch is complete and uses the
distinct perspective projector at its own frozen source revision. It must not
be conflated with corrected parallel objective timing. See
docs/JOURNAL_SESSION_ANALYSIS.md for explicit device-stratified analysis.
