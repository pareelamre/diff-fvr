# Parallel physical geometry review disposition

Audit aud_2K3wEjdVBw returned three completed voices and one failed model channel.
The terminal result is partial panel coverage, not a unanimous four-voice review.
All 15 returned findings were checked against source and regressions. Duplicate
root causes share remedies. The frozen perspective benchmark is unchanged.

| finding | decision | action | verification |
|---|---|---|---|
| voice0/f1 major blocking; target/prediction origin mismatch; torch_engine.py target preparation | accepted | Apply spatial ifftshift before target FFT and retain identical image-space convention | Odd/even true-pose loss and gradient regression reproduced the defect before repair and passes afterward |
| voice0/f2 major; detector-period wrap; torch_engine.py frequency grid | accepted | Choose enclosing default pitch and reject full-box support outside the detector; cropped rendering remains outside this API | Out-of-frame translation and too-small detector regressions were red before guard, now pass |
| voice0/f3 major; default CUDA/Triton unverified | accepted | Disable default Triton selection for this engine; qualify explicit opt-in as experimental until exclusive GPU forward/backward parity is available | Constructor default false; CPU explicit-stencil values/gradients pass; actual GPU parity remains unexecuted and unclaimed |
| voice1/f1 major blocking; centered target phase | accepted | Same origin repair as voice0/f1 | True-pose objective near zero and gradient near zero on odd/even grids, both canonical views |
| voice1/f2 major; Parseval identity alone masks geometric mismatch | accepted | Add independent axis references and true-pose zero-loss regression; retain descent check only as descent evidence | 14 geometry tests plus5 existing tests pass; do not require single-view six-DOF convergence because ray-axis translation is unobservable |
| voice1/f3 minor; image-space FFT convention differs | accepted | Apply ifftshift to both centered images before their FFTs | Existing image-loss descent and separate displayed-image finite differences pass |
| voice2/f1 major; insufficient detector field of view | accepted | Same support guard as voice0/f2; automatic field encloses any zero-translation rotation | Small-window and large-shift cases raise explicit ValueError; physical shift tests remain inside valid support |
| voice2/f2 major; off-grid spatial roll-off and ghosts understated | accepted | Name approximation explicitly and quantify padding sensitivity with a declared analytic fixture; retain padding4 relative L2 bound of3% | Padding2 AP/LAT9.6385/9.5030%; padding4 2.3800/2.4847%; on-grid line sums remain exact; unconditional precompensation was not adopted because it would bias those sums |
| voice2/f3 major; no oblique/fractional independent reference | accepted | Add closed-form rotated anisotropic Gaussian with30/17/9-degree pose and fractional translation | Both padding4 views meet predeclared3% relative L2 and improve over padding2; no universal error guarantee |
| voice2/f4 major; unverified default Triton and stencil | accepted | Default false plus explicit eight-neighbor comparison using actual engine frequencies, anisotropic axes and guard center | CPU forward/axis-gradient parity passes including outside-guard samples; GPU kernel parity remains required before CUDA validation claim |
| voice2/f5 minor; narrow finite differences and interpolation kinks | accepted | Add second oblique pose for real/ReLU image loss; document one-sided grid-node and ReLU kinks | Spectral6-axis central differences within2%; displayed-image loss within3% on declared non-kink fixtures |
| voice2/f6 minor; low-precision grid and fp16 overflow | accepted | Upcast Torch sampling values to float32 on both CPU/CUDA; reject overflowing float16 casts before storage | f16/bf16 fallback equals float32 sampling of quantized amplitudes; constructor and to_precision overflow regressions pass |
| voice2/f7 minor; changed semantics under same class | accepted | Expose DFVR-PARALLEL-2 geometry_version and document migration; search all call sites | Only tests and historical exp24/exp29 callers found; old scripts/CSVs explicitly not certified for new geometry, exp29 explicitly opts into Triton |
| voice2/f8 minor; independent-reference coverage overstated | accepted | Distinguish independent-reference, equivariance, input and self-consistency checks | Test docstring, physical-coordinate guide and this record use the narrower descriptions |
| voice2/f9 minor; padding comment contradicts accepted range | accepted | Remove false universal sqrt3 bound; accept factors>=1 with explicit fidelity/storage tradeoff | Constructor check and guide agree; analytic padding2/4 diagnostics quantify fixture-specific cost of interpolation |

Fresh verification: `python -m unittest tests.test_parallel_geometry tests.test_torch_engine tests.test_process_resources tests.test_journal_summary tests.test_journal_protocol` passed42 tests on CPU. This is not GPU kernel validation or a performance benchmark.
