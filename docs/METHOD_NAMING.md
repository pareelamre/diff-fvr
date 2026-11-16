# Diff-FVR method and citation identifiers

Use **Diff-FVR** for the software framework and **PoU-FVR** for its
partition-of-unity perspective Fourier projector. These names describe related
components, not interchangeable claims of exact cone-beam line integration.
PoU-FVR is a finite-subvolume approximation with pinhole resampling.

## Stable configuration names

| Identifier | Meaning |
|---|---|
| `Diff-FVR-Spectral` | Parallel-beam Fourier slicing with spectral similarity. |
| `PoU-FVR-G4x4x4-R64` | Standalone perspective projector: 4 by 4 by 4 spatial blocks, local resolution 64. |
| `PoU-FVR+nanodrr-K4-FP32` | Hybrid: PoU-FVR candidate screening, four surviving candidates, float32 nanodrr refinement. |
| `PoU-FVR+DiffDRR-K4-FP32` | Hybrid: PoU-FVR candidate screening, four surviving candidates, float32 DiffDRR refinement. |

The exp33 hybrids use the coarse `G2x2x2-R64` projector for screening; they do
not use the standalone `G4x4x4-R64` configuration. Record this in captions and
experiment manifests. These are descriptive configuration identifiers within
Diff-FVR, not names for separately published methods or separate DOIs.

Use `N_b` for spatial block count, `R` for local block resolution, `C` for the
number of optimization candidates, and `K` for retained candidates. Avoid using
`B` for both blocks and optimization batch size in the same manuscript. State
precision per stage, step counts, learning rates, target source, calibration,
and initialization protocol alongside the identifier.

## Citing the software

The verified release is **Diff-FVR v1.2.0**, released 15 November 2026:

- Release: <https://github.com/pareelamre/diff-fvr/releases/tag/v1.2.0>
- Source commit: `73c60bec9ef5b9951e25d9e0df11c919ffa097f9`
- Machine-readable metadata: `CITATION.cff` at the cited release.

Suggested software citation:

> Amre, P. (2026). Diff-FVR: Differentiable Fourier Volume Rendering for 3D-2D
> Registration (Version 1.2.0) [Computer software].
> https://github.com/pareelamre/diff-fvr/releases/tag/v1.2.0

The working-tree citation metadata was corrected after the release; the tag's
original CFF may still say 1.1.1. Cite the release URL and exact source commit
until a subsequent release includes the corrected metadata. Do not rewrite the tag.

No DOI or accepted journal article is asserted by this citation. If an archive
later assigns a DOI, add its verified identifier. Cite upstream DiffDRR and
nanodrr separately for hybrid results, using their own verified citation metadata.

## Figure identifiers

The private journal-review figure pack uses stable identifiers `DFVR-REG-01`
(cadaveric registration), `DFVR-RES-02` (reported resource accounting), and
`DFVR-SIM-03` (single-volume CT simulations). Filenames include the method,
scientific subject, and revision. Each figure's provenance manifest records the
source CSV hashes. The pack is a review artifact, not an archived publication.
