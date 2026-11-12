# Historical results: regeneration required

Existing CSVs and figures were produced before the physical-unit correction.
They are preserved as historical artifacts, not verified physical accuracy data.

CT experiments computed mTRE in voxel coordinates but labeled it mm. For the
hospital CT resampled to the former 96 grid, the nominal scale was about 2.604 mm
per voxel; other grids have different scales. Do not apply one global conversion
to this directory. Re-run the source experiments and recompute threshold-based
success rates. The corrected CLI also uses a physically consistent resampling
grid, which can change results beyond a unit conversion.

New CT mTRE tables include `Metric_units=mm` and `Target_type=synthetic` when the
script writes tabular records. Experiment 16 refuses to aggregate legacy tables
without explicit metric units, and no longer invents fallback baseline numbers.
These markers describe units and simulation provenance; they do not establish
clinical validity or independent-renderer accuracy.

Experiments 7, 12, and 13 use one CT source volume each, multiple random poses,
and synthetic projections generated with the registration renderer. Their titles
and filenames are retained for discoverability; they do not represent validation
against acquired patient fluoroscopy or a multi-patient cohort. Old figures and
the browser demonstration must not be used as evidence of surgical suitability.
