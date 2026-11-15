# DFVR-J34-r1 journal replication

This is a **development replication**, not independent validation. All six
DeepFluoro cadaveric anatomies were used during development. A frozen rerun
does not create an external test set. No journal submission is claimed.

Run from a clean committed checkout with CUDA, DiffDRR 0.6.1 and nanodrr:

```powershell
python experiments/exp34_journal_replication.py --data-root data/deepfluoro/deepfluoro --output outputs/journal-j34-r1
```

Use a new output directory for every attempt. Existing manifests/logs/results
are exclusive: partial runs are not silently overwritten or resumed. Add
`--smoke` for H=64, four candidates and two-step checks; this is not a benchmark.
CPU checks additionally require `--device cpu`.

Six methods run in separate processes per specimen. Their order is shuffled
deterministically. Protocol parameters, analysis family, source/dependency
hashes, package versions and clean Git identity are recorded and checked by
each worker. A fixed pair-selection rule and shared initial/candidate hashes
support pairing. Spatial microbatching changes execution cost, not the
candidate pool. The single-start arm receives an identity update.

Synthetic canonical poses calibrate renderer conventions without evaluation
poses or target pixels. Camera buffers and selected isocenter are recorded.
Relative inter-view geometry is nevertheless supplied by reference annotations.
This is a known-geometry assumption, not estimated calibration performance.

Memory fields are MiB CUDA allocator counters: resident, full allocated and
reserved peaks, and incremental peak allocated. They exclude driver/context
allocations. Setup includes common data loading. Warm-up executes one full
optimization. The combined hypothetical one-registration cost is not full
process wall time or a claim that setup repeats before every trial.

Successful rows are flushed to JSONL. A failed worker retains those rows, log
and failure sidecar; any failed/missing result makes the batch exit nonzero.
No paired inference is allowed without the expected trial roster and a full
failure accounting. The primary analysis is per-specimen effects, not pooled
trials. At n=6, five Holm-adjusted comparisons cannot attain alpha=.05 even
with every effect in the same direction; lack of significance is not equivalence.
Bootstrap intervals are exploratory, with uncertain small-sample coverage.

Historical exp32/33 CSVs belong to pre-fix v1.2.0 commit
`73c60bec9ef5b9951e25d9e0df11c919ffa097f9`. The exp32 wide-tier and CT
single-start arms used the first randomized candidate rather than the common
initial pose. The current source repairs that defect; historic CSVs are not
rewritten or presented as corrected evidence. Reference-pose calibration and
partial preparation/incremental-memory limitations also remain in those CSVs.

The CT loader now requires one explicit single-frame regular series directory,
a common coordinate frame, single-sample MONOCHROME2 pixels and valid geometry.
It sorts along the slice normal and returns local voxel axes and spacing,
without patient origin/direction. It must not transform patient-frame landmarks.
Compressed transfer syntaxes require a supported pydicom decoder. Irregular
GE QA data are rejected pending geometry-aware reconstruction. No mean-spacing
substitution or fallback manufactures a regular physical grid.

Code audit: `aud_qN2vevJ1F1`; 33 source findings adjudicated in
`JOURNAL_PROTOCOL_ADJUDICATION.md`. The audit did not execute GPU runs.
