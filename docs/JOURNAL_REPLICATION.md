# DFVR-J34-r2 journal replication

This is a **development replication**, not independent validation. All six
DeepFluoro cadaveric anatomies were used during development. A frozen rerun
does not create an external test set. No journal submission is claimed.

Run from a clean committed checkout with CUDA, DiffDRR 0.6.1, nanodrr,
psutil and NVIDIA's `nvidia-ml-py` binding. Resource sampling requires a Linux
GPU runtime with one NVML-visible GPU and no other compute process on it:

```powershell
python experiments/exp34_journal_replication.py --data-root data/deepfluoro/deepfluoro --output outputs/journal-j34-r2
```

Use a new output directory for every attempt. Existing manifests/logs/results
are exclusive: partial runs are not silently overwritten or resumed. Add
`--smoke` for H=64, four candidates and two-step checks; this is not a benchmark.
CPU checks additionally require `--device cpu`. Smoke outputs, including CPU
outputs, are verification artifacts and cannot produce a performance summary.
The full-batch summary requires all six CUDA specimens. Do not replace missing
GPU measurements with zero when inspecting a CPU smoke result.

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

Each worker is held at a stdin gate until an external sampler has taken its
first observation. The sampler records host root RSS, contemporaneous process
tree RSS and NVML process GPU bytes from startup through exit, including data
loading, renderer preparation, a full optimization warm-up, all four timed
registrations and result/provenance serialization. GPU context, resident CT,
Fourier buffers and workspaces contribute to this process footprint. The
parent monitor and notebook are outside the worker footprint. NVML telemetry
errors or concurrent GPU compute invalidate the profile and terminate the worker.
Separate workers start in separate sessions; interrupting a notebook monitor
does not intentionally signal the worker process group.

The 20 ms requested interval is sampling, not an exact high-water mark.
Profiles report sample count, actual maximum gap and monitor sampling cost.
At least two samples and a maximum gap of 0.5 s are required. The worker-tree
wall-time limit is four hours per worker. These limits are protocol-hashed;
exceeding either invalidates the run. Known descendants are tracked after root
exit; a final same-process-group scan also finds surviving POSIX children.
Interruption/telemetry failure terminates and reaps the worker group, preserves
an invalid sidecar where writable and propagates interruption to the coordinator.
Short transients can be missed. Summed tree RSS can double-count shared pages.
Process GPU memory and RSS peaks occur independently: never add their maxima.
These measurements do not establish the minimum GPU capacity needed to run.
Both compute and graphics process lists are checked, with duplicate PID memory
counted once. This protocol deliberately supports only a host with one physical
NVML-visible GPU; CUDA_VISIBLE_DEVICES does not hide physical adapters from NVML.
NVML and psutil must share PID visibility. A CUDA probe on the intended Colab L4
host must confirm matching PIDs before the benchmark; mismatched/hidden PIDs are
unsupported rather than silently attributed. A descendant list is refreshed
before a newly observed GPU PID is declared foreign. MPS workloads are outside
this exclusive-device protocol.
See [NVIDIA NVML](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)
and [psutil RSS semantics](https://psutil.io/faq/).

CUDA allocator resident, allocated/reserved peaks and incremental allocation
remain separate counters, including preparation and warm-up peaks. Camera
metadata stores small calibration matrices plus shapes/dtypes/hashes for large
buffers; the actual detector grids remain resident and measured, without
serializing their values repeatedly into JSON. Estimated final transforms are
retained for reproducible result panels.

Worker wall time covers the complete four-trial worker, not one registration.
Setup includes data loading and renderer preparation; benchmark warm-up is one
discarded full optimization. Warm registration time excludes setup, compilation
and provenance. Deployment estimates treat that first optimization as productive:
`(setup + warmup + (N-1)*mean_warm_registration)/N` for N=1,10,100 on one CT.
They exclude process imports/provenance and are not measured throughput or cold
process latency. The separate discarded-warmup-plus-one-registration field is
explicitly hypothetical. Paired tables retain every error and runtime, report
success counts separately by initialization tier and show speed ratios only
for the explicitly labelled subset where both methods reach <2 mm mTRE.
This threshold does not establish clinical equivalence or generalization.

The former r1 L4 pilot is separate evidence and cannot supply r2 memory claims.

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

This replication reads the dataset's reconstructed `volume.nii.gz` and validates
its NIfTI affine; it does not reconstruct raw CT DICOM. Separately, the library
CT DICOM loader requires one explicit single-frame regular series directory,
a common coordinate frame, single-sample MONOCHROME2 pixels and valid geometry.
It sorts along the slice normal and returns local voxel axes and spacing,
without patient origin/direction. It must not transform patient-frame landmarks.
Compressed transfer syntaxes require a supported pydicom decoder. Irregular
GE QA data are rejected pending geometry-aware reconstruction. No mean-spacing
substitution or fallback manufactures a regular physical grid.

Code audit: `aud_qN2vevJ1F1`; 33 source findings adjudicated in
`JOURNAL_PROTOCOL_ADJUDICATION.md`. The audit did not execute GPU runs.

After every worker has finished, validate the full 36-result batch before
producing any paired table:

```powershell
python -m experiments.summarize_journal_replication outputs/journal-j34-r2 --output outputs/journal-j34-summary.json
```

The summary rejects missing whole specimens, duplicate trials, changed
code/protocol/data, different initial poses and mixed hardware. For r2 every
worker must have a valid process resource profile bound to its exact result
hash and protocol; unavailable process memory is never replaced with zero.
It emits
specimen effects and descriptive multiplicity-adjusted probabilities only
after those checks. The median effect and its exploratory interval are labelled
separately from the mean specimen effect tested by the sign-flip probability.
Synthetic regressions cover missing subjects, mismatched/duplicate inputs,
invalid resource profiles, full aggregation and speed-subset interpretation.
The resource-monitor tests cover a real descendant outliving its root,
timeout/interruption cleanup, unavailable/foreign NVML records and duplicate
compute/graphics contexts. The existing protocol audit is linked in
[JOURNAL_PROTOCOL_ADJUDICATION.md](JOURNAL_PROTOCOL_ADJUDICATION.md); the new
resource audit and its dispositions are recorded separately.
