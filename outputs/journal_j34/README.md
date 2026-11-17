# Derived journal measurements

See docs/JOURNAL_SESSION_ANALYSIS.md for interpretation, validation and commands.

complete-numeric-records.tar.gz contains only protocol and36 derived result/resource JSON pairs. No CT, radiograph, patient-identifying headers or other source image arrays are included. Extract into an empty folder and run the explicit session summary. The default summary intentionally rejects the two physical GPU UUIDs.

Measured acquired-image source: frozen cba875a implementation. All six anatomies are development data. This release adds descriptive analysis, not a new GPU rerun.

The independent study has a separate six-case preselection, protocol, measured report and post-run source/input validation addendum. The measured-source snapshot preserves exact script hashes before later guard/metadata hardening; renderer math and measured outcomes did not change. Use pinned selection for future runs and preserve the original selection for provenance.

Original datasets: DeepFluoro (Grupp et al.2020), CC BY-NC4.0, https://github.com/rg2/DeepFluoroLabeling-IPCAI2020; standardized distribution https://huggingface.co/datasets/eigenvivek/xvr-data. VerSe19 (Sekuboyina et al.2021;Loffler et al.2020), CC BY-SA4.0, https://github.com/anjany/verse. Their licenses remain in force; no dataset license is replaced by this numerical release.

The author's derived numeric measurements, report text and software retain the repository MIT license. This does not grant or replace rights in the original dataset images, which are not included. Cite Diff-FVR and the original dataset/method authors. No clinical efficacy, equivalence, universal memory benefit or independent acquired-image generalization is asserted.
