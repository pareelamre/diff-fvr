"""Submission-critical regressions using synthetic inputs only."""
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.exp32_reviewer_rebuttal_gpu import _sample_multistart_theta
from src.fvr.dicom_loader import load_dicom_series
from experiments.journal_protocol import paired_specimen_effect, holm_adjust, protocol_hash, PROTOCOL
from experiments.exp34_journal_replication import main as replication_main


class JournalProtocolTests(unittest.TestCase):
    def test_worker_failures_make_batch_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out'
            argv=['replicate','--smoke','--data-root',tmp,'--output',str(out)]
            with patch.object(sys,'argv',argv), patch('experiments.exp34_journal_replication.code_identity',return_value={}), patch('experiments.exp34_journal_replication.run_profiled',return_value={'returncode':1,'status':'ok'}):
                with self.assertRaisesRegex(SystemExit,'6 workers failed'):
                    replication_main()
            self.assertEqual(len(list(out.glob('*.failed.json'))),6)

    def test_specimens_not_trials_are_inference_units(self):
        rows = [dict(subject=s, tier='wide', init_idx=i, pair='A+B', method=m,
                     final_mtre_mm=v) for s in ('a','b','c') for i in range(4)
                for m,v in [('new',1.),('reference',3.)]]
        expected = {(r['subject'],r['tier'],r['init_idx'],r['pair']) for r in rows}
        effect = paired_specimen_effect(rows, 'new', 'reference', expected_keys=expected, bootstrap=100)
        self.assertEqual(effect['n_specimens'], 3)
        self.assertEqual(effect['n_trials'], 12)
        self.assertEqual(effect['exploratory_cluster_percentile95_mm'], [-2., -2.])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            paired_specimen_effect(rows[:-1], 'new', 'reference', expected_keys=expected)
        with self.assertRaisesRegex(ValueError, 'expected trial roster'):
            paired_specimen_effect([r for r in rows if r['subject']!='c'], 'new', 'reference', expected_keys=expected)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            paired_specimen_effect(rows + [rows[0]], 'new', 'reference', expected_keys=expected)

    def test_holm_and_protocol_hash(self):
        np.testing.assert_allclose(holm_adjust([.03,.01,.2]), [.06,.03,.2])
        self.assertNotEqual(protocol_hash(PROTOCOL), protocol_hash(dict(PROTOCOL, candidates=8)))

    def test_single_candidate_keeps_common_initial_pose(self):
        seed = _sample_multistart_theta(1, torch.device('cpu'), 42, 16, 22, keep_zero=True)
        torch.testing.assert_close(seed, torch.zeros(1, 6))

    def write_series(self, folder, positions, orientations=None):
        series_uid = generate_uid()
        frame_uid = generate_uid()
        for i, z in enumerate(positions):
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(str(folder / f'{i}.dcm'), {}, file_meta=meta, preamble=b'\0' * 128)
            ds.SOPClassUID = CTImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.SeriesInstanceUID = series_uid
            ds.FrameOfReferenceUID = frame_uid
            ds.Modality = 'CT'
            ds.ImagePositionPatient = [0., 0., z]
            ds.ImageOrientationPatient = orientations[i] if orientations else [1., 0., 0., 0., 1., 0.]
            ds.PixelSpacing = [0.5, 0.75]
            ds.Rows = ds.Columns = 4
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            ds.RescaleIntercept = -1000
            ds.RescaleSlope = 2
            ds.PixelData = np.full((4, 4), i, dtype=np.int16).tobytes()
            ds.save_as(ds.filename, enforce_file_format=True)

    def test_geometry_failures_and_order(self):
        changes = [('FrameOfReferenceUID', generate_uid(), 'coordinate frame'),
                   ('SeriesInstanceUID', generate_uid(), 'Mixed'),
                   ('ImagePositionPatient', [0,0,0], 'Duplicate'),
                   ('ImagePositionPatient', [1,0,4], 'shear'),
                   ('NumberOfFrames', 2, 'multi-frame'),
                   ('PhotometricInterpretation', 'MONOCHROME1', 'MONOCHROME2'),
                   ('SamplesPerPixel', 3, 'single-sample')]
        for tag, value, message in changes:
            with self.subTest(tag=tag,value=value), tempfile.TemporaryDirectory() as tmp:
                self.write_series(Path(tmp), [0.,2.,4.])
                path=Path(tmp)/'2.dcm'
                ds=pydicom.dcmread(path); setattr(ds,tag,value); ds.save_as(path)
                with self.assertRaisesRegex(ValueError,message):
                    load_dicom_series(tmp)
        with tempfile.TemporaryDirectory() as tmp:
            self.write_series(Path(tmp), [4.,2.,0.])
            volume,_=load_dicom_series(tmp)
            np.testing.assert_allclose(volume[:,0,0],[-996,-998,-1000])

    def test_irregular_slice_spacing_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_series(Path(tmp), [0., 1., 3.])
            with self.assertRaisesRegex(ValueError, 'nonuniform'):
                load_dicom_series(tmp)

    def test_changing_orientation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_series(Path(tmp), [0., 1., 2.], [[1,0,0,0,1,0], [1,0,0,0,1,0], [1,0,0,0,.99,.1]])
            with self.assertRaises(ValueError):
                load_dicom_series(tmp)

    def test_regular_grid_preserves_hu_and_spacing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_series(Path(tmp), [0., 2., 4.])
            volume, spacing = load_dicom_series(tmp)
            np.testing.assert_allclose(spacing, (2., .5, .75))
            np.testing.assert_allclose(volume[:, 0, 0], [-1000, -998, -996])


if __name__ == '__main__':
    unittest.main()
