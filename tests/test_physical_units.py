"""Synthetic regressions for physical units and supported projection geometry."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fvr.cli import load_volume, main_register, main_render
from src.fvr.core_engine import RobustDiffFVREngine
from src.fvr.registration_3d2d import compute_mtre_6dof
from experiments.exp16_unified_ced_curves import load_metric_table


def phantom(size=24):
    z, y, x = np.indices((size,) * 3)
    return (np.exp(-((z-10)**2 + (y-11)**2 + (x-10)**2) / 8)
            + .7 * np.exp(-((z-15)**2 + (y-15)**2 + (x-15)**2) / 4))


class PhysicalUnitsTests(unittest.TestCase):
    def test_mm_translation_moves_expected_number_of_pixels(self):
        engine = RobustDiffFVREngine(phantom(), spacing=(2.,) * 3)
        for view in ('ap', 'lateral'):
            with self.subTest(view=view):
                image = engine.render_calibrated_drr(view=view, crop_to_roi=False)
                shifted = engine.render_calibrated_drr(tx=4, ty=-2, view=view, crop_to_roi=False)
                np.testing.assert_allclose(shifted, np.roll(image, (-1, 2), axis=(0, 1)), atol=1e-9)

    def test_registration_returns_mm_in_both_loss_paths(self):
        engine = RobustDiffFVREngine(phantom(), spacing=(2.,) * 3)
        true_pose = (0., 0., 0., 2., -2., 2.)
        for metal in (False, True):
            with self.subTest(metal=metal):
                ap = engine.render_calibrated_drr(tx=2, ty=2, view='ap')
                lat = engine.render_calibrated_drr(tx=-2, ty=2, view='lateral')
                if metal:
                    ap[1:4, 1:8] = ap.max() * 10
                    lat[1:4, 1:8] = lat.max() * 10
                result = engine.register_biplanar_robust(ap, lat, handle_metal=metal)
                error = compute_mtre_6dof(true_pose, result['final_pose'], phantom().shape,
                                          spacing=(2.,) * 3, translation_units='mm')
                self.assertLess(error, .3)
                self.assertEqual(result['metal_detected'], metal)

    def test_mtre_scales_voxel_translations_and_rotations(self):
        origin = (0.,) * 6
        self.assertAlmostEqual(compute_mtre_6dof(origin, (0, 0, 0, 3, 0, 0),
                               (16,) * 3, spacing=(2.,) * 3), 6.)
        self.assertAlmostEqual(compute_mtre_6dof(origin, (0, 0, 0, 3, 0, 0),
                               (16,) * 3, spacing=(2.,) * 3, translation_units='mm'), 3.)
        self.assertAlmostEqual(compute_mtre_6dof(origin, (90, 0, 0, 0, 0, 0),
                               (16,) * 3, spacing=(2.,) * 3), 16.)

    def test_explicit_cone_beam_geometry_is_rejected(self):
        for geometry in ({'sdd_mm': 1000, 'sad_mm': 750}, {'sdd_mm': 1000}):
            with self.subTest(geometry=geometry), self.assertRaisesRegex(ValueError, 'parallel-beam'):
                RobustDiffFVREngine(phantom(), **geometry)

    def test_invalid_physical_grids_are_rejected(self):
        for spacing in ((0, 0, 0), (float('nan'),) * 3, (1, 2, 1)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                RobustDiffFVREngine(phantom(), spacing=spacing)

    def test_registration_rejects_invalid_or_uninformative_targets(self):
        engine = RobustDiffFVREngine(phantom())
        valid = engine.render_calibrated_drr(view='ap')
        for target in (np.zeros((24, 24)), np.full((24, 24), np.nan), np.ones((12, 12))):
            with self.subTest(shape=target.shape), self.assertRaises(ValueError):
                engine.register_biplanar_robust(target, valid)

    def test_cli_resampling_preserves_spacing_and_long_fov(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ct.npy'
            np.save(path, np.zeros((40, 16, 16)))
            volume, spacing = load_volume(path, target_res=16, spacing_mm=(3, 1, 1))
            self.assertEqual(volume.shape, (16,) * 3)
            np.testing.assert_allclose(spacing, (7.5,) * 3)
            self.assertGreater(volume[:, 8, 8].min(), .99)

    def test_dicom_is_resampled_even_if_y_already_matches_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('src.fvr.cli.load_dicom_series', return_value=(np.zeros((8, 16, 16)), (4, 1, 1))):
                volume, spacing = load_volume(Path(folder), target_res=16)
            self.assertEqual(volume.shape, (16,) * 3)
            np.testing.assert_allclose(spacing, (2.,) * 3)

    def test_render_cli_uses_xyz_in_the_selected_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            ct = Path(folder) / 'ct.npy'
            np.save(ct, phantom() * 1000 - 1000)
            for view, expected in [('ap', (4., 6.)), ('lateral', (8., 6.))]:
                out = Path(folder) / f'{view}.png'
                args = ['fvr-render', '--ct', str(ct), '--out', str(out), '--res', '24',
                        '--view', view, '--tx', '4', '--ty', '8', '--tz', '6',
                        '--spacing-mm', '2', '2', '2']
                with patch.object(sys, 'argv', args), patch.object(RobustDiffFVREngine,
                        'render_calibrated_drr', return_value=np.ones((24, 24))) as render:
                    with contextlib.redirect_stdout(io.StringIO()):
                        main_render()
                self.assertEqual((render.call_args.kwargs['tx'], render.call_args.kwargs['ty']), expected)
                self.assertTrue(out.exists())

    def test_failed_cli_registration_is_reported_in_json_and_exit_code(self):
        with tempfile.TemporaryDirectory() as folder:
            ct, image, out = [Path(folder) / name for name in ('ct.npy', 'ap.npy', 'result.json')]
            np.save(ct, phantom())
            np.save(image, np.ones((24, 24)))
            args = ['fvr-register', '--ct', str(ct), '--ap', str(image), '--lat', str(image),
                    '--res', '24', '--out', str(out)]
            result = dict(success=False, final_pose=np.zeros(6), final_loss=.5,
                          time_sec=.1, evaluations=5, metal_detected=False)
            with patch.object(sys, 'argv', args), patch.object(RobustDiffFVREngine,
                    'register_biplanar_robust', return_value=result):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as failure:
                        main_register()
            self.assertEqual(failure.exception.code, 1)
            self.assertFalse(json.loads(out.read_text())['success'])

    def test_summary_rejects_unverified_historical_units(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'results.csv'
            path.write_text('Final_mTRE_mm\n0.1\n')
            with self.assertRaisesRegex(ValueError, 'regenerate'):
                load_metric_table(path)
            path.write_text('Final_mTRE_mm,Metric_units\n0.1,mm\n')
            self.assertEqual(load_metric_table(path)['Final_mTRE_mm'].iloc[0], .1)


if __name__ == '__main__':
    unittest.main()
