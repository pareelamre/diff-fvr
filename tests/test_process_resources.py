"""Resource measurements must include startup and fail visibly on lost telemetry."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock
import psutil

from experiments.process_resources import ResourceAccumulator, run_profiled, NvmlMemory


class ResourceTests(unittest.TestCase):
    def test_peaks_are_contemporaneous_not_sums_of_independent_maxima(self):
        samples = ResourceAccumulator(.02)
        samples.add(1., 100, 120, 700)
        samples.add(1.05, 90, 200, 500)
        report = samples.report(2., 0, gpu_required=True)
        self.assertEqual(report['peak_root_rss_bytes'], 100)
        self.assertEqual(report['peak_tree_rss_bytes'], 200)
        self.assertEqual(report['peak_process_gpu_bytes'], 700)
        self.assertAlmostEqual(report['max_sample_gap_s'], .05)
        self.assertEqual(report['samples'], 2)

    def test_missing_gpu_telemetry_is_not_zero_memory(self):
        samples = ResourceAccumulator(.02)
        samples.add(1., 100, 100, None)
        report = samples.report(2., 0, gpu_required=True)
        self.assertEqual(report['status'], 'invalid')
        self.assertIsNone(report['peak_process_gpu_bytes'])

    def test_sparse_sampling_is_invalid(self):
        samples=ResourceAccumulator(.02)
        samples.add(1.,100,100,100)
        samples.add(3.,100,100,100)
        self.assertEqual(samples.report(4.,0,gpu_required=True,max_gap_s=.5)['status'],'invalid')

    def test_nvml_rejects_foreign_graphics_and_unavailable_values(self):
        fake=SimpleNamespace(nvmlInit=Mock(),nvmlShutdown=Mock(),
            nvmlDeviceGetCount=lambda:1,nvmlDeviceGetHandleByIndex=lambda i:i,
            nvmlDeviceGetName=lambda h:'fixture',nvmlDeviceGetUUID=lambda h:'uuid',
            nvmlSystemGetDriverVersion=lambda:'driver',
            nvmlDeviceGetComputeRunningProcesses=Mock(return_value=[]),
            nvmlDeviceGetGraphicsRunningProcesses=Mock(return_value=[]))
        with patch.dict(sys.modules,{'pynvml':fake}):
            gpu=NvmlMemory()
            fake.nvmlDeviceGetGraphicsRunningProcesses.return_value=[SimpleNamespace(pid=42,usedGpuMemory=100)]
            with self.assertRaisesRegex(RuntimeError,'Foreign'):
                gpu.read({1})
            self.assertEqual(gpu.read({1},refresh=lambda:{1,42}),100)
            fake.nvmlDeviceGetComputeRunningProcesses.return_value=[SimpleNamespace(pid=42,usedGpuMemory=100)]
            self.assertEqual(gpu.read({42}),100)  # Same context cannot be counted twice.
            for value in [None,-1,2**64-1]:
                fake.nvmlDeviceGetComputeRunningProcesses.return_value=[SimpleNamespace(pid=42,usedGpuMemory=value)]
                with self.assertRaisesRegex(RuntimeError,'unavailable'):
                    gpu.read({42})
            gpu.close()
            fake.nvmlShutdown.assert_called_once()
            fake.nvmlDeviceGetCount=lambda:2
            with self.assertRaisesRegex(RuntimeError,'exactly one'):
                NvmlMemory()
            self.assertEqual(fake.nvmlShutdown.call_count,2)

    def test_descendant_outliving_root_is_measured_until_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            child="import time; a=bytearray(32*1024*1024); a[::4096]=b'x'*len(a[::4096]); time.sleep(.7)"
            code=f"import sys,subprocess,time;sys.stdin.buffer.read(1);subprocess.Popen([sys.executable,'-c',{child!r}]);time.sleep(.25)"
            report=run_profiled([sys.executable,'-c',code],Path(tmp)/'r.json',gpu_required=False)
            self.assertEqual(report['status'],'ok')
            self.assertGreater(report['peak_tree_rss_bytes'],report['peak_root_rss_bytes']+16*1024*1024)
            self.assertGreaterEqual(len(report['observed_pids']),2)
            self.assertGreater(report['worker_wall_s'],.7)

    def test_timeout_cleans_descendants_and_records_invalid_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            child='import time; time.sleep(20)'
            code=f"import sys,subprocess,time;sys.stdin.buffer.read(1);subprocess.Popen([sys.executable,'-c',{child!r}]);time.sleep(20)"
            report=run_profiled([sys.executable,'-c',code],Path(tmp)/'r.json',gpu_required=False,timeout_s=.6)
            self.assertEqual(report['status'],'invalid')
            self.assertIn('TimeoutError',report['error'])
            self.assertGreaterEqual(len(report['observed_pids']),2)
            for pid in report['observed_pids']:
                if psutil.pid_exists(pid):
                    self.assertEqual(psutil.Process(pid).status(),psutil.STATUS_ZOMBIE)

    def test_monitor_interruption_is_recorded_and_rethrown_after_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu=SimpleNamespace(identity={},close=Mock(),read=Mock(side_effect=[1024,KeyboardInterrupt()]))
            path=Path(tmp)/'r.json'
            with patch('experiments.process_resources.NvmlMemory',return_value=gpu):
                with self.assertRaises(KeyboardInterrupt):
                    run_profiled([sys.executable,'-c',"import sys,time;sys.stdin.buffer.read(1);time.sleep(20)"],path,gpu_required=True)
            report=json.loads(path.read_text())
            self.assertEqual(report['status'],'invalid')
            self.assertIn('KeyboardInterrupt',report['error'])
            gpu.close.assert_called_once()

    def test_driver_allocation_is_followed_after_os_process_exit(self):
        class DelayedRelease:
            identity={}
            drained=0
            def read(self,pids,refresh=None):
                if pids and all(not psutil.pid_exists(pid) for pid in pids):
                    self.drained+=1
                    return 4096 if self.drained<3 else 0
                return 8192
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            gpu=DelayedRelease()
            with patch('experiments.process_resources.NvmlMemory',return_value=gpu):
                report=run_profiled([sys.executable,'-c',"import sys,time;sys.stdin.buffer.read(1);time.sleep(.1)"],Path(tmp)/'r.json',gpu_required=True)
            self.assertEqual(report['status'],'ok')
            self.assertGreaterEqual(gpu.drained,3)

    def test_telemetry_error_records_invalid_profile_and_closes_driver(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu=SimpleNamespace(identity={},close=Mock(),read=Mock(side_effect=[1024,RuntimeError('unavailable')]))
            with patch('experiments.process_resources.NvmlMemory',return_value=gpu):
                report=run_profiled([sys.executable,'-c',"import sys,time;sys.stdin.buffer.read(1);time.sleep(20)"],Path(tmp)/'r.json',gpu_required=True)
            self.assertEqual(report['status'],'invalid')
            self.assertIn('unavailable',report['error'])
            gpu.close.assert_called_once()

    def test_gate_covers_startup_and_preserves_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'resources.json'
            # Outlive one allowed .5s sampling gap, including on a busy Windows host.
            code = "import sys,time; assert sys.stdin.buffer.read(1)==b'1'; a=bytearray(8*1024*1024); a[::4096]=b'x'*len(a[::4096]); time.sleep(.6); sys.exit(7)"
            report = run_profiled([sys.executable, '-c', code], out,
                                  gpu_required=False, interval_s=.01,
                                  stdout=subprocess.DEVNULL)
            self.assertEqual(report['returncode'], 7)
            self.assertEqual(report['status'], 'ok')
            self.assertGreater(report['peak_root_rss_bytes'], 0)
            self.assertGreater(report['peak_tree_rss_bytes'], 8*1024*1024)
            self.assertGreater(report['samples'], 2)
            self.assertGreaterEqual(report['worker_wall_s'], .6)
            self.assertEqual(json.loads(out.read_text()), report)


if __name__ == '__main__':
    unittest.main()
