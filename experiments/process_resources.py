"""External sampled worker footprint, including CUDA context and setup.

Peaks are sampled lower bounds, not a minimum-VRAM requirement. Tree RSS sums
resident sets and can count shared pages more than once. No allocator subtraction.
"""
import json
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import time


class ResourceAccumulator:
    def __init__(self, interval_s):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError('Sampling interval must be finite and positive')
        self.interval = interval_s
        self.samples = 0
        self.previous = None
        self.gap = 0.
        self.root = self.tree = 0
        self.gpu = None

    def add(self, timestamp, root_rss, tree_rss, gpu_bytes):
        if self.previous is not None:
            self.gap = max(self.gap, timestamp-self.previous)
        self.previous = timestamp
        self.samples += 1
        self.root = max(self.root, root_rss)
        self.tree = max(self.tree, tree_rss)
        if gpu_bytes is not None:
            self.gpu = max(self.gpu or 0, gpu_bytes)

    def report(self, wall_s, returncode, *, gpu_required, max_gap_s=.5):
        valid = self.samples >= 2 and self.gap<=max_gap_s and (not gpu_required or bool(self.gpu))
        return dict(schema='DFVR-RESOURCE-1', status='ok' if valid else 'invalid',
                    returncode=returncode, worker_wall_s=wall_s,
                    requested_interval_s=self.interval, max_sample_gap_s=self.gap,
                    max_allowed_gap_s=max_gap_s, samples=self.samples, peak_root_rss_bytes=self.root,
                    peak_tree_rss_bytes=self.tree, peak_process_gpu_bytes=self.gpu,
                    gpu_required=gpu_required,
                    scope='gated worker startup through process exit; all trials, setup, warmup and provenance',
                    limitations=['sampled peaks can miss short transients',
                                 'tree RSS double-counts shared resident pages',
                                 'GPU bytes sum worker and live descendants on one exclusive GPU'])


class NvmlMemory:
    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() != 1:
                raise RuntimeError('Resource protocol requires exactly one visible NVML GPU')
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.identity = dict(name=pynvml.nvmlDeviceGetName(self.handle),
                                 uuid=pynvml.nvmlDeviceGetUUID(self.handle),
                                 driver=pynvml.nvmlSystemGetDriverVersion())
            self.read(set())  # Reject concurrent GPU workloads before launch.
        except Exception:  # aqg: top-level boundary - release driver handle on init failure
            pynvml.nvmlShutdown()
            raise

    def read(self, pids, refresh=None):
        values = {}
        processes=list(self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle))
        processes+=list(self.nvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle))
        for proc in processes:
            if proc.pid not in pids and refresh is not None:
                pids=refresh()  # A CUDA descendant may have started between queries.
            if proc.pid not in pids:
                raise RuntimeError('Foreign GPU PID: concurrent workload or PID namespace mismatch')
            value = proc.usedGpuMemory
            if value is None or value < 0 or value >= 2**64-1:
                raise RuntimeError('NVML process memory unavailable')
            values[proc.pid]=max(values.get(proc.pid,0),value)  # Compute/graphics can list the same PID.
        return sum(values.values())

    def close(self):
        self.nvml.nvmlShutdown()


def run_profiled(cmd, output, *, gpu_required, interval_s=.02, stdout=None, env=None,
                 result_path=None, protocol_sha256=None, max_gap_s=.5, timeout_s=14400):
    """Launch a stdin-gated worker; telemetry failure terminates it and is recorded."""
    import psutil
    if Path(output).exists():
        raise FileExistsError(output)  # Refuse before starting costly work.
    samples = ResourceAccumulator(interval_s)
    gpu = None
    started = time.perf_counter()
    process = None
    error = None
    sampling_s = 0.
    tracked = {}
    interrupted = False

    def live_processes():
        for proc in list(tracked.values()):
            try:
                for child in proc.children(recursive=True):
                    tracked.setdefault(child.pid,child)
            except psutil.NoSuchProcess:
                continue
        alive=[]
        for proc in tracked.values():
            try:
                if proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE:
                    alive.append(proc)
            except psutil.NoSuchProcess:
                continue
        return alive

    def group_members():
        if os.name=='posix' and process is not None:
            for proc in psutil.process_iter(['pid']):
                try:
                    if os.getpgid(proc.pid)==process.pid:
                        tracked.setdefault(proc.pid,proc)
                except (ProcessLookupError,PermissionError):
                    continue

    def owned_pids():
        live_processes()
        # NVML can retain a worker's context briefly after OS process exit.
        # Keep that known identity, but never accept a reused live PID.
        return {pid for pid,proc in tracked.items() if proc.is_running() or not psutil.pid_exists(pid)}

    def cleanup():
        if process is None:
            return
        group_members()
        members=live_processes()
        for proc in reversed(members):
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                continue
        if os.name=='posix':
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
        _,alive=psutil.wait_procs(members,timeout=2)
        for proc in alive:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                continue
        if os.name=='posix':
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
        psutil.wait_procs(alive,timeout=2)
        process.wait(timeout=5)

    try:
        gpu = NvmlMemory() if gpu_required else None
        started = time.perf_counter()
        samples.previous=started
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stdout,
                                   stderr=subprocess.STDOUT, env=env, start_new_session=True)
        root = psutil.Process(process.pid)
        tracked[root.pid]=root

        def sample():
            nonlocal sampling_s
            before = time.perf_counter()
            root_rss = rss = 0
            pids = set()
            for child in live_processes():
                try:
                    value=child.memory_info().rss
                    rss += value
                    if child.pid==root.pid:
                        root_rss=value
                    pids.add(child.pid)
                except psutil.NoSuchProcess:
                    continue  # An exited child's released memory is no longer resident.
            value = gpu.read(owned_pids(),refresh=owned_pids) if gpu else None
            samples.add(before, root_rss, rss, value)
            sampling_s += time.perf_counter()-before
            return value

        sample()
        process.stdin.write(b'1')
        process.stdin.close()
        while True:
            tick = time.perf_counter()
            if process.poll() is not None:
                group_members()  # Find same-session descendants even after root exit.
            remaining_gpu=sample()
            if tick-started>timeout_s:
                raise TimeoutError('Worker tree exceeded protocol wall-time limit')
            if samples.gap>max_gap_s:
                raise RuntimeError('Resource sampling gap exceeded protocol limit')
            if not live_processes() and process.poll() is not None and not remaining_gpu:
                break
            time.sleep(max(0., interval_s-(time.perf_counter()-tick)))
        process.wait()
    except (Exception,KeyboardInterrupt) as exc:  # aqg: top-level boundary - clean up and preserve invalid telemetry
        error = f'{type(exc).__name__}: {exc}'
        interrupted=isinstance(exc,KeyboardInterrupt)
        cleanup()
    finally:
        if gpu:
            gpu.close()
    if samples.previous is not None:
        samples.gap=max(samples.gap,time.perf_counter()-samples.previous)
    report = samples.report(time.perf_counter()-started,
                            process.returncode if process else None, gpu_required=gpu_required,max_gap_s=max_gap_s)
    report.update(sampling_overhead_s=sampling_s, gpu_identity=gpu.identity if gpu else None)
    report.update(observed_pids=sorted(tracked),timeout_s=timeout_s)
    report['protocol_sha256'] = protocol_sha256
    report['result_sha256'] = (hashlib.sha256(Path(result_path).read_bytes()).hexdigest()
                              if result_path and Path(result_path).is_file() else None)
    if error:
        report.update(status='invalid', error=error)
    with Path(output).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    if interrupted:
        raise KeyboardInterrupt
    return report
