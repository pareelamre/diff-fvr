"""DFVR-J34: isolated-method development replication; never overwrite exp32/33.

Run without --worker to launch sequential clean processes. This is deliberately
labelled development replication: all six anatomies and selected pairs were
used in previous experiments. A frozen protocol is not an external test set.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import importlib.metadata
import importlib.util
import random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.journal_protocol import PROTOCOL, protocol_hash


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_identity():
    files = sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'experiments').glob('*.py'))
    packages = {}
    for name in ('torch','diffdrr','nanodrr','scipy','triton','nibabel','pydicom'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in files}
    for name in ('diffdrr','nanodrr'):
        spec = importlib.util.find_spec(name)
        if spec and spec.origin:
            directory = Path(spec.origin).parent
            hashes.update({f'dependency/{name}/{p.relative_to(directory)}':digest(p) for p in directory.rglob('*.py')})
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                git_status=subprocess.check_output(['git','status','--porcelain','--','.',':!outputs',':!.aqg'],cwd=ROOT,text=True),
                code_sha256=hashes, packages=packages)


def worker(args, protocol):
    import numpy as np
    import torch
    from diffdrr.data import read
    from diffdrr.pose import RigidTransform, convert
    from experiments.exp32_reviewer_rebuttal_gpu import (
        _find_max_optical_pair, _load_deepfluoro_view, _compute_mtre,
        _sample_multistart_theta, _optimize_multistart_microbatched,
        _diff_pose_to_nano_rt, _se3_exp, _biplanar_edge_loss,
    )
    from experiments.exp33_softspot_fixes_gpu import _calibrate_nanodrr_pair, _optimize_multistart_topk
    from src.fvr.cone_beam_fvr import PerspectiveSlabFVR

    identity = code_identity()
    if identity['git_status'] and not args.smoke:
        raise ValueError('Full runs require a clean committed code tree')
    manifest = json.loads((args.output/'protocol.json').read_text(encoding='utf-8'))
    if manifest['sha256'] != protocol_hash(protocol) or manifest['code_identity'] != identity:
        raise ValueError('Worker code/protocol differs from frozen manifest')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required for full journal replication; use --smoke --device cpu for checks')
    if device.type == 'cpu' and not args.smoke:
        raise ValueError('Full protocol requires CUDA; CPU is smoke-only')
    torch.manual_seed(protocol['seed'])
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    def memory():
        if device.type != 'cuda':
            return {k: None for k in ('allocated_MiB','reserved_MiB','peak_allocated_MiB','peak_reserved_MiB')}
        return dict(allocated_MiB=torch.cuda.memory_allocated()/2**20,
                    reserved_MiB=torch.cuda.memory_reserved()/2**20,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)

    subj_dir = args.data_root / args.subject
    started = time.perf_counter()
    source_files = [subj_dir/'volume.nii.gz', subj_dir/'fiducials.pt']
    import nibabel as nib
    affine = nib.load(source_files[0]).affine
    axes = affine[:3,:3]
    scales = np.linalg.norm(axes,axis=0)
    if (not np.isfinite(affine).all() or np.any(scales<=0)
            or not np.allclose((axes/scales).T@(axes/scales),np.eye(3),atol=1e-4)):
        raise ValueError('CT affine must be finite, invertible and free of shear')
    subject = read(str(source_files[0]), orientation='AP')
    fids = torch.load(source_files[1], weights_only=True).to(device=device, dtype=torch.float32)
    if fids.ndim not in (2,3) or fids.shape[-1] != 3 or not torch.isfinite(fids).all():
        raise ValueError('Fiducials must be finite L by 3 physical coordinates')
    vA, vB, angle = _find_max_optical_pair(subject, str(subj_dir))
    H = protocol['height']
    poseA, xA, dA, intrA = _load_deepfluoro_view(subject, str(subj_dir), vA, H=H, device=device)
    poseB, xB, dB, intrB = _load_deepfluoro_view(subject, str(subj_dir), vB, H=H, device=device)
    source_files += sorted((subj_dir/'xrays').glob('*.pt'))
    source_files += [subj_dir/'xrays'/f'{vid}.dcm' for vid in (vA,vB)]
    # Relative camera pose is provided by the reference annotations, a known-
    # geometry assumption shared by all arms, not an estimated calibration.
    relative = poseA.matrix.inverse() @ poseB.matrix
    method = args.worker
    parameters = protocol['parameters']
    pou = nanoA = nanoB = nano_subject = iso = None
    calibration = None
    if method.startswith('PoU'):
        blocks = tuple(parameters['standalone_grid'] if method == 'PoU-FVR-G4x4x4-R64' else parameters['screen_grid'])
        if args.smoke:
            blocks = (2,2,2)
        pou = PerspectiveSlabFVR.from_diffdrr_subject(subject, grid_blocks=blocks,
                  block_res=32 if args.smoke else parameters['block_res'], pad_mult=parameters['padding'], use_triton=device.type=='cuda').to(device)
        if device.type == 'cuda':
            pou.to_precision(torch.bfloat16)
        latticeA = pou.cache_diffdrr_geometry(dA, H)
        latticeB = pou.cache_diffdrr_geometry(dB, H)
    if 'nanodrr' in method:
        # Calibration receives only fixed synthetic camera poses and metadata.
        # It never receives acquired target pixels or reference evaluation poses.
        fixed = torch.tensor(protocol['canonical_poses'], device=device, dtype=torch.float32)
        canonical = convert(fixed[:, :3], fixed[:, 3:], parameterization='euler_angles', convention='ZYX')
        nano_subject, unused_bf16, nanoA, nanoB, iso, ncc = _calibrate_nanodrr_pair(
            str(source_files[0]), dA, RigidTransform(canonical.matrix[:1]), intrA,
            dB, RigidTransform(canonical.matrix[1:]), intrB, device, min_ncc=parameters['calibration_min_ncc'])
        del unused_bf16
        calibration = {'canonical_ncc': float(ncc), 'isocenter_mm': iso.detach().cpu().tolist(),
                       'poses': protocol['canonical_poses'], 'cameraA':{k:v.detach().cpu().tolist() for k,v in nanoA.named_buffers()},
                       'cameraB':{k:v.detach().cpu().tolist() for k,v in nanoB.named_buffers()}}
        # Explicit eager execution for both spatial arms; no unmatched graph replay.
    if method not in ('DiffDRR-SS-FP32','DiffDRR-C64-FP32','PoU-FVR+DiffDRR-K4-FP32'):
        del dA, dB
    if device.type=='cuda':
        torch.cuda.empty_cache()
    sync()
    setup_s = time.perf_counter() - started
    setup_memory = memory()
    rows = []
    index = int(args.subject.removeprefix('subject'))
    warmup_s = None
    for tier_index, (tier, rlim, tlim, sr, st) in enumerate(protocol['tiers']):
        for init_index in range(protocol['inits_per_tier']):
            rng = np.random.default_rng(protocol['seed'] + index*100 + tier_index*20 + init_index)
            perturb = np.concatenate([rng.uniform(-rlim,rlim,(1,3)), rng.uniform(-tlim,tlim,(1,3))],axis=1)
            perturb = torch.tensor(perturb,device=device,dtype=torch.float32)
            initial = convert(perturb[:,:3], perturb[:,3:], parameterization='euler_angles',convention='ZYX').matrix @ poseA.matrix
            initialB = initial @ relative
            seeds = _sample_multistart_theta(protocol['candidates'], device, protocol['seed']+3000+index*100+tier_index*20+init_index, sr, st, keep_zero=tier=='bounded')
            def spatial(dT):
                return dA(RigidTransform(dT@initial)), dB(RigidTransform(dT@initialB))
            def spectral(dT):
                return pou._forward_from_lattice(dT@initial,latticeA), pou._forward_from_lattice(dT@initialB,latticeB)
            def nano(dT):
                backend = 'auto' if device.type=='cuda' else 'torch'
                return (nanoA(nano_subject,_diff_pose_to_nano_rt(dT@initial,iso),n_samples=protocol['nano_samples'],backend=backend),
                        nanoB(nano_subject,_diff_pose_to_nano_rt(dT@initialB,iso),n_samples=protocol['nano_samples'],backend=backend))
            def run():
                steps = protocol['steps']
                if method == 'DiffDRR-SS-FP32':
                    return _optimize_multistart_microbatched(spatial,torch.zeros_like(seeds[:1]),xA,xB,steps['single'])
                if method == 'DiffDRR-C64-FP32':
                    return _optimize_multistart_microbatched(spatial,seeds,xA,xB,steps['spatial'],micro_batch=protocol['diffdrr_microbatch'])
                if method == 'nanodrr-C64-FP32':
                    return _optimize_multistart_topk(nano,seeds,xA,xB,steps['spatial'])
                if method == 'PoU-FVR-G4x4x4-R64':
                    return _optimize_multistart_topk(spectral,seeds,xA,xB,steps['pou'])
                th, _ = _optimize_multistart_topk(spectral,seeds,xA,xB,steps['screen'],top_k=protocol['survivors'])
                if 'nanodrr' in method:
                    return _optimize_multistart_topk(nano,th,xA,xB,steps['polish'],lr_rot=.008,lr_trans=.8)
                return _optimize_multistart_microbatched(spatial,th,xA,xB,steps['polish'],micro_batch=protocol['diffdrr_microbatch'],lr_rot=.008,lr_trans=.8)
            if warmup_s is None:
                # Include lazy CUDA/Triton compilation in explicit measured warmup.
                # One real optimization runs and is discarded before timed trials.
                sync(); tw = time.perf_counter()
                warm_theta, warm_loss = run()
                del warm_theta, warm_loss
                sync(); warmup_s = time.perf_counter()-tw
            resident = memory()
            if device.type=='cuda':
                torch.cuda.reset_peak_memory_stats()
            sync(); t0=time.perf_counter()
            theta, loss = run()
            sync(); elapsed=time.perf_counter()-t0
            mem=memory()
            mem['incremental_peak_allocated_MiB'] = (mem['peak_allocated_MiB']-resident['allocated_MiB']) if device.type=='cuda' else None
            error=_compute_mtre(RigidTransform(_se3_exp(theta[:1])@initial),poseA,fids)
            if not np.isfinite(error) or not torch.isfinite(loss).all():
                raise RuntimeError('Nonfinite registration outcome; worker failure preserved in log')
            row=dict(subject=args.subject,tier=tier,init_idx=init_index+1,pair=f'{vA}+{vB}',
                     method=method,status='ok',angle_deg=angle,final_mtre_mm=error,
                     init_mtre_mm=_compute_mtre(RigidTransform(initial),poseA,fids),
                     initial_delta=perturb.cpu().tolist(),candidate_sha256=hashlib.sha256(seeds.cpu().numpy().tobytes()).hexdigest(),
                     reg_time_s=elapsed,setup_s=setup_s,warmup_s=warmup_s,
                     hypothetical_setup_warmup_one_reg_s=setup_s+warmup_s+elapsed,
                     resident=resident,memory=mem,final_edge_loss=float(loss[0]),
                     protocol_sha256=protocol_hash(protocol),calibration=calibration)
            rows.append(row)
            with (args.output/f'{args.subject}_{method}.trials.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
            print(json.dumps(row,allow_nan=False),flush=True)
    result=dict(protocol=protocol,protocol_sha256=protocol_hash(protocol),rows=rows,
                setup_memory=setup_memory,python=platform.python_version(),torch=torch.__version__,
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
                code_identity=identity,
                source_sha256={str(p.relative_to(subj_dir)):digest(p) for p in source_files},
                limitations=['development pairs','reference-derived relative camera geometry',
                  'allocator counters exclude driver/context allocations; not device/process footprint'])
    path=args.output/f'{args.subject}_{method}.json'
    with path.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--worker',choices=PROTOCOL['methods'])
    p.add_argument('--subject',default='subject01',choices=[f'subject{i:02d}' for i in range(1,7)])
    args=p.parse_args()
    protocol=copy.deepcopy(PROTOCOL)
    if args.smoke:
        protocol.update(id='DFVR-J34-SMOKE',height=64,candidates=4,survivors=2,inits_per_tier=1)
        protocol['steps']={k:2 for k in protocol['steps']}
        protocol['tiers']=protocol['tiers'][:1]
    args.output.mkdir(parents=True,exist_ok=True)
    if args.worker:
        if args.smoke and not (args.output/'protocol.json').exists():
            (args.output/'protocol.json').write_text(json.dumps(dict(protocol=protocol,sha256=protocol_hash(protocol),code_identity=code_identity())),encoding='utf-8')
        worker(args,protocol)
        return
    manifest=args.output/'protocol.json'
    with manifest.open('x',encoding='utf-8') as stream:
        json.dump(dict(protocol=protocol,sha256=protocol_hash(protocol),code_identity=code_identity()),stream,indent=2)
    subjects=['subject01'] if args.smoke else [f'subject{i:02d}' for i in range(1,7)]
    failures = 0
    for index, subject in enumerate(subjects):
        order = list(protocol['methods'])
        random.Random(protocol['seed']+index).shuffle(order)
        for method in order:
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',method,'--subject',subject,
                 '--data-root',str(args.data_root),'--output',str(args.output),'--device',args.device]
            if args.smoke:
                cmd.append('--smoke')
            log=args.output/f'{subject}_{method}.log'
            with log.open('x',encoding='utf-8') as stream:
                completed=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
            print(subject,method,'exit',completed.returncode,flush=True)
            if completed.returncode:
                failures += 1
                # Preserve failure; continue independent methods. Never create success rows.
                (args.output/f'{subject}_{method}.failed.json').write_text(json.dumps(dict(subject=subject,method=method,status='failed',returncode=completed.returncode)),encoding='utf-8')
            elif not (args.output/f'{subject}_{method}.json').is_file():
                failures += 1
    if failures:
        raise SystemExit(f'{failures} workers failed or did not produce a complete result')


if __name__=='__main__':
    main()
