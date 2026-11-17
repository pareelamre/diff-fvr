"""Fail-closed summary of a complete DFVR-J34 batch; no partial significance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from experiments.journal_protocol import protocol_hash, paired_specimen_effect, holm_adjust


def read_complete_batch(root, subjects=None, *, resource_policy='single-device'):
    if resource_policy not in ('single-device','stratified'):
        raise ValueError('Unknown resource analysis policy')
    subjects = subjects or [f'subject{i:02d}' for i in range(1,7)]
    root=Path(root)
    if list(root.glob('*.failed.json')):
        raise ValueError('Failed workers present; report failures before paired inference')
    manifest=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    protocol=manifest['protocol']
    if resource_policy=='stratified' and not protocol.get('resources',{}).get('required'):
        raise ValueError('Stratified analysis requires bound resource profiles')
    if manifest['sha256'] != protocol_hash(protocol):
        raise ValueError('Invalid protocol digest')
    rows=[]; gpu=set(); gpu_ids=set(); shared={}; data_hashes={}
    expected={(t[0],i+1) for t in protocol['tiers'] for i in range(protocol['inits_per_tier'])}
    for subject in subjects:
        for method in protocol['methods']:
            path=root/f'{subject}_{method}.json'
            if not path.is_file():
                raise ValueError(f'Missing expected result: {path.name}')
            data=json.loads(path.read_text(encoding='utf-8'))
            if protocol.get('resources',{}).get('required'):
                resource_path=root/f'{subject}_{method}.resources.json'
                if not resource_path.is_file():
                    raise ValueError(f'Missing resource profile: {path.name}')
                profile=json.loads(resource_path.read_text(encoding='utf-8'))
                if (profile.get('schema')!='DFVR-RESOURCE-1' or profile.get('status')!='ok'
                        or profile.get('returncode')!=0 or profile.get('gpu_required') is not True
                        or profile.get('protocol_sha256')!=manifest['sha256']
                        or profile.get('result_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest()):
                    raise ValueError(f'Invalid resource identity/status: {path.name}')
                for key in ('peak_process_gpu_bytes','peak_root_rss_bytes','peak_tree_rss_bytes',
                            'samples','worker_wall_s','requested_interval_s'):
                    value=profile.get(key)
                    if value is None or not math.isfinite(float(value)) or value<=0:
                        raise ValueError(f'Invalid resource measurement: {key}')
                for key in ('max_sample_gap_s','sampling_overhead_s'):
                    if not math.isfinite(float(profile.get(key,-1))) or profile.get(key,-1)<0:
                        raise ValueError(f'Invalid resource measurement: {key}')
                if profile['samples']<2 or profile['max_sample_gap_s']>protocol['resources'].get('max_gap_s',.5):
                    raise ValueError('Invalid resource sampling coverage')
                if profile['requested_interval_s']!=protocol['resources'].get('interval_s',.02):
                    raise ValueError('Invalid resource sampling interval')
                gpu_identity=profile.get('gpu_identity') or {}
                if gpu_identity.get('name')!=data['gpu'] or not gpu_identity.get('uuid'):
                    raise ValueError('Resource GPU identity mismatch')
                gpu_ids.add(gpu_identity['uuid'])
                # One worker profile describes all its trials; never sum repeated peaks.
                for row in data['rows']:
                    row['process_resources']=profile
                    row['worker_allocator_phases']={k:data[k] for k in ('setup_memory','warmup_memory') if k in data}
            if (data['protocol_sha256']!=manifest['sha256'] or data['protocol']!=protocol
                    or data['code_identity']!=manifest['code_identity']):
                raise ValueError(f'Code/protocol identity mismatch: {path.name}')
            gpu.add(data['gpu'])
            if subject in data_hashes and data_hashes[subject]!=data['source_sha256']:
                raise ValueError('Paired input data hashes differ')
            data_hashes[subject]=data['source_sha256']
            selected=data['rows']
            if len(selected)!=len(expected) or {(r['tier'],r['init_idx']) for r in selected}!=expected:
                raise ValueError(f'Incomplete/duplicate trial roster: {path.name}')
            for row in selected:
                if row['subject']!=subject or row['method']!=method or row['status']!='ok':
                    raise ValueError(f'Failed/misidentified trial: {path.name}')
                for key in ('final_mtre_mm','reg_time_s','setup_s','warmup_s'):
                    if not math.isfinite(float(row[key])) or float(row[key])<0:
                        raise ValueError(f'Invalid numeric result: {key}')
                key=(subject,row['tier'],row['init_idx'])
                initial=(row['pair'],row['initial_delta'],row['candidate_sha256'])
                if key in shared and shared[key]!=initial:
                    raise ValueError('Paired initial pose/candidate pool mismatch')
                shared[key]=initial
                rows.append(row)
    if len(gpu)!=1:
        raise ValueError('Mixed GPU models; split resource analyses by hardware')
    if len(gpu_ids)>1 and resource_policy=='single-device':
        raise ValueError('Mixed GPU devices; split resource analyses by hardware')
    return manifest,rows,next(iter(gpu))


def summarize(root, subjects=None):
    manifest,rows,gpu=read_complete_batch(root,subjects=subjects)
    protocol=manifest['protocol']
    if 'SMOKE' in protocol.get('id',''):
        raise ValueError('Smoke output is verification only; full CUDA protocol required for performance summary')
    threshold=protocol.get('resources',{}).get('success_threshold_mm',2.)
    summaries=[]
    for method in protocol['methods']:
        selected=[r for r in rows if r['method']==method]
        errors=[r['final_mtre_mm'] for r in selected]
        item=dict(method=method,n_specimens=len({r['subject'] for r in selected}),
             n_trials=len(selected),median_mtre_mm=statistics.median(errors),
             success_threshold_mm=threshold,successes_below_threshold=sum(e<threshold for e in errors),
             mean_reg_s=statistics.mean(r['reg_time_s'] for r in selected),
             mean_setup_s=statistics.mean(r['setup_s'] for r in selected),
             mean_warmup_s=statistics.mean(r['warmup_s'] for r in selected),
             max_peak_allocated_MiB=max(r['memory']['peak_allocated_MiB'] for r in selected),
             max_peak_reserved_MiB=max(r['memory']['peak_reserved_MiB'] for r in selected))
        if protocol.get('resources',{}).get('required'):
            profiles={r['subject']:r['process_resources'] for r in selected}
            item.update(max_worker_process_gpu_MiB=max(p['peak_process_gpu_bytes'] for p in profiles.values())/2**20,
                        max_worker_root_rss_MiB=max(p['peak_root_rss_bytes'] for p in profiles.values())/2**20,
                        max_worker_tree_rss_MiB=max(p['peak_tree_rss_bytes'] for p in profiles.values())/2**20,
                        mean_worker_wall_s=statistics.mean(p['worker_wall_s'] for p in profiles.values()),
                        max_sample_gap_s=max(p['max_sample_gap_s'] for p in profiles.values()),
                        mean_sampling_overhead_s=statistics.mean(p['sampling_overhead_s'] for p in profiles.values()),
                        estimated_amortized_s_per_registration={str(n):
                            (item['mean_setup_s']+item['mean_warmup_s']+n*item['mean_reg_s'])/n
                            for n in protocol['resources']['amortization_registrations']})
            phase_values=[v for r in selected for v in r['worker_allocator_phases'].values()]+[r['memory'] for r in selected]
            item['max_worker_allocator_allocated_MiB']=max(v['peak_allocated_MiB'] for v in phase_values)
            item['max_worker_allocator_reserved_MiB']=max(v['peak_reserved_MiB'] for v in phase_values)
        summaries.append(item)
    expected={(r['subject'],r['tier'],r['init_idx'],r['pair']) for r in rows}
    reference=protocol['analysis']['reference']
    comparisons=[]
    for method in protocol['methods']:
        if method==reference: continue
        effect=paired_specimen_effect(rows,method,reference,expected_keys=expected)
        comparisons.append(dict(method=method,reference=reference,**effect))
    for effect,p in zip(comparisons,holm_adjust([r['p_unadjusted'] for r in comparisons])):
        effect['p_holm_descriptive']=p
    return dict(gpu=gpu,protocol_sha256=manifest['sha256'],code_identity=manifest['code_identity'],
                summaries=summaries,comparisons=comparisons,
                practical_comparisons=practical_comparisons(rows,reference,threshold_mm=threshold),
                analysis=protocol['analysis'],status='complete development replication',
                limitations=['previously used anatomies','known relative camera geometry',
                    'sampled process peaks can miss transients; allocator counters reported separately',
                    'amortized estimates reuse one CT setup and do not include process startup/provenance',
                    'small-n exploratory inference'])


def practical_comparisons(rows, reference, threshold_mm=2.):
    """Raw paired costs plus jointly successful subset, without discarding failures."""
    key=lambda r:(r['subject'],r['tier'],r['init_idx'],r['pair'])
    references={key(r):r for r in rows if r['method']==reference}
    reports=[]
    for method in sorted({r['method'] for r in rows}-{reference}):
        for tier in sorted({r['tier'] for r in rows}):
            paired=[(r,references[key(r)]) for r in rows if r['method']==method and r['tier']==tier]
            joint=[(a,b) for a,b in paired if a['final_mtre_mm']<threshold_mm and b['final_mtre_mm']<threshold_mm]
            ratios=[b['reg_time_s']/a['reg_time_s'] for a,b in joint if a['reg_time_s']>0]
            reports.append(dict(method=method,reference=reference,tier=tier,n_paired=len(paired),
                threshold_mm=threshold_mm,method_successes=sum(a['final_mtre_mm']<threshold_mm for a,b in paired),
                reference_successes=sum(b['final_mtre_mm']<threshold_mm for a,b in paired),
                n_joint_success=len(joint),median_warm_speed_ratio_joint_success=statistics.median(ratios) if ratios else None,
                paired_trials=[dict(subject=a['subject'],init_idx=a['init_idx'],
                    method_mtre_mm=a['final_mtre_mm'],reference_mtre_mm=b['final_mtre_mm'],
                    method_reg_s=a['reg_time_s'],reference_reg_s=b['reg_time_s']) for a,b in paired],
                interpretation='Ratio >1 favors method; selected joint-success subset is descriptive, not equivalence or clinical generalization'))
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=summarize(args.input)
    report['result_file_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(args.input.glob('*.json'))}
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(f'Validated {sum(r["n_trials"] for r in report["summaries"])} rows on {report["gpu"]}')


if __name__=='__main__': main()
