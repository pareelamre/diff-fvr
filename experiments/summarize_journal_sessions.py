"""Explicit descriptive device strata for a complete frozen journal batch.

The default single-device summary remains strict. Accuracy retains every
specimen; resource comparisons use complete, same-device specimen blocks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.summarize_journal_replication import read_complete_batch


def median(values):
    values=list(values)
    return statistics.median(values) if values else None


def summarize_sessions(root, subjects=None):
    manifest,rows,gpu=read_complete_batch(root,subjects=subjects,resource_policy='stratified')
    protocol=manifest['protocol']
    if any(r['reg_time_s']<=0 for r in rows):
        raise ValueError('Positive measured registration costs required for ratios')
    if 'SMOKE' in protocol.get('id',''):
        raise ValueError('Smoke output is verification only')
    methods=protocol['methods'];reference=protocol['analysis']['reference']
    if reference not in methods:raise ValueError('Reference absent from protocol methods')
    threshold=protocol['resources'].get('success_threshold_mm',2.)
    all_subjects=sorted({r['subject'] for r in rows})
    profiles={(r['subject'],r['method']):r['process_resources'] for r in rows}
    device=lambda s,m:profiles[s,m]['gpu_identity']['uuid']
    accuracy=[]
    all_worker_memory=[]
    for method in methods:
        selected=[r for r in rows if r['method']==method]
        accuracy.append(dict(method=method,n_specimens=len(all_subjects),n_trials=len(selected),
                             median_mtre_mm=median(r['final_mtre_mm'] for r in selected),
                             successes=sum(r['final_mtre_mm']<threshold for r in selected),
                             success_threshold_mm=threshold,
                             specimen_results=[dict(subject=s,median_mtre_mm=median(r['final_mtre_mm'] for r in selected if r['subject']==s),
                                 successes=sum(r['final_mtre_mm']<threshold for r in selected if r['subject']==s)) for s in all_subjects]))
        resources=[profiles[s,method] for s in all_subjects]
        all_worker_memory.append(dict(method=method,n_workers=len(resources),
            scope='Maximum over all observed specimens, including crossed-device specimens; not a paired device comparison.',
            max_process_gpu_MiB=max(p['peak_process_gpu_bytes'] for p in resources)/2**20,
            max_process_tree_rss_MiB=max(p['peak_tree_rss_bytes'] for p in resources)/2**20))
    strata=[]
    for uuid in sorted({device(s,m) for s,m in profiles}):
        matched=[s for s in all_subjects if all(device(s,m)==uuid for m in methods)]
        observed=[dict(subject=s,method=m) for s,m in sorted(profiles) if device(s,m)==uuid]
        stratum=dict(gpu_uuid=uuid,gpu_name=gpu,observed_workers=observed,
                     matched_subjects=matched,methods=[],comparisons=[])
        if not matched:
            strata.append(stratum);continue
        selected=[r for r in rows if r['subject'] in matched]
        for method in methods:
            trials=[r for r in selected if r['method']==method]
            resources=[profiles[s,method] for s in matched]
            phases=[v for r in trials for v in r.get('worker_allocator_phases',{}).values()]+[r['memory'] for r in trials]
            stratum['methods'].append(dict(method=method,n_specimens=len(matched),n_trials=len(trials),
                median_reg_s=median(r['reg_time_s'] for r in trials),
                median_setup_s=median(r['setup_s'] for r in trials),
                median_full_optimization_warmup_s=median(r['warmup_s'] for r in trials),
                median_setup_plus_warmup_s=median(r['setup_s']+r['warmup_s'] for r in trials),
                median_worker_wall_s=median(p['worker_wall_s'] for p in resources),
                max_process_gpu_MiB=max(p['peak_process_gpu_bytes'] for p in resources)/2**20,
                max_process_tree_rss_MiB=max(p['peak_tree_rss_bytes'] for p in resources)/2**20,
                max_allocator_allocated_MiB=max(v['peak_allocated_MiB'] for v in phases),
                max_allocator_reserved_MiB=max(v['peak_reserved_MiB'] for v in phases),
                max_sample_gap_s=max(p['max_sample_gap_s'] for p in resources)))
        key=lambda r:(r['subject'],r['tier'],r['init_idx'],r['pair'])
        # Include a practical modern spatial comparator without changing the
        # protocol's prespecified reference or introducing a significance test.
        comparators=[reference]+([m for m in methods if m=='nanodrr-C64-FP32' and m!=reference])
        for comparator in comparators:
            refs={key(r):r for r in selected if r['method']==comparator}
            for method in methods:
                if method==comparator:continue
                paired=[(r,refs[key(r)]) for r in selected if r['method']==method]
                joint=[(a,b) for a,b in paired if a['final_mtre_mm']<threshold and b['final_mtre_mm']<threshold]
                stratum['comparisons'].append(dict(method=method,reference=comparator,
                    reference_role='prespecified' if comparator==reference else 'additional descriptive comparator',
                    n_specimens=len(matched),n_pairs=len(paired),n_joint_success=len(joint),
                    method_successes=sum(a['final_mtre_mm']<threshold for a,b in paired),
                    reference_successes=sum(b['final_mtre_mm']<threshold for a,b in paired),
                    median_warm_speed_ratio_all_pairs=median(b['reg_time_s']/a['reg_time_s'] for a,b in paired),
                    median_warm_speed_ratio_joint_success=median(b['reg_time_s']/a['reg_time_s'] for a,b in joint),
                    median_worker_wall_ratio=median(profiles[s,comparator]['worker_wall_s']/profiles[s,method]['worker_wall_s'] for s in matched),
                    median_process_gpu_ratio=median(profiles[s,comparator]['peak_process_gpu_bytes']/profiles[s,method]['peak_process_gpu_bytes'] for s in matched),
                    median_host_rss_ratio=median(profiles[s,comparator]['peak_tree_rss_bytes']/profiles[s,method]['peak_tree_rss_bytes'] for s in matched),
                    interpretation='Every ratio is comparator/method; values above1 favor the method. Joint-success ratios retain their stated selected-pair denominator.'))
        strata.append(stratum)
    cross=[s for s in all_subjects if len({device(s,m) for m in methods})>1]
    policy=dict(prespecified_reference=reference,additional_descriptive_comparators=[m for m in methods if m=='nanodrr-C64-FP32' and m!=reference],
        comparator_selection='Added after acquiring the frozen batch to contextualize gains against a faster spatial implementation; no confirmatory significance inference.',
        resource_matching='Complete specimen blocks within physical GPU UUID; all-worker maxima also reported separately.')
    return dict(status='complete descriptive device-stratified development replication',
        protocol_sha256=manifest['sha256'],code_identity=manifest['code_identity'],
        accuracy=accuracy,all_worker_memory=all_worker_memory,device_strata=strata,cross_device_specimens=cross,
        analysis_policy=policy,analysis_policy_sha256=hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        n_workers=len(profiles),n_trials=len(rows),
        excluded_from_matched_cost_workers=sum(s in cross for s,m in profiles),
        limitations=['Accuracy retains all six previously used development anatomies; no confirmatory inference.',
            'Cross-device specimens remain in accuracy and are excluded from complete-block resource comparisons.',
            'Device cohorts contain different anatomies; no between-device performance comparison is inferred.',
            'The additional nanodrr comparator and device-stratified reporting were chosen after observing the batch; all are descriptive.',
            'Warm registration excludes preparation; worker wall includes startup, data/provenance, one discarded optimization warmup and all four measured registrations.',
            'Setup plus warmup is an implementation estimate, not measured clinical first-use latency.',
            'Process peaks are sampled; allocated and reserved tensor peaks are separate quantities.',
            'Joint-success speed ratios describe selected successful pairs and do not establish equivalence.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=summarize_sessions(args.input)
    report['input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.input.glob('*.json'))}
    report['analysis_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),Path(__file__).with_name('summarize_journal_replication.py'))}
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(f'Validated {report["n_trials"]} trials; device strata remain separate')


if __name__=='__main__':main()
