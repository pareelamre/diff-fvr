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


def read_complete_batch(root, subjects=None):
    subjects = subjects or [f'subject{i:02d}' for i in range(1,7)]
    root=Path(root)
    if list(root.glob('*.failed.json')):
        raise ValueError('Failed workers present; report failures before paired inference')
    manifest=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    protocol=manifest['protocol']
    if manifest['sha256'] != protocol_hash(protocol):
        raise ValueError('Invalid protocol digest')
    rows=[]; gpu=set(); shared={}; data_hashes={}
    expected={(t[0],i+1) for t in protocol['tiers'] for i in range(protocol['inits_per_tier'])}
    for subject in subjects:
        for method in protocol['methods']:
            path=root/f'{subject}_{method}.json'
            if not path.is_file():
                raise ValueError(f'Missing expected result: {path.name}')
            data=json.loads(path.read_text(encoding='utf-8'))
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
    return manifest,rows,next(iter(gpu))


def summarize(root):
    manifest,rows,gpu=read_complete_batch(root)
    protocol=manifest['protocol']
    summaries=[]
    for method in protocol['methods']:
        selected=[r for r in rows if r['method']==method]
        errors=[r['final_mtre_mm'] for r in selected]
        summaries.append(dict(method=method,n_specimens=len({r['subject'] for r in selected}),
             n_trials=len(selected),median_mtre_mm=statistics.median(errors),
             successes_below_2mm=sum(e<2 for e in errors),
             mean_reg_s=statistics.mean(r['reg_time_s'] for r in selected),
             mean_setup_s=statistics.mean(r['setup_s'] for r in selected),
             mean_warmup_s=statistics.mean(r['warmup_s'] for r in selected),
             max_peak_allocated_MiB=max(r['memory']['peak_allocated_MiB'] for r in selected),
             max_peak_reserved_MiB=max(r['memory']['peak_reserved_MiB'] for r in selected)))
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
                analysis=protocol['analysis'],status='complete development replication',
                limitations=['previously used anatomies','known relative camera geometry',
                    'allocator peaks exclude driver/context memory','small-n exploratory inference'])


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
