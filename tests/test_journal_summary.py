"""A full batch, not the presence of a few JSON files, authorizes a summary."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.journal_protocol import protocol_hash
from experiments.summarize_journal_replication import read_complete_batch, practical_comparisons,summarize


class SummaryTests(unittest.TestCase):
    def batch(self, root):
        protocol={'methods':['a','b'],'tiers':[['bounded',0,0,0,0]],'inits_per_tier':1,
                  'analysis':{'reference':'a'}}
        identity={'git_commit':'fixture','git_status':'','code_sha256':{'fixture':'abc'}}
        manifest={'protocol':protocol,'sha256':protocol_hash(protocol),'code_identity':identity}
        (root/'protocol.json').write_text(json.dumps(manifest))
        for s in ['subject01','subject02']:
            for m in ['a','b']:
                row=dict(subject=s,method=m,tier='bounded',init_idx=1,pair='A+B',status='ok',
                         initial_delta=[[0]*6],candidate_sha256='same',final_mtre_mm=1,
                         reg_time_s=1,setup_s=2,warmup_s=3,
                         memory={'peak_allocated_MiB':4,'peak_reserved_MiB':5})
                result=dict(protocol=protocol,protocol_sha256=manifest['sha256'],
                            code_identity=identity,source_sha256={'CT':'abc'},gpu='fixture-GPU',rows=[row])
                (root/f'{s}_{m}.json').write_text(json.dumps(result))

    def test_requires_entire_expected_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.batch(root)
            _,rows,_=read_complete_batch(root,subjects=['subject01','subject02'])
            self.assertEqual(len(rows),4)
            for m in ['a','b']: (root/f'subject02_{m}.json').unlink()
            with self.assertRaisesRegex(ValueError,'Missing'):
                read_complete_batch(root,subjects=['subject01','subject02'])

    def test_rejects_duplicate_and_mismatched_initial_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.batch(root)
            p=root/'subject01_b.json'; data=json.loads(p.read_text())
            data['rows'][0]['initial_delta']=[[1]*6]; p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'initial'):
                read_complete_batch(root,subjects=['subject01','subject02'])
            data['rows']*=2; p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'roster'):
                read_complete_batch(root,subjects=['subject01','subject02'])

    def test_full_memory_protocol_rejects_missing_or_invalid_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.batch(root)
            manifest=json.loads((root/'protocol.json').read_text())
            manifest['protocol']['resources']={'required':True,'amortization_registrations':[1,10,100],
                                                'success_threshold_mm':.5}
            manifest['sha256']=protocol_hash(manifest['protocol'])
            (root/'protocol.json').write_text(json.dumps(manifest))
            for p in root.glob('subject*.json'):
                data=json.loads(p.read_text()); data['protocol']=manifest['protocol']
                data['protocol_sha256']=manifest['sha256']; p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'resource'):
                read_complete_batch(root,subjects=['subject01','subject02'])
            for p in root.glob('subject*.json'):
                profile=dict(schema='DFVR-RESOURCE-1',status='ok',returncode=0,gpu_required=True,
                    protocol_sha256=manifest['sha256'],result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    peak_process_gpu_bytes=1024,peak_root_rss_bytes=1024,peak_tree_rss_bytes=2048,
                    samples=10,worker_wall_s=7,requested_interval_s=.02,max_sample_gap_s=.03,
                    sampling_overhead_s=.1,gpu_identity={'name':'fixture-GPU','uuid':'fixture'})
                p.with_suffix('.resources.json').write_text(json.dumps(profile))
            _,rows,_=read_complete_batch(root,subjects=['subject01','subject02'])
            self.assertEqual(len(rows),4)
            report=summarize(root,subjects=['subject01','subject02'])
            self.assertEqual(report['summaries'][0]['successes_below_threshold'],0)
            self.assertEqual(report['summaries'][0]['estimated_amortized_s_per_registration']['1'],6.)
            self.assertEqual(report['summaries'][0]['estimated_amortized_s_per_registration']['10'],1.5)
            self.assertEqual(report['summaries'][0]['max_worker_process_gpu_MiB'],1024/2**20)
            p=root/'subject01_a.resources.json'; profile=json.loads(p.read_text())
            profile['peak_process_gpu_bytes']=None; p.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError,'resource'):
                read_complete_batch(root,subjects=['subject01','subject02'])

    def test_speed_comparison_retains_failures_and_labels_success_subset(self):
        rows=[dict(subject='s',tier='wide',init_idx=i,pair='A+B',method=m,
                   final_mtre_mm=e,reg_time_s=t) for i in (1,2)
              for m,e,t in [('reference',1.,10.),('new',1. if i==1 else 8.,2.)]]
        report=practical_comparisons(rows,'reference')[0]
        self.assertEqual(report['n_paired'],2)
        self.assertEqual(report['method_successes'],1)
        self.assertEqual(report['reference_successes'],2)
        self.assertEqual(report['n_joint_success'],1)
        self.assertEqual(report['median_warm_speed_ratio_joint_success'],5.)
        self.assertEqual(len(report['paired_trials']),2)

    def test_complete_aggregation_runs_and_does_not_certify_smoke_as_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.batch(root)
            report=summarize(root,subjects=['subject01','subject02'])
            self.assertEqual(len(report['summaries']),2)
            self.assertEqual(report['summaries'][0]['successes_below_threshold'],2)
            self.assertEqual(report['comparisons'][0]['mean_specimen_effect_mm_tested'],0.)
            manifest=json.loads((root/'protocol.json').read_text())
            manifest['protocol']['id']='DFVR-J34-SMOKE'
            manifest['sha256']=protocol_hash(manifest['protocol'])
            (root/'protocol.json').write_text(json.dumps(manifest))
            for p in root.glob('subject*.json'):
                data=json.loads(p.read_text());data['protocol']=manifest['protocol']
                data['protocol_sha256']=manifest['sha256'];p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Smoke output'):
                summarize(root,subjects=['subject01','subject02'])


if __name__=='__main__': unittest.main()
